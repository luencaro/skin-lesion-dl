"""Constructor común de clasificadores por transferencia sobre keras.applications.

Todos los modelos comparten la misma estructura exterior:

    Input float32 en [0, 255] -> Lambda(preprocess_input) -> backbone (submodelo anidado,
    llamado con training=False) -> cabeza -> Dense(n_classes, softmax, float32)

El backbone es un submodelo con nombre BACKBONE_NAME que contiene solo los bloques
convolucionales de la aplicación; las capas de la cabeza van después, en una cadena lineal.
Grad-CAM (src/gradcam.py) se apoya en esa estructura.
"""

from __future__ import annotations

from collections.abc import Callable

import keras
import numpy as np

BACKBONE_NAME = "backbone"
PREPROCESS_NAME = "preprocess"
OUTPUT_NAME = "predictions"


def check_application(application: Callable, model_cfg: dict) -> None:
    """Error si configs/<modelo>.yaml declara otra keras_application."""
    expected = model_cfg.get("keras_application")
    if expected and expected != application.__name__:
        raise ValueError(
            f"{model_cfg['name']}: keras_application={expected} no coincide con {application.__name__}"
        )


def input_shape(model_cfg: dict) -> tuple[int, int, int]:
    size = int(model_cfg["input_size"])
    return (size, size, 3)


def preprocessed_input(preprocess_input: Callable, model_cfg: dict):
    """(inputs, x): entrada float32 en [0, 255] y su salida tras preprocess_input (dentro del modelo)."""
    inputs = keras.Input(shape=input_shape(model_cfg), dtype="float32", name="image")
    x = keras.layers.Lambda(preprocess_input, dtype="float32", name=PREPROCESS_NAME)(inputs)
    return inputs, x


def output_layer(n_classes: int) -> keras.layers.Dense:
    """Capa de salida: Dense(n_classes, softmax) en float32 (estable con precisión mixta)."""
    return keras.layers.Dense(n_classes, activation="softmax", dtype="float32", name=OUTPUT_NAME)


def build_transfer_model(
    application: Callable,
    preprocess_input: Callable,
    model_cfg: dict,
    n_classes: int,
    dropout: float,
) -> keras.Model:
    """Entrada float32 en [0, 255] -> preprocess_input -> backbone(training=False) ->
    GlobalAveragePooling2D -> Dropout -> Dense(n_classes, softmax, float32).

    El backbone se llama con training=False, así que BatchNormalization queda en modo
    inferencia también cuando se descongela. El backbone empieza congelado.
    """
    check_application(application, model_cfg)
    backbone = application(
        include_top=False,
        weights=model_cfg.get("weights", "imagenet"),
        input_shape=input_shape(model_cfg),
        name=BACKBONE_NAME,
    )
    backbone.trainable = False

    inputs, x = preprocessed_input(preprocess_input, model_cfg)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = output_layer(n_classes)(x)
    return keras.Model(inputs, outputs, name=model_cfg["name"])


def application_backbone(application: Callable, model_cfg: dict, last_layer: str | None = None,
                         **kwargs) -> tuple[keras.Model, keras.Model]:
    """(backbone, full): carga `application` con su cabeza original (include_top=True, para
    conservar las capas con pesos de ImageNet de la cabeza) y define el backbone anidado como el
    submodelo que termina en la entrada de `last_layer` (la primera capa de la cabeza). El
    backbone empieza congelado. `full` se devuelve para copiar los pesos de la cabeza."""
    check_application(application, model_cfg)
    full = application(include_top=True, weights=model_cfg.get("weights", "imagenet"),
                       input_shape=input_shape(model_cfg), **kwargs)
    backbone = keras.Model(full.inputs, full.get_layer(last_layer).input, name=BACKBONE_NAME)
    backbone.trainable = False
    return backbone, full


def get_backbone(model: keras.Model) -> keras.Model:
    return model.get_layer(BACKBONE_NAME)


def set_backbone_trainable(model: keras.Model, trainable: bool) -> None:
    """Congela (False) o descongela (True) todo el backbone. Hay que recompilar después.
    Las capas de la cabeza (fuera del backbone) quedan siempre entrenables."""
    get_backbone(model).trainable = trainable


def pre_backbone_layers(model: keras.Model) -> list:
    """Capas entre la entrada y el backbone, en orden: preprocess en los modelos del benchmark;
    la capa CCI y preprocess en el modelo original (src/original_model.py)."""
    names = [layer.name for layer in model.layers]
    return [layer for layer in model.layers[:names.index(BACKBONE_NAME)]
            if not isinstance(layer, keras.layers.InputLayer)]


def apply_pre_backbone(model: keras.Model, x):
    """Entrada del backbone para las imágenes `x` (float32 en [0, 255])."""
    for layer in pre_backbone_layers(model):
        x = layer(x)
    return x


def head_layers(model: keras.Model) -> list:
    """Capas posteriores al backbone, en orden (la cabeza, terminada en la salida)."""
    names = [layer.name for layer in model.layers]
    return model.layers[names.index(BACKBONE_NAME) + 1:]


def count_params(model: keras.Model) -> dict:
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    return {"trainable": trainable, "non_trainable": non_trainable, "total": trainable + non_trainable}
