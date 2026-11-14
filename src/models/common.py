"""Constructor común de clasificadores por transferencia sobre keras.applications."""

from __future__ import annotations

from collections.abc import Callable

import keras
import numpy as np

BACKBONE_NAME = "backbone"


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
    expected = model_cfg.get("keras_application")
    if expected and expected != application.__name__:
        raise ValueError(
            f"{model_cfg['name']}: keras_application={expected} no coincide con {application.__name__}"
        )
    size = int(model_cfg["input_size"])
    shape = (size, size, 3)
    backbone = application(
        include_top=False,
        weights=model_cfg.get("weights", "imagenet"),
        input_shape=shape,
        name=BACKBONE_NAME,
    )
    backbone.trainable = False

    inputs = keras.Input(shape=shape, dtype="float32", name="image")
    x = keras.layers.Lambda(preprocess_input, dtype="float32", name="preprocess")(inputs)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inputs, outputs, name=model_cfg["name"])


def get_backbone(model: keras.Model) -> keras.Model:
    return model.get_layer(BACKBONE_NAME)


def set_backbone_trainable(model: keras.Model, trainable: bool) -> None:
    """Congela (False) o descongela (True) todo el backbone. Hay que recompilar después."""
    get_backbone(model).trainable = trainable


def count_params(model: keras.Model) -> dict:
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    return {"trainable": trainable, "non_trainable": non_trainable, "total": trainable + non_trainable}
