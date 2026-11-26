"""EfficientNet-B0 (ImageNet) con su cabeza nativa: GAP -> Dropout -> Dense softmax.

La cabeza original de keras.applications.EfficientNetB0 es GlobalAveragePooling2D (avg_pool),
Dropout (top_dropout, 0,2) y la capa densa de 1000 clases; aquí el Dropout usa la tasa de los
hiperparámetros del modelo y la densa se sustituye por Dense(n_classes, softmax, float32). El
backbone (include_top=False) incluye la convolución 1 x 1 final (top_conv, top_bn,
top_activation). La normalización de la entrada va dentro de la aplicación (capas rescaling y
normalization), de modo que `efficientnet.preprocess_input` es la identidad.

Keras compone el nombre del archivo de pesos de EfficientNet con el nombre del modelo
(`name + "_notop.h5"`), y el backbone se llama BACKBONE_NAME, así que la aplicación se construye
sin pesos y los de ImageNet se cargan después con el archivo y el hash de B0.
"""

from __future__ import annotations

import keras
from keras.applications import efficientnet
from keras.src.applications import efficientnet as _efficientnet_src

from src.models.common import build_transfer_model

WEIGHTS_NAME = "efficientnetb0"


def weights_file(include_top: bool) -> tuple[str, str, str]:
    """(nombre del archivo, URL, hash MD5) de los pesos de ImageNet de EfficientNet-B0."""
    file_name = WEIGHTS_NAME + (".h5" if include_top else "_notop.h5")
    file_hash = _efficientnet_src.WEIGHTS_HASHES["b0"][0 if include_top else 1]
    return file_name, _efficientnet_src.BASE_WEIGHTS_PATH + file_name, file_hash


def EfficientNetB0(include_top=True, weights="imagenet", input_shape=None, name=WEIGHTS_NAME, **kwargs):
    """keras.applications.EfficientNetB0 con cualquier `name` y los pesos de ImageNet de B0."""
    model = efficientnet.EfficientNetB0(include_top=include_top, weights=None, input_shape=input_shape,
                                        name=name, **kwargs)
    if weights == "imagenet":
        file_name, url, file_hash = weights_file(include_top)
        model.load_weights(keras.utils.get_file(file_name, url, cache_subdir="models", file_hash=file_hash))
    elif weights is not None:
        model.load_weights(weights)
    return model


def build(model_cfg: dict, n_classes: int, dropout: float) -> keras.Model:
    return build_transfer_model(EfficientNetB0, efficientnet.preprocess_input, model_cfg, n_classes, dropout)
