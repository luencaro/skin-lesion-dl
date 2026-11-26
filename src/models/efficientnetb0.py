"""EfficientNet-B0 (ImageNet) con su cabeza nativa: GAP -> Dropout -> Dense softmax.

La cabeza original de keras.applications.EfficientNetB0 es GlobalAveragePooling2D (avg_pool),
Dropout (top_dropout, 0,2) y la capa densa de 1000 clases; aquí el Dropout usa la tasa de los
hiperparámetros del modelo y la densa se sustituye por Dense(n_classes, softmax, float32). El
backbone (include_top=False) incluye la convolución 1 x 1 final (top_conv, top_bn,
top_activation). La normalización de la entrada va dentro de la aplicación (capas rescaling y
normalization), de modo que `efficientnet.preprocess_input` es la identidad.
"""

from __future__ import annotations

import keras
from keras.applications import efficientnet

from src.models.common import build_transfer_model


def build(model_cfg: dict, n_classes: int, dropout: float) -> keras.Model:
    return build_transfer_model(
        efficientnet.EfficientNetB0, efficientnet.preprocess_input, model_cfg, n_classes, dropout
    )
