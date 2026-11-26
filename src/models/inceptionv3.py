"""Inception-V3 (ImageNet) con su cabeza nativa: GAP -> Dropout -> Dense softmax.

La cabeza original de keras.applications.InceptionV3 es GlobalAveragePooling2D (avg_pool) seguido
de la capa densa de 1000 clases; aquí la densa se sustituye por Dense(n_classes, softmax, float32)
y se añade Dropout antes de ella. La resolución nativa es 299 x 299; se usa 224 x 224 como en el
resto del benchmark (include_top=False admite entradas de al menos 75 x 75), así que el último
módulo, `mixed10`, produce un mapa de 5 x 5.
"""

from __future__ import annotations

import keras
from keras.applications import inception_v3

from src.models.common import build_transfer_model


def build(model_cfg: dict, n_classes: int, dropout: float) -> keras.Model:
    return build_transfer_model(
        inception_v3.InceptionV3, inception_v3.preprocess_input, model_cfg, n_classes, dropout
    )
