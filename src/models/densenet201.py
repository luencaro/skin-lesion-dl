"""DenseNet-201 (ImageNet) con su cabeza nativa: GAP -> Dropout -> Dense softmax.

La cabeza original de keras.applications.DenseNet201 es GlobalAveragePooling2D seguido de la
capa densa de 1000 clases; aquí la densa se sustituye por Dense(n_classes, softmax, float32) y se
añade Dropout antes de ella. El backbone (include_top=False) termina en la capa `relu` que sigue a
la última normalización por lotes.
"""

from __future__ import annotations

import keras
from keras.applications import densenet

from src.models.common import build_transfer_model


def build(model_cfg: dict, n_classes: int, dropout: float) -> keras.Model:
    return build_transfer_model(densenet.DenseNet201, densenet.preprocess_input, model_cfg, n_classes, dropout)
