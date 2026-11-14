"""MobileNetV2 (ImageNet) con cabeza GAP -> Dropout -> Dense softmax."""

from __future__ import annotations

import keras
from keras.applications import mobilenet_v2

from src.models.common import build_transfer_model


def build(model_cfg: dict, n_classes: int, dropout: float) -> keras.Model:
    return build_transfer_model(
        mobilenet_v2.MobileNetV2, mobilenet_v2.preprocess_input, model_cfg, n_classes, dropout
    )
