"""ConvNeXt-Tiny (ImageNet) con su cabeza nativa: GAP -> LayerNorm -> Dropout -> Dense softmax.

Se carga ConvNeXtTiny(include_top=True) para conservar la normalización de capa de la cabeza
(`convnext_tiny_head_layernorm`) con sus pesos de ImageNet; la capa densa de 1000 clases se
sustituye por Dense(n_classes, softmax, float32) y se añade Dropout antes de ella. El backbone
anidado termina en la salida de la última etapa (la entrada de `convnext_tiny_head_gap`). La
normalización de la entrada va dentro de la aplicación (include_preprocessing=True), de modo que
`convnext.preprocess_input` es la identidad.
"""

from __future__ import annotations

import keras
from keras.applications import convnext

from src.models.common import application_backbone, output_layer, preprocessed_input

PREFIX = "convnext_tiny"


def build(model_cfg: dict, n_classes: int, dropout: float) -> keras.Model:
    backbone, full = application_backbone(convnext.ConvNeXtTiny, model_cfg, f"{PREFIX}_head_gap")
    original_ln = full.get_layer(f"{PREFIX}_head_layernorm")

    inputs, x = preprocessed_input(convnext.preprocess_input, model_cfg)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    layer_norm = keras.layers.LayerNormalization(epsilon=original_ln.epsilon, name="head_layernorm")
    x = layer_norm(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = output_layer(n_classes)(x)
    model = keras.Model(inputs, outputs, name=model_cfg["name"])
    layer_norm.set_weights(original_ln.get_weights())  # pesos de ImageNet de la normalización de la cabeza
    return model
