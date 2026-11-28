"""VGG-16 (ImageNet) con su cabeza nativa: Flatten -> fc1 -> Dropout -> fc2 -> Dropout -> Dense softmax.

Se carga VGG16(include_top=True) para conservar fc1 y fc2 con sus pesos de ImageNet; la capa
`predictions` de 1000 clases se sustituye por Dense(n_classes, softmax, float32) y se añade
Dropout después de fc1 y de fc2. El backbone anidado contiene solo los cinco bloques
convolucionales (hasta block5_pool), así que en la fase A (backbone congelado) se entrenan
fc1, fc2 y la salida, y en la fase B todo.
"""

from __future__ import annotations

import keras
from keras.applications import vgg16

from src.models.common import (
    BACKBONE_NAME,
    check_application,
    input_shape,
    output_layer,
    preprocessed_input,
)

DENSE_LAYERS = ("fc1", "fc2")


def build(model_cfg: dict, n_classes: int, dropout: float) -> keras.Model:
    application = vgg16.VGG16
    check_application(application, model_cfg)
    full = application(
        include_top=True,
        weights=model_cfg.get("weights", "imagenet"),
        input_shape=input_shape(model_cfg),  # con include_top=True debe ser 224 x 224 x 3
    )
    backbone = keras.Model(full.inputs, full.get_layer("block5_pool").output, name=BACKBONE_NAME)
    backbone.trainable = False

    inputs, x = preprocessed_input(vgg16.preprocess_input, model_cfg)
    x = backbone(x, training=False)
    x = keras.layers.Flatten(name="flatten")(x)
    dense = {}
    for i, name in enumerate(DENSE_LAYERS, start=1):
        original = full.get_layer(name)
        dense[name] = keras.layers.Dense(original.units, activation=original.activation, name=name)
        x = dense[name](x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = output_layer(n_classes)(x)
    model = keras.Model(inputs, outputs, name=model_cfg["name"])
    for name, layer in dense.items():  # pesos de ImageNet de fc1 y fc2
        layer.set_weights(full.get_layer(name).get_weights())
    return model
