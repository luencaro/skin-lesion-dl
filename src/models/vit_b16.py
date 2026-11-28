"""ViT-B/16 (keras_hub, ImageNet) con su cabeza nativa: token de clase -> Dropout -> Dense softmax.

El backbone es `keras_hub.models.ViTBackbone` del preset `keras_hub_preset` de configs/vit_b16.yaml
(pesos de ImageNet; con weights: null se construye la misma arquitectura sin descargar pesos). Su
salida es la secuencia de 197 tokens (1 de clase + 14 x 14 parches) tras la normalización de capa
final del encoder; la clasificación se hace sobre el token de clase, como la cabeza original
(ViTImageClassifier con pooling="token"), con la densa de 1000 clases sustituida por
Dense(n_classes, softmax, float32) y Dropout antes de ella.

El preprocesamiento del preset (ViTImageConverter: escala a [-1, 1]) va dentro del modelo como
capa `preprocess`, igual que preprocess_input en los modelos de keras.applications. El backbone se
llama con training=False (desactiva el dropout interno, que en el preset es 0) y tiene el nombre
BACKBONE_NAME, así que set_backbone_trainable y el resto del pipeline funcionan sin cambios.
Grad-CAM no aplica: src/interpret.py usa attention rollout sobre los bloques del encoder.
"""

from __future__ import annotations

import keras

from src.models.common import BACKBONE_NAME, PREPROCESS_NAME, input_shape, output_layer

CLASS_TOKEN_NAME = "class_token"
EMBEDDING_LAYER = "vit_patching_and_embedding"
ENCODER_LAYER = "vit_encoder"


def _preset(model_cfg: dict) -> str:
    preset = model_cfg.get("keras_hub_preset")
    if not preset:
        raise ValueError(f"configs/{model_cfg['name']}.yaml no define keras_hub_preset")
    return preset


def load_backbone(model_cfg: dict):
    """ViTBackbone del preset, con pesos de ImageNet si weights == "imagenet" (sin pesos si es null)."""
    import keras_hub

    load_weights = model_cfg.get("weights", "imagenet") is not None
    backbone = keras_hub.models.ViTBackbone.from_preset(_preset(model_cfg), load_weights=load_weights,
                                                        name=BACKBONE_NAME)
    expected = tuple(input_shape(model_cfg))
    if tuple(backbone.image_shape) != expected:
        raise ValueError(f"El preset {_preset(model_cfg)} espera {backbone.image_shape}, no {expected}")
    return backbone


def load_converter(model_cfg: dict):
    """ViTImageConverter del preset (resize a la resolución del preset, escala y desplazamiento)
    en float32 y con el nombre PREPROCESS_NAME."""
    import keras_hub

    ref = keras_hub.layers.ViTImageConverter.from_preset(_preset(model_cfg))
    config = {**ref.get_config(), "name": PREPROCESS_NAME, "dtype": "float32"}
    return keras_hub.layers.ViTImageConverter.from_config(config)


def build(model_cfg: dict, n_classes: int, dropout: float) -> keras.Model:
    backbone = load_backbone(model_cfg)
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape(model_cfg), dtype="float32", name="image")
    x = load_converter(model_cfg)(inputs)
    x = backbone(x, training=False)
    x = keras.layers.Lambda(lambda t: t[:, 0, :], name=CLASS_TOKEN_NAME)(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = output_layer(n_classes)(x)
    return keras.Model(inputs, outputs, name=model_cfg["name"])
