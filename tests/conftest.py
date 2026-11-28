"""Utilidades comunes de las pruebas: solo CPU, float32 y un clasificador diminuto con la misma
estructura que los modelos del REGISTRY (Input -> preprocess -> backbone -> GAP -> Dropout ->
Dense softmax), para no descargar pesos de ImageNet."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # solo CPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np
import pytest

SIZE = 32          # lado de las imágenes de prueba
REDUCED = 8        # reduced_size de la capa en las pruebas (32 = 4 x 8, como 224 = 4 x 56)
N_CLASSES = 4
LAYER_CFG = {"reduced_size": REDUCED, "filters": 4, "kernel_size": 3, "eps": 1e-3, "kappa": 170.0,
             "p_init": 6.0, "input_range": [0.0, 255.0]}


@pytest.fixture(autouse=True)
def _float32_session():
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("float32")
    yield
    keras.backend.clear_session()


def tiny_classifier(n_classes: int = N_CLASSES, size: int = SIZE, dropout: float = 0.0, name: str = "tiny",
                    layernorm: bool = False) -> keras.Model:
    """Clasificador diminuto con la estructura de src/models/common.py."""
    from src.models.common import BACKBONE_NAME, OUTPUT_NAME, PREPROCESS_NAME

    inner_in = keras.Input((size, size, 3))
    h = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inner_in)
    h = keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(h)
    backbone = keras.Model(inner_in, h, name=BACKBONE_NAME)

    inputs = keras.Input((size, size, 3), dtype="float32", name="image")
    x = keras.layers.Rescaling(1 / 127.5, offset=-1.0, name=PREPROCESS_NAME)(inputs)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    if layernorm:
        x = keras.layers.LayerNormalization(name="head_layernorm")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax", dtype="float32", name=OUTPUT_NAME)(x)
    return keras.Model(inputs, outputs, name=name)


def tiny_cci(variant: str = "full", **kw) -> keras.Model:
    from src.original_model import build_cci_model

    return build_cci_model(tiny_classifier(**kw), variant, layer_cfg=LAYER_CFG)


def randomize_weight_net(model, seed: int = 0, scale: float = 0.5) -> None:
    """Pesos aleatorios en la última convolución de g_φ (que empieza en cero), para que los pesos
    espaciales no sean uniformes."""
    from src.original_model import CCI_NAME

    layer = model.get_layer(CCI_NAME)
    if layer.weight_net is None:
        return
    rng = np.random.default_rng(seed)
    for w in layer.weight_net.layers[-1].weights:
        w.assign(rng.normal(0.0, scale, size=w.shape).astype("float32"))


def sharpen_head(model, factor: float = 50.0) -> None:
    """Multiplica los pesos de la capa de salida: un clasificador entrenado es mucho más sensible a
    su entrada que uno recién inicializado, y así la prueba de invarianza es más exigente."""
    kernel = model.get_layer("predictions").kernel
    kernel.assign(kernel.numpy() * factor)


def positive_images(n: int = 4, size: int = SIZE, seed: int = 0, low: float = 20.0, high: float = 120.0) -> np.ndarray:
    """Imágenes sin ceros ni saturación: con ganancias en [0,5; 2] quedan dentro de [10, 240]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, size, size, 3)).astype("float32")


def structured_images(n: int = 4, size: int = SIZE, seed: int = 0) -> np.ndarray:
    """Imágenes con regiones de colores distintos (una "lesión" oscura y rojiza sobre un fondo más
    claro), en [20, 120]: los pesos espaciales cambian el iluminante estimado."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    out = []
    for _ in range(n):
        cy, cx, r = rng.uniform(0.3, 0.7) * size, rng.uniform(0.3, 0.7) * size, rng.uniform(0.15, 0.35) * size
        lesion = ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2)[..., None]
        skin = rng.uniform(80, 115, size=3)
        spot = rng.uniform(25, 60, size=3)
        img = np.where(lesion, spot, skin) + rng.normal(0, 3, (size, size, 3))
        out.append(np.clip(img, 20, 120))
    return np.asarray(out, dtype="float32")


def log_uniform_gains(n: int, seed: int = 1, low: float = 0.5, high: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(low), np.log(high), size=(n, 1, 1, 3))).astype("float32")


def pattern_dataset(n_per_class: int = 4, size: int = SIZE, seed: int = 0):
    """Lote sintético de 4 clases que se distinguen por la textura (no por el color, que la capa
    elimina): rayas horizontales, verticales, tablero y diagonales, con ruido."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    patterns = [(yy // 2) % 2, (xx // 2) % 2, ((yy // 2) + (xx // 2)) % 2, ((yy + xx) // 3) % 2]
    xs, ys = [], []
    for k, pat in enumerate(patterns):
        for _ in range(n_per_class):
            base = 60 + 120 * pat[..., None] + rng.normal(0, 8, (size, size, 3))
            gain = rng.uniform(0.7, 1.2, size=(1, 1, 3))
            xs.append(np.clip(base * gain, 5, 250))
            ys.append(k)
    return np.asarray(xs, dtype="float32"), np.asarray(ys, dtype="int32")
