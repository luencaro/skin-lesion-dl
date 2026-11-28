"""Configuración del dashboard (Cromia): rutas, límites, clases, fuentes, modelos y colores.

Los colores de clase y de interfaz se definen aquí una vez para Python y otra en
assets/cromia.css (variables --c-* y tokens); deben coincidir.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DASHBOARD_DATA_DIR", APP_DIR / "data"))
ASSETS_DIR = APP_DIR / "assets"

BRAND = "Cromia"
TITLE = "Cromia · Lesiones de la piel"
SUBTITLE = "Clasificación de lesiones cutáneas de HAM10000 con aprendizaje profundo y constancia de color invariante"
DISCLAIMER = ("Herramienta académica de demostración.",
              "No es un dispositivo médico ni sustituye el diagnóstico de un dermatólogo.")

# Carga de imágenes: se rechaza en el navegador (dcc.Upload.max_size) y en el servidor.
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PIXELS = 40_000_000          # protección frente a imágenes descomprimidas enormes
MIN_SIDE = 64                    # lado mínimo en píxeles
ALLOWED_FORMATS = {"JPEG": "JPEG", "PNG": "PNG"}
SESSION_CACHE = 2                # modelos ONNX en memoria a la vez (lru_cache)
ORT_THREADS = int(os.environ.get("ORT_THREADS", "2"))
INPUT_SIZE = 224                 # todos los modelos (también Inception-V3) usan 224 x 224

# La capa CCI va dentro del grafo ONNX del modelo original (src/export.py exporta el modelo
# completo, capa incluida): el dashboard no la aplica por su cuenta.
CCI_IN_GRAPH = True

# ---------------------------------------------------------------------------
# Clases (orden de presentación) y colores Okabe-Ito
# ---------------------------------------------------------------------------

CLASS_ORDER = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_COLORS = {"mel": "#D55E00", "nv": "#0072B2", "bcc": "#E69F00", "akiec": "#CC79A7",
                "bkl": "#56B4E9", "df": "#009E73", "vasc": "#F0E442"}
CLASS_NAMES = {"mel": "Melanoma", "nv": "Nevus melanocítico", "bcc": "Carcinoma basocelular",
               "akiec": "Queratosis actínica / Bowen", "bkl": "Lesión queratósica benigna",
               "df": "Dermatofibroma", "vasc": "Lesión vascular"}
CLASS_SHORT = {c: c.upper() for c in CLASS_ORDER}

# Fuentes de adquisición (columna dataset de HAM10000) y su color de serie (--s1..--s4)
SOURCE_ORDER = ["vidir_molemax", "vidir_modern", "rosendahl", "vienna_dias"]
SOURCE_NAMES = {"vidir_molemax": "ViDIR MoleMax", "vidir_modern": "ViDIR moderno", "rosendahl": "Rosendahl",
                "vienna_dias": "Viena DIAS"}
SOURCE_VARS = {"vidir_molemax": "var(--s1)", "vidir_modern": "var(--s2)", "rosendahl": "var(--s3)",
               "vienna_dias": "var(--s4)"}

SEX_NAMES = {"male": "Hombre", "female": "Mujer", "unknown": "Sin dato"}
LOCALIZATION_NAMES = {
    "back": "Espalda", "lower extremity": "Extremidad inferior", "trunk": "Tronco",
    "upper extremity": "Extremidad superior", "abdomen": "Abdomen", "face": "Cara", "chest": "Pecho",
    "foot": "Pie", "unknown": "Desconocida", "neck": "Cuello", "scalp": "Cuero cabelludo", "hand": "Mano",
    "ear": "Oreja", "genital": "Genital", "acral": "Acral",
}

# ---------------------------------------------------------------------------
# Modelos (identificadores de src.models.REGISTRY y de configs/base.yaml)
# ---------------------------------------------------------------------------

ORIGINAL_MODEL = "cci"
BENCHMARK_MODELS = ["mobilenetv2", "resnet50", "vgg16", "densenet201", "inceptionv3", "convnext_tiny",
                    "vit_b16", "efficientnetb0"]
BASELINES = ["trivial", "logreg"]
MODEL_NAMES = {
    "cci": "Nuestro modelo", "mobilenetv2": "MobileNetV2", "resnet50": "ResNet-50", "vgg16": "VGG-16",
    "densenet201": "DenseNet-201", "inceptionv3": "Inception-V3", "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b16": "ViT-B/16", "efficientnetb0": "EfficientNet-B0",
    "trivial": "Clase mayoritaria", "logreg": "Regresión logística",
}
CCI_FAMILIES = {"Original (CCI)", "Ablación (CCI)"}

# Umbrales de la etiqueta de confianza de la demo (probabilidad top-1)
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.5
DEFAULT_ALPHA = "0.1"            # cobertura objetivo del conjunto conforme: 1 − α = 90 %

# Mapa de atención: azul cielo -> amarillo -> naranja -> bermellón (Okabe-Ito, sin jet)
HEATMAP_STOPS = ["#56B4E9", "#F0E442", "#E69F00", "#D55E00"]

FONT = "Inter, system-ui, sans-serif"


def model_name(model_id: str | None) -> str:
    if not model_id:
        return ""
    return MODEL_NAMES.get(model_id, model_id)
