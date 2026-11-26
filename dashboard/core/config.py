"""Configuración del dashboard: rutas, límites, clases, fuentes y paletas."""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DASHBOARD_DATA_DIR", APP_DIR / "data"))

TITLE = "Lesiones pigmentadas de la piel"
SUBTITLE = "Clasificación de HAM10000 con aprendizaje profundo y constancia de color invariante"

# Carga de imágenes: se rechaza en el navegador (dcc.Upload.max_size) y en el servidor.
MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PIXELS = 40_000_000          # protección frente a imágenes descomprimidas enormes
MIN_SIDE = 64                    # lado mínimo en píxeles
ALLOWED_FORMATS = {"JPEG": "JPEG", "PNG": "PNG"}
SESSION_CACHE = 2                # modelos ONNX en memoria a la vez (lru_cache)
ORT_THREADS = int(os.environ.get("ORT_THREADS", "2"))

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Queratosis actínica / carcinoma intraepitelial",
    "bcc": "Carcinoma basocelular",
    "bkl": "Lesión queratósica benigna",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus melanocítico",
    "vasc": "Lesión vascular",
}
SOURCES = ["rosendahl", "vidir_modern", "vidir_molemax", "vienna_dias"]

# Paleta Okabe–Ito (apta para daltonismo), asignada de forma fija en todo el dashboard.
OKABE_ITO = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
CLASS_COLORS = {"akiec": OKABE_ITO["orange"], "bcc": OKABE_ITO["purple"], "bkl": OKABE_ITO["green"],
                "df": OKABE_ITO["yellow"], "mel": OKABE_ITO["vermillion"], "nv": OKABE_ITO["blue"],
                "vasc": OKABE_ITO["sky"]}
SOURCE_COLORS = {"rosendahl": OKABE_ITO["orange"], "vidir_modern": OKABE_ITO["sky"],
                 "vidir_molemax": OKABE_ITO["green"], "vienna_dias": OKABE_ITO["purple"]}
FAMILY_COLORS = {"CNN": OKABE_ITO["blue"], "Transformer": OKABE_ITO["orange"], "Original (CCI)": OKABE_ITO["vermillion"],
                 "Ablación (CCI)": OKABE_ITO["purple"], "Baseline": OKABE_ITO["sky"]}

# Interfaz: 5 colores (deben coincidir con las variables CSS de assets/style.css).
UI = {"dark": "#1B2631", "light": "#F6F5F1", "primary": "#0F5E6E", "accent": "#9C4A00", "alert": "#B3261E"}
FONT = "Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

KEY_FIGURE = {
    "value": "−25 puntos",
    "label": "de exactitud balanceada al evaluar en BCN20000 modelos entrenados en HAM10000",
    "source": "Pınar et al. (2026), Journal of Clinical Medicine 15(19), 7422",
}
