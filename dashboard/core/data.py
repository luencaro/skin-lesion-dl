"""Acceso a los datos precalculados de dashboard/data/ (solo lectura, en memoria).

Todas las funciones toleran archivos ausentes: devuelven None o estructuras vacías, de modo que
el dashboard arranca con el EDA aunque aún no haya resultados de entrenamiento.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import DATA_DIR


def _read_json(rel: str):
    path = DATA_DIR / rel
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def manifest() -> dict:
    return _read_json("manifest.json") or {}


@lru_cache(maxsize=1)
def eda_summary() -> dict:
    return _read_json("eda/summary.json") or {}


@lru_cache(maxsize=1)
def images() -> pd.DataFrame:
    """Una fila por imagen: metadatos, partición y color (solo train)."""
    raw = _read_json("eda/images.json")
    if not raw:
        return pd.DataFrame(columns=["image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization",
                                     "dataset", "split", "fold"])
    df = pd.DataFrame(raw["data"], columns=raw["columns"])
    for c in df.columns:
        if c not in ("image_id", "lesion_id", "dx", "dx_type", "sex", "localization", "dataset", "split"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def interpretation(key: str) -> dict | None:
    item = (eda_summary().get("interpretations") or {}).get(key)
    return item if item and item.get("text") else None


@lru_cache(maxsize=1)
def models_index() -> dict:
    return _read_json("models/index.json") or {"status": "pendiente", "artifacts": [], "models_with_runs": []}


def results_ready() -> bool:
    return models_index().get("status") == "listo"


@lru_cache(maxsize=1)
def leaderboard() -> dict | None:
    return _read_json("models/leaderboard.json")


@lru_cache(maxsize=1)
def global_comparison() -> dict | None:
    return _read_json("models/global.json")


@lru_cache(maxsize=1)
def original_section() -> dict:
    return _read_json("models/original.json") or {}


@lru_cache(maxsize=16)
def model_metadata(model: str) -> dict | None:
    return _read_json(f"models/{model}/metadata.json")


@lru_cache(maxsize=16)
def diagnostics(model: str) -> dict | None:
    return _read_json(f"models/{model}/diagnostics.json")


@lru_cache(maxsize=8)
def precomputed(model: str) -> dict | None:
    return _read_json(f"models/{model}/precomputed.json")


@lru_cache(maxsize=8)
def ood_reference(model: str) -> np.ndarray | None:
    path = DATA_DIR / "models" / model / "ood_reference.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return z["embeddings"].astype(np.float32)


def model_file(model: str) -> Path | None:
    path = DATA_DIR / "models" / model / "model.onnx"
    return path if path.exists() else None


def available_models() -> list[dict]:
    """Modelos con artefactos: [{"model", "family", "live", "cam", "size_mb"}]."""
    return list(models_index().get("artifacts") or [])


@lru_cache(maxsize=1)
def examples() -> list[dict]:
    return _read_json("examples/examples.json") or []


def example_path(image_id: str) -> Path | None:
    for ex in examples():
        if ex["image_id"] == image_id:
            p = DATA_DIR / "examples" / ex["file"]
            return p if p.exists() else None
    return None


@lru_cache(maxsize=4)
def content(name: str) -> str | None:
    path = DATA_DIR / "content" / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def media_path(kind: str, name: str) -> Path | None:
    """Ruta de una imagen servible (whitelist de carpetas y extensión)."""
    folders = {"gallery": DATA_DIR / "eda" / "gallery", "examples": DATA_DIR / "examples"}
    if kind not in folders or "/" in name or "\\" in name or not name.endswith(".jpg"):
        return None
    path = (folders[kind] / name).resolve()
    if folders[kind].resolve() not in path.parents or not path.exists():
        return None
    return path
