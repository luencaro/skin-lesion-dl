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

from core.config import (
    BASELINES,
    BENCHMARK_MODELS,
    CCI_FAMILIES,
    DATA_DIR,
    MODEL_NAMES,
    ORIGINAL_MODEL,
    SOURCE_ORDER,
)


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


# ---------------------------------------------------------------------------
# Cifras derivadas (sin inventar nada: todo sale de los JSON exportados)
# ---------------------------------------------------------------------------


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def model_row(model: str) -> dict:
    """Cifras de un modelo para la tabla de líderes; None donde no hay dato.

    ba y f1 (exactitud balanceada y F1 macro medios en test), ece y GFLOPs/parámetros/latencia
    salen de models/leaderboard.json (tabla maestra y outputs/cost); el tamaño, del ONNX servido
    (metadata.json)."""
    board = {r["modelo"]: r for r in (leaderboard() or {}).get("rows", [])}
    r = board.get(model, {})
    art = {a["model"]: a for a in available_models()}.get(model, {})
    meta = model_metadata(model) or {}
    params = _num(r.get("params_total"))
    latency = _num(r.get("latencia_ms"))
    if latency is None:
        latency = _num((meta.get("dashboard") or {}).get("latency_ms_median"))
    return {
        "id": model, "name": MODEL_NAMES.get(model, model), "family": r.get("familia") or art.get("family"),
        "is_baseline": model in BASELINES or r.get("familia") == "Baseline",
        "ba": _num(r.get("ba_media")), "ba_std": _num(r.get("ba_std")), "runs": r.get("corridas"),
        "ece": _num(r.get("ece_media")),
        "f1": _num(r.get("f1_media")), "f1_std": _num(r.get("f1_std")),
        "latency_ms": latency, "size_mb": _num(art.get("size_mb") if art else meta.get("size_mb")),
        "params_m": params / 1e6 if params is not None else None, "gflops": _num(r.get("gflops")),
        "has_artifact": bool(art), "live": bool(art.get("live")) if art else False,
    }


def leaderboard_models() -> list[str]:
    """Los 8 modelos del benchmark y los 2 baselines, más cualquier otro modelo de la tabla
    maestra que no sea el original ni una ablación."""
    extra = [r["modelo"] for r in (leaderboard() or {}).get("rows", [])
             if r.get("familia") not in CCI_FAMILIES and r["modelo"] not in BENCHMARK_MODELS + BASELINES
             and r["modelo"] != ORIGINAL_MODEL]
    return BENCHMARK_MODELS + extra + BASELINES


def leaderboard_rows() -> list[dict]:
    return [model_row(m) for m in leaderboard_models()]


def ranked(rows: list[dict]) -> list[dict]:
    """Modelos (sin baselines) con exactitud balanceada, de mayor a menor."""
    return sorted([r for r in rows if r["ba"] is not None and not r["is_baseline"]], key=lambda r: -r["ba"])


def best_model() -> dict | None:
    top = ranked(leaderboard_rows())
    return top[0] if top else None


def ours() -> dict:
    """Modelo original (CCI) frente a su backbone, si hay resultados."""
    o = original_section()
    backbone = o.get("backbone")
    cci = model_row(ORIGINAL_MODEL)
    base = model_row(backbone) if backbone else None
    spread = o.get("source_spread") or {}
    return {"backbone": backbone, "backbone_name": MODEL_NAMES.get(backbone, backbone) if backbone else None,
            "cci": cci, "base": base, "has_runs": bool(o.get("has_runs")) and cci["ba"] is not None,
            "source_std": _num(spread.get(ORIGINAL_MODEL)), "backbone_source_std": _num(spread.get(backbone or ""))}


def demo_models() -> list[dict]:
    """Pastillas de la demo: el modelo original y los 8 del benchmark, con su disponibilidad."""
    arts = {a["model"]: a for a in available_models()}
    out = []
    for m in [ORIGINAL_MODEL, *BENCHMARK_MODELS]:
        a = arts.get(m)
        out.append({"id": m, "name": MODEL_NAMES.get(m, m), "available": a is not None,
                    "live": bool(a and a.get("live")), "cam": bool(a and a.get("cam")), "ours": m == ORIGINAL_MODEL})
    return out


def _srgb_to_lab(rgb) -> tuple[float, float, float]:
    """sRGB en [0, 1] -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = m @ lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return float(116 * f[1] - 16), float(500 * (f[0] - f[1])), float(200 * (f[1] - f[2]))


@lru_cache(maxsize=1)
def source_color() -> dict:
    """Color medio por fuente (media de las medias RGB por imagen de train de eda/images.json,
    las del notebook 06) en sRGB y en Lab. Vacío si el EDA se exportó sin color."""
    df = images()
    cols = ["media_r", "media_g", "media_b"]
    if not set(cols) <= set(df.columns):
        return {}
    out = {}
    for s in SOURCE_ORDER:
        sub = df[(df["dataset"] == s)][cols].dropna()
        if len(sub):
            rgb = sub.mean().to_numpy()
            out[s] = {"rgb": [float(v) for v in rgb], "lab": _srgb_to_lab(rgb), "n": len(sub),
                      "hex": "#" + "".join(f"{round(255 * float(v)):02X}" for v in np.clip(rgb, 0, 1))}
    return out
