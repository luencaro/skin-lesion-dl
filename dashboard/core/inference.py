"""Inferencia en CPU con ONNX Runtime: validación de la imagen, preprocesamiento, probabilidades
calibradas, conjunto conforme, CAM y aviso fuera de dominio.

Pipeline: imagen (en memoria) -> RGB -> 224 x 224 bilineal -> float32 en [0, 255] -> ONNX. La
normalización de cada backbone y, en el modelo original, la capa CCI van dentro del grafo
(config.CCI_IN_GRAPH), así que aquí no se aplican.

- CAM: el ONNX de los modelos con cabeza GAP -> [LayerNorm] -> lineal expone `feature_map` y
  `head_weights` (src/export.py) y el mapa es Σ_c A[..., c] W[c, k] (equivale a Grad-CAM en esa
  arquitectura). Los modelos sin esa cabeza (ViT-B/16: su attention rollout solo existe en el
  pipeline de TensorFlow, src/interpret.py) no traen mapa.
- Conjunto conforme: LAC, {k : 1 − p_k <= q̂(α)} con los cuantiles de metadata.json.
- Fuera de distribución: distancia coseno media a los k vecinos de train en el espacio de
  embeddings frente al umbral fijado en validación (metadata.json y ood_reference.npz).

Las imágenes solo se procesan en memoria: no se escriben en disco ni se registran.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError

from core import data
from core.config import (
    ALLOWED_FORMATS,
    HEATMAP_STOPS,
    INPUT_SIZE,
    MAX_PIXELS,
    MAX_UPLOAD_BYTES,
    MIN_SIDE,
    ORT_THREADS,
    SESSION_CACHE,
)

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
log = logging.getLogger(__name__)
_OOD_WARNED: set[str] = set()


class InvalidImage(ValueError):
    """Imagen rechazada (formato, tamaño o contenido); el mensaje se muestra al usuario."""


# ---------------------------------------------------------------------------
# Imagen
# ---------------------------------------------------------------------------


def decode_upload(contents: str) -> bytes:
    """Bytes de un data URL de dcc.Upload ("data:image/jpeg;base64,...")."""
    if not contents or "," not in contents:
        raise InvalidImage("No se recibió ninguna imagen.")
    header, b64 = contents.split(",", 1)
    if not header.startswith("data:image/"):
        raise InvalidImage("El archivo no es una imagen. Usa JPEG o PNG.")
    if len(b64) * 3 / 4 > MAX_UPLOAD_BYTES * 1.01:
        raise InvalidImage(f"La imagen supera el límite de {MAX_UPLOAD_BYTES // 2 ** 20} MB.")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise InvalidImage("No se pudo leer el archivo.") from e
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InvalidImage(f"La imagen supera el límite de {MAX_UPLOAD_BYTES // 2 ** 20} MB.")
    return raw


def open_image(raw: bytes) -> Image.Image:
    """Abre y valida una imagen JPEG o PNG; devuelve RGB de 8 bits."""
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            fmt = probe.format
            probe.verify()
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise InvalidImage("El archivo no es una imagen JPEG o PNG válida.") from e
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImage(f"Formato {fmt} no admitido. Usa JPEG o PNG.")
    if min(img.size) < MIN_SIDE:
        raise InvalidImage(f"La imagen es demasiado pequeña ({img.size[0]} × {img.size[1]}); el lado mínimo es {MIN_SIDE} px.")
    return img.convert("RGB")


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """(1, size, size, 3) float32 en [0, 255]: redimensionado bilineal con antialias y sin conservar
    la relación de aspecto, como en el entrenamiento (la normalización va dentro del modelo)."""
    resized = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 255.0)[None]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


@lru_cache(maxsize=SESSION_CACHE)
def session(model: str):
    """Sesión de ONNX Runtime en CPU (se carga bajo demanda; como máximo SESSION_CACHE a la vez)."""
    import onnxruntime as ort

    path = data.model_file(model)
    if path is None:
        raise FileNotFoundError(f"El modelo {model} no tiene un archivo ONNX servible.")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_THREADS
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def conformal_sets(probs: np.ndarray, meta: dict, classes) -> dict:
    """{alpha: [clases]} con 1 − p_k <= q̂(alpha) (LAC, cuantiles de metadata.json)."""
    conf = (meta.get("conformal") or {}).get("quantiles") or {}
    return {a: [c for c, p in zip(classes, probs) if 1.0 - p <= float(q) + 1e-12] for a, q in conf.items()}


def cam_maps(feature_map: np.ndarray, head_weights: np.ndarray, k: int) -> np.ndarray:
    """CAM de la clase k sin gradientes: Σ_c A[..., c] W[c, k], con ReLU y normalización a [0, 1]."""
    cam = np.maximum(feature_map @ head_weights[:, k], 0.0)
    top = cam.max()
    return cam / top if top > 0 else cam


def ood_check(embedding: np.ndarray, model: str, meta: dict) -> dict | None:
    """Distancia coseno media a los k vecinos de train frente al umbral fijado en validación."""
    ref = data.ood_reference(model)
    spec = meta.get("ood")
    if ref is None or not spec:
        if model not in _OOD_WARNED:
            _OOD_WARNED.add(model)
            log.warning("%s: sin ood_reference.npz o sin umbral en metadata.json; detección fuera de distribución desactivada", model)
        return None
    e = embedding.astype(np.float32)
    e = e / max(float(np.linalg.norm(e)), 1e-12)
    sims = ref @ e
    k = min(int(spec["k"]), len(sims))
    dist = 1.0 - float(np.sort(sims)[-k:].mean())
    return {"distance": dist, "threshold": float(spec["threshold"]), "out_of_domain": dist > float(spec["threshold"]),
            "quantile": spec.get("quantile")}


def _result(model: str, meta: dict, logits: np.ndarray, embedding, cam, latency_ms, image: np.ndarray, source: str) -> dict:
    classes = meta["classes"]
    T = float(meta.get("temperature") or 1.0)
    probs = softmax(logits / T)
    return {"model": model, "classes": classes, "probs": probs.tolist(), "raw_probs": softmax(logits).tolist(),
            "temperature": T, "pred": classes[int(np.argmax(probs))], "sets": conformal_sets(probs, meta, classes),
            "cam": None if cam is None else cam.tolist(), "latency_ms": latency_ms,
            "ood": None if embedding is None else ood_check(embedding, model, meta),
            "image": image, "source": source}


def predict(model: str, img: Image.Image) -> dict:
    """Predicción en vivo de un modelo servible (metadata `live`)."""
    meta = data.model_metadata(model)
    if meta is None:
        raise FileNotFoundError(f"No hay metadatos de {model}.")
    size = int((meta.get("resolution") or [INPUT_SIZE])[0])
    x = preprocess(img, size)
    sess = session(model)
    names = [o.name for o in sess.get_outputs()]
    t0 = time.perf_counter()
    out = dict(zip(names, sess.run(None, {"image": x})))
    latency = 1000 * (time.perf_counter() - t0)
    logits = out["logits"][0].astype(np.float64)
    k = int(np.argmax(logits))
    cam = cam_maps(out["feature_map"][0], out["head_weights"], k) if "feature_map" in out else None
    emb = out.get("embedding")
    return _result(model, meta, logits, None if emb is None else emb[0], cam, latency, x[0].astype(np.uint8), "en vivo")


def predict_precomputed(model: str, image_id: str) -> dict:
    """Resultado precalculado de un ejemplo de la galería (modelos que superan el límite de tamaño)."""
    meta = data.model_metadata(model)
    pre = data.precomputed(model) or {}
    if meta is None or image_id not in pre:
        raise InvalidImage("Este modelo se muestra solo con resultados precalculados de la galería de ejemplos.")
    entry = pre[image_id]
    logits = np.asarray(entry["logits"], dtype=np.float64)
    k = int(np.argmax(logits))
    cam = np.asarray(entry["cam"][k]) if entry.get("cam") else None
    if cam is not None:
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
    path = data.example_path(image_id)
    image = preprocess(Image.open(path), int((meta.get("resolution") or [INPUT_SIZE])[0]))[0].astype(np.uint8) if path else None
    emb = np.asarray(entry["embedding"], dtype=np.float32) if entry.get("embedding") else None
    return _result(model, meta, logits, emb, cam, None, image, "precalculado")


# ---------------------------------------------------------------------------
# Imágenes de salida (PNG en base64, en memoria)
# ---------------------------------------------------------------------------


def png_data_uri(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr).astype(np.uint8)).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _stops() -> np.ndarray:
    return np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in HEATMAP_STOPS], dtype=np.float32)


def heatmap_rgba(cam, size: int) -> np.ndarray:
    """Mapa (h, w) -> RGBA (size, size, 4): ReLU, normalización 0–1, redimensionado bilineal,
    paleta azul cielo -> amarillo -> naranja -> bermellón y alfa proporcional a la intensidad."""
    m = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    top = float(m.max())
    m = m / top if top > 0 else m
    m = np.asarray(Image.fromarray(m).resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    m = np.clip(m, 0.0, 1.0)
    stops = _stops()
    pos = m * (len(stops) - 1)
    i = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    t = (pos - i)[..., None]
    rgb = stops[i] * (1 - t) + stops[i + 1] * t
    alpha = 255.0 * np.clip(m, 0.0, 1.0) ** 0.8
    return np.concatenate([rgb, alpha[..., None]], axis=-1).round().astype(np.uint8)


def heatmap_png(cam, size: int) -> str | None:
    return None if cam is None else png_data_uri(heatmap_rgba(cam, size))
