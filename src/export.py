"""Exportación de corridas guardadas a ONNX para el dashboard (sin entrenar).

    python -m src.export --model cci                       # corrida export.run de export.label
    python -m src.export --model resnet50 --seed 1 --fold 1 --fp16
    python -m src.export --all                              # todos los modelos con model.weights.h5

Por modelo se escribe artifacts/<modelo>/ (configs/export.yaml):

- model.onnx: el modelo en float32 (reconstruido sin precisión mixta, con model.weights.h5) con
  entrada `image` (N, H, W, 3) float32 RGB en [0, 255] y salidas
      logits (N, K)          logits antes de la softmax
      probabilities (N, K)   softmax(logits), sin calibrar
      embedding (N, D)       entrada de la capa densa final (para el aviso fuera de dominio)
  y, en los modelos con pooling global (GAP -> [LayerNorm] -> Dropout -> Dense),
      feature_map (N, h, w, C)  último mapa de activación del backbone
      head_weights (C, K)       pesos de la cabeza, para el CAM sin gradientes:
                                CAM_k = Σ_c feature_map[..., c] · head_weights[c, k]
  (con LayerNorm en la cabeza, como ConvNeXt, head_weights = γ ⊙ W y el CAM es exacto salvo una
  transformación afín positiva por imagen, que la normalización min-max del mapa elimina);
- model_fp16.onnx (opcional): cuantización FP16 con entradas y salidas en float32, aceptada solo si
  el TOST por bootstrap de lesiones muestra que la pérdida de exactitud balanceada en validación
  está dentro de ±δ (benchmark.delta);
- ood_reference.npz: embeddings normalizados de una muestra de train de la corrida y el umbral de
  distancia coseno (k vecinos) fijado en validación;
- metadata.json: versiones, semilla, hiperparámetros, preprocesamiento, resolución, esquema de
  entrada y salida, temperatura de calibración, cuantiles conformes, métricas de validación y de
  prueba, prueba de ida y vuelta, latencia en CPU (mediana y p95), memoria pico, tamaño y si el
  modelo supera export.max_size_mb (se marca `live: false` y el dashboard muestra resultados
  precalculados).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import data as D

log = logging.getLogger(__name__)

EXPORT_CONFIG = D.PROJECT_ROOT / "configs" / "export.yaml"
WEIGHTS_FILE = "model.weights.h5"
ONNX_FILE = "model.onnx"
ONNX_FP16_FILE = "model_fp16.onnx"
OOD_FILE = "ood_reference.npz"
METADATA_FILE = "metadata.json"
INPUT_NAME = "image"
BASE_OUTPUTS = ("logits", "probabilities", "embedding")
CAM_OUTPUTS = ("feature_map", "head_weights")
CAM_HIDDEN = ("GlobalAveragePooling2D", "Dropout", "LayerNormalization")


def load_export_config(path: str | Path | None = None) -> dict:
    with open(path or EXPORT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def artifacts_dir(ecfg: dict, model_name: str) -> Path:
    return D.PROJECT_ROOT / ecfg["export"]["artifacts_dir"] / model_name


# ---------------------------------------------------------------------------
# Grafo de inferencia
# ---------------------------------------------------------------------------


def cam_head(model) -> dict | None:
    """Pesos efectivos de la cabeza para el CAM, o None si la cabeza no es GAP -> lineal.

    Devuelve {"kind": "gap_linear" | "gap_layernorm_linear", "weights": (C, K)}."""
    from src.models.common import head_layers

    *hidden, out = head_layers(model)
    kinds = [type(layer).__name__ for layer in hidden]
    if not kinds or kinds[0] != "GlobalAveragePooling2D" or any(k not in CAM_HIDDEN for k in kinds):
        return None
    w = np.asarray(out.kernel.numpy(), dtype=np.float32)
    kind = "gap_linear"
    for layer in hidden:
        if type(layer).__name__ == "LayerNormalization":
            gamma = layer.gamma.numpy() if layer.scale else np.ones(w.shape[0], np.float32)
            w = gamma[:, None].astype(np.float32) * w
            kind = "gap_layernorm_linear"
    return {"kind": kind, "weights": w}


def serving_function(model):
    """tf.function image -> (logits, probabilities, embedding[, feature_map, head_weights])."""
    import tensorflow as tf

    from src.models.common import apply_pre_backbone, get_backbone, head_layers

    backbone = get_backbone(model)
    *hidden, out = head_layers(model)
    cam = cam_head(model)
    shape = [None, *model.inputs[0].shape[1:]]

    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32, name=INPUT_NAME)], autograph=False)
    def serve(image):
        x = apply_pre_backbone(model, image)
        fmap = backbone(x, training=False)
        y = fmap
        for layer in hidden:
            y = layer(y, training=False)
        y = tf.cast(y, tf.float32)
        logits = tf.matmul(y, tf.cast(out.kernel, tf.float32)) + tf.cast(out.bias, tf.float32)
        outputs = [logits, tf.nn.softmax(logits, axis=-1), y]
        if cam is not None:
            outputs += [tf.cast(fmap, tf.float32), tf.identity(tf.constant(cam["weights"]))]
        return tuple(outputs)

    names = list(BASE_OUTPUTS) + (list(CAM_OUTPUTS) if cam is not None else [])
    return serve, names, cam


def _rename_outputs(proto, names) -> None:
    """Da a las salidas del grafo los nombres de `names`. Una salida que es directamente un
    initializer (head_weights) se emite con un nodo Identity, para que la conversión a FP16 no
    duplique el nombre."""
    import onnx

    g = proto.graph
    inits = {i.name for i in g.initializer}
    for out, new in zip(g.output, names):
        old = out.name
        if old in inits:
            g.node.append(onnx.helper.make_node("Identity", [old], [new], name=f"{new}_identity"))
            out.name = new
            continue
        for node in g.node:
            for i, o in enumerate(node.output):
                if o == old:
                    node.output[i] = new
            for i, o in enumerate(node.input):
                if o == old:
                    node.input[i] = new
        out.name = new


def export_keras_model(model, path: str | Path, opset: int = 17) -> dict:
    """Convierte `model` (estructura de src/models/common.py) a ONNX con tf2onnx. Devuelve
    {"outputs": nombres, "cam": dict | None}."""
    import tf2onnx

    serve, names, cam = serving_function(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    proto, _ = tf2onnx.convert.from_function(serve, input_signature=serve.input_signature, opset=int(opset))
    _rename_outputs(proto, names)
    proto.doc_string = f"skin-lesion-dl {model.name}: entrada image float32 RGB [0, 255]"
    import onnx

    onnx.save(proto, str(path))
    return {"outputs": names, "cam": cam}


def onnx_session(path: str | Path, threads: int | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def onnx_predict(sess, images: np.ndarray, batch_size: int = 16) -> dict:
    """{salida: array} de la sesión ONNX sobre `images` (N, H, W, 3) float32 en [0, 255]."""
    names = [o.name for o in sess.get_outputs()]
    parts = {n: [] for n in names}
    for start in range(0, len(images), batch_size):
        res = sess.run(None, {INPUT_NAME: np.asarray(images[start:start + batch_size], dtype=np.float32)})
        for n, r in zip(names, res):
            parts[n].append(r)
    return {n: (v[0] if n == "head_weights" else np.concatenate(v)) for n, v in parts.items()}


def roundtrip_check(model, sess, images: np.ndarray, atol: float) -> dict:
    """Error absoluto máximo en probabilidades y coincidencia de la clase predicha (Keras frente a ONNX)."""
    keras_p = np.asarray(model.predict(images, verbose=0, batch_size=16), dtype=np.float64)
    onnx_p = onnx_predict(sess, images)["probabilities"].astype(np.float64)
    err = float(np.abs(keras_p - onnx_p).max())
    same = bool((keras_p.argmax(1) == onnx_p.argmax(1)).all())
    return {"n_images": len(images), "max_abs_error": err, "atol": float(atol),
            "same_predicted_class": same, "passed": bool(err < atol and same)}


# ---------------------------------------------------------------------------
# FP16
# ---------------------------------------------------------------------------


FP32_NODE_PREFIXES = ("color_constancy",)   # la capa CCI (logaritmos, potencias de orden p) queda en float32
FP32_OPS = ("IsInf", "IsNaN")                # sin núcleo float16 en ONNX Runtime


def convert_fp16(src: str | Path, dst: str | Path) -> None:
    """Pesos y operaciones en float16 con entradas y salidas en float32 (onnxruntime.transformers).
    Los nodos de la capa CCI y los operadores sin soporte float16 se conservan en float32."""
    import onnx
    from onnxruntime.transformers.float16 import (
        DEFAULT_OP_BLOCK_LIST,
        convert_float_to_float16,
    )

    proto = onnx.load(str(src))
    keep = [n.name for n in proto.graph.node if any(n.name.startswith(p) for p in FP32_NODE_PREFIXES)]
    proto = convert_float_to_float16(proto, keep_io_types=True, node_block_list=keep,
                                     op_block_list=list(DEFAULT_OP_BLOCK_LIST) + list(FP32_OPS))
    onnx.save(proto, str(dst))


def fp16_equivalence(y, groups, pred32, pred16, delta: float, alpha: float, n_resamples: int, seed: int) -> dict:
    """TOST por bootstrap de lesiones de d = BA(fp16) − BA(fp32): equivalentes si el IC percentil
    1 − 2α de d queda dentro de (−δ, δ). p_tost = max(P*(d <= −δ), P*(d >= δ))."""
    from sklearn.metrics import balanced_accuracy_score

    from src import stats as S

    y = np.asarray(y)
    boot = S.cluster_bootstrap_balanced_accuracy(
        y, groups, {"fp32": (pred32 == y)[None], "fp16": (pred16 == y)[None]}, n_resamples, seed)
    d = boot["fp16"]["remuestreos"] - boot["fp32"]["remuestreos"]
    lo, hi = np.quantile(d, [alpha, 1 - alpha])
    p = max(float((d <= -delta).mean()), float((d >= delta).mean()))
    return {"ba_fp32": float(balanced_accuracy_score(y, pred32)), "ba_fp16": float(balanced_accuracy_score(y, pred16)),
            "diferencia": float(boot["fp16"]["observado"] - boot["fp32"]["observado"]),
            "ic_bajo": float(lo), "ic_alto": float(hi), "nivel_ic": 1 - 2 * alpha, "delta": float(delta),
            "p_tost": p, "equivalentes": bool(-delta < lo and hi < delta), "n_images": len(y),
            "same_class_frac": float((pred32 == pred16).mean()), "n_resamples": int(n_resamples)}


# ---------------------------------------------------------------------------
# Latencia, calibración y fuera de dominio
# ---------------------------------------------------------------------------


def benchmark_onnx(path: str | Path, lcfg: dict) -> dict:
    """src.onnx_bench en un subproceso (memoria pico de un proceso solo de inferencia)."""
    cmd = [sys.executable, "-m", "src.onnx_bench", str(path), "--repeats", str(lcfg.get("repeats", 100)),
           "--warmup", str(lcfg.get("warmup", 10)), "--threads", str(lcfg.get("intra_op_threads", 2))]
    res = subprocess.run(cmd, cwd=D.PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=1800)
    return json.loads(res.stdout.strip().splitlines()[-1])


def calibration_info(cfg: dict, run_dir: Path, classes) -> dict:
    """Temperatura (validación de la corrida) y cuantiles conformes LAC con esa temperatura."""
    from src import calibration as C

    path = Path(run_dir) / "probabilities_val.csv"
    if not path.exists():
        return {"temperature": None, "conformal": None}
    df = pd.read_csv(path)
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    p = df[[f"p_{c}" for c in classes]].to_numpy(np.float64)
    c, ccfg = cfg["calibration"], cfg["conformal"]
    T = C.fit_temperature(y, p, tuple(c["temperature_bounds"]), float(c["eps"]))
    pc = C.softmax_t(p, T, float(c["eps"])) if ccfg.get("use_temperature", True) else p
    q = {str(float(a)): C.conformal_threshold(y, pc, float(a)) for a in ccfg["alphas"]}
    return {"temperature": float(T),
            "conformal": {"score": ccfg.get("score", "lac"), "use_temperature": bool(ccfg.get("use_temperature", True)),
                          "calibration_split": "val", "n_calibration": len(y), "quantiles": q}}


def knn_distance(query: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    """Distancia coseno media a los k vecinos más cercanos (embeddings ya normalizados)."""
    sim = query @ reference.T
    k = min(k, reference.shape[0])
    top = -np.sort(-sim, axis=1)[:, :k]
    return 1.0 - top.mean(axis=1)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def ood_reference(sess, train_images: np.ndarray, val_images: np.ndarray, ocfg: dict) -> tuple[np.ndarray, dict]:
    """Embeddings de referencia (train) y umbral = cuantil de la distancia k-NN en validación."""
    ref = l2_normalize(onnx_predict(sess, train_images)["embedding"])
    dist = knn_distance(l2_normalize(onnx_predict(sess, val_images)["embedding"]), ref, int(ocfg["k"]))
    q = float(ocfg["quantile"])
    return ref.astype(np.float16), {"metric": "cosine_knn_mean", "k": int(ocfg["k"]), "quantile": q,
                                    "threshold": float(np.quantile(dist, q)), "n_reference": len(ref),
                                    "n_calibration": len(dist), "calibration_split": "val",
                                    "distance_median_val": float(np.median(dist))}


# ---------------------------------------------------------------------------
# Corrida -> artefactos
# ---------------------------------------------------------------------------


def versions() -> dict:
    import keras
    import onnx
    import onnxruntime
    import tensorflow as tf
    import tf2onnx

    return {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
            "tf2onnx": tf2onnx.__version__, "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__}


def family_of(cfg: dict, model_name: str) -> str:
    e2, e3 = cfg.get("e2", {}), cfg.get("e3", {})
    if model_name == e3.get("original_model", "cci"):
        return "Original (CCI)"
    if model_name in e3.get("ablation_models", []):
        return "Ablación (CCI)"
    if model_name in e2.get("transformer_models", []):
        return "Transformer"
    if model_name in e2.get("baselines", []):
        return "Baseline"
    return "CNN"


def _sample_ids(ids, n, seed: int):
    ids = np.asarray(ids)
    if n is None or n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=int(n), replace=False))


def _images_available(cfg: dict, ids) -> bool:
    return len(ids) > 0 and all(D.image_path(cfg, i).exists() for i in ids[:5])


def preprocessing_spec(cfg: dict, model_cfg: dict) -> dict:
    t = cfg["train"]
    size = int(t["image_size"])
    return {"color": "RGB", "layout": "NHWC", "dtype": "float32", "range": [0.0, 255.0],
            "resize": [size, size], "resize_method": t.get("resize_method", "bilinear"),
            "antialias": bool(t.get("resize_antialias", True)), "keep_aspect_ratio": False, "clip": [0.0, 255.0],
            "normalization": "dentro del modelo (capa preprocess"
                             + (" precedida de la capa CCI)" if model_cfg.get("backbone") else ")"),
            "steps": ["decodificar JPEG/PNG a RGB de 8 bits", (f"redimensionar a {size} x {size} ({t.get('resize_method', 'bilinear')}, "
                       "antialias, sin conservar la relación de aspecto)"), "convertir a float32 sin normalizar, en [0, 255]"]}


def export_run(cfg: dict, ecfg: dict, model_name: str, run_dir: Path, fp16: bool | None = None,
               out_dir: Path | None = None) -> dict:
    """Exporta la corrida `run_dir` de `model_name` a `out_dir` (artifacts/<modelo>/). Devuelve metadata."""
    from src.gradcam import load_model_float32
    from src.train import load_model_config

    e = ecfg["export"]
    run_dir = Path(run_dir)
    out_dir = Path(out_dir or artifacts_dir(ecfg, model_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / WEIGHTS_FILE).exists():
        raise FileNotFoundError(f"{run_dir} no tiene {WEIGHTS_FILE}")
    info = D.load_json(run_dir / "run_info.json")
    metrics = D.load_json(run_dir / "metrics.json")
    classes = D.class_names(cfg)
    model_cfg = load_model_config(model_name)
    model = load_model_float32(cfg, model_name, run_dir)
    onnx_path = out_dir / ONNX_FILE
    log.info(f"{model_name}: exportando {run_dir} -> {onnx_path}")
    graph = export_keras_model(model, onnx_path, int(e["opset"]))
    sess = onnx_session(onnx_path)

    # Imágenes de la corrida: validación (ida y vuelta, FP16, umbral OOD) y train (referencia OOD)
    splits = D.load_splits(cfg["paths"]["splits_csv"])
    fold = int(info["val_fold"])
    val_df = D.select_split(splits, "val", fold)
    train_df = D.select_split(splits, "train", fold)
    have_data = _images_available(cfg, val_df["image_id"].to_numpy())
    rt = e["roundtrip"]
    size = int(cfg["train"]["image_size"])
    if have_data:
        rt_ids = _sample_ids(val_df["image_id"], int(rt["n_images"]), int(info["seed"]))
        rt_images = D.load_resized_images(cfg, rt_ids)
        rt_source = "val"
    else:
        rt_images = np.random.default_rng(0).uniform(1, 254, (int(rt["n_images"]), size, size, 3)).astype(np.float32)
        rt_source = "sintéticas (no hay imágenes en data/raw/images)"
    roundtrip = {**roundtrip_check(model, sess, rt_images, float(rt["atol"])), "images": rt_source}
    if not roundtrip["passed"]:
        log.warning(f"{model_name}: la prueba de ida y vuelta falla: {roundtrip}")

    # FP16 con TOST
    fp16 = bool(e["fp16"]["enabled"]) if fp16 is None else fp16
    fp16_res = {"enabled": fp16}
    if fp16 and have_data:
        fcfg, b = e["fp16"], cfg["benchmark"]
        convert_fp16(onnx_path, out_dir / ONNX_FP16_FILE)
        ev = val_df.set_index("image_id").loc[_sample_ids(val_df["image_id"], fcfg.get("n_eval"), int(fcfg["seed"]))]
        imgs = D.load_resized_images(cfg, ev.index)
        y = ev["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        pred32 = onnx_predict(sess, imgs)["probabilities"].argmax(1)
        pred16 = onnx_predict(onnx_session(out_dir / ONNX_FP16_FILE), imgs)["probabilities"].argmax(1)
        fp16_res.update(fp16_equivalence(y, ev["lesion_id"].to_numpy(), pred32, pred16, float(b["delta"]),
                                         float(b["alpha"]), int(fcfg["n_resamples"]), int(fcfg["seed"])))
        fp16_res["size_mb"] = (out_dir / ONNX_FP16_FILE).stat().st_size / 2 ** 20
        fp16_res["benchmark"] = benchmark_onnx(out_dir / ONNX_FP16_FILE, e["latency"])
        if not fp16_res["equivalentes"]:
            log.warning(f"{model_name}: FP16 no es equivalente (TOST); se conserva el archivo pero no se usa")
    elif fp16:
        fp16_res["skipped"] = "no hay imágenes de validación para el TOST"

    # Fuera de dominio
    ocfg = e["ood"]
    ood = None
    if have_data:
        ref_ids = _sample_ids(train_df["image_id"], int(ocfg["n_reference"]), int(ocfg["seed"]))
        cal_ids = _sample_ids(val_df["image_id"], int(ocfg["n_calibration"]), int(ocfg["seed"]))
        ref, ood = ood_reference(sess, D.load_resized_images(cfg, ref_ids), D.load_resized_images(cfg, cal_ids), ocfg)
        np.savez_compressed(out_dir / OOD_FILE, embeddings=ref)
        ood["file"] = OOD_FILE

    bench = benchmark_onnx(onnx_path, e["latency"])
    size_mb = onnx_path.stat().st_size / 2 ** 20
    limit = float(e["max_size_mb"])
    serving = ONNX_FP16_FILE if fp16_res.get("equivalentes") else ONNX_FILE
    serving_mb = (out_dir / serving).stat().st_size / 2 ** 20
    outputs = {n: {"shape": [None, *map(int, sess.get_outputs()[i].shape[1:])] if n != "head_weights"
                   else list(map(int, sess.get_outputs()[i].shape))} for i, n in enumerate(graph["outputs"])}
    meta = {
        "model": model_name,
        "family": family_of(cfg, model_name),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "format": "onnx", "opset": int(e["opset"]),
        "run": {"label": info["label"], "seed": int(info["seed"]), "val_fold": fold,
                "run_dir": str(run_dir.relative_to(D.PROJECT_ROOT)) if run_dir.is_relative_to(D.PROJECT_ROOT) else str(run_dir)},
        "global_seed": int(cfg["seed"]),
        "versions": versions(),
        "hyperparameters": info["hyperparameters"],
        "model_config": {k: model_cfg.get(k) for k in ("backbone", "variant", "cci", "keras_application",
                                                       "keras_hub_preset", "weights") if k in model_cfg},
        "classes": classes,
        "resolution": [size, size],
        "input": {"name": INPUT_NAME, "shape": [None, size, size, 3], "dtype": "float32", "range": [0.0, 255.0],
                  "layout": "NHWC", "color": "RGB"},
        "preprocessing": preprocessing_spec(cfg, model_cfg),
        "outputs": outputs,
        "cam": {"supported": graph["cam"] is not None, "kind": graph["cam"]["kind"] if graph["cam"] else None,
                "formula": "CAM_k = sum_c feature_map[..., c] * head_weights[c, k]" if graph["cam"] else None},
        **calibration_info(cfg, run_dir, classes),
        "metrics": {s: metrics[s] for s in ("val", "test") if s in metrics},
        "roundtrip": roundtrip,
        "fp16": fp16_res,
        "benchmark": bench,
        "params": info.get("params", {}).get("finetune"),
        "size_mb": size_mb,
        "size_limit_mb": limit,
        "serving_file": serving,
        "live": bool(serving_mb <= limit),
        "ood": ood,
    }
    if not meta["live"]:
        log.warning(f"{model_name}: {serving} ocupa {serving_mb:.1f} MB > {limit} MB; se mostrará con resultados precalculados")
    D.save_json(meta, out_dir / METADATA_FILE)
    log.info(f"{model_name}: listo ({size_mb:.1f} MB, ida y vuelta {roundtrip['max_abs_error']:.2e}, "
             f"latencia mediana {bench['latency_ms_median']:.1f} ms) -> {out_dir}")
    return meta


def exportable_runs(cfg: dict, ecfg: dict, models=None) -> dict:
    """{modelo: carpeta} de la corrida export.run con pesos guardados de cada modelo."""
    from src.models import REGISTRY
    from src.train import run_dir_for

    e = ecfg["export"]
    seed, fold = (int(v) for v in e["run"])
    out = {}
    for m in models or sorted(REGISTRY):
        d = run_dir_for(cfg, m, e["label"], seed, fold)
        if (d / WEIGHTS_FILE).exists():
            out[m] = d
    return out


def main(argv=None) -> None:
    from src.models import REGISTRY
    from src.train import run_dir_for
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Exporta corridas guardadas a ONNX (artifacts/<modelo>/).")
    parser.add_argument("--model", action="append", choices=sorted(REGISTRY), default=None)
    parser.add_argument("--all", action="store_true", help="todos los modelos con model.weights.h5")
    parser.add_argument("--label", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", help="exporta también model_fp16.onnx con TOST")
    parser.add_argument("--config", default=None)
    parser.add_argument("--export-config", default=None)
    args = parser.parse_args(argv)
    if not args.model and not args.all:
        parser.error("indica --model o --all")

    cfg = D.load_config(args.config)
    ecfg = load_export_config(args.export_config)
    e = ecfg["export"]
    if args.label:
        e["label"] = args.label
    if args.seed is not None or args.fold is not None:
        e["run"] = [args.seed if args.seed is not None else e["run"][0], args.fold if args.fold is not None else e["run"][1]]
    setup_logging(cfg["paths"]["logs_dir"] / "export.log")
    import keras

    keras.mixed_precision.set_global_policy("float32")
    runs = exportable_runs(cfg, ecfg, args.model)
    for m in args.model or []:
        if m not in runs:
            d = run_dir_for(cfg, m, e["label"], *map(int, e["run"]))
            log.error(f"{m}: no hay {WEIGHTS_FILE} en {d}")
    if not runs:
        sys.exit("No hay corridas con pesos guardados que exportar.")
    for m, d in runs.items():
        export_run(cfg, ecfg, m, d, fp16=True if args.fp16 else None)


if __name__ == "__main__":
    main()
