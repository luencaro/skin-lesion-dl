"""Costo computacional de cada modelo, medido en la implementación (sin entrenar ni usar datos reales).

    python -m src.cost --model vit_b16 [--model ...]      # o --all

Por modelo, con la arquitectura de configs/<modelo>.yaml (weights=None: mismos parámetros que con
ImageNet) y la sección `cost` de base.yaml:

- parámetros totales y entrenables en las fases A y B;
- FLOPs de una inferencia a image_size x image_size (profiler de TensorFlow sobre el grafo
  congelado, que cuenta 2 FLOPs por multiplicación-suma; se registra la estadística de Einsum,
  que el profiler no cuenta y que usa la atención de ViT). GMACs = GFLOPs / 2;
- memoria pico de la GPU en un paso de entrenamiento de la fase B (todo descongelado) con un lote
  sintético de train.batch_size y la política de precisión de train.mixed_precision, y en
  inferencia con el mismo lote;
- latencia por imagen (lote de 1, mediana de `latency_repeats` llamadas tras `warmup` de
  calentamiento) e imágenes por segundo con lotes de train.batch_size.

Resultado: outputs/cost/<modelo>.json. El tiempo de ajuste y de entrenamiento se lee de las
corridas guardadas (cost_from_runs). Las funciones de tablas son tolerantes (src/report.py).
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src.report import Missing

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

_EINSUM_REGISTERED = False


def _register_einsum_flops() -> None:
    """Registra en el profiler de TF los FLOPs de Einsum: 2 x producto de los tamaños de todos
    los índices de la ecuación (la suma sobre los índices contraídos cuenta como una
    multiplicación-suma por término)."""
    global _EINSUM_REGISTERED
    if _EINSUM_REGISTERED:
        return
    from tensorflow.python.framework import graph_util, ops

    def _labels(term: str, ndim: int) -> list[str]:
        if "..." not in term:
            return list(term)
        pre, post = term.split("...")  # la elipsis cubre las dimensiones restantes
        return list(pre) + [f"_{i}" for i in range(ndim - len(pre) - len(post))] + list(post)

    def _einsum_flops(graph, node):
        eq = node.attr["equation"].s.decode()
        lhs = eq.split("->")[0].replace(" ", "")
        sizes = {}
        for term, name in zip(lhs.split(","), node.input):
            dims = graph_util.tensor_shape_from_node_def_name(graph, name).as_list()
            for lab, d in zip(_labels(term, len(dims)), dims):
                if d is not None:
                    sizes[lab] = max(sizes.get(lab, 1), int(d))
        return ops.OpStats("flops", 2 * int(np.prod(list(sizes.values()))) if sizes else 0)

    try:
        ops.RegisterStatistics("Einsum", "flops")(_einsum_flops)
    except KeyError:  # ya registrada
        pass
    _EINSUM_REGISTERED = True


def count_flops(model, image_size: int, batch: int = 1) -> int:
    """FLOPs de una inferencia de `batch` imágenes (profiler de TF sobre el grafo congelado)."""
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    _register_einsum_flops()

    @tf.autograph.experimental.do_not_convert
    def forward(x):
        return model(x, training=False)

    spec = tf.TensorSpec([batch, image_size, image_size, 3], tf.float32)
    graph = convert_variables_to_constants_v2(tf.function(forward).get_concrete_function(spec)).graph
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    info = tf.compat.v1.profiler.profile(graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd="op",
                                         options=opts)
    return int(info.total_float_ops)


# ---------------------------------------------------------------------------
# Memoria, latencia y rendimiento
# ---------------------------------------------------------------------------


def _gpu_name() -> str | None:
    from src.utils import list_gpus

    gpus = list_gpus()
    return gpus[0].name.replace("/physical_device:", "") if gpus else None


def _reset_peak(device: str | None) -> None:
    import tensorflow as tf

    if device:
        tf.config.experimental.reset_memory_stats(device)


def _peak_mb(device: str | None) -> float:
    import tensorflow as tf

    if not device:
        return float("nan")
    return tf.config.experimental.get_memory_info(device)["peak"] / 2 ** 20


def measure_model(cfg: dict, model_name: str, accumulation: int | None = None) -> dict:
    """Mide parámetros, FLOPs, memoria pico, latencia y rendimiento de `model_name` con datos
    sintéticos. El entrenamiento usa micro-lotes de train.batch_size / gradient_accumulation_steps
    (del YAML del modelo, o `accumulation`) y el tiempo por actualización suma sus micro-pasos. Si
    la GPU se queda sin memoria en el paso de entrenamiento, lo registra en `oom_entrenamiento`."""
    import keras
    import tensorflow as tf

    from src.models import REGISTRY, count_params, set_backbone_trainable
    from src.train import (
        accumulation_steps,
        compile_model,
        configure_runtime,
        load_model_config,
        make_optimizer,
    )
    from src.utils import environment_info

    ccfg = cfg["cost"]
    tcfg = cfg["train"]
    size, bs = int(tcfg["image_size"]), int(tcfg["batch_size"])
    k = len(D.class_names(cfg))
    model_cfg = dict(load_model_config(model_name), weights=None)
    if accumulation is None:
        accumulation = accumulation_steps(tcfg, model_cfg)
    micro = D.micro_batch_size(bs, accumulation)
    rng = np.random.default_rng(int(ccfg.get("seed", 0)))
    x = rng.uniform(0, 255, size=(micro, size, size, 3)).astype(np.float32)
    y = rng.integers(0, k, size=micro).astype(np.int32)
    out = {"model": model_name, "image_size": size, "batch_size": bs, "micro_batch_size": micro,
           "gradient_accumulation_steps": accumulation, "mixed_precision": bool(tcfg.get("mixed_precision"))}

    # FLOPs en float32 (no dependen de la política de precisión)
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("float32")
    model = REGISTRY[model_name](model_cfg, k, 0.0)
    flops = count_flops(model, size)
    set_backbone_trainable(model, False)
    pa = count_params(model)
    set_backbone_trainable(model, True)
    pb = count_params(model)
    out.update({"params_total": pb["total"], "params_trainable_warmup": pa["trainable"],
                "params_trainable_finetune": pb["trainable"], "flops": flops,
                "gflops": flops / 1e9, "gmacs": flops / 2e9})
    del model
    gc.collect()

    # Memoria y tiempos con la política del entrenamiento
    device = _gpu_name()
    out["device"] = device or "cpu"
    keras.backend.clear_session()
    configure_runtime(cfg)
    model = REGISTRY[model_name](model_cfg, k, 0.0)
    set_backbone_trainable(model, True)
    compile_model(model, make_optimizer("adamw", 1e-5, 1e-4, tcfg.get("weight_decay_exclude"), accumulation), k)
    try:
        for _ in range(int(ccfg["warmup"]) * accumulation):  # compilación (XLA si jit_compile="auto") y asignación
            model.train_on_batch(x, y)
        _reset_peak(device)
        steps = []
        for _ in range(int(ccfg["train_steps"])):  # cada paso medido es una actualización completa
            t0 = time.perf_counter()
            for _ in range(accumulation):
                model.train_on_batch(x, y)
            steps.append(time.perf_counter() - t0)
        out["train_step_s"] = float(np.median(steps))
        out["peak_mem_train_mb"] = _peak_mb(device)
        out["oom_entrenamiento"] = False
    except tf.errors.ResourceExhaustedError as e:
        out.update({"train_step_s": float("nan"), "peak_mem_train_mb": float("nan"), "oom_entrenamiento": True,
                    "error_entrenamiento": str(e).splitlines()[0][:300]})

    # Inferencia con la función de predicción compilada de Keras (la misma ruta que model.predict)
    x = rng.uniform(0, 255, size=(bs, size, size, 3)).astype(np.float32)
    x1 = x[:1]
    for _ in range(int(ccfg["warmup"])):
        model.predict_on_batch(x)
        model.predict_on_batch(x1)
    _reset_peak(device)
    model.predict_on_batch(x)
    out["peak_mem_inference_mb"] = _peak_mb(device)
    lat = []
    for _ in range(int(ccfg["latency_repeats"])):
        t0 = time.perf_counter()
        model.predict_on_batch(x1)
        lat.append(time.perf_counter() - t0)
    out["latency_ms_per_image"] = 1000 * float(np.median(lat))
    thr = []
    for _ in range(int(ccfg["throughput_repeats"])):
        t0 = time.perf_counter()
        model.predict_on_batch(x)
        thr.append(bs / (time.perf_counter() - t0))
    out["images_per_second"] = float(np.median(thr))
    steps_per_epoch = int(np.ceil(cfg_train_size(cfg) / bs))
    out["epoch_estimate_min"] = out["train_step_s"] * steps_per_epoch / 60
    out["environment"] = environment_info()
    del model
    keras.backend.clear_session()
    gc.collect()
    return out


def cfg_train_size(cfg: dict) -> int:
    """Imágenes de train de una corrida (train del pliegue 0), para estimar la duración de una época."""
    splits = D.load_splits(cfg["paths"]["splits_csv"])
    return int(((splits["split"] == "train") & (splits["fold"] != 0)).sum())


def cost_path(cfg: dict, model_name: str) -> Path:
    return cfg["paths"]["cost_dir"] / f"{model_name}.json"


def run_cost(cfg: dict, models) -> pd.DataFrame:
    rows = []
    for m in models:
        log.info(f"Midiendo el costo de {m}")
        res = measure_model(cfg, m)
        D.save_json(res, cost_path(cfg, m))
        rows.append({k: v for k, v in res.items() if k != "environment"})
        log.info(f"{m}: {res['gflops']:.2f} GFLOPs, pico de entrenamiento {res['peak_mem_train_mb']:.0f} MB, "
                 f"{res['latency_ms_per_image']:.2f} ms/imagen -> {cost_path(cfg, m)}")
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------


def load_cost(cfg: dict, models) -> pd.DataFrame | Missing:
    """Mediciones guardadas en outputs/cost/<modelo>.json de `models` (los que existan)."""
    rows = []
    for m in models:
        p = cost_path(cfg, m)
        if p.exists():
            r = D.load_json(p)
            r.pop("environment", None)
            rows.append(r)
    if not rows:
        return Missing("no hay mediciones de costo en outputs/cost/; ejecuta `python -m src.cost --all`.")
    return pd.DataFrame(rows).set_index("model").reindex([m for m in models if cost_path(cfg, m).exists()])


# Notas sobre tiempos registrados que no reflejan el costo con el pipeline actual.
TIME_NOTES = {
    "mobilenetv2": ("los tiempos registrados de t00 a t06 de MobileNetV2 están inflados unas 5 veces por el uso de "
                    "swap del 14 de noviembre de 2026 y no reflejan el costo con el pipeline actual "
                    "(experiments/presupuesto_computo.md)"),
}


def cost_from_runs(cfg: dict, models) -> pd.DataFrame | Missing:
    """Por modelo: tiempo total del ajuste (suma de los ensayos de trials.csv), tiempo de
    entrenamiento medio por corrida definitiva y total, en horas, y corridas definitivas. La
    columna `nota` (y attrs["nota"]) señala los tiempos de TIME_NOTES."""
    from src.final import finished_runs

    rows = {}
    for m in models:
        row = {}
        trials = cfg["paths"]["tuning_dir"] / m / "trials.csv"
        if trials.exists():
            t = pd.read_csv(trials)
            row["ajuste_h"] = float(t["train_time_s"].sum()) / 3600
            row["ensayos"] = len(t)
            if m in TIME_NOTES:
                row["nota"] = TIME_NOTES[m]
        runs = finished_runs(cfg, m)
        if runs:
            infos = [D.load_json(Path(d) / "run_info.json") for d in runs]
            times = [i["train_time_s"]["total"] for i in infos]
            row.update({"entrenamiento_h_por_corrida": float(np.mean(times)) / 3600,
                        "entrenamiento_h_total": float(np.sum(times)) / 3600, "corridas": len(times),
                        "micro_lote": infos[0].get("micro_batch_size", infos[0].get("inference_batch_size")),
                        "pasos_de_acumulacion": infos[0].get("gradient_accumulation_steps", 1)})
        if row:
            rows[m] = row
    if not rows:
        return Missing("no hay ajustes ni corridas definitivas con tiempos registrados.")
    out = pd.DataFrame(rows).T
    notes = [TIME_NOTES[m] for m in rows if m in TIME_NOTES and "nota" in rows[m]]
    if notes:
        out.attrs["nota"] = "; ".join(notes)
    return out


def cost_table(measured, times) -> pd.DataFrame | Missing:
    """Tabla de costo: parámetros, GFLOPs, GMACs, memoria pico, latencia, imágenes por segundo
    (outputs/cost) y tiempo de ajuste y de entrenamiento (corridas). Si falta una de las dos
    fuentes, devuelve la otra; si faltan ambas, Missing."""
    cols = ["micro_batch_size", "gradient_accumulation_steps", "params_total", "params_trainable_finetune",
            "gflops", "gmacs", "peak_mem_train_mb",
            "peak_mem_inference_mb", "oom_entrenamiento", "latency_ms_per_image", "images_per_second",
            "epoch_estimate_min"]
    parts = []
    if isinstance(measured, pd.DataFrame):
        parts.append(measured[[c for c in cols if c in measured.columns]])
    if isinstance(times, pd.DataFrame):
        parts.append(times)
    if not parts:
        return measured
    return parts[0] if len(parts) == 1 else parts[0].join(parts[1], how="outer")


def main(argv=None) -> None:
    from src.models import REGISTRY
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Costo computacional de los modelos (datos sintéticos).")
    parser.add_argument("--model", action="append", choices=sorted(REGISTRY), default=None)
    parser.add_argument("--all", action="store_true", help="todos los modelos de e2.deep_models")
    parser.add_argument("--config", default=None, help="YAML base (por defecto configs/base.yaml)")
    args = parser.parse_args(argv)
    cfg = D.load_config(args.config)
    models = list(args.model or [])
    if args.all:
        models = [m for m in deep_models(cfg) if m not in models] + models
    if not models:
        sys.exit("Indica --model o --all")
    setup_logging(cfg["paths"]["logs_dir"] / "cost.log")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(run_cost(cfg, models).T.to_string())


def deep_models(cfg: dict) -> list[str]:
    e2 = cfg["e2"]
    return list(e2["base_models"]) + list(e2["sota_models"])


if __name__ == "__main__":
    main()
