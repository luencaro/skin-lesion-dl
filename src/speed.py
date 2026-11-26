"""Velocidad y memoria del entrenamiento de un modelo, sin escribir corridas ni tocar el ajuste.

    python -m src.speed --model mobilenetv2 [--steps 50] [--warmup-steps 10] [--phases A B]
                        [--variants pipeline no_aug synthetic_gpu ...] [--accumulation N]
                        [--backbone mobilenetv2] [--out outputs/speed/<modelo>.csv]

Usa solo el train del pliegue 0 (y su validación para medir la predicción de una época). Por fase
(A: backbone congelado, optimizador de train.warmup; B: todo descongelado, optimizador de
train.finetune con tasa constante, que no cambia el costo) y por variante, mide `--steps` pasos
de entrenamiento tras `--warmup-steps` de calentamiento (el primero incluye la compilación).
Con acumulación de gradientes, un paso es un micro-lote. Variantes:

    pipeline              pipeline de train de la config (train.cache) con jit_compile="auto"
    no_aug                igual, sin aumento de datos
    synthetic_gpu         un lote sintético ya en la GPU: cota superior del modelo sin la entrada
    pipeline_nojit        pipeline con jit_compile=False
    synthetic_gpu_nojit   lote sintético con jit_compile=False
    cache_memory_float32  pipeline con train.cache = memory y cache_dtype = float32 (el anterior)
    cache_memory_uint8    pipeline con train.cache = memory y cache_dtype = uint8
    cache_disk            pipeline con train.cache = disk (float32 en disco)
    val_predict           predicción del pliegue de validación completo (segundos por época)

Por fila: imágenes por segundo, segundos por paso (media y mediana), tiempo del primer paso, uso
medio de la GPU muestreado con nvidia-smi durante la ventana medida, memoria pico de la GPU
(asignador de TensorFlow y nvidia-smi), RSS y swap del proceso, política de precisión, clase del
optimizador (LossScaleOptimizer con mixed_float16) y jit_compile efectivo. Un error de memoria de
la GPU se registra como status = OOM. `--backbone` fija el backbone de un modelo CCI solo para
esta medición (configs/cci.yaml no se modifica).
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import subprocess
import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from src import data as D

log = logging.getLogger(__name__)

PIPELINE_VARIANTS = ("pipeline", "no_aug", "pipeline_nojit", "cache_memory_float32", "cache_memory_uint8",
                     "cache_disk")
SYNTHETIC_VARIANTS = ("synthetic_gpu", "synthetic_gpu_nojit")
ALL_VARIANTS = ("pipeline", "no_aug", "synthetic_gpu", "pipeline_nojit", "synthetic_gpu_nojit",
                "cache_memory_float32", "cache_memory_uint8", "cache_disk", "val_predict")
CACHE_VARIANTS = {"cache_memory_float32": {"cache": "memory", "cache_dtype": "float32"},
                  "cache_memory_uint8": {"cache": "memory", "cache_dtype": "uint8"},
                  "cache_disk": {"cache": "disk"}}
SPEED_HPARAMS = {"learning_rate": 1e-4, "weight_decay": 1e-5, "dropout": 0.2}  # no afectan al costo


# ---------------------------------------------------------------------------
# Memoria y GPU
# ---------------------------------------------------------------------------


def process_memory_gib() -> dict:
    """VmRSS, VmHWM y VmSwap del proceso en GiB (Linux)."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS", "VmHWM", "VmSwap")):
                    key, value = line.split(":")
                    out[key.lower()] = int(value.split()[0]) / 2**20
    except OSError:
        pass
    return {"rss_gib": out.get("vmrss"), "rss_peak_gib": out.get("vmhwm"), "swap_gib": out.get("vmswap")}


def gpu_memory_gib() -> dict:
    import tensorflow as tf

    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except (ValueError, RuntimeError):
        return {"gpu_peak_tf_gib": None}
    return {"gpu_peak_tf_gib": info["peak"] / 2**30}


def reset_gpu_peak() -> None:
    import tensorflow as tf

    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (ValueError, RuntimeError):
        pass


class GpuSampler:
    """Muestrea utilization.gpu y memory.used con `nvidia-smi -lms` mientras está activo."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.proc = None
        self.samples: list[tuple[float, float]] = []

    def start(self) -> None:
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                 "-i", "0", f"-lms={self.interval_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.proc = None

    def stop(self) -> dict:
        if self.proc is None:
            return {"gpu_util_mean": None, "gpu_util_samples": 0, "gpu_mem_smi_max_mib": None}
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=10)
        for line in out.splitlines():
            try:
                util, mem = (float(v) for v in line.split(","))
                self.samples.append((util, mem))
            except ValueError:
                continue
        if not self.samples:
            return {"gpu_util_mean": None, "gpu_util_samples": 0, "gpu_mem_smi_max_mib": None}
        util = np.array([s[0] for s in self.samples])
        mem = np.array([s[1] for s in self.samples])
        return {"gpu_util_mean": float(util.mean()), "gpu_util_samples": len(util),
                "gpu_mem_smi_max_mib": float(mem.max())}


def _is_oom(e: Exception) -> bool:
    import tensorflow as tf

    text = str(e)
    return isinstance(e, tf.errors.ResourceExhaustedError) or "RESOURCE_EXHAUSTED" in text or "OOM" in text


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def speed_model_config(model_name: str, backbone: str | None = None) -> dict:
    """configs/<modelo>.yaml; con `backbone`, lo fija solo en memoria (modelos CCI)."""
    from src import train as T

    if backbone is None:
        return T.load_model_config(model_name)
    from src.original_model import resolve_model_config

    model_cfg = T._read_model_yaml(model_name)
    if "backbone" not in model_cfg:
        raise ValueError(f"{model_name} no tiene backbone configurable")
    model_cfg["backbone"] = backbone
    return resolve_model_config(model_cfg, T.load_model_config)


def build_speed_model(cfg: dict, model_cfg: dict):
    from src.models import REGISTRY

    return REGISTRY[model_cfg["name"]](model_cfg, len(D.class_names(cfg)), SPEED_HPARAMS["dropout"])


def compile_for_phase(model, cfg: dict, model_cfg: dict, phase: str, accum: int, jit) -> None:
    """Congela o descongela el backbone y compila como src.train (mismo optimizador, pérdida y
    métricas); en la fase B con tasa constante."""
    from src import train as T
    from src.metrics import SparseBalancedAccuracy
    from src.models import set_backbone_trainable

    tcfg = cfg["train"]
    exclude = list(tcfg.get("weight_decay_exclude") or []) + list(model_cfg.get("weight_decay_exclude") or [])
    pcfg = tcfg["warmup"] if phase == "A" else tcfg["finetune"]
    lr = float(tcfg["warmup"]["learning_rate"]) if phase == "A" else SPEED_HPARAMS["learning_rate"]
    set_backbone_trainable(model, phase == "B")
    opt = T.make_optimizer(pcfg.get("optimizer", "adamw"), lr, SPEED_HPARAMS["weight_decay"], exclude, accum)
    n_classes = len(D.class_names(cfg))
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"), SparseBalancedAccuracy(n_classes)],
        jit_compile=jit,
    )


def precision_info(model) -> dict:
    opt = model.optimizer
    inner = getattr(opt, "inner_optimizer", None)
    return {
        "policy": keras.mixed_precision.global_policy().name,
        "optimizer": type(opt).__name__,
        "inner_optimizer": type(inner).__name__ if inner is not None else None,
        "jit_compile": bool(model.jit_compile),
    }


# ---------------------------------------------------------------------------
# Mediciones
# ---------------------------------------------------------------------------


class StepTimer(keras.callbacks.Callback):
    """Instante del final de cada paso (los logs se convierten a Python en cada paso, lo que
    sincroniza con la GPU) y muestreo de la GPU durante la ventana medida."""

    def __init__(self, warmup: int, steps: int):
        super().__init__()
        self.warmup, self.steps = warmup, steps
        self.t_begin = 0.0
        self.times: list[float] = []
        self.sampler = GpuSampler()
        self.gpu: dict = {}

    def on_train_begin(self, logs=None):
        self.t_begin = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        self.times.append(time.perf_counter())
        if batch == self.warmup - 1:
            self.sampler.start()
        elif batch == self.warmup + self.steps - 1:
            self.gpu = self.sampler.stop()


def _summary(times: list[float], t_begin: float, warmup: int, steps: int, micro: int) -> dict:
    t = np.asarray(times)
    if len(t) < warmup + steps:
        raise RuntimeError(f"Solo {len(t)} pasos de {warmup + steps}")
    d = np.diff(t[warmup - 1:warmup + steps])
    return {
        "first_step_s": float(t[0] - t_begin),
        "warmup_s": float(t[warmup - 1] - t_begin),
        "step_s_mean": float(d.mean()),
        "step_s_median": float(np.median(d)),
        "step_s_p90": float(np.quantile(d, 0.9)),
        "images_per_s": float(micro * steps / d.sum()),
    }


def measure_fit(model, ds, cw: dict, warmup: int, steps: int, micro: int) -> dict:
    timer = StepTimer(warmup, steps)
    model.fit(ds.repeat(), epochs=1, steps_per_epoch=warmup + steps, class_weight=cw, verbose=0,
              callbacks=[timer])
    return {**_summary(timer.times, timer.t_begin, warmup, steps, micro), **timer.gpu}


def measure_synthetic(model, cw: dict, micro: int, size: int, warmup: int, steps: int, seed: int = 0) -> dict:
    """Pasos de train_on_batch con un lote fijo creado en la GPU (sin pipeline de entrada)."""
    import tensorflow as tf

    n_classes = len(cw)
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, micro).astype(np.int32)
    with tf.device("/GPU:0"):
        x = tf.random.stateless_uniform((micro, size, size, 3), seed=[seed, 0], minval=0.0, maxval=255.0)
        y = tf.identity(tf.constant(labels))
        sw = tf.identity(tf.constant(np.array([cw[int(k)] for k in labels], np.float32)))
    sampler = GpuSampler()
    t_begin = time.perf_counter()
    times = []
    for i in range(warmup + steps):
        model.train_on_batch(x, y, sample_weight=sw)  # devuelve valores de NumPy: sincroniza
        times.append(time.perf_counter())
        if i == warmup - 1:
            sampler.start()
    return {**_summary(times, t_begin, warmup, steps, micro), **sampler.stop()}


def measure_predict(model, ds, n: int) -> dict:
    model.predict(ds.take(2), verbose=0)  # compila los tamaños de lote completos
    t0 = time.perf_counter()
    proba = model.predict(ds, verbose=0)  # incluye la compilación del último lote incompleto
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = model.predict(ds, verbose=0)
    dt = time.perf_counter() - t0
    if len(proba) != n:
        raise RuntimeError(f"Se esperaban {n} predicciones y hay {len(proba)}")
    return {"predict_first_s": first, "predict_s": dt, "images_per_s": n / dt}


def prefill(source: D.SplitSource) -> float:
    """Recorre una vez `stored` para completar la caché en memoria (no hace nada útil con disk)."""
    t0 = time.perf_counter()
    for _ in source.stored.batch(256).prefetch(2):
        pass
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def _variant_cfg(cfg: dict, variant: str) -> dict:
    out = copy.deepcopy(cfg)
    if variant in CACHE_VARIANTS:
        out["train"].update(CACHE_VARIANTS[variant])
    if variant == "no_aug":
        out["train"]["augmentation"] = {}
    return out


def _cache_key(cfg: dict) -> tuple:
    t = cfg["train"]
    return (D.cache_mode(t), t.get("cache_dtype", "float32") if D.cache_mode(t) == "memory" else None)


def run_speed(cfg: dict, model_name: str, *, phases=("A", "B"), variants=ALL_VARIANTS, steps: int = 50,
              warmup_steps: int = 10, accumulation: int | None = None, backbone: str | None = None,
              val_fold: int = 0, seed: int = 0, out: Path | None = None) -> pd.DataFrame:
    """Mide las variantes de cada fase y devuelve una fila por medición. Con `out`, guarda el CSV
    tras cada fila (un fallo interno de TensorFlow que aborte el proceso conserva lo medido)."""
    import tensorflow as tf

    from src import train as T
    from src.utils import set_seed

    unknown = sorted(set(variants) - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(f"Variantes desconocidas: {unknown}")
    model_cfg = speed_model_config(model_name, backbone)
    tcfg = cfg["train"]
    accum = int(accumulation) if accumulation else T.accumulation_steps(tcfg, model_cfg)
    micro = D.micro_batch_size(int(tcfg["batch_size"]), accum)
    size = int(tcfg["image_size"])
    splits = D.load_splits(cfg["paths"]["splits_csv"])
    classes = D.class_names(cfg)
    train_df = D.select_split(splits, "train", val_fold)
    cw = D.class_weights(train_df["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy(), len(classes))

    T.configure_runtime(cfg)
    set_seed(seed, deterministic=False)
    model = build_speed_model(cfg, model_cfg)

    base = {"model": model_name, "backbone": backbone or model_cfg.get("backbone"), "batch_size": int(tcfg["batch_size"]),
            "micro_batch_size": micro, "gradient_accumulation_steps": accum, "steps": steps,
            "warmup_steps": warmup_steps, "n_train": len(train_df), "cache_default": D.cache_mode(tcfg)}
    rows = []
    sources_cache: dict = {}

    def sources_for(vcfg: dict) -> D.SplitSource:
        key = _cache_key(vcfg)
        if key not in sources_cache:
            for k in [k for k in sources_cache if k != key and k[0] == "memory"]:  # libera la RAM
                del sources_cache[k]
            gc.collect()
            src = D._split_source(vcfg, train_df, classes)
            fill = prefill(src) if key[0] == "memory" else 0.0
            sources_cache[key] = src
            log.info(f"{model_name} | caché {key}: preparada en {fill:.1f}s | RSS {process_memory_gib()['rss_gib']:.2f} GiB")
        return sources_cache[key]

    def record(phase, variant, jit, result, status="ok", error=None):
        row = {**base, "phase": phase, "variant": variant, "jit_requested": jit, "status": status, **result,
               **gpu_memory_gib(), **process_memory_gib(), **precision_info(model), "error": error}
        rows.append(row)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)
        ips = row.get("images_per_s")
        log.info(f"{model_name} | fase {phase} | {variant:22s} | {status} | "
                 + (f"{ips:7.1f} img/s | {row.get('step_s_mean', float('nan')) * 1e3:7.1f} ms/paso | "
                    f"GPU {row.get('gpu_util_mean') or float('nan'):5.1f} % | "
                    f"pico TF {row.get('gpu_peak_tf_gib') or float('nan'):.2f} GiB | "
                    f"RSS {row.get('rss_gib') or float('nan'):.2f} GiB" if ips else str(error)[:200]))

    def run_variant(phase, variant, jit):
        reset_gpu_peak()
        try:
            if variant in SYNTHETIC_VARIANTS:
                res = measure_synthetic(model, cw, micro, size, warmup_steps, steps, seed)
            elif variant == "val_predict":
                vsrc = D._split_source(cfg, D.select_split(splits, "val", val_fold), classes)
                res = measure_predict(model, vsrc.base.batch(micro).prefetch(tf.data.AUTOTUNE), len(vsrc.labels))
            else:
                vcfg = _variant_cfg(cfg, variant)
                src = sources_for(vcfg)
                ds = D.make_datasets(vcfg, val_fold, seed, sources={"train": src, "val": src},
                                     accumulation_steps=accum).train
                res = measure_fit(model, ds, cw, warmup_steps, steps, micro)
            record(phase, variant, jit, res)
        except Exception as e:  # se registra un OOM y se sigue con la siguiente variante
            if not _is_oom(e):
                raise
            record(phase, variant, jit, {}, status="OOM", error=type(e).__name__)
            gc.collect()

    jit_on = [v for v in variants if v not in ("pipeline_nojit", "synthetic_gpu_nojit")]
    jit_off = [v for v in variants if v in ("pipeline_nojit", "synthetic_gpu_nojit")]
    # Las cachés en memoria float32 van al final de cada fase para no tenerlas junto a las demás.
    order = sorted(jit_on, key=lambda v: v == "cache_memory_float32")
    for phase in phases:
        for jit, group in (("auto", order), (False, jit_off)):
            if not group:
                continue
            compile_for_phase(model, cfg, model_cfg, phase, accum, jit)
            for variant in group:
                if variant == "val_predict" and phase != phases[-1]:
                    continue  # la predicción no depende de la fase: se mide una vez
                run_variant(phase, variant, jit)
        sources_cache = {k: v for k, v in sources_cache.items() if k[0] != "memory"}
        gc.collect()
    model = None
    keras.backend.clear_session()
    gc.collect()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Presupuesto de cómputo (experiments/presupuesto_computo.md)
# ---------------------------------------------------------------------------

BUDGET_MODELS = ("mobilenetv2", "resnet50", "vgg16", "densenet201", "inceptionv3", "convnext_tiny", "vit_b16",
                 "efficientnetb0", "cci")
BUDGET = {"tuning": {"runs": 6, "epochs_a": 5, "epochs_b": 15},
          "final_worst": {"runs": 5, "epochs_a": 5, "epochs_b": 30},
          "final_typical": {"runs": 5, "epochs_a": 5, "epochs_b": 15},
          # Protocolo original (antes de D1 a D3): 12 configuraciones y 25 corridas por modelo
          "tuning_original": {"runs": 12, "epochs_a": 5, "epochs_b": 15},
          "final_original_worst": {"runs": 25, "epochs_a": 5, "epochs_b": 30},
          "final_original_typical": {"runs": 25, "epochs_a": 5, "epochs_b": 15}}


def _row(df: pd.DataFrame, phase: str, variant: str) -> pd.Series | None:
    sel = df[(df["phase"] == phase) & (df["variant"] == variant) & (df["status"] == "ok")]
    return sel.iloc[-1] if len(sel) else None


def model_budget(df: pd.DataFrame, n: dict, startup_s: float) -> dict:
    """Segundos por época y horas del ajuste y de la ejecución definitiva de un modelo a partir de
    su CSV de src.speed (variante pipeline de las fases A y B y val_predict). `n`: imágenes de
    train, val y test por corrida."""
    a, b, p = _row(df, "A", "pipeline"), _row(df, "B", "pipeline"), _row(df, "B", "val_predict")
    if a is None or b is None or p is None:
        raise ValueError("faltan las filas pipeline (fases A y B) o val_predict")
    val_s = n["val"] / p["images_per_s"]
    epoch_a = n["train"] / a["images_per_s"] + val_s
    epoch_b = n["train"] / b["images_per_s"] + val_s
    predict_compile = max(p["predict_first_s"] - p["predict_s"], 0.0)
    # Por corrida: arranque del proceso y carga de datos; en cada fase, dos compilaciones del paso
    # (lote completo y último lote incompleto, cada una ~ primer paso menos un paso normal) y la de
    # la predicción (se recompila tras cada compile); y la predicción final de val.
    overhead = (startup_s + 2 * (a["first_step_s"] - a["step_s_mean"]) + 2 * (b["first_step_s"] - b["step_s_mean"])
                + 2 * predict_compile + val_s)
    final_eval = (n["test"] + n["train"]) / p["images_per_s"] + 2 * predict_compile  # test y train sin aumento
    hours = {}
    for key, spec in BUDGET.items():
        per_run = spec["epochs_a"] * epoch_a + spec["epochs_b"] * epoch_b + overhead
        if key.startswith("final"):
            per_run += final_eval
        hours[key] = spec["runs"] * per_run / 3600
    peak = df.loc[df["variant"].isin(["pipeline", "val_predict"]) & (df["status"] == "ok"), "gpu_peak_tf_gib"]
    smi = df.loc[df["variant"].isin(["pipeline", "val_predict"]) & (df["status"] == "ok"), "gpu_mem_smi_max_mib"]
    return {
        "img_s_a": a["images_per_s"], "img_s_b": b["images_per_s"], "epoch_a_s": epoch_a, "epoch_b_s": epoch_b,
        "gpu_peak_tf_gib": float(peak.max()), "gpu_smi_max_gib": float(smi.max()) / 1024,
        "micro_batch_size": int(b["micro_batch_size"]), "accumulation": int(b["gradient_accumulation_steps"]),
        "overhead_s": overhead, **{f"h_{k}": v for k, v in hours.items()},
        "h_total_worst": hours["tuning"] + hours["final_worst"],
        "h_total_typical": hours["tuning"] + hours["final_typical"],
        "h_total_original_worst": hours["tuning_original"] + hours["final_original_worst"],
        "h_total_original_typical": hours["tuning_original"] + hours["final_original_typical"],
    }


def budget_frame(cfg: dict, speed_dir: Path, startup_s: float, models=BUDGET_MODELS) -> pd.DataFrame:
    splits = D.load_splits(cfg["paths"]["splits_csv"])
    folds = list(cfg["final"]["folds"])
    n = {"train": float(np.mean([len(D.select_split(splits, "train", f)) for f in folds])),
         "val": float(np.mean([len(D.select_split(splits, "val", f)) for f in folds])),
         "test": float(len(D.select_split(splits, "test", 0)))}
    rows = []
    for m in models:
        path = speed_dir / f"{m}.csv"
        if not path.exists():
            log.warning(f"Sin mediciones de {m}: {path}")
            continue
        rows.append({"model": m, **model_budget(pd.read_csv(path), n, startup_s)})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    from src.models import REGISTRY
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Velocidad y memoria del entrenamiento (sin escribir corridas).")
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--steps", type=int, default=50, help="pasos medidos tras el calentamiento")
    parser.add_argument("--warmup-steps", type=int, default=10, help="pasos de calentamiento (el primero compila)")
    parser.add_argument("--phases", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS), choices=list(ALL_VARIANTS))
    parser.add_argument("--accumulation", type=int, default=None,
                        help="gradient_accumulation_steps para esta medición (por defecto, el del modelo)")
    parser.add_argument("--backbone", default=None, help="backbone provisional de un modelo CCI (solo medición)")
    parser.add_argument("--out", default=None, help="CSV de salida (por defecto outputs/speed/<modelo>.csv)")
    parser.add_argument("--config", default=None, help="YAML base (por defecto configs/base.yaml)")
    args = parser.parse_args(argv)

    cfg = D.load_config(args.config)
    out = Path(args.out) if args.out else D.PROJECT_ROOT / "outputs" / "speed" / f"{args.model}.csv"
    setup_logging(cfg["paths"]["logs_dir"] / f"speed_{args.model}.log")
    log.info(f"python -m src.speed {' '.join(sys.argv[1:] if argv is None else argv)}")
    df = run_speed(cfg, args.model, phases=args.phases, variants=args.variants, steps=args.steps,
                   warmup_steps=args.warmup_steps, accumulation=args.accumulation, backbone=args.backbone,
                   out=out)
    log.info(f"{len(df)} filas -> {out}")


if __name__ == "__main__":
    main()
