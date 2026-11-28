"""Entrenamiento de una corrida = (modelo, semilla, pliegue de validación, hiperparámetros).

Fase A: backbone congelado, AdamW con tasa constante (train.warmup).
Fase B: todo descongelado, AdamW con decaimiento coseno y parada temprana sobre la exactitud
balanceada de validación (train.finetune). Con los mejores pesos se calculan las métricas de
validación; solo con --test, las de test, y solo con --train-metrics, las del train de la
corrida sin aumento de datos ni dropout (para medir la brecha entre train y validación).

    python -m src.train --model mobilenetv2 --seed 0 --val-fold 0 [--test] [--train-metrics] [--label etiqueta]
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import yaml

from src import data as D
from src.metrics import SparseBalancedAccuracy, classification_metrics, cross_entropy
from src.models import REGISTRY, count_params, set_backbone_trainable
from src.utils import (
    enable_memory_growth,
    environment_info,
    format_duration,
    set_seed,
    setup_logging,
)

log = logging.getLogger(__name__)

CONFIGS_DIR = D.PROJECT_ROOT / "configs"
HPARAM_KEYS = ("learning_rate", "weight_decay", "dropout")


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------


def _read_model_yaml(name: str) -> dict:
    """configs/<name>.yaml; con `extends: <base>`, la config de <base> actualizada con esta
    (las secciones dict se combinan un nivel; hyperparameters no se hereda)."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    with open(path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}
    parent = model_cfg.pop("extends", None)
    if parent:
        base = _read_model_yaml(parent)
        base.pop("hyperparameters", None)
        for k, v in model_cfg.items():
            base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        model_cfg = base
    return model_cfg


def load_model_config(name: str) -> dict:
    """Lee configs/<name>.yaml y comprueba que el modelo esté en REGISTRY. Los modelos con
    `backbone` (el modelo original, src/original_model.py) heredan de configs/<backbone>.yaml las
    claves de src.original_model.INHERITED_KEYS que no fijan."""
    if name not in REGISTRY:
        raise ValueError(f"Modelo '{name}' no registrado. Disponibles: {sorted(REGISTRY)}")
    model_cfg = _read_model_yaml(name)
    if model_cfg.get("name") != name:
        raise ValueError(f"configs/{name}.yaml: name={model_cfg.get('name')!r} no coincide con '{name}'")
    if model_cfg.get("backbone"):
        from src.original_model import resolve_model_config

        model_cfg = resolve_model_config(model_cfg, load_model_config)
    return model_cfg


def resolve_hparams(model_cfg: dict, overrides: dict | None = None) -> dict:
    """Hiperparámetros del YAML del modelo, sustituidos por `overrides`. Error si alguno es null."""
    hp = dict(model_cfg.get("hyperparameters") or {})
    hp.update(overrides or {})
    missing = [k for k in HPARAM_KEYS if hp.get(k) is None]
    if missing:
        name = model_cfg.get("name")
        raise ValueError(
            f"configs/{name}.yaml tiene hiperparámetros en null: {missing}. "
            f"Ejecuta `python -m src.tune --model {name}` y copia los valores de "
            f"outputs/tuning/{name}/best.json en la sección hyperparameters."
        )
    return {k: float(hp[k]) for k in HPARAM_KEYS}


def configure_xla_env(cfg: dict) -> None:
    """Variables de entorno que XLA y la GPU leen al inicializarse (no cambian ningún valor de la
    configuración experimental): con train.xla_autotune_cache, los resultados del autotuning de
    XLA se guardan en paths.xla_cache_dir y los reutilizan las corridas siguientes (cada proceso
    hijo de isolate_runs volvería a medir los mismos algoritmos); con
    train.gpu_min_system_memory_mb, la memoria de la GPU que TensorFlow deja fuera de su asignador
    (por defecto 1536 MiB en GPU de capacidad 8.x); con train.gpu_allocator, TF_GPU_ALLOCATOR
    (cuda_malloc_async: el asignador de CUDA en lugar del BFC de TensorFlow). Llamar antes de la
    primera operación en la GPU."""
    import os

    tcfg = cfg["train"]
    if tcfg.get("xla_autotune_cache") and cfg["paths"].get("xla_cache_dir"):
        cache = Path(cfg["paths"]["xla_cache_dir"]) / "autotune"
        cache.mkdir(parents=True, exist_ok=True)
        if '"' in str(cache):
            raise ValueError(f"paths.xla_cache_dir no puede contener comillas: {cache}")
        # XLA_FLAGS se separa por espacios: la ruta va entre comillas (la del proyecto tiene espacios)
        flag = f'--xla_gpu_per_fusion_autotune_cache_dir="{cache}"'
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_gpu_per_fusion_autotune_cache_dir" not in flags:
            os.environ["XLA_FLAGS"] = f"{flags} {flag}".strip()
    if tcfg.get("gpu_min_system_memory_mb") is not None:
        os.environ.setdefault("TF_DEVICE_MIN_SYS_MEMORY_IN_MB", str(int(tcfg["gpu_min_system_memory_mb"])))
    if tcfg.get("gpu_allocator"):
        os.environ.setdefault("TF_GPU_ALLOCATOR", str(tcfg["gpu_allocator"]))


def configure_runtime(cfg: dict) -> None:
    """Memory growth en la GPU y política de precisión. Llamar antes de crear tensores."""
    tcfg = cfg["train"]
    configure_xla_env(cfg)
    if tcfg.get("gpu_memory_growth", True):
        enable_memory_growth()
    keras.mixed_precision.set_global_policy(
        "mixed_float16" if tcfg.get("mixed_precision") else "float32"
    )


def run_dir_for(cfg: dict, model_name: str, label: str, seed: int, val_fold: int) -> Path:
    return cfg["paths"]["runs_dir"] / model_name / label / f"s{seed}_f{val_fold}"


def _relpath(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(D.PROJECT_ROOT))
    except ValueError:
        return str(p)


def _save_yaml(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Optimizadores y callbacks
# ---------------------------------------------------------------------------


def accumulation_steps(tcfg: dict, model_cfg: dict) -> int:
    """gradient_accumulation_steps de configs/<modelo>.yaml si lo define; si no, el de train."""
    value = model_cfg.get("gradient_accumulation_steps", tcfg.get("gradient_accumulation_steps", 1))
    return int(value or 1)


@keras.saving.register_keras_serializable(package="skin_lesion_dl")
class UpdateStepSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Evalúa `schedule` en el número de actualizaciones, floor(micro-pasos / accumulation_steps).

    Con gradient_accumulation_steps, el optimizador de Keras 3 llama al programa de la tasa con
    su contador de micro-pasos; así el decaimiento coseno (decay_steps en actualizaciones) avanza
    una vez por actualización, igual que sin acumulación."""

    def __init__(self, schedule, accumulation_steps: int):
        self.schedule = schedule
        self.accumulation_steps = int(accumulation_steps)

    def __call__(self, step):
        return self.schedule(keras.ops.floor_divide(step, self.accumulation_steps))

    def get_config(self):
        return {"schedule": keras.saving.serialize_keras_object(self.schedule),
                "accumulation_steps": self.accumulation_steps}

    @classmethod
    def from_config(cls, config):
        return cls(keras.saving.deserialize_keras_object(config["schedule"]), config["accumulation_steps"])


def make_optimizer(name: str, learning_rate, weight_decay: float, exclude,
                   accumulation: int = 1) -> keras.optimizers.Optimizer:
    """AdamW; con `accumulation` > 1, gradient_accumulation_steps de Keras 3: el optimizador
    promedia los gradientes de `accumulation` micro-lotes y aplica una actualización (con el
    decaimiento de pesos) cada `accumulation` pasos."""
    if name != "adamw":
        raise ValueError(f"Optimizador no soportado: {name}")
    if accumulation > 1 and isinstance(learning_rate, keras.optimizers.schedules.LearningRateSchedule):
        learning_rate = UpdateStepSchedule(learning_rate, accumulation)
    opt = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay,
                                 gradient_accumulation_steps=accumulation if accumulation > 1 else None)
    if exclude:
        opt.exclude_from_weight_decay(var_names=[f"^{n}$" for n in exclude])
    return opt


def make_schedule(fcfg: dict, learning_rate: float, decay_steps: int):
    """`decay_steps` en actualizaciones del optimizador (no en micro-lotes)."""
    schedule = fcfg.get("schedule")
    if schedule == "cosine":
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            alpha=float(fcfg.get("cosine_alpha", 0.0)),
        )
    if schedule in (None, "constant"):
        return learning_rate
    raise ValueError(f"Schedule no soportado: {schedule}")


def current_lr(optimizer) -> float:
    try:
        return float(keras.ops.convert_to_numpy(optimizer.learning_rate))
    except Exception:  # noqa: BLE001 - solo informativo
        return float("nan")


class ValidationMonitor(keras.callbacks.Callback):
    """Al final de cada época predice val y añade a `logs` val_loss, val_accuracy,
    val_balanced_accuracy y val_f1_macro (numéricos: los leen EarlyStopping y la barra de Keras)."""

    def __init__(self, val_ds, y_val):
        super().__init__()
        self.val_ds = val_ds
        self.y_val = np.asarray(y_val)

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import balanced_accuracy_score, f1_score

        logs = logs if logs is not None else {}
        proba = self.model.predict(self.val_ds, verbose=0)
        y_pred = proba.argmax(axis=1)
        logs["val_loss"] = cross_entropy(self.y_val, proba)
        logs["val_accuracy"] = float((y_pred == self.y_val).mean())
        logs["val_balanced_accuracy"] = float(balanced_accuracy_score(self.y_val, y_pred))
        logs["val_f1_macro"] = float(f1_score(self.y_val, y_pred, average="macro", zero_division=0))


class EpochRecorder(keras.callbacks.Callback):
    """Añade a `logs` phase, global_epoch, learning_rate y epoch_time_s (columnas extra de
    history.csv) y registra con logging la línea resumida de la época.

    Va después de ProgbarLogger (que no admite valores de texto como `phase` y así cierra la
    barra antes de la línea resumida) y antes de CSVLogger."""

    def __init__(self, phase: str, epoch_offset: int, max_epochs: int, run_name: str):
        super().__init__()
        self.phase = phase
        self.epoch_offset = epoch_offset
        self.max_epochs = max_epochs
        self.run_name = run_name
        self.best = -np.inf  # mejor val_balanced_accuracy de esta fase
        self.epochs_done = 0
        self._t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        dt = time.perf_counter() - self._t0
        bal = float(logs.get("val_balanced_accuracy", float("nan")))
        logs["phase"] = self.phase
        logs["global_epoch"] = self.epoch_offset + epoch + 1
        logs["learning_rate"] = current_lr(self.model.optimizer)
        logs["epoch_time_s"] = dt
        self.best = max(self.best, bal)
        self.epochs_done += 1
        log.info(
            f"{self.run_name} | {self.phase} | época {epoch + 1}/{self.max_epochs} | "
            f"loss {float(logs.get('loss', float('nan'))):.4f} | val_bal_acc {bal:.4f} | "
            f"mejor {self.best:.4f} | {format_duration(dt)}"
        )


def phase_callbacks(data, recorder: EpochRecorder, history_csv: Path, append: bool, verbose: int,
                    extra=()) -> list:
    """Orden: ValidationMonitor -> `extra` (EarlyStopping) -> ProgbarLogger (si verbose) ->
    EpochRecorder -> CSVLogger. Todos comparten el mismo dict `logs`. La barra de Keras va a
    stdout con print, no por logging, así que no llega al archivo de outputs/logs/."""
    cbs = [ValidationMonitor(data.val, data.val_labels), *extra]
    if verbose:
        cbs.append(keras.callbacks.ProgbarLogger())  # explícito: Keras no añade otro
    cbs += [recorder, keras.callbacks.CSVLogger(str(history_csv), append=append)]
    return cbs


def compile_model(model, optimizer, n_classes: int) -> None:
    """Entropía cruzada; métricas de train: accuracy y balanced_accuracy (acumuladas en la época)."""
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"), SparseBalancedAccuracy(n_classes)],
    )


def predict_timed(model, ds, n: int) -> tuple[np.ndarray, float]:
    """Probabilidades y segundos por imagen (tras una predicción de calentamiento)."""
    model.predict(ds.take(1), verbose=0)
    t0 = time.perf_counter()
    proba = model.predict(ds, verbose=0)
    dt = time.perf_counter() - t0
    if len(proba) != n:
        raise RuntimeError(f"Se esperaban {n} predicciones y hay {len(proba)}")
    return proba.astype(np.float64), dt / n


def probabilities_frame(ids, y_true, proba, classes) -> pd.DataFrame:
    df = pd.DataFrame({"image_id": ids, "label": [classes[i] for i in y_true],
                       "pred": [classes[i] for i in proba.argmax(axis=1)]})
    for i, c in enumerate(classes):
        df[f"p_{c}"] = proba[:, i]
    return df


# ---------------------------------------------------------------------------
# Corrida
# ---------------------------------------------------------------------------


def build_model(cfg: dict, model_name: str, dropout: float | None = None) -> keras.Model:
    """Modelo de `model_name` con el número de clases de la config. `dropout` por defecto es el
    del YAML del modelo, o 0.0 si aún es null (no cambia el número de parámetros)."""
    model_cfg = load_model_config(model_name)
    if dropout is None:
        dropout = (model_cfg.get("hyperparameters") or {}).get("dropout") or 0.0
    return REGISTRY[model_name](model_cfg, len(D.class_names(cfg)), float(dropout))


def load_run_summary(run_dir: Path) -> dict:
    """Resumen de una corrida terminada (metrics.json + run_info.json)."""
    metrics = D.load_json(run_dir / "metrics.json")
    info = D.load_json(run_dir / "run_info.json")
    extra = {f"{split}_balanced_accuracy": metrics[split]["balanced_accuracy"]
             for split in ("test", "train") if split in metrics}
    return {
        "run_dir": _relpath(run_dir),
        "seed": info["seed"],
        "val_fold": info["val_fold"],
        "label": info["label"],
        **info["hyperparameters"],
        "val_balanced_accuracy": metrics["val"]["balanced_accuracy"],
        **extra,
        "best_epoch": info["epochs"]["finetune_best"],
        "epochs_warmup": info["epochs"]["warmup"],
        "epochs_finetune": info["epochs"]["finetune_run"],
        "train_time_s": info["train_time_s"]["total"],
    }


def run_training(
    cfg: dict,
    model_name: str,
    seed: int,
    val_fold: int,
    *,
    hparams: dict | None = None,
    label: str = "default",
    evaluate_test: bool = False,
    evaluate_train: bool = False,
    max_epochs: int | None = None,
    save_weights: bool | None = None,
    sources: dict | None = None,
    overwrite: bool = False,
    hparams_source: dict | None = None,
) -> dict:
    """Ejecuta una corrida y guarda sus artefactos. Devuelve load_run_summary(run_dir).

    `hparams` sustituye a los del YAML del modelo; `max_epochs` a finetune.max_epochs;
    `sources` (de data.load_split_sources con el mismo val_fold) reutiliza la cache.
    `hparams_source` (de dónde salen los hiperparámetros, p. ej. best.json del ajuste) se registra
    en config.yaml (run.hyperparameters_source).
    Con `evaluate_train`, metrics.json incluye también "train": las métricas sobre las imágenes
    de train de la corrida sin aumento y en modo inferencia (sin dropout).
    """
    model_cfg = load_model_config(model_name)
    hp = resolve_hparams(model_cfg, hparams)
    tcfg = copy.deepcopy(cfg["train"])
    if max_epochs is not None:
        tcfg["finetune"]["max_epochs"] = int(max_epochs)
    if save_weights is None:
        save_weights = bool(tcfg.get("save_weights", False))
    if int(model_cfg["input_size"]) != int(tcfg["image_size"]):
        raise ValueError(
            f"input_size de {model_name} ({model_cfg['input_size']}) != train.image_size ({tcfg['image_size']})"
        )

    run_dir = run_dir_for(cfg, model_name, label, seed, val_fold)
    if (run_dir / "metrics.json").exists() and not overwrite:
        raise FileExistsError(f"{_relpath(run_dir)} ya tiene metrics.json; usa --overwrite u otra --label")
    run_dir.mkdir(parents=True, exist_ok=True)

    keras.backend.clear_session()  # borra también la política de precisión global
    configure_runtime(cfg)
    set_seed(seed, deterministic=bool(tcfg.get("deterministic", False)))

    accum = accumulation_steps(tcfg, model_cfg)
    micro = D.micro_batch_size(int(tcfg["batch_size"]), accum)
    data = D.make_datasets(cfg, val_fold, seed, include_test=evaluate_test, sources=sources,
                           accumulation_steps=accum)
    classes = data.class_names
    k = len(classes)
    if tcfg.get("class_weight") == "inverse_frequency":
        cw = D.class_weights(data.train_labels, k)
    elif tcfg.get("class_weight") in (None, "none"):
        cw = None
    else:
        raise ValueError(f"class_weight no soportado: {tcfg.get('class_weight')}")

    resolved = {
        "run": {"model": model_name, "label": label, "seed": seed, "val_fold": val_fold,
                "evaluate_test": evaluate_test, "evaluate_train": evaluate_train,
                "save_weights": save_weights, "micro_batch_size": micro, "gradient_accumulation_steps": accum,
                **({"hyperparameters_source": hparams_source} if hparams_source else {})},
        "model": {**model_cfg, "hyperparameters": hp},
        "train": tcfg,
        "classes": classes,
        "class_weights": {classes[i]: w for i, w in (cw or {}).items()},
        "paths": {"splits_csv": _relpath(cfg["paths"]["splits_csv"]),
                  "images_dir": _relpath(cfg["paths"]["images_dir"])},
    }
    _save_yaml(resolved, run_dir / "config.yaml")

    model = build_model(cfg, model_name, hp["dropout"])
    exclude = list(tcfg.get("weight_decay_exclude") or []) + list(model_cfg.get("weight_decay_exclude") or [])
    params = {}
    times = {}
    n_train = len(data.train_labels)
    steps = math.ceil(n_train / int(tcfg["batch_size"]))  # actualizaciones por época (lote efectivo)
    run_name = f"{model_name} | {label} s{seed}_f{val_fold}"
    history_csv = run_dir / "history.csv"
    verbose = int(tcfg.get("verbose", 1))

    # Fase A: backbone congelado
    wcfg = tcfg["warmup"]
    epochs_a = int(wcfg.get("epochs", 0))
    set_backbone_trainable(model, False)
    params["warmup"] = count_params(model)
    t0 = time.perf_counter()
    if epochs_a > 0:
        compile_model(model, make_optimizer(wcfg.get("optimizer", "adamw"), float(wcfg["learning_rate"]),
                                            hp["weight_decay"], exclude, accum), k)
        log.info(f"{run_name} | Fase A: {epochs_a} épocas, backbone congelado")
        recorder_a = EpochRecorder("warmup", 0, epochs_a, run_name)
        model.fit(data.train, epochs=epochs_a, class_weight=cw, verbose=verbose,
                  callbacks=phase_callbacks(data, recorder_a, history_csv, append=False, verbose=verbose))
    times["warmup"] = time.perf_counter() - t0

    # Fase B: todo descongelado, coseno + parada temprana
    fcfg = tcfg["finetune"]
    epochs_b = int(fcfg["max_epochs"])
    set_backbone_trainable(model, True)
    params["finetune"] = count_params(model)
    es = fcfg["early_stopping"]
    early = keras.callbacks.EarlyStopping(
        monitor=es["monitor"], mode=es["mode"], patience=int(es["patience"]),
        min_delta=float(es.get("min_delta", 0.0)),
        restore_best_weights=bool(es.get("restore_best_weights", True)), verbose=0,
    )
    lr = make_schedule(fcfg, hp["learning_rate"], decay_steps=epochs_b * steps)
    compile_model(model, make_optimizer(fcfg.get("optimizer", "adamw"), lr, hp["weight_decay"], exclude, accum), k)
    log.info(f"{run_name} | Fase B: hasta {epochs_b} épocas, todo descongelado")
    recorder_b = EpochRecorder("finetune", epochs_a, epochs_b, run_name)
    t0 = time.perf_counter()
    model.fit(data.train, epochs=epochs_b, class_weight=cw, verbose=verbose,
              callbacks=phase_callbacks(data, recorder_b, history_csv, append=epochs_a > 0,  # tras la fase A
                                        verbose=verbose, extra=[early]))
    times["finetune"] = time.perf_counter() - t0
    times["total"] = times["warmup"] + times["finetune"]
    epochs_run_b = recorder_b.epochs_done
    best_b = int(early.best_epoch) + 1 if early.best_weights is not None else epochs_run_b
    if early.stopped_epoch > 0:
        log.info(f"{run_name} | parada temprana en la época {early.stopped_epoch + 1}")
    log.info(f"{run_name} | se restauran los pesos de la época {best_b} de la fase B")

    # Evaluación con los mejores pesos
    ece_bins = int(tcfg["ece_bins"])
    metrics = {}
    inference = {}
    proba_val, inference["val"] = predict_timed(model, data.val, len(data.val_labels))
    metrics["val"] = classification_metrics(data.val_labels, proba_val, classes, ece_bins)
    probabilities_frame(data.val_ids, data.val_labels, proba_val, classes).to_csv(
        run_dir / "probabilities_val.csv", index=False)
    if evaluate_test:
        proba_test, inference["test"] = predict_timed(model, data.test, len(data.test_labels))
        metrics["test"] = classification_metrics(data.test_labels, proba_test, classes, ece_bins)
        probabilities_frame(data.test_ids, data.test_labels, proba_test, classes).to_csv(
            run_dir / "probabilities_test.csv", index=False)
    if evaluate_train:  # sin aumento; predict usa el modo inferencia (dropout inactivo)
        proba_train, inference["train"] = predict_timed(model, data.train_eval, n_train)
        metrics["train"] = classification_metrics(data.train_labels, proba_train, classes, ece_bins)

    if save_weights:
        model.save_weights(run_dir / "model.weights.h5")

    info = {
        "model": model_name,
        "label": label,
        "seed": seed,
        "val_fold": val_fold,
        "hyperparameters": hp,
        "evaluated_test": evaluate_test,
        "evaluated_train": evaluate_train,
        "n_images": {"train": n_train, "val": len(data.val_labels),
                     "test": len(data.test_labels) if evaluate_test else None},
        "epochs": {"warmup": epochs_a, "finetune_max": epochs_b, "finetune_run": epochs_run_b,
                   "finetune_best": best_b},
        "best_val_balanced_accuracy_during_training": float(early.best) if early.best_weights is not None else None,
        "train_time_s": times,
        "inference_s_per_image": inference,
        "inference_batch_size": micro,
        "batch_size": int(tcfg["batch_size"]),            # lote efectivo de cada actualización
        "micro_batch_size": micro,                         # lote del tf.data
        "gradient_accumulation_steps": accum,
        "params": params,
        "mixed_precision": bool(tcfg.get("mixed_precision")),
        "deterministic": bool(tcfg.get("deterministic")),
        "environment": environment_info(),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    D.save_json(info, run_dir / "run_info.json")
    D.save_json(metrics, run_dir / "metrics.json")  # se escribe al final: marca la corrida como terminada
    from src.tracking import safe_log_run

    safe_log_run(cfg, run_dir)  # MLflow (mlflow.enabled); un fallo solo avisa

    log.info(f"{run_name} | val balanced_accuracy = {metrics['val']['balanced_accuracy']:.4f} "
             f"(época {best_b} de la fase B) | entrenamiento {format_duration(times['total'])} -> {_relpath(run_dir)}")
    del model, data
    keras.backend.clear_session()
    gc.collect()
    return load_run_summary(run_dir)


def _isolated_child(cfg: dict, model_name: str, seed: int, val_fold: int, kwargs: dict, log_file) -> None:
    if log_file is not None:
        setup_logging(log_file)
    configure_runtime(cfg)
    run_training(cfg, model_name, seed, val_fold, **kwargs)


def run_training_isolated(cfg: dict, model_name: str, seed: int, val_fold: int, *, log_file=None,
                          **kwargs) -> dict:
    """run_training en un proceso hijo (multiprocessing con spawn) y load_run_summary de la corrida.

    Con XLA (jit_compile="auto" en la GPU), cada modelo compilado deja en el proceso alrededor de
    1 GB de RAM que keras.backend.clear_session() no libera; al terminar el hijo, el sistema la
    recupera. El hijo carga sus propias imágenes (con train.cache = disk solo lee la caché en
    disco), así que `sources` no se admite. RuntimeError si el hijo no termina la corrida."""
    import multiprocessing as mp

    if kwargs.get("sources") is not None:
        raise ValueError("run_training_isolated no admite `sources`: cada proceso carga sus imágenes")
    kwargs.pop("sources", None)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_isolated_child, args=(cfg, model_name, seed, val_fold, kwargs, log_file),
                       name=f"train-{model_name}-s{seed}-f{val_fold}")
    proc.start()
    proc.join()
    run_dir = run_dir_for(cfg, model_name, kwargs.get("label", "default"), seed, val_fold)
    if not (run_dir / "metrics.json").exists():
        hint = " (señal 9: p. ej. el OOM killer del sistema)" if proc.exitcode == -9 else ""
        raise RuntimeError(f"La corrida {_relpath(run_dir)} no terminó: el proceso hijo salió con código "
                           f"{proc.exitcode}{hint}")
    if proc.exitcode != 0:  # metrics.json se escribe al final: la corrida terminó y el fallo fue al liberar
        log.warning(f"{_relpath(run_dir)}: terminada, pero el proceso hijo salió con código {proc.exitcode}")
    return load_run_summary(run_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Entrena una corrida (modelo, semilla, pliegue de validación).")
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--val-fold", type=int, required=True)
    parser.add_argument("--test", action="store_true", help="evalúa también en test")
    parser.add_argument("--train-metrics", action="store_true",
                        help="calcula también las métricas de train sin aumento ni dropout")
    parser.add_argument("--label", default="default", help="subcarpeta en outputs/runs/<modelo>/")
    parser.add_argument("--overwrite", action="store_true", help="sobrescribe una corrida terminada")
    parser.add_argument("--config", default=None, help="YAML base (por defecto configs/base.yaml)")
    args = parser.parse_args(argv)

    cfg = D.load_config(args.config)
    try:
        resolve_hparams(load_model_config(args.model))
    except ValueError as e:
        sys.exit(f"Error: {e}")
    log_file = cfg["paths"]["logs_dir"] / f"train_{args.model}.log"
    setup_logging(log_file)
    log.info(f"python -m src.train {' '.join(sys.argv[1:] if argv is None else argv)} | log: {_relpath(log_file)}")
    configure_runtime(cfg)
    run_training(cfg, args.model, args.seed, args.val_fold, label=args.label,
                 evaluate_test=args.test, evaluate_train=args.train_metrics, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
