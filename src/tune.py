"""Búsqueda aleatoria de hiperparámetros sobre un pliegue de validación (nunca evalúa test).

    python -m src.tune --model mobilenetv2

Las configuraciones dependen solo de train.tuning (space, n_trials, sampler_seed), así que
todos los modelos reciben las mismas. Cada ensayo es una corrida en
outputs/runs/<modelo>/tuning_tXX/s<seed>_f<fold>/; si ya terminó, se reutiliza (reanudable).
Resultados: outputs/tuning/<modelo>/trials.csv y best.json. No modifica configs/<modelo>.yaml.
La selección considera solo t00 a t(n_trials - 1), después de comprobar que sus hiperparámetros
coinciden con el muestreador. Los ensayos terminados con índice mayor (p. ej., t06 de MobileNetV2,
evaluado antes de la desviación D1) se conservan y se listan en trials.csv con excluded = True.
"""

from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import pandas as pd

from src import data as D
from src.models import REGISTRY
from src.train import (
    configure_runtime,
    load_model_config,
    load_run_summary,
    run_dir_for,
    run_training,
    run_training_isolated,
)
from src.utils import RunProgress, set_seed, setup_logging

log = logging.getLogger(__name__)


def sample_configs(space: dict, n_trials: int, sampler_seed: int) -> list[dict]:
    """`n_trials` configuraciones muestreadas en el orden de `space` con default_rng(sampler_seed)."""
    rng = np.random.default_rng(sampler_seed)
    out = []
    for _ in range(n_trials):
        hp = {}
        for name, spec in space.items():
            kind = spec["type"]
            if kind == "loguniform":
                hp[name] = float(math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"]))))
            elif kind == "uniform":
                hp[name] = float(rng.uniform(spec["low"], spec["high"]))
            elif kind == "choice":
                hp[name] = spec["values"][int(rng.integers(len(spec["values"])))]
            else:
                raise ValueError(f"Tipo de distribución no soportado para {name}: {kind}")
        out.append(hp)
    return out


SUMMARY_KEYS = ("val_balanced_accuracy", "best_epoch", "epochs_warmup", "epochs_finetune", "train_time_s", "run_dir")
TRIAL_PREFIX = "tuning_t"


def check_trials_match_sampler(trials: pd.DataFrame, configs: list[dict]) -> None:
    """RuntimeError si algún ensayo incluido no tiene los hiperparámetros de su índice en el muestreador
    (p. ej., t00 a t05 ya evaluados frente a sample_configs con n_trials = 6)."""
    for _, row in trials[~trials["excluded"]].iterrows():
        hp = configs[int(row["trial"])]
        if not all(np.isclose(float(row[k]), float(v), rtol=0.0, atol=0.0) for k, v in hp.items()):
            raise RuntimeError(f"El ensayo {int(row['trial'])} ({row['run_dir']}) no coincide con el muestreador: "
                               f"{ {k: row[k] for k in hp} } frente a {hp}")


def excluded_trials(cfg: dict, model_name: str, seed: int, val_fold: int, n_trials: int, space) -> list[dict]:
    """Ensayos terminados con índice >= n_trials (p. ej., t06 de MobileNetV2, evaluado antes de la
    desviación D1): se conservan y se listan como excluidos de la selección; no se borran."""
    model_dir = cfg["paths"]["runs_dir"] / model_name
    rows = []
    for d in sorted(model_dir.glob(f"{TRIAL_PREFIX}[0-9][0-9]")) if model_dir.exists() else []:
        i = int(d.name[len(TRIAL_PREFIX):])
        run_dir = run_dir_for(cfg, model_name, d.name, seed, val_fold)
        if i < n_trials or not (run_dir / "metrics.json").exists():
            continue
        summary = load_run_summary(run_dir)
        rows.append({"trial": i, **{k: summary[k] for k in space}, **{k: summary[k] for k in SUMMARY_KEYS},
                     "excluded": True, "excluded_reason": f"índice >= n_trials ({n_trials}); desviación D1"})
    return rows


def run_tuning(cfg: dict, model_name: str) -> pd.DataFrame:
    """Ejecuta (o reanuda) la búsqueda aleatoria de train.tuning para `model_name`, sin test.

    Guarda outputs/tuning/<modelo>/trials.csv y best.json y devuelve la tabla de ensayos.
    Registra el progreso en outputs/logs/tune_<modelo>.log.
    """
    tun = cfg["train"]["tuning"]
    if tun.get("method") != "random":
        raise ValueError(f"tuning.method no soportado: {tun.get('method')}")
    load_model_config(model_name)
    seed, val_fold = int(tun["seed"]), int(tun["val_fold"])
    configs = sample_configs(tun["space"], int(tun["n_trials"]), int(tun["sampler_seed"]))

    out_dir = cfg["paths"]["tuning_dir"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg["paths"]["logs_dir"] / f"tune_{model_name}.log"
    setup_logging(log_file)
    log.info(f"Ajuste de hiperparámetros de {model_name} | log: {log_file}")
    isolate = bool(cfg["train"].get("isolate_runs", False))
    if not isolate:  # con isolate_runs, cada hijo configura la GPU y las semillas
        configure_runtime(cfg)
        set_seed(seed, deterministic=bool(cfg["train"].get("deterministic", False)))
    sources = None  # cache de train/val compartida por todos los ensayos (sin isolate_runs)

    labels = [f"tuning_t{i:02d}" for i in range(len(configs))]
    run_dirs = [run_dir_for(cfg, model_name, lb, seed, val_fold) for lb in labels]
    progress = RunProgress(len(configs), [i for i, d in enumerate(run_dirs) if (d / "metrics.json").exists()], log)

    rows = []
    for i, hp in enumerate(configs):
        label, run_dir = labels[i], run_dirs[i]
        desc = f"{label}: " + ", ".join(f"{k}={v:.3g}" for k, v in hp.items())
        if (run_dir / "metrics.json").exists():
            summary = load_run_summary(run_dir)
            stored = {k: summary[k] for k in hp}
            if not all(np.isclose(stored[k], hp[k]) for k in hp):
                raise RuntimeError(f"{run_dir} tiene otros hiperparámetros ({stored}); bórralo para repetir el ajuste")
            progress.reused(i, f"{desc}; val_bal_acc {summary['val_balanced_accuracy']:.4f}")
        else:
            progress.start(i, desc)
            kwargs = {"hparams": hp, "label": label, "evaluate_test": False, "max_epochs": int(tun["max_epochs"]),
                      "save_weights": False}
            if isolate:
                summary = run_training_isolated(cfg, model_name, seed, val_fold, log_file=log_file, **kwargs)
            else:
                if sources is None:
                    sources = D.load_split_sources(cfg, val_fold, include_test=False)
                summary = run_training(cfg, model_name, seed, val_fold, sources=sources, **kwargs)
            progress.finish(i, label, f"val_bal_acc {summary['val_balanced_accuracy']:.4f}")
        rows.append({"trial": i, **hp, **{k: summary[k] for k in SUMMARY_KEYS}, "excluded": False,
                     "excluded_reason": ""})
        pd.DataFrame(rows).to_csv(out_dir / "trials.csv", index=False)

    rows += excluded_trials(cfg, model_name, seed, val_fold, len(configs), tun["space"])
    trials = pd.DataFrame(rows)
    trials.to_csv(out_dir / "trials.csv", index=False)
    check_trials_match_sampler(trials, configs)  # antes de seleccionar
    included = trials[~trials["excluded"]]
    best = included.sort_values(["val_balanced_accuracy", "trial"], ascending=[False, True]).iloc[0]
    D.save_json(
        {
            "model": model_name,
            "trial": int(best["trial"]),
            "hyperparameters": {k: float(best[k]) for k in tun["space"]},
            "val_balanced_accuracy": float(best["val_balanced_accuracy"]),
            "best_epoch": int(best["best_epoch"]),
            "selection": "máxima val_balanced_accuracy; en empate, el ensayo de menor índice",
            "tuning": tun,
            "n_trials": len(included),
            "excluded_trials": [int(t) for t in trials.loc[trials["excluded"], "trial"]],
            "run_dir": best["run_dir"],
        },
        out_dir / "best.json",
    )
    log.info("Ensayos:\n" + trials.to_string(index=False))
    log.info(f"Mejor ensayo: {int(best['trial'])} -> {out_dir / 'best.json'}")
    log.info(f"Copia sus hiperparámetros en configs/{model_name}.yaml (sección hyperparameters).")
    return trials


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Búsqueda aleatoria de hiperparámetros (sin test).")
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--config", default=None, help="YAML base (por defecto configs/base.yaml)")
    args = parser.parse_args(argv)
    run_tuning(D.load_config(args.config), args.model)


if __name__ == "__main__":
    main()
