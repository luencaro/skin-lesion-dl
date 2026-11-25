"""Ejecución definitiva: validación cruzada repetida semilla x pliegue (sección `final` de base.yaml).

    python -m src.final --model mobilenetv2

Las corridas son el producto final.seeds x final.folds, con los hiperparámetros de
configs/<modelo>.yaml, evaluación de test, métricas de train sin aumento ni dropout y pesos
guardados. Cada corrida va en outputs/runs/<modelo>/<final.label>/s<seed>_f<fold>/; si ya
terminó, se reutiliza (reanudable). El progreso y el tiempo restante se registran en
outputs/logs/final_<modelo>.log. Resultado: outputs/final/<modelo>/summary.csv (una fila por
corrida y filas mean, std, ci95_low y ci95_high).

Las corridas se recorren por pliegue (y por semilla dentro de cada pliegue) para reutilizar la
cache de imágenes de train, val y test entre las semillas de un mismo pliegue.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src.models import REGISTRY
from src.report import Missing
from src.results import add_summary_rows
from src.train import (
    _relpath,
    configure_runtime,
    load_model_config,
    load_run_summary,
    resolve_hparams,
    run_dir_for,
    run_training,
)
from src.utils import RunProgress, setup_logging

log = logging.getLogger(__name__)

SUMMARY_COLUMNS = ["seed", "val_fold", "val_balanced_accuracy", "test_balanced_accuracy", "train_balanced_accuracy",
                   "best_epoch", "epochs_warmup", "epochs_finetune", "train_time_s", "run_dir"]


def final_label(cfg: dict) -> str:
    return str(cfg["final"].get("label", "final"))


def final_pairs(cfg: dict) -> list[tuple[int, int]]:
    """(semilla, pliegue) de final.seeds x final.folds, ordenados por pliegue y luego por semilla."""
    fcfg = cfg["final"]
    seeds = [int(s) for s in fcfg["seeds"]]
    folds = [int(f) for f in fcfg["folds"]]
    return [(s, f) for f in folds for s in seeds]


def summary_path(cfg: dict, model_name: str) -> Path:
    return cfg["paths"]["final_dir"] / model_name / "summary.csv"


def _check_run(run_dir: Path, hp: dict, need_train: bool) -> dict:
    """Resumen de una corrida terminada; error si sus hiperparámetros difieren o le falta test/train."""
    summary = load_run_summary(run_dir)
    if not all(np.isclose(summary[k], v) for k, v in hp.items()):
        raise RuntimeError(f"{run_dir} se entrenó con otros hiperparámetros; bórralo para repetir la corrida")
    metrics = D.load_json(run_dir / "metrics.json")
    if "test" not in metrics or (need_train and "train" not in metrics):
        raise RuntimeError(f"{run_dir} no tiene métricas de test o de train; bórralo para repetir la corrida")
    return summary


def run_status(cfg: dict, model_name: str) -> pd.DataFrame:
    """Estado (terminada o pendiente) de cada corrida definitiva prevista, sin ejecutar nada."""
    label = final_label(cfg)
    rows = []
    for seed, fold in final_pairs(cfg):
        d = run_dir_for(cfg, model_name, label, seed, fold)
        done = (d / "metrics.json").exists()
        rows.append({"seed": seed, "val_fold": fold, "estado": "terminada" if done else "pendiente",
                     "run_dir": _relpath(d)})
    return pd.DataFrame(rows)


def finished_runs(cfg: dict, model_name: str) -> list[Path] | Missing:
    """Carpetas de las corridas definitivas terminadas (en el orden de final_pairs), o
    `Missing` si no hay ninguna."""
    label = final_label(cfg)
    dirs = [run_dir_for(cfg, model_name, label, s, f) for s, f in final_pairs(cfg)]
    done = [d for d in dirs if (d / "metrics.json").exists()]
    if not done:
        return Missing(f"no hay corridas definitivas terminadas de {model_name} en "
                       f"outputs/runs/{model_name}/{label}/. Ejecuta `python -m src.final --model {model_name}` "
                       f"o pon RERUN_FINAL = True.")
    return done


def summarize(rows: list[dict], alpha: float) -> pd.DataFrame:
    """Tabla por corrida con filas mean, std, ci95_low y ci95_high (t de Student) de las columnas numéricas."""
    runs = pd.DataFrame(rows)
    runs = runs[[c for c in SUMMARY_COLUMNS if c in runs.columns]]
    numeric = [c for c in runs.select_dtypes("number").columns if c not in ("seed", "val_fold")]
    table = add_summary_rows(runs.set_index(runs["seed"].astype(str) + "_" + runs["val_fold"].astype(str)),
                             numeric, alpha)
    table.insert(0, "row", ["run"] * len(runs) + ["mean", "std", "ci95_low", "ci95_high"])
    return table.reset_index(drop=True)


def run_final(cfg: dict, model_name: str) -> pd.DataFrame:
    """Ejecuta las corridas definitivas pendientes y reutiliza las terminadas. Guarda
    outputs/final/<modelo>/summary.csv y lo devuelve. ValueError si hay hiperparámetros null."""
    fcfg = cfg["final"]
    hp = resolve_hparams(load_model_config(model_name))
    label = final_label(cfg)
    evaluate_train = bool(fcfg.get("evaluate_train", True))
    pairs = final_pairs(cfg)
    alpha = float(cfg.get("benchmark", {}).get("alpha", 0.05))

    log_file = cfg["paths"]["logs_dir"] / f"final_{model_name}.log"
    setup_logging(log_file)
    log.info(f"Ejecución definitiva de {model_name} | {len(pairs)} corridas (semillas {fcfg['seeds']} x "
             f"pliegues {fcfg['folds']}) | log: {log_file}")
    configure_runtime(cfg)

    run_dirs = [run_dir_for(cfg, model_name, label, s, f) for s, f in pairs]
    progress = RunProgress(len(pairs), [i for i, d in enumerate(run_dirs) if (d / "metrics.json").exists()], log)

    rows = []
    sources, sources_fold = None, None  # cache de imágenes compartida por las semillas de un pliegue
    for i, (seed, fold) in enumerate(pairs):
        run_dir = run_dirs[i]
        desc = f"semilla {seed}, pliegue {fold}"
        if (run_dir / "metrics.json").exists():
            summary = _check_run(run_dir, hp, evaluate_train)
            progress.reused(i, f"{desc}; val_bal_acc {summary['val_balanced_accuracy']:.4f}")
        else:
            if sources_fold != fold:
                sources = None
                gc.collect()
                sources, sources_fold = D.load_split_sources(cfg, fold, include_test=True), fold
            progress.start(i, desc)
            summary = run_training(cfg, model_name, seed, fold, label=label, evaluate_test=True,
                                   evaluate_train=evaluate_train, save_weights=True, sources=sources)
            progress.finish(i, desc, f"val_bal_acc {summary['val_balanced_accuracy']:.4f}, "
                                     f"test_bal_acc {summary['test_balanced_accuracy']:.4f}")
        rows.append(summary)

    out = summary_path(cfg, model_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = summarize(rows, alpha)
    table.to_csv(out, index=False)
    log.info("Resumen de la ejecución definitiva:\n" + table.to_string(index=False))
    log.info(f"-> {out}")
    return table


def load_or_run_final(cfg: dict, model_name: str, rerun: bool = False) -> tuple[list[Path] | Missing, pd.DataFrame]:
    """Con `rerun`, llama a run_final (ejecuta las pendientes y reutiliza las terminadas). Sin
    `rerun`, no ejecuta nada. Devuelve (corridas terminadas o Missing, estado de cada corrida).
    Si run_final lanza ValueError (hiperparámetros null), devuelve un Missing con el mensaje."""
    if rerun:
        try:
            run_final(cfg, model_name)
        except ValueError as e:
            return Missing(f"no se pudo ejecutar: {e}"), run_status(cfg, model_name)
    return finished_runs(cfg, model_name), run_status(cfg, model_name)


def status_summary(status: pd.DataFrame) -> pd.DataFrame:
    """Corridas previstas, terminadas y pendientes."""
    counts = status["estado"].value_counts()
    return pd.Series({"previstas": len(status), "terminadas": int(counts.get("terminada", 0)),
                      "pendientes": int(counts.get("pendiente", 0))}, name="corridas").to_frame()


def config_table(cfg: dict) -> pd.DataFrame:
    """Parámetros de la ejecución definitiva (sección final de base.yaml)."""
    fcfg = cfg["final"]
    return pd.Series({"etiqueta": final_label(cfg), "semillas": fcfg["seeds"], "pliegues": fcfg["folds"],
                      "corridas": len(final_pairs(cfg)), "evaluación de test": True,
                      "métricas de train sin aumento": bool(fcfg.get("evaluate_train", True)),
                      "pesos guardados": True}, name="valor").to_frame()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ejecución definitiva: semillas x pliegues, con test.")
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--config", default=None, help="YAML base (por defecto configs/base.yaml)")
    args = parser.parse_args(argv)
    try:
        run_final(D.load_config(args.config), args.model)
    except ValueError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
