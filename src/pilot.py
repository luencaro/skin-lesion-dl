"""Piloto de variabilidad: corridas (semilla, pliegue) de train.pilot.pairs, sin test.

    python -m src.pilot --model mobilenetv2

Usa los hiperparámetros de configs/<modelo>.yaml. Cada corrida va en
outputs/runs/<modelo>/pilot/s<seed>_f<fold>/; si ya terminó, se reutiliza (reanudable).
Resultado: outputs/pilot/<modelo>/summary.csv con la exactitud balanceada de validación de
cada corrida y, en filas aparte, media, desviación estándar (ddof = 1) y coeficiente de variación.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from src import data as D
from src.models import REGISTRY
from src.train import (
    configure_runtime,
    load_model_config,
    load_run_summary,
    resolve_hparams,
    run_dir_for,
    run_training,
)
from src.utils import RunProgress, setup_logging

LABEL = "pilot"
log = logging.getLogger(__name__)


def summarize(runs: pd.DataFrame, col: str = "val_balanced_accuracy") -> pd.DataFrame:
    """Añade filas mean, std (ddof = 1) y cv = std / mean de `col`."""
    x = runs[col].to_numpy(dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    stats = pd.DataFrame({"row": ["mean", "std", "cv"], col: [mean, std, std / mean if mean else float("nan")]})
    runs = runs.assign(row="run")
    return pd.concat([runs, stats], ignore_index=True)[["row"] + [c for c in runs.columns if c != "row"]]


def run_pilot(cfg: dict, model_name: str) -> pd.DataFrame:
    """Ejecuta (o reanuda) las corridas de train.pilot.pairs con los hiperparámetros de
    configs/<modelo>.yaml, sin test. Guarda outputs/pilot/<modelo>/summary.csv y lo devuelve.
    Registra el progreso en outputs/logs/pilot_<modelo>.log. ValueError si hay hiperparámetros null.
    """
    hp = resolve_hparams(load_model_config(model_name))
    pairs = [(int(s), int(f)) for s, f in cfg["train"]["pilot"]["pairs"]]
    out_dir = cfg["paths"]["pilot_dir"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg["paths"]["logs_dir"] / f"pilot_{model_name}.log"
    setup_logging(log_file)
    log.info(f"Piloto de {model_name} | log: {log_file}")
    configure_runtime(cfg)

    run_dirs = [run_dir_for(cfg, model_name, LABEL, s, f) for s, f in pairs]
    progress = RunProgress(len(pairs), [i for i, d in enumerate(run_dirs) if (d / "metrics.json").exists()], log)

    rows = []
    for i, (seed, fold) in enumerate(pairs):
        run_dir = run_dirs[i]
        desc = f"semilla {seed}, pliegue {fold}"
        if (run_dir / "metrics.json").exists():
            summary = load_run_summary(run_dir)
            if not all(np.isclose(summary[k], v) for k, v in hp.items()):
                raise RuntimeError(f"{run_dir} se entrenó con otros hiperparámetros; bórralo para repetir el piloto")
            progress.reused(i, f"{desc}; val_bal_acc {summary['val_balanced_accuracy']:.4f}")
        else:
            progress.start(i, desc)
            summary = run_training(cfg, model_name, seed, fold, label=LABEL, evaluate_test=False, save_weights=False)
            progress.finish(i, desc, f"val_bal_acc {summary['val_balanced_accuracy']:.4f}")
        rows.append({k: summary[k] for k in ("seed", "val_fold", "val_balanced_accuracy", "best_epoch",
                                             "epochs_warmup", "epochs_finetune", "train_time_s", "run_dir")})

    table = summarize(pd.DataFrame(rows))
    table.to_csv(out_dir / "summary.csv", index=False)
    log.info("Resumen del piloto:\n" + table.to_string(index=False))
    log.info(f"-> {out_dir / 'summary.csv'}")
    return table


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Piloto de variabilidad entre semillas y pliegues (sin test).")
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--config", default=None, help="YAML base (por defecto configs/base.yaml)")
    args = parser.parse_args(argv)
    try:
        run_pilot(D.load_config(args.config), args.model)
    except ValueError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
