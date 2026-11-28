"""Registro de corridas en MLflow local (sección `mlflow` de base.yaml).

    python -m src.tracking [--label final] [--model ...]     # registra las corridas ya guardadas
    uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db

Cada corrida (carpeta outputs/runs/<modelo>/<etiqueta>/s<semilla>_f<pliegue>/) se registra como
una run de MLflow con:
- parámetros: modelo, etiqueta, semilla, pliegue, hiperparámetros y los ajustes de train
  (aplanados) de config.yaml;
- métricas: las escalares de metrics.json por partición (p. ej. test_balanced_accuracy), la
  sensibilidad por clase, tiempos y parámetros del modelo de run_info.json, y la curva de
  history.csv por época;
- artefactos: config.yaml, metrics.json, run_info.json, history.csv y probabilities_*.csv (los
  pesos no, por tamaño).

Las corridas ya registradas (misma carpeta y misma fecha de metrics.json) se omiten. src.train y
src.baselines llaman a log_run al terminar cada corrida si mlflow.enabled; un fallo de MLflow
solo emite un aviso y nunca interrumpe el entrenamiento.
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from pathlib import Path

import pandas as pd
import yaml

from src import data as D

log = logging.getLogger(__name__)
ARTIFACTS = ("config.yaml", "metrics.json", "run_info.json", "history.csv", "probabilities_val.csv",
             "probabilities_test.csv")


def _rel(path) -> str:
    from src.train import _relpath

    return _relpath(path)


def tracking_uri(cfg: dict) -> str:
    d = Path(cfg["paths"]["mlflow_dir"])
    return f"sqlite:///{d / 'mlflow.db'}"


def _experiment(cfg: dict):
    import mlflow

    d = Path(cfg["paths"]["mlflow_dir"])
    d.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(cfg))
    name = cfg["mlflow"]["experiment"]
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name, artifact_location=(d / "artifacts").as_uri())
    else:
        exp_id = exp.experiment_id
    return exp_id


def _flatten(obj, prefix: str = "") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        out[prefix.rstrip(".")] = ",".join(map(str, obj))[:500]
    else:
        out[prefix.rstrip(".")] = obj
    return out


def _params(run_dir: Path, info: dict) -> dict:
    params = {"model": info["model"], "label": info["label"], "seed": info["seed"], "val_fold": info["val_fold"],
              **{f"hp.{k}": v for k, v in (info.get("hyperparameters") or {}).items()}}
    cfg_file = run_dir / "config.yaml"
    if cfg_file.exists():
        with open(cfg_file, encoding="utf-8") as f:
            c = yaml.safe_load(f) or {}
        for section in ("train", "baseline"):
            if section in c:
                params.update({f"{section}.{k}": v for k, v in _flatten(c[section]).items()})
    return {k[:250]: str(v)[:500] for k, v in params.items()}


def _metrics(metrics: dict, info: dict) -> dict:
    out = {}
    for split, m in metrics.items():
        for k, v in m.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                out[f"{split}_{k}"] = float(v)
        for c, pc in (m.get("per_class") or {}).items():
            out[f"{split}_recall_{c}"] = float(pc["recall"])
    for k, v in (info.get("train_time_s") or {}).items():
        out[f"train_time_s_{k}"] = float(v)
    p = (info.get("params") or {}).get("finetune") or {}
    if "total" in p:
        out["params_total"] = float(p["total"])
    return out


def already_logged(client, exp_id: str, run_dir: Path, stamp: str) -> bool:
    rel = _rel(run_dir)
    runs = client.search_runs([exp_id], filter_string=f"tags.run_dir = '{rel}' and tags.metrics_mtime = '{stamp}'",
                              max_results=1)
    return len(runs) > 0


def log_run(cfg: dict, run_dir: str | Path, force: bool = False) -> str | None:
    """Registra la corrida de `run_dir` en MLflow; devuelve el id de la run (None si ya estaba)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    run_dir = Path(run_dir)
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"{run_dir} no tiene metrics.json (corrida sin terminar)")
    exp_id = _experiment(cfg)
    stamp = str(metrics_file.stat().st_mtime_ns)
    client = MlflowClient(tracking_uri(cfg))
    if not force and already_logged(client, exp_id, run_dir, stamp):
        return None
    info = D.load_json(run_dir / "run_info.json")
    metrics = D.load_json(metrics_file)
    name = f"{info['model']}/{info['label']}/s{info['seed']}_f{info['val_fold']}"
    with mlflow.start_run(experiment_id=exp_id, run_name=name) as run:
        mlflow.set_tags({"run_dir": _rel(run_dir), "metrics_mtime": stamp,
                         "model": info["model"], "label": info["label"], "baseline": str(bool(info.get("baseline")))})
        mlflow.log_params(_params(run_dir, info))
        mlflow.log_metrics(_metrics(metrics, info))
        hist = run_dir / "history.csv"
        if hist.exists():
            h = pd.read_csv(hist)
            for _, r in h.iterrows():
                step = int(r.get("global_epoch", 0))
                mlflow.log_metrics({k: float(r[k]) for k in ("loss", "val_loss", "val_balanced_accuracy",
                                                             "balanced_accuracy", "learning_rate")
                                    if k in r and pd.notna(r[k])}, step=step)
        for a in ARTIFACTS:
            if (run_dir / a).exists():
                mlflow.log_artifact(str(run_dir / a))
        return run.info.run_id


def safe_log_run(cfg: dict, run_dir: str | Path) -> None:
    """log_run si mlflow.enabled; cualquier error se convierte en un aviso."""
    if not (cfg.get("mlflow") or {}).get("enabled", False):
        return
    try:
        log_run(cfg, run_dir)
    except Exception as e:  # noqa: BLE001 - el registro nunca debe interrumpir el entrenamiento
        warnings.warn(f"MLflow no registró {run_dir}: {e}", stacklevel=2)


def sync(cfg: dict, labels=None, models=None) -> pd.DataFrame:
    """Registra todas las corridas terminadas de outputs/runs (filtradas por etiqueta y modelo)."""
    rows = []
    for metrics_file in sorted(Path(cfg["paths"]["runs_dir"]).glob("*/*/s*_f*/metrics.json")):
        d = metrics_file.parent
        model, label = d.parent.parent.name, d.parent.name
        if (labels and label not in labels) or (models and model not in models):
            continue
        rid = log_run(cfg, d)
        rows.append({"run_dir": _rel(d), "estado": "registrada" if rid else "ya estaba"})
    return pd.DataFrame(rows, columns=["run_dir", "estado"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Registra en MLflow las corridas guardadas en outputs/runs.")
    parser.add_argument("--label", action="append", default=None, help="etiqueta (final, pilot, tuning_t00, ...)")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    t = sync(D.load_config(args.config), args.label, args.model)
    print(t["estado"].value_counts().to_string() if len(t) else "No hay corridas terminadas.")
    print(f"MLflow: {tracking_uri(D.load_config(args.config))}")


if __name__ == "__main__":
    main()
