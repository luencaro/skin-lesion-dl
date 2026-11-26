"""Baselines del benchmark: trivial (siempre nv) y regresión logística sobre embeddings congelados.

    python -m src.baselines [--only trivial|logreg] [--compute-embeddings]

Se evalúan en las mismas corridas (semilla, pliegue de validación) que la ejecución definitiva de
los modelos profundos (final.seeds x final.folds) y guardan sus artefactos con el mismo formato en
outputs/runs/<baseline>/<final.label>/s<semilla>_f<pliegue>/: config.yaml, metrics.json (train,
val y test), run_info.json y probabilities_val.csv / probabilities_test.csv. Así el benchmark los
carga igual que a cualquier modelo. Resumen en outputs/final/<baseline>/summary.csv.

- trivial: predice siempre baselines.trivial.majority_class; sus probabilidades son las
  frecuencias de clase del train de la corrida (DummyClassifier(strategy="prior")), cuyo argmax es
  la clase mayoritaria. No depende de la semilla.
- logreg: StandardScaler + LogisticRegression multinomial con class_weight="balanced" (los mismos
  pesos N / (K N_k) que los modelos profundos) sobre los embeddings de MobileNetV2 (pooling
  promedio, ImageNet) del EDA. El escalado y el modelo se ajustan solo con el train de cada
  corrida. C se elige en train.tuning.val_fold con la exactitud balanceada de validación, como el
  ajuste de los modelos profundos (outputs/tuning/logreg/). Con lbfgs el ajuste es determinista,
  así que las corridas con el mismo pliegue y distinta semilla coinciden.

Los embeddings de train se leen de paths.embeddings_cache (los calculó el EDA); los de test, de
paths.embeddings_test_cache, que se calcula con la misma función (src.eda.mobilenet_embeddings)
solo con --compute-embeddings (es inferencia de MobileNetV2 sin ajustar; nunca usa etiquetas).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import data as D
from src.metrics import classification_metrics
from src.report import Missing, is_missing

log = logging.getLogger(__name__)

BASELINES = ("trivial", "logreg")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _read_cache(path: Path) -> tuple[np.ndarray, np.ndarray, int] | None:
    if not Path(path).exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return z["embeddings"], z["image_ids"].astype(str), int(z["input_size"])


def load_embeddings(cfg: dict, image_ids, compute_missing: bool = False) -> np.ndarray | Missing:
    """Embeddings de MobileNetV2 de `image_ids` (en ese orden), de las caches de train y test. Con
    `compute_missing`, calcula y guarda en embeddings_test_cache los de test que falten (todas las
    imágenes de test, con src.eda.load_or_compute_embeddings). Si faltan, devuelve Missing."""
    ids = [str(i) for i in image_ids]
    input_size = int(cfg["eda_embeddings_input_size"])
    lookup = {}
    for key in ("embeddings_cache", "embeddings_test_cache"):
        cached = _read_cache(cfg["paths"][key])
        if cached is not None and cached[2] == input_size:
            lookup.update(zip(cached[1], cached[0]))
    missing = [i for i in ids if i not in lookup]
    if missing and compute_missing:
        from src import eda

        splits = D.load_splits(cfg["paths"]["splits_csv"])
        test_ids = sorted(splits.loc[splits["split"] == "test", "image_id"])
        emb = eda.load_or_compute_embeddings(cfg["paths"]["embeddings_test_cache"], test_ids,
                                             D.image_paths(cfg, test_ids), input_size,
                                             int(cfg["eda_embeddings_batch_size"]))
        lookup.update(zip(test_ids, emb))
        missing = [i for i in ids if i not in lookup]
    if missing:
        return Missing(f"faltan los embeddings de {len(missing)} imágenes (p. ej. {missing[0]}); ejecuta "
                       "`python -m src.baselines --compute-embeddings` (los de train los calcula "
                       "book/06-eda_imagenes.ipynb).")
    return np.stack([lookup[i] for i in ids]).astype(np.float64)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


def make_logreg(cfg: dict, C: float, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lcfg = cfg["baselines"]["logreg"]
    steps = [StandardScaler()] if lcfg.get("standardize", True) else []
    steps.append(LogisticRegression(C=float(C), class_weight=lcfg.get("class_weight"), solver=lcfg["solver"],
                                    max_iter=int(lcfg["max_iter"]), random_state=seed))
    return make_pipeline(*steps)


def prior_probabilities(y_train, k: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y_train), minlength=k).astype(float)
    return counts / counts.sum()


def fit_predict(cfg: dict, name: str, X: dict, y: dict, k: int, hparams: dict, seed: int) -> tuple[dict, dict]:
    """Ajusta el baseline `name` con X["train"], y["train"] y devuelve ({split: probabilidades},
    info con número de parámetros, tiempo de ajuste e inferencia por imagen)."""
    t0 = time.perf_counter()
    if name == "trivial":
        prior = prior_probabilities(y["train"], k)
        classes = D.class_names(cfg)
        majority = cfg["baselines"]["trivial"]["majority_class"]
        if classes[int(prior.argmax())] != majority:
            raise ValueError(f"La clase más frecuente del train es {classes[int(prior.argmax())]}, no {majority}")
        fit_s = time.perf_counter() - t0
        proba, inf = {}, {}
        for split, Xs in X.items():
            t1 = time.perf_counter()
            proba[split] = np.tile(prior, (len(Xs), 1))
            inf[split] = (time.perf_counter() - t1) / max(len(Xs), 1)
        params = k
    elif name == "logreg":
        model = make_logreg(cfg, hparams["C"], seed).fit(X["train"], y["train"])
        fit_s = time.perf_counter() - t0
        proba, inf = {}, {}
        for split, Xs in X.items():
            t1 = time.perf_counter()
            p = np.zeros((len(Xs), k))
            p[:, model.classes_] = model.predict_proba(Xs)
            proba[split] = p
            inf[split] = (time.perf_counter() - t1) / max(len(Xs), 1)
        clf = model[-1]
        params = int(clf.coef_.size + clf.intercept_.size)
    else:
        raise ValueError(f"Baseline desconocido: {name}")
    return proba, {"fit_s": fit_s, "inference_s_per_image": inf, "params": params}


# ---------------------------------------------------------------------------
# Datos por corrida
# ---------------------------------------------------------------------------


def split_frames(cfg: dict, val_fold: int) -> dict:
    splits = D.load_splits(cfg["paths"]["splits_csv"])
    return {s: D.select_split(splits, s, val_fold) for s in ("train", "val", "test")}


def labels_of(cfg: dict, df: pd.DataFrame) -> np.ndarray:
    idx = {c: i for i, c in enumerate(D.class_names(cfg))}
    return df["dx"].map(idx).to_numpy(np.int64)


def _features(cfg: dict, name: str, frames: dict, compute_missing: bool):
    if name == "trivial":
        return {s: np.zeros((len(f), 1)) for s, f in frames.items()}
    X = {}
    for s, f in frames.items():
        e = load_embeddings(cfg, f["image_id"], compute_missing)
        if is_missing(e):
            return e
        X[s] = e
    return X


# ---------------------------------------------------------------------------
# Ajuste de C (regresión logística)
# ---------------------------------------------------------------------------


def tune_logreg(cfg: dict, compute_missing: bool = False) -> dict:
    """Elige C de baselines.logreg.C_grid por la exactitud balanceada de validación en
    train.tuning.val_fold (en empate, el menor C). Guarda outputs/tuning/logreg/trials.csv y best.json."""
    from sklearn.metrics import balanced_accuracy_score

    tun = cfg["train"]["tuning"]
    fold, seed = int(tun["val_fold"]), int(tun["seed"])
    frames = split_frames(cfg, fold)
    frames = {s: frames[s] for s in ("train", "val")}
    X = _features(cfg, "logreg", frames, compute_missing)
    if is_missing(X):
        raise ValueError(X.message)
    y = {s: labels_of(cfg, f) for s, f in frames.items()}
    k = len(D.class_names(cfg))
    rows = []
    for i, C in enumerate(cfg["baselines"]["logreg"]["C_grid"]):
        proba, info = fit_predict(cfg, "logreg", X, y, k, {"C": C}, seed)
        rows.append({"trial": i, "C": float(C),
                     "val_balanced_accuracy": float(balanced_accuracy_score(y["val"], proba["val"].argmax(1))),
                     "train_time_s": info["fit_s"]})
        log.info(f"logreg C={C}: val_bal_acc {rows[-1]['val_balanced_accuracy']:.4f}")
    trials = pd.DataFrame(rows)
    out = cfg["paths"]["tuning_dir"] / "logreg"
    out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out / "trials.csv", index=False)
    best = trials.sort_values(["val_balanced_accuracy", "C"], ascending=[False, True]).iloc[0]
    res = {"model": "logreg", "trial": int(best["trial"]), "hyperparameters": {"C": float(best["C"])},
           "val_balanced_accuracy": float(best["val_balanced_accuracy"]), "val_fold": fold,
           "selection": "máxima val_balanced_accuracy; en empate, el menor C"}
    D.save_json(res, out / "best.json")
    return res


# ---------------------------------------------------------------------------
# Corridas
# ---------------------------------------------------------------------------


def _save_run(cfg: dict, name: str, seed: int, fold: int, frames: dict, y: dict, proba: dict, info: dict,
              hparams: dict) -> dict:
    from src.final import final_label
    from src.train import _relpath, probabilities_frame, run_dir_for
    from src.utils import environment_info

    classes = D.class_names(cfg)
    label = final_label(cfg)
    run_dir = run_dir_for(cfg, name, label, seed, fold)
    run_dir.mkdir(parents=True, exist_ok=True)
    ece_bins = int(cfg["train"]["ece_bins"])
    metrics = {s: classification_metrics(y[s], proba[s], classes, ece_bins) for s in ("val", "test", "train")}
    for s in ("val", "test"):
        probabilities_frame(frames[s]["image_id"].to_numpy(), y[s], proba[s], classes).to_csv(
            run_dir / f"probabilities_{s}.csv", index=False)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"run": {"model": name, "label": label, "seed": seed, "val_fold": fold},
                        "baseline": cfg["baselines"][name], "hyperparameters": hparams, "classes": classes},
                       f, sort_keys=False, allow_unicode=True)
    n_params = int(info["params"])
    run_info = {
        "model": name, "label": label, "seed": seed, "val_fold": fold, "hyperparameters": hparams,
        "baseline": True, "evaluated_test": True, "evaluated_train": True,
        "n_images": {s: len(y[s]) for s in ("train", "val", "test")},
        "epochs": {"warmup": 0, "finetune_max": 0, "finetune_run": 0, "finetune_best": 0},
        "best_val_balanced_accuracy_during_training": None,
        "train_time_s": {"warmup": 0.0, "finetune": info["fit_s"], "total": info["fit_s"]},
        "inference_s_per_image": info["inference_s_per_image"], "inference_batch_size": None,
        "params": {p: {"trainable": n_params, "non_trainable": 0, "total": n_params} for p in ("warmup", "finetune")},
        "mixed_precision": False, "deterministic": True, "environment": environment_info(),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    D.save_json(run_info, run_dir / "run_info.json")
    D.save_json(metrics, run_dir / "metrics.json")  # al final: marca la corrida como terminada
    from src.tracking import safe_log_run

    safe_log_run(cfg, run_dir)
    return {"seed": seed, "val_fold": fold, "val_balanced_accuracy": metrics["val"]["balanced_accuracy"],
            "test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
            "train_balanced_accuracy": metrics["train"]["balanced_accuracy"], "train_time_s": info["fit_s"],
            "run_dir": _relpath(run_dir)}


def run_baseline(cfg: dict, name: str, compute_missing: bool = False, overwrite: bool = False) -> pd.DataFrame:
    """Corridas de `name` en final.seeds x final.folds (las terminadas se reutilizan salvo con
    `overwrite`). Guarda outputs/final/<name>/summary.csv y lo devuelve."""
    from src.final import final_label, final_pairs, summarize
    from src.train import load_run_summary, run_dir_for

    k = len(D.class_names(cfg))
    hparams = {}
    if name == "logreg":
        best = cfg["paths"]["tuning_dir"] / "logreg" / "best.json"
        hparams = (D.load_json(best) if best.exists() and not overwrite else tune_logreg(cfg, compute_missing))["hyperparameters"]
    rows, cache = [], {}
    for seed, fold in final_pairs(cfg):
        run_dir = run_dir_for(cfg, name, final_label(cfg), seed, fold)
        if (run_dir / "metrics.json").exists() and not overwrite:
            s = load_run_summary(run_dir)
            rows.append({k_: s.get(k_) for k_ in ("seed", "val_fold", "val_balanced_accuracy", "test_balanced_accuracy",
                                                   "train_balanced_accuracy", "train_time_s", "run_dir")})
            continue
        if fold not in cache:
            frames = split_frames(cfg, fold)
            X = _features(cfg, name, frames, compute_missing)
            if is_missing(X):
                raise ValueError(X.message)
            cache = {fold: (frames, X, {s: labels_of(cfg, f) for s, f in frames.items()})}
        frames, X, y = cache[fold]
        proba, info = fit_predict(cfg, name, X, y, k, hparams, seed)
        rows.append(_save_run(cfg, name, seed, fold, frames, y, proba, info, hparams))
        log.info(f"{name} s{seed}_f{fold}: test_bal_acc {rows[-1]['test_balanced_accuracy']:.4f}")
    table = summarize(rows, float(cfg["benchmark"]["alpha"]))
    out = cfg["paths"]["final_dir"] / name / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def main(argv=None) -> None:
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Baselines trivial y regresión logística (mismas corridas que final).")
    parser.add_argument("--only", choices=BASELINES, default=None)
    parser.add_argument("--compute-embeddings", action="store_true",
                        help="calcula los embeddings de test que falten (inferencia de MobileNetV2)")
    parser.add_argument("--overwrite", action="store_true", help="repite el ajuste de C y las corridas terminadas")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = D.load_config(args.config)
    setup_logging(cfg["paths"]["logs_dir"] / "baselines.log")
    try:
        for name in [args.only] if args.only else BASELINES:
            table = run_baseline(cfg, name, args.compute_embeddings, args.overwrite)
            log.info(f"{name}:\n" + table.to_string(index=False))
    except ValueError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
