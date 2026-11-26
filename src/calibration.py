"""Calibración e incertidumbre sobre las probabilidades guardadas de las corridas definitivas.

- Temperature scaling (Guo et al., 2017): una temperatura T por corrida, ajustada minimizando la
  log-loss de validación de esa corrida (nunca test) sobre los logits log p; se aplica a test.
- Métricas antes y después: ECE de `calibration.n_bins` intervalos de igual ancho, ECE adaptativo
  (intervalos de igual frecuencia), Brier multiclase y log-loss; diagramas de fiabilidad.
- Deep ensemble: media de las probabilidades de test de todas las corridas de un modelo, con
  entropía predictiva, entropía esperada e información mutua, y AUROC de la incertidumbre para
  detectar errores.
- Predicción conforme dividida (split conformal): umbral calibrado en validación de cada corrida
  con la puntuación 1 - p(y | x) y evaluado en test: cobertura empírica frente a 1 - alpha y
  tamaño medio del conjunto, global, por fuente y por clase.

Parámetros: secciones `calibration` y `conformal` de base.yaml. Todas las funciones de tablas son
tolerantes (src/report.py).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.report import Missing, tolerant

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def _norm(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(axis=1, keepdims=True)


def log_loss(y, p, eps: float = 1e-12) -> float:
    p = _norm(p)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), np.asarray(y)], eps, 1.0))))


def brier(y, p) -> float:
    """Brier multiclase: media de la suma de (p_k - 1[y = k])^2 sobre las clases."""
    p = _norm(p)
    onehot = np.eye(p.shape[1])[np.asarray(y)]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def ece(y, p, n_bins: int) -> float:
    """ECE de la clase predicha con intervalos de igual ancho (como src.metrics)."""
    from src.metrics import expected_calibration_error

    return expected_calibration_error(y, p, n_bins)


def adaptive_ece(y, p, n_bins: int) -> float:
    """ECE con intervalos de igual frecuencia en la confianza (Nixon et al., 2019)."""
    p = _norm(p)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == np.asarray(y)).astype(float)
    order = np.argsort(conf, kind="mergesort")
    total = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            total += len(idx) / len(conf) * abs(correct[idx].mean() - conf[idx].mean())
    return float(total)


def calibration_metrics(y, p, cfg: dict) -> dict:
    c = cfg["calibration"]
    return {"ece": ece(y, p, int(c["n_bins"])), "ece_adaptativo": adaptive_ece(y, p, int(c["adaptive_bins"])),
            "brier": brier(y, p), "log_loss": log_loss(y, p, float(c["eps"]))}


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def softmax_t(p, T: float, eps: float = 1e-12) -> np.ndarray:
    """softmax(log p / T): probabilidades con temperatura T (T > 1 suaviza, T < 1 agudiza)."""
    z = np.log(np.clip(_norm(p), eps, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(y_val, p_val, bounds=(0.05, 20.0), eps: float = 1e-12) -> float:
    """T que minimiza la log-loss de validación (búsqueda acotada sobre log T)."""
    y = np.asarray(y_val)

    def nll(log_t):
        return log_loss(y, softmax_t(p_val, float(np.exp(log_t)), eps), eps)

    res = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x))


# ---------------------------------------------------------------------------
# Datos por corrida
# ---------------------------------------------------------------------------


def _arrays(df: pd.DataFrame, classes) -> tuple[np.ndarray, np.ndarray]:
    idx = {c: i for i, c in enumerate(classes)}
    return df["label"].map(idx).to_numpy(), df[[f"p_{c}" for c in classes]].to_numpy(np.float64)


def run_predictions(rs, model: str, classes) -> dict:
    """{(semilla, pliegue): {"val": (y, p, ids), "test": (y, p, ids)}} de las corridas de `model`
    que tienen probabilidades de validación y de test."""
    out = {}
    for key, d in rs.runs.get(model, {}).items():
        if not ((d / "probabilities_val.csv").exists() and (d / "probabilities_test.csv").exists()):
            continue
        entry = {}
        for split in ("val", "test"):
            df = rs.predictions(model, *key, split).sort_values("image_id")
            y, p = _arrays(df, classes)
            entry[split] = (y, p, df["image_id"].to_numpy())
        out[key] = entry
    return out


@tolerant()
def temperature_table(rs, cfg: dict, models, classes) -> pd.DataFrame:
    """Por modelo y corrida: T ajustada en validación y métricas de test antes y después."""
    c = cfg["calibration"]
    bounds, eps = tuple(c["temperature_bounds"]), float(c["eps"])
    rows = []
    for m in models:
        for (s, f), e in run_predictions(rs, m, classes).items():
            T = fit_temperature(e["val"][0], e["val"][1], bounds, eps)
            y, p, _ = e["test"]
            before = calibration_metrics(y, p, cfg)
            after = calibration_metrics(y, softmax_t(p, T, eps), cfg)
            rows.append({"modelo": m, "run": f"s{s}_f{f}", "T": T,
                         **{f"{k}_antes": v for k, v in before.items()},
                         **{f"{k}_despues": v for k, v in after.items()}})
    if not rows:
        return Missing("no hay corridas con probabilidades de validación y test.")
    return pd.DataFrame(rows).set_index(["modelo", "run"])


@tolerant()
def temperature_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación estándar entre corridas de T y de cada métrica antes y después."""
    g = table.groupby(level="modelo", sort=False)
    out = pd.concat({"media": g.mean(), "std": g.std(ddof=1)}, axis=1).swaplevel(axis=1)
    return out.reindex(columns=[(c, s) for c in table.columns for s in ("media", "std")])


def _reliability(y, p, n_bins: int) -> pd.DataFrame:
    from src.evaluation import reliability_table

    p = _norm(p)
    return reliability_table(p.argmax(axis=1) == y, p.max(axis=1), n_bins)


@tolerant()
def plot_reliability_before_after(rs, cfg: dict, models, classes, ncols: int = 4):
    """Diagrama de fiabilidad de test (todas las corridas juntas) antes (gris) y después (negro)
    del temperature scaling, un panel por modelo."""
    c = cfg["calibration"]
    n_bins, eps, bounds = int(c["n_bins"]), float(c["eps"]), tuple(c["temperature_bounds"])
    data = {}
    for m in models:
        preds = run_predictions(rs, m, classes)
        if not preds:
            continue
        ys, before, after = [], [], []
        for e in preds.values():
            T = fit_temperature(e["val"][0], e["val"][1], bounds, eps)
            ys.append(e["test"][0])
            before.append(e["test"][1])
            after.append(softmax_t(e["test"][1], T, eps))
        data[m] = (np.concatenate(ys), np.vstack(before), np.vstack(after))
    if not data:
        return Missing("no hay corridas con probabilidades de validación y test.")
    nrows = int(np.ceil(len(data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.ravel()[len(data):]:
        ax.axis("off")
    for ax, (m, (y, pb, pa)) in zip(axes.ravel(), data.items()):
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
        for p, color, name in ((pb, "darkgray", "antes"), (pa, "black", "después")):
            t = _reliability(y, p, n_bins).dropna()
            ax.plot(t["confianza_media"], t["exactitud"], marker="o", ms=3, color=color,
                    label=f"{name} (ECE {ece(y, p, n_bins):.3f})")
        ax.set_title(m)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.supxlabel("confianza (probabilidad de la clase predicha)")
    fig.supylabel("exactitud")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Deep ensemble
# ---------------------------------------------------------------------------


def _entropy(p, eps: float = 1e-12) -> np.ndarray:
    return -(p * np.log(np.clip(p, eps, 1.0))).sum(axis=1)


@tolerant()
def ensemble_table(rs, cfg: dict, models, classes, temperature: bool = False) -> pd.DataFrame:
    """Por modelo: métricas de test del deep ensemble (media de las probabilidades de todas sus
    corridas) frente a la media de las corridas individuales, y AUROC de la entropía predictiva
    y de la información mutua para separar errores de aciertos del ensemble."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    c = cfg["calibration"]
    eps, bounds = float(c["eps"]), tuple(c["temperature_bounds"])
    rows = []
    for m in models:
        preds = run_predictions(rs, m, classes)
        if not preds:
            continue
        probs, singles = [], []
        y = ids = None
        for e in preds.values():
            yt, p, it = e["test"]
            if ids is None:
                y, ids = yt, it
            elif not np.array_equal(ids, it):
                raise ValueError(f"Las corridas de {m} no tienen las mismas imágenes de test")
            if temperature:
                p = softmax_t(p, fit_temperature(e["val"][0], e["val"][1], bounds, eps), eps)
            probs.append(p)
            singles.append({"balanced_accuracy": balanced_accuracy_score(y, p.argmax(1)), **calibration_metrics(y, p, cfg)})
        P = np.stack(probs)                          # (R, N, K)
        mean_p = P.mean(axis=0)
        total = _entropy(mean_p, eps)
        expected = np.mean([_entropy(p, eps) for p in P], axis=0)
        mi = total - expected
        err = mean_p.argmax(1) != y
        ens = {"balanced_accuracy": balanced_accuracy_score(y, mean_p.argmax(1)), **calibration_metrics(y, mean_p, cfg)}
        single = pd.DataFrame(singles).mean()
        row = {"modelo": m, "corridas": len(P)}
        for k, v in ens.items():
            row[f"{k}_ensemble"] = v
            row[f"{k}_corrida_media"] = single[k]
        row["ganancia_balanced_accuracy"] = ens["balanced_accuracy"] - single["balanced_accuracy"]
        row["entropia_media"] = float(total.mean())
        row["informacion_mutua_media"] = float(mi.mean())
        row["auroc_error_entropia"] = float(roc_auc_score(err, total)) if 0 < err.sum() < len(err) else np.nan
        row["auroc_error_info_mutua"] = float(roc_auc_score(err, mi)) if 0 < err.sum() < len(err) and np.ptp(mi) > 0 else np.nan
        rows.append(row)
    if not rows:
        return Missing("no hay corridas con probabilidades de test.")
    return pd.DataFrame(rows).set_index("modelo")


# ---------------------------------------------------------------------------
# Predicción conforme dividida
# ---------------------------------------------------------------------------


def conformal_threshold(y_cal, p_cal, alpha: float) -> float:
    """Cuantil corregido ceil((n + 1)(1 - alpha)) / n de las puntuaciones 1 - p(y | x) de calibración."""
    scores = 1.0 - _norm(p_cal)[np.arange(len(y_cal)), np.asarray(y_cal)]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(p, qhat: float) -> np.ndarray:
    """Conjuntos de predicción (N, K) booleanos: clases con 1 - p_k <= qhat."""
    return (1.0 - _norm(p)) <= qhat + 1e-12


@tolerant()
def conformal_results(rs, cfg: dict, models, classes, meta: pd.DataFrame) -> pd.DataFrame:
    """Formato largo: modelo, alpha, corrida, grupo (global, fuente=<x> o clase=<x>), n,
    cobertura y tamaño medio del conjunto en test."""
    ccfg, c = cfg["conformal"], cfg["calibration"]
    eps, bounds = float(c["eps"]), tuple(c["temperature_bounds"])
    src = cfg["e2_stats"]["source_column"]
    rows = []
    for m in models:
        for (s, f), e in run_predictions(rs, m, classes).items():
            (yv, pv, _), (yt, pt, ids) = e["val"], e["test"]
            if ccfg.get("use_temperature", True):
                T = fit_temperature(yv, pv, bounds, eps)
                pv, pt = softmax_t(pv, T, eps), softmax_t(pt, T, eps)
            sources = meta[src].reindex(ids).to_numpy() if src in meta.columns else np.full(len(ids), "?")
            for alpha in ccfg["alphas"]:
                sets = prediction_sets(pt, conformal_threshold(yv, pv, float(alpha)))
                covered = sets[np.arange(len(yt)), yt]
                size = sets.sum(axis=1)
                groups = [("global", np.ones(len(yt), bool))]
                groups += [(f"fuente={v}", sources == v) for v in pd.unique(sources)]
                groups += [(f"clase={classes[k]}", yt == k) for k in range(len(classes))]
                for name, mask in groups:
                    if mask.any():
                        rows.append({"modelo": m, "alpha": float(alpha), "run": f"s{s}_f{f}", "grupo": name,
                                     "n": int(mask.sum()), "cobertura": float(covered[mask].mean()),
                                     "tamano_medio": float(size[mask].mean())})
    if not rows:
        return Missing("no hay corridas con probabilidades de validación y test.")
    return pd.DataFrame(rows)


@tolerant()
def conformal_summary(results: pd.DataFrame, by_group: str = "global") -> pd.DataFrame:
    """Cobertura y tamaño medio (media y std entre corridas) frente a 1 - alpha. `by_group`:
    "global", "fuente" o "clase"."""
    sub = results[results["grupo"] == "global"] if by_group == "global" else \
        results[results["grupo"].str.startswith(f"{by_group}=")]
    if sub.empty:
        return Missing(f"no hay resultados conformes para {by_group}.")
    g = sub.groupby(["modelo", "alpha", "grupo"], sort=False)
    out = pd.DataFrame({"n": g["n"].median(), "corridas": g.size(), "cobertura_media": g["cobertura"].mean(),
                        "cobertura_std": g["cobertura"].std(ddof=1), "tamano_medio": g["tamano_medio"].mean(),
                        "tamano_std": g["tamano_medio"].std(ddof=1)})
    out["objetivo"] = 1 - out.index.get_level_values("alpha")
    out["cobertura_menos_objetivo"] = out["cobertura_media"] - out["objetivo"]
    return out


@tolerant()
def plot_conformal(summary: pd.DataFrame):
    """Cobertura frente a tamaño medio del conjunto por modelo y alpha (global)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for alpha, g in summary.groupby(level="alpha"):
        ax.scatter(g["tamano_medio"], g["cobertura_media"], label=f"alpha = {alpha}")
        for (m, _, _), r in g.iterrows():
            ax.annotate(m, (r["tamano_medio"], r["cobertura_media"]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.axhline(1 - alpha, ls="--", lw=1, color="gray")
    ax.set_xlabel("tamaño medio del conjunto de predicción")
    ax.set_ylabel("cobertura empírica en test")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def probability_metrics_by_run(rs, cfg: dict, models, classes, split: str = "test") -> pd.DataFrame | Missing:
    """Formato largo (model, seed, val_fold, split, metric, value) con brier, ece_adaptativo y
    log_loss calculados desde las probabilidades guardadas (para la tabla principal)."""
    rows = []
    for m in models:
        for (s, f), d in rs.runs.get(m, {}).items():
            if not (d / f"probabilities_{split}.csv").exists():
                continue
            y, p = _arrays(rs.predictions(m, s, f, split), classes)
            met = calibration_metrics(y, p, cfg)
            for k in ("brier", "ece_adaptativo", "log_loss"):
                rows.append({"model": m, "seed": s, "val_fold": f, "split": split, "metric": k, "value": met[k]})
    if not rows:
        return Missing(f"no hay probabilidades de {split}.")
    return pd.DataFrame(rows)
