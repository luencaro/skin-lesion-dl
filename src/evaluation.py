"""Evaluación de un modelo sobre sus corridas definitivas (genérico para cualquier modelo).

Parte de las probabilidades guardadas por corrida (probabilities_<split>.csv), de metrics.json
y de los metadatos de HAM10000. Secciones:

- Robustez por subgrupos: métricas por fuente (dataset), dx_type y localización.
- Sobreajuste: brecha entre train sin aumento, validación y test.
- Errores: sensibilidad y precisión por clase, pares de clases más confundidos y tasa de
  error en las imágenes atípicas de data/atipicos_imagenes.csv.
- Casos difíciles: imágenes que fallan en la mayoría de las corridas y errores más confiados.
- Calibración: diagrama de fiabilidad y ECE.

Todas las funciones de tablas y gráficos son tolerantes (src/report.py): con corridas
ausentes devuelven un `Missing` que el notebook muestra como mensaje.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src import data as D
from src.report import Missing, tolerant
from src.results import (
    NO_RUNS,
    STAT_ROWS,
    add_mean_std,
    load_run_info,
    load_run_metrics,
    run_label,
)

METRIC_FUNCS = {
    "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, p),
    "accuracy": lambda y, p: accuracy_score(y, p),
    # F1 macro sobre las clases presentes en las etiquetas reales del subconjunto
    "f1_macro": lambda y, p: f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0),
}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


@tolerant(NO_RUNS, check="run_dirs")
def load_predictions(run_dirs, split: str = "test") -> pd.DataFrame | Missing:
    """Probabilidades de `split` de todas las corridas en formato largo: run, seed, val_fold,
    image_id, label, pred, correct, confidence (probabilidad de la clase predicha) y p_<clase>."""
    frames = []
    for d in run_dirs:
        path = Path(d) / f"probabilities_{split}.csv"
        if not path.exists():
            continue
        info = load_run_info(d)
        df = pd.read_csv(path)
        df.insert(0, "run", run_label(info))
        df.insert(1, "seed", info["seed"])
        df.insert(2, "val_fold", info["val_fold"])
        frames.append(df)
    if not frames:
        return Missing(f"ninguna corrida tiene probabilities_{split}.csv.")
    out = pd.concat(frames, ignore_index=True)
    pcols = [c for c in out.columns if c.startswith("p_")]
    out["correct"] = out["label"] == out["pred"]
    out["confidence"] = out[pcols].max(axis=1)
    return out


def probability_columns(preds: pd.DataFrame) -> list[str]:
    return [c for c in preds.columns if c.startswith("p_")]


def class_order(preds: pd.DataFrame) -> list[str]:
    return [c[2:] for c in probability_columns(preds)]


def load_metadata_or_missing(cfg: dict):
    """Metadatos de HAM10000 indexados por image_id, o `Missing` si falta el CSV."""
    path = cfg["paths"]["metadata_csv"]
    if not Path(path).exists():
        return Missing(f"no existe {path}; descarga los datos (README) y ejecuta `python -m src.verify_data`.")
    return D.load_metadata(cfg).set_index("image_id")


# ---------------------------------------------------------------------------
# Robustez por subgrupos
# ---------------------------------------------------------------------------


def _metrics_by_group(preds: pd.DataFrame, group_col: str, metrics) -> pd.DataFrame:
    rows = []
    for (run, group), g in preds.groupby(["run", group_col], sort=True, observed=True):
        row = {"run": run, group_col: group, "n": len(g)}
        for m in metrics:
            row[m] = float(METRIC_FUNCS[m](g["label"], g["pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


@tolerant()
def subgroup_metrics(preds: pd.DataFrame, meta: pd.DataFrame, by: str, metrics, min_n: int = 0) -> pd.DataFrame:
    """Métricas de las predicciones por categoría de la columna de metadatos `by`: media y
    desviación estándar (ddof = 1) entre corridas, imágenes por corrida y número de corridas.
    `pocas_imagenes` marca las categorías con menos de `min_n` imágenes."""
    if by not in meta.columns:
        return Missing(f"los metadatos no tienen la columna {by}.")
    df = preds.join(meta[[by]], on="image_id")
    per_run = _metrics_by_group(df, by, metrics)
    g = per_run.groupby(by, observed=True)
    out = pd.DataFrame({"n_imagenes": g["n"].median(), "corridas": g["run"].nunique()})
    for m in metrics:
        out[f"{m}_mean"] = g[m].mean()
        out[f"{m}_std"] = g[m].std(ddof=1)
    out["pocas_imagenes"] = out["n_imagenes"] < min_n
    return out.sort_values("n_imagenes", ascending=False)


@tolerant()
def subgroup_report(preds: pd.DataFrame, meta: pd.DataFrame, subgroups, metrics, min_n: int = 0) -> dict:
    """subgroup_metrics para cada columna de `subgroups`: {columna: tabla}."""
    return {by: subgroup_metrics(preds, meta, by, metrics, min_n) for by in subgroups}


@tolerant()
def plot_subgroup_metric(table: pd.DataFrame, metric: str = "balanced_accuracy", title: str | None = None):
    """Barras horizontales de la media de `metric` por categoría con la desviación estándar
    entre corridas como barra de error. Las categorías con pocas imágenes van en gris."""
    t = table.sort_values(f"{metric}_mean")
    colors = ["lightgray" if small else "tab:blue" for small in t["pocas_imagenes"]]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(t) + 1.2))
    ax.barh([str(i) for i in t.index], t[f"{metric}_mean"], xerr=t[f"{metric}_std"].fillna(0),
            color=colors, capsize=3)
    for y, (_, r) in enumerate(t.iterrows()):
        ax.text(0.01, y, f"n={int(r['n_imagenes'])}", va="center", fontsize=7, color="black")
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(title or t.index.name)
    fig.tight_layout()
    return fig


@tolerant()
def plot_subgroup_report(report: dict, metric: str = "balanced_accuracy") -> dict:
    """plot_subgroup_metric para cada tabla de subgroup_report: {columna: figura o Missing}."""
    return {by: plot_subgroup_metric(t, metric, title=f"{metric} por {by}") for by, t in report.items()}


# ---------------------------------------------------------------------------
# Sobreajuste
# ---------------------------------------------------------------------------


@tolerant(NO_RUNS, check="run_dirs")
def overfitting_table(run_dirs, metrics) -> pd.DataFrame:
    """Por corrida y métrica: valor en train (sin aumento ni dropout), validación y test, y las
    brechas train - val y val - test. Filas mean y std al final. Las particiones que falten en
    metrics.json quedan en NaN."""
    rows = {}
    for d in run_dirs:
        m = load_run_metrics(d)
        row = {}
        for k in metrics:
            vals = {s: (m[s].get(k) if s in m else np.nan) for s in ("train", "val", "test")}
            for s, v in vals.items():
                row[f"{k}_{s}"] = v
            row[f"{k}_gap_train_val"] = vals["train"] - vals["val"]
            row[f"{k}_gap_val_test"] = vals["val"] - vals["test"]
        rows[run_label(load_run_info(d))] = row
    table = pd.DataFrame(rows).T.astype(float).rename_axis("run")
    return add_mean_std(table)


@tolerant()
def overfitting_summary(table: pd.DataFrame, metrics) -> pd.DataFrame:
    """Media y desviación estándar entre corridas de train, val, test y brechas, una fila por métrica."""
    cols = ("train", "val", "test", "gap_train_val", "gap_val_test")
    out = {}
    for k in metrics:
        out[k] = {f"{c}_{s}": table.loc[s, f"{k}_{c}"] for c in cols for s in STAT_ROWS}
    return pd.DataFrame(out).T


@tolerant()
def plot_overfitting(table: pd.DataFrame, metric: str = "balanced_accuracy"):
    """`metric` en train, val y test por corrida (líneas finas) y su media (línea gruesa)."""
    runs = table.drop(index=STAT_ROWS, errors="ignore")
    splits = ["train", "val", "test"]
    cols = [f"{metric}_{s}" for s in splits]
    fig, ax = plt.subplots(figsize=(5, 4))
    for _, r in runs[cols].iterrows():
        ax.plot(splits, r.to_numpy(), color="tab:blue", alpha=0.3, marker=".")
    ax.plot(splits, runs[cols].mean().to_numpy(), color="black", lw=2, marker="o", label="media")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: train sin aumento, validación y test ({len(runs)} corridas)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Errores
# ---------------------------------------------------------------------------


@tolerant()
def confused_pairs(cm, labels, top: int = 10) -> pd.DataFrame:
    """Pares (real, predicha) fuera de la diagonal de la matriz acumulada `cm`, de más a menos
    frecuentes: conteo, porcentaje sobre las imágenes de la clase real y sobre todos los errores."""
    cm = np.asarray(cm)
    rows = []
    errors = cm.sum() - np.trace(cm)
    for i, real in enumerate(labels):
        for j, pred in enumerate(labels):
            if i != j and cm[i, j] > 0:
                rows.append({"real": real, "predicha": pred, "conteo": int(cm[i, j]),
                             "pct_de_la_clase_real": 100 * cm[i, j] / max(cm[i].sum(), 1),
                             "pct_de_los_errores": 100 * cm[i, j] / max(errors, 1)})
    out = pd.DataFrame(rows, columns=["real", "predicha", "conteo", "pct_de_la_clase_real", "pct_de_los_errores"])
    return out.sort_values("conteo", ascending=False).head(top).reset_index(drop=True)


@tolerant(NO_RUNS, check="run_dirs")
def confused_pairs_from_runs(run_dirs, split: str = "test", top: int = 10) -> pd.DataFrame:
    """confused_pairs sobre la matriz de confusión de `split` acumulada en las corridas."""
    from src.results import pooled_confusion

    cm, labels = pooled_confusion(run_dirs, split)
    return confused_pairs(cm, labels, top)


@tolerant()
def outlier_flags(cfg: dict) -> pd.DataFrame:
    """Marcas booleanas por image_id a partir de data/atipicos_imagenes.csv: mahalanobis,
    univariante (al menos univariate_min_marks marcas) y cualquiera."""
    path = cfg["paths"]["outliers_csv"]
    if not Path(path).exists():
        return Missing(f"no existe {path}; se genera en book/06-eda_imagenes.ipynb.")
    ocfg = cfg["evaluation"]["outliers"]
    df = pd.read_csv(path).set_index("image_id")
    flags = pd.DataFrame(index=df.index)
    flags["mahalanobis"] = df[ocfg["mahalanobis_column"]].astype(str).str.lower().eq("true")
    flags["univariante"] = df[ocfg["univariate_column"]] >= int(ocfg["univariate_min_marks"])
    flags["cualquiera"] = flags["mahalanobis"] | flags["univariante"]
    return flags


@tolerant()
def outlier_error_rates(preds: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Tasa de error de las imágenes marcadas frente a las no marcadas, por tipo de marca:
    media y desviación estándar entre corridas, imágenes por corrida y diferencia de medias.
    Solo se usan las imágenes de `preds` que están en `flags`."""
    df = preds.join(flags, on="image_id", how="inner")
    if df.empty:
        return Missing("ninguna imagen de las predicciones está en data/atipicos_imagenes.csv.")
    df["error"] = ~df["correct"]
    rows = []
    for flag in flags.columns:
        per_run = df.groupby(["run", flag])["error"].agg(["mean", "size"]).reset_index()
        row = {"marca": flag}
        for value, name in ((True, "marcadas"), (False, "no_marcadas")):
            sub = per_run[per_run[flag] == value]
            row[f"n_{name}"] = float(sub["size"].median()) if len(sub) else 0.0
            row[f"error_{name}_mean"] = float(sub["mean"].mean()) if len(sub) else np.nan
            row[f"error_{name}_std"] = float(sub["mean"].std(ddof=1)) if len(sub) > 1 else np.nan
        row["diferencia"] = row["error_marcadas_mean"] - row["error_no_marcadas_mean"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("marca")


# ---------------------------------------------------------------------------
# Casos difíciles
# ---------------------------------------------------------------------------


@tolerant()
def image_error_frequency(preds: pd.DataFrame) -> pd.DataFrame:
    """Por imagen: corridas, errores, fracción de corridas con error, clase real, predicción
    errónea más frecuente y probabilidad media de la clase real."""
    p_true = preds.apply(lambda r: r[f"p_{r['label']}"], axis=1)
    df = preds.assign(error=~preds["correct"], p_true=p_true)
    g = df.groupby("image_id")
    wrong = df[df["error"]].groupby("image_id")["pred"].agg(lambda s: s.value_counts().index[0])
    out = pd.DataFrame({"label": g["label"].first(), "corridas": g.size(), "errores": g["error"].sum(),
                        "p_real_media": g["p_true"].mean()})
    out["frac_error"] = out["errores"] / out["corridas"]
    out["pred_erronea_frecuente"] = wrong.reindex(out.index)
    return out.sort_values(["frac_error", "p_real_media"], ascending=[False, True])


@tolerant()
def hard_cases(freq: pd.DataFrame, min_error_frac: float = 0.5, top: int | None = None) -> pd.DataFrame:
    """Imágenes que fallan en más de `min_error_frac` de las corridas (las `top` primeras si se indica)."""
    out = freq[freq["frac_error"] > min_error_frac]
    return out.head(top) if top is not None else out


@tolerant()
def hard_cases_by_class(freq: pd.DataFrame, min_error_frac: float = 0.5) -> pd.DataFrame:
    """Imágenes por clase real, cuántas son casos difíciles y su porcentaje."""
    g = freq.groupby("label")
    out = pd.DataFrame({"imagenes": g.size(), "dificiles": g["frac_error"].apply(lambda s: int((s > min_error_frac).sum()))})
    out["pct_dificiles"] = 100 * out["dificiles"] / out["imagenes"]
    return out


@tolerant()
def confident_errors(preds: pd.DataFrame, n: int = 16) -> pd.DataFrame:
    """Los `n` errores con mayor probabilidad asignada a la clase predicha (una fila por imagen:
    la corrida con el error más confiado)."""
    wrong = preds[~preds["correct"]].sort_values("confidence", ascending=False)
    wrong = wrong.drop_duplicates("image_id")
    return wrong[["image_id", "run", "label", "pred", "confidence"]].head(n).reset_index(drop=True)


@tolerant()
def plot_case_grid(cfg: dict, cases: pd.DataFrame, pred_col: str, extra_col: str | None = None,
                   n: int = 16, ncols: int = 8, title: str | None = None, thumb: int = 200):
    """Cuadrícula de imágenes originales con image_id, clase real y predicha (`pred_col`) y,
    opcionalmente, el valor de `extra_col`. `cases` debe tener image_id (columna o índice) y label."""
    from PIL import Image

    df = cases.reset_index() if "image_id" not in cases.columns else cases
    df = df.head(n)
    if df.empty:
        return Missing("no hay imágenes que mostrar.")
    nrows = int(np.ceil(len(df) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        with Image.open(D.image_path(cfg, r["image_id"])) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
            ax.imshow(np.asarray(im))
        txt = f"{r['image_id']}\nreal: {r['label']} | pred: {r[pred_col]}"
        if extra_col is not None:
            v = r[extra_col]
            txt += f"\n{extra_col}: {v:.2f}" if isinstance(v, float) else f"\n{extra_col}: {v}"
        ax.set_title(txt, fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------


def _bin_index(conf: np.ndarray, n_bins: int) -> np.ndarray:
    """Intervalo (lo, hi] de igual ancho de cada confianza (el primero incluye 0), como
    metrics.expected_calibration_error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)


def reliability_table(correct, confidence, n_bins: int) -> pd.DataFrame:
    """Por intervalo de confianza: imágenes, confianza media, exactitud y |exactitud - confianza|.
    ECE = suma de (n_b / N) * |exactitud_b - confianza_b|."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    idx = _bin_index(conf, n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"desde": edges[b], "hasta": edges[b + 1], "n": int(m.sum()),
                     "confianza_media": conf[m].mean() if m.any() else np.nan,
                     "exactitud": corr[m].mean() if m.any() else np.nan})
    t = pd.DataFrame(rows)
    t["brecha"] = (t["exactitud"] - t["confianza_media"]).abs()
    return t


def ece_from_table(t: pd.DataFrame) -> float:
    n = t["n"].sum()
    return float((t["n"] / n * t["brecha"]).fillna(0).sum()) if n else float("nan")


@tolerant()
def calibration_by_run(preds: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """ECE de cada corrida (recalculada desde las probabilidades), con filas mean y std."""
    rows = {run: {"ece": ece_from_table(reliability_table(g["correct"], g["confidence"], n_bins)),
                  "confianza_media": g["confidence"].mean(), "exactitud": g["correct"].mean()}
            for run, g in preds.groupby("run", sort=False)}
    return add_mean_std(pd.DataFrame(rows).T.rename_axis("run").astype(float))


@tolerant()
def pooled_reliability(preds: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """reliability_table con las predicciones de todas las corridas juntas."""
    return reliability_table(preds["correct"], preds["confidence"], n_bins)


@tolerant()
def plot_reliability(preds: pd.DataFrame, n_bins: int, title: str | None = None):
    """Diagrama de fiabilidad: exactitud frente a confianza media por intervalo, una línea fina
    por corrida y una gruesa con todas las corridas juntas; debajo, el histograma de confianzas."""
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="calibración perfecta")
    for _, g in preds.groupby("run", sort=False):
        t = reliability_table(g["correct"], g["confidence"], n_bins).dropna()
        ax.plot(t["confianza_media"], t["exactitud"], color="tab:blue", alpha=0.2, lw=0.8)
    pooled = pooled_reliability(preds, n_bins)
    t = pooled.dropna()
    ax.plot(t["confianza_media"], t["exactitud"], color="black", marker="o", lw=2,
            label=f"todas las corridas (ECE {ece_from_table(pooled):.3f})")
    ax.set_ylabel("exactitud")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title or f"Diagrama de fiabilidad ({preds['run'].nunique()} corridas, {n_bins} intervalos)")
    axh.hist(preds["confidence"], bins=np.linspace(0, 1, n_bins + 1), color="tab:blue", alpha=0.7)
    axh.set_xlabel("confianza (probabilidad de la clase predicha)")
    axh.set_ylabel("predicciones")
    fig.tight_layout()
    return fig


@tolerant()
def predictions_summary(preds: pd.DataFrame) -> pd.DataFrame:
    """Corridas, imágenes y predicciones cargadas."""
    return pd.Series({"corridas": preds["run"].nunique(), "imagenes": preds["image_id"].nunique(),
                      "predicciones": len(preds)}, name="valor").to_frame()
