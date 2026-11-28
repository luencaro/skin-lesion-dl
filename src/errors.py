"""Análisis de errores entre modelos, atajos por artefactos y taxonomía de errores.

Parte de las predicciones de test alineadas de src.benchmark.test_data (todas las corridas
definitivas de cada modelo) y de la sección `errors` de base.yaml.

- Errores compartidos: por imagen, fracción de corridas en que cada modelo falla; una imagen es
  un error del modelo si falla en más de la mitad de sus corridas. Conteo de imágenes según
  cuántos modelos fallan y correlación (phi) de los indicadores de error por corrida pareada.
- Ensemble simple: media de las probabilidades de los errors.ensemble_top_k mejores modelos en
  cada corrida pareada, frente al mejor de ellos: diferencia por corrida, IC por t y Wilcoxon.
- Atajos: exactitud balanceada y exactitud con y sin cada artefacto de
  data/anotacion_artefactos.csv, en las imágenes anotadas de cada partición de
  errors.shortcut_splits (en val, con las predicciones fuera de pliegue), con IC por bootstrap
  por lesión de la diferencia.
- Taxonomía: plantilla data/taxonomia_errores.csv con una columna 0/1 por categoría de
  errors.taxonomy_categories y una muestra estratificada por clase real de errors.taxonomy_n
  errores del mejor modelo, para etiquetado manual; resumen de las etiquetas.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data as D
from src import stats as S
from src.report import Missing, is_missing, tolerant

# ---------------------------------------------------------------------------
# Errores compartidos y correlación
# ---------------------------------------------------------------------------


@tolerant()
def error_fraction(td, models) -> pd.DataFrame:
    """Fracción de corridas con error por imagen (filas) y modelo (columnas)."""
    models = [m for m in models if m in td.keys]
    if not models:
        return Missing("ningún modelo tiene predicciones de test.")
    return pd.DataFrame({m: 1.0 - td.correct(m).mean(axis=0) for m in models}, index=td.image_ids)


@tolerant()
def shared_errors_table(frac: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Imágenes según el número de modelos que fallan (error = falla en más de `threshold`, errors.error_frac, de sus corridas)."""
    err = frac > threshold
    counts = err.sum(axis=1).value_counts().reindex(range(err.shape[1] + 1), fill_value=0)
    out = pd.DataFrame({"imagenes": counts, "porcentaje": 100 * counts / len(err)})
    out.index.name = "modelos_que_fallan"
    return out


@tolerant()
def errors_per_model(frac: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Errores por modelo (falla en más de `threshold` de sus corridas), exclusivos y compartidos por todos."""
    err = frac > threshold
    n_fail = err.sum(axis=1)
    return pd.DataFrame({"errores": err.sum(), "exclusivos": (err & (n_fail == 1).to_numpy()[:, None]).sum(),
                         "compartidos_por_todos": (err & (n_fail == err.shape[1]).to_numpy()[:, None]).sum()})


@tolerant()
def error_correlation(td, models) -> pd.DataFrame:
    """Correlación phi media entre los indicadores de error por imagen de cada par de modelos,
    sobre sus corridas pareadas comunes (una correlación por corrida y su media)."""
    models = [m for m in models if m in td.keys]
    out = pd.DataFrame(np.eye(len(models)), index=models, columns=models)
    for a, b in combinations(models, 2):
        ia, ib = td.common(a, b)
        ea, eb = ~td.correct(a)[ia], ~td.correct(b)[ib]
        vals = [np.corrcoef(x, y)[0, 1] for x, y in zip(ea, eb) if x.std() > 0 and y.std() > 0]
        out.loc[a, b] = out.loc[b, a] = float(np.mean(vals)) if vals else np.nan
    return out


@tolerant()
def plot_error_correlation(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(0.7 * len(corr) + 2.5, 0.7 * len(corr) + 2))
    im = ax.imshow(corr.to_numpy(float), cmap="viridis", vmin=0, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if corr.iat[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.set_title("Correlación phi de los errores por imagen (media entre corridas)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ensemble simple
# ---------------------------------------------------------------------------


@tolerant()
def ensemble_gain(td, ranked_models, k: int) -> pd.DataFrame:
    """Por corrida pareada de los k primeros modelos de `ranked_models`: exactitud balanceada del
    ensemble (media de probabilidades) y del mejor modelo individual (el primero), su diferencia
    y, en filas finales, media, IC 95 % por t y p de Wilcoxon exacto bilateral."""
    from sklearn.metrics import balanced_accuracy_score

    models = [m for m in ranked_models if m in td.keys][:k]
    if len(models) < 2:
        return Missing("el ensemble necesita al menos 2 modelos con predicciones de test.")
    keys = sorted(set.intersection(*(set(td.keys[m]) for m in models)))
    rows = []
    for key in keys:
        P = np.mean([td.proba[m][td.keys[m].index(key)] for m in models], axis=0)
        best = td.proba[models[0]][td.keys[models[0]].index(key)]
        e, b = balanced_accuracy_score(td.y, P.argmax(1)), balanced_accuracy_score(td.y, best.argmax(1))
        rows.append({"run": f"s{key[0]}_f{key[1]}", "ensemble": e, models[0]: b, "ganancia": e - b})
    if not rows:
        return Missing("no hay corridas pareadas comunes a los modelos del ensemble.")
    t = pd.DataFrame(rows).set_index("run")
    lo, hi = S.t_confidence_interval(t["ganancia"], 0.05)
    t.attrs["resumen"] = pd.Series({"modelos": ", ".join(models), "corridas": len(t),
                                    "ganancia_media": t["ganancia"].mean(), "ic_t_bajo": lo, "ic_t_alto": hi,
                                    "p_wilcoxon": S.wilcoxon_exact(t["ganancia"].to_numpy())["p"]},
                                   name="valor").to_frame()
    return t


@tolerant()
def ensemble_summary(table: pd.DataFrame) -> pd.DataFrame:
    return table.attrs["resumen"]


# ---------------------------------------------------------------------------
# Atajos por artefactos
# ---------------------------------------------------------------------------


def _annotations(cfg: dict) -> pd.DataFrame | Missing:
    p = cfg["paths"]["artifacts_annotation_csv"]
    if not Path(p).exists():
        return Missing(f"no existe {p}.")
    a = pd.read_csv(p)
    cols = [c for c in cfg["artifact_columns"] if c in a.columns]
    a = a.dropna(subset=cols)
    return a.set_index("image_id")[cols].astype(int)


def _split_correct(rs, cfg: dict, model: str, ids, split: str) -> np.ndarray:
    """Fracción de corridas con acierto por imagen de `ids` en `split` (NaN si ninguna corrida la predice)."""
    hits = pd.DataFrame(index=pd.Index(ids))
    for key in sorted(rs.runs.get(model, {})):
        p = rs.predictions(model, *key, split).set_index("image_id")
        hits[key] = (p["label"] == p["pred"]).reindex(hits.index).astype(float)
    return hits.mean(axis=1).to_numpy()


@tolerant()
def shortcut_table(rs, cfg: dict, models, split: str) -> pd.DataFrame:
    """Por modelo y artefacto, en las imágenes anotadas de `split`: imágenes con y sin artefacto,
    exactitud balanceada y exactitud de cada grupo (aciertos medios entre corridas) y diferencia
    de exactitud balanceada (con - sin) con IC 95 % y p bilateral por bootstrap por lesión
    (e2_stats.exploratory_resamples), con ajuste e2_stats.exploratory_p_adjust entre las filas."""
    ann = _annotations(cfg)
    if is_missing(ann):
        return ann
    splits = D.load_splits(cfg["paths"]["splits_csv"]).set_index("image_id")
    in_split = splits.loc[ann.index, "split"] == ("test" if split == "test" else "train")
    ann = ann[in_split.to_numpy()]
    if ann.empty:
        n = len(_annotations(cfg))
        return Missing(f"ninguna de las {n} imágenes anotadas de data/anotacion_artefactos.csv está en {split}.")
    ids = ann.index.to_numpy()
    classes = D.class_names(cfg)
    y = splits.loc[ids, "dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    lesions = splits.loc[ids, "lesion_id"].to_numpy()
    st, alpha = cfg["e2_stats"], float(cfg["benchmark"]["alpha"])
    rows = []
    for m in models:
        if m not in rs.runs:
            continue
        acc = _split_correct(rs, cfg, m, ids, split)
        ok = np.isfinite(acc)
        if not ok.any():
            continue
        c = np.nan_to_num(acc)[None, :]
        masks = {}
        for art in ann.columns:
            has = ann[art].to_numpy() == 1
            masks[(art, 1)], masks[(art, 0)] = (has & ok).astype(float), (~has & ok).astype(float)
        boot = {k: [] for k in masks}
        for w in S.cluster_weight_batches(lesions, int(st["exploratory_resamples"]), int(cfg["benchmark"]["lesion_bootstrap"]["seed"])):
            for k, mk in masks.items():
                boot[k].append(S.weighted_balanced_accuracy(w, y, c, mk)[:, 0])
        one = np.ones((1, len(y)))
        for art in ann.columns:
            n1, n0 = int(masks[(art, 1)].sum()), int(masks[(art, 0)].sum())
            ba1 = float(S.weighted_balanced_accuracy(one, y, c, masks[(art, 1)])[0, 0]) if n1 else np.nan
            ba0 = float(S.weighted_balanced_accuracy(one, y, c, masks[(art, 0)])[0, 0]) if n0 else np.nan
            dist = np.concatenate(boot[(art, 1)]) - np.concatenate(boot[(art, 0)])
            lo, hi = S.percentile_interval(dist, alpha) if n1 and n0 else (np.nan, np.nan)
            rows.append({"modelo": m, "artefacto": art, "con": n1, "sin": n0,
                         "ba_con": ba1, "ba_sin": ba0, "exactitud_con": float(acc[masks[(art, 1)] > 0].mean()) if n1 else np.nan,
                         "exactitud_sin": float(acc[masks[(art, 0)] > 0].mean()) if n0 else np.nan,
                         "diferencia_ba": ba1 - ba0, "ic_bajo": lo, "ic_alto": hi,
                         "p": S.bootstrap_p(dist, 0.0, "two-sided") if n1 and n0 else np.nan})
    if not rows:
        return Missing(f"ningún modelo tiene predicciones de {split} para las imágenes anotadas.")
    out = pd.DataFrame(rows).set_index(["modelo", "artefacto"])
    out[f"p_{st['exploratory_p_adjust']}"] = S.adjust_pvalues(out["p"], st["exploratory_p_adjust"])
    out.attrs["nota"] = f"partición {split}; {len(ids)} imágenes anotadas"
    return out


# ---------------------------------------------------------------------------
# Taxonomía de errores
# ---------------------------------------------------------------------------


def taxonomy_sample(td, model: str, cfg: dict) -> pd.DataFrame:
    """Muestra estratificada por clase real de errors.taxonomy_n errores de `model` (imágenes que
    fallan en más de errors.error_frac de sus corridas), con asignación proporcional y al menos uno por
    clase con errores. Columnas informativas y una columna vacía por categoría."""
    ecfg = cfg["errors"]
    frac = 1.0 - td.correct(model).mean(axis=0)
    P = td.proba[model]
    pred_freq = pd.DataFrame(P.argmax(axis=2)).mode(axis=0).iloc[0].to_numpy().astype(int)
    df = pd.DataFrame({"image_id": td.image_ids, "lesion_id": td.lesions,
                       "label": [td.classes[i] for i in td.y], "pred_frecuente": [td.classes[i] for i in pred_freq],
                       "frac_error": frac, "confianza_media": P.max(axis=2).mean(axis=0)})
    if td.meta is not None:
        for c in ("dataset", "dx_type", "localization"):
            df[c] = td.meta[c].to_numpy()
    errors = df[df["frac_error"] > float(ecfg["error_frac"])]
    n = min(int(ecfg["taxonomy_n"]), len(errors))
    counts = errors["label"].value_counts()
    alloc = np.maximum(1, np.floor(n * counts / counts.sum())).astype(int).clip(upper=counts)
    while alloc.sum() > n:  # el mínimo de uno por clase puede pasarse de n: se quita de las clases mayores
        alloc[alloc.idxmax()] -= 1
    rest = n - alloc.sum()
    for c in (counts / counts.sum() * n - np.floor(counts / counts.sum() * n)).sort_values(ascending=False).index:
        if rest <= 0:
            break
        if alloc[c] < counts[c]:
            alloc[c] += 1
            rest -= 1
    rng = np.random.default_rng(int(ecfg["taxonomy_seed"]))
    parts = []
    for c, k in alloc.items():
        pool = errors[errors["label"] == c].sort_values("image_id")
        parts.append(pool.iloc[np.sort(rng.choice(len(pool), size=int(k), replace=False))])
    out = pd.concat(parts).sort_values(["label", "image_id"]).reset_index(drop=True)
    out.insert(0, "modelo", model)
    for cat in ecfg["taxonomy_categories"]:
        out[cat] = pd.NA
    out["comentario"] = ""
    return out


def write_taxonomy_template(cfg: dict, td, model: str, overwrite: bool = False) -> pd.DataFrame:
    """Escribe la plantilla en paths.error_taxonomy_csv si no existe (nunca sobrescribe etiquetas
    manuales salvo con `overwrite`) y la devuelve."""
    path = Path(cfg["paths"]["error_taxonomy_csv"])
    if path.exists() and not overwrite:
        return pd.read_csv(path)
    t = taxonomy_sample(td, model, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(path, index=False)
    return t


@tolerant()
def taxonomy_template(cfg: dict, td, ranked_models, write: bool = True) -> pd.DataFrame:
    """Plantilla de taxonomía con los errores del mejor modelo (el primero de `ranked_models`):
    con `write`, la escribe en paths.error_taxonomy_csv solo si no existe y la devuelve (si ya
    existe, devuelve la guardada, con las etiquetas manuales)."""
    if write:
        return write_taxonomy_template(cfg, td, ranked_models[0])
    return taxonomy_sample(td, ranked_models[0], cfg)


@tolerant()
def taxonomy_status(cfg: dict) -> dict:
    """Estado de la plantilla (filas etiquetadas con todas las categorías en 0/1 y valores no
    válidos) y conteo de errores por categoría."""
    path = Path(cfg["paths"]["error_taxonomy_csv"])
    if not path.exists():
        return Missing(f"no existe {path.name}; se genera en la sección de taxonomía de book/20-benchmark-e2.ipynb.")
    t = pd.read_csv(path)
    cats = [c for c in cfg["errors"]["taxonomy_categories"] if c in t.columns]
    vals = t[cats].apply(pd.to_numeric, errors="coerce")
    complete = vals.isin([0, 1]).all(axis=1)
    invalid = int((t[cats].notna() & ~vals.isin([0, 1])).sum().sum())
    out = pd.DataFrame({"errores_con_categoria": vals[complete].sum().astype(int)})
    out["porcentaje"] = 100 * out["errores_con_categoria"] / max(int(complete.sum()), 1)
    state = pd.Series({"filas": len(t), "etiquetadas": int(complete.sum()), "valores_no_validos": invalid,
                       "modelo": t["modelo"].iloc[0] if len(t) else ""}, name="valor").to_frame()
    return {"Estado de la plantilla": state, "Errores por categoría (filas etiquetadas)": out}


@tolerant()
def taxonomy_by_class(cfg: dict) -> pd.DataFrame:
    """Porcentaje de errores etiquetados con cada categoría por clase real."""
    path = Path(cfg["paths"]["error_taxonomy_csv"])
    if not path.exists():
        return Missing(f"no existe {path.name}.")
    t = pd.read_csv(path)
    cats = [c for c in cfg["errors"]["taxonomy_categories"] if c in t.columns]
    vals = t[cats].apply(pd.to_numeric, errors="coerce")
    done = vals.isin([0, 1]).all(axis=1)
    if not done.any():
        return Missing("la plantilla de taxonomía aún no tiene filas etiquetadas.")
    return 100 * vals[done].groupby(t.loc[done, "label"]).mean()
