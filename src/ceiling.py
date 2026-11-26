"""Techo de desempeño: error de Bayes estimado con k vecinos y cota de lectores.

- k vecinos sobre los embeddings congelados de MobileNetV2 (los de src.baselines; train y, si
  están calculados, test), con la métrica ceiling.metric y sin contar como vecinos las imágenes de
  la misma lesión (ceiling.exclude_same_lesion), para no medir la memoria de la lesión. Para cada k
  de ceiling.knn_k se calcula la exactitud balanceada de la regla de k vecinos (voto mayoritario;
  en empate, la clase del vecino más cercano).
- Cota inferior del error de Bayes de Cover y Hart (1967) a partir del error de 1-NN R:
  R* >= ((K - 1) / K) (1 - sqrt(1 - K R / (K - 1))). Se aplica al error balanceado 1 - BA (que es
  el error con clases equiprobables), de modo que el techo es s_techo = 1 - R*_inferior, una cota
  optimista de la exactitud balanceada alcanzable con esa representación. Es una aproximación
  asintótica y depende de la representación (los embeddings de ImageNet).
- Cota de lectores: la de benchmark.reference_bounds con nombre ceiling.readers_bound_name
  (tschandl2019), con su métrica original.

γ y las comparaciones con los modelos se calculan en src.benchmark.gamma_table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data as D
from src.report import Missing, is_missing, tolerant


def knn_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray, ks, metric: str = "cosine",
                    exclude_same_group: bool = True) -> dict:
    """{k: predicción de k vecinos de cada punto} sin el propio punto y, si `exclude_same_group`,
    sin los puntos de su mismo grupo."""
    from sklearn.neighbors import NearestNeighbors

    ks = sorted(int(k) for k in ks)
    max_group = int(pd.Series(groups).value_counts().max()) if exclude_same_group else 1
    nn = NearestNeighbors(n_neighbors=min(len(X), ks[-1] + max_group), metric=metric).fit(X)
    _, idx = nn.kneighbors(X)
    k_cls = int(y.max()) + 1
    out = {k: np.empty(len(X), dtype=int) for k in ks}
    for i in range(len(X)):
        cand = idx[i][idx[i] != i]
        if exclude_same_group:
            cand = cand[groups[cand] != groups[i]]
        for k in ks:
            nb = y[cand[:k]]
            votes = np.bincount(nb, minlength=k_cls)
            best = np.flatnonzero(votes == votes.max())
            out[k][i] = best[0] if len(best) == 1 else next(c for c in nb if c in best)
    return out


def cover_hart_lower(r1: float, k: int) -> float:
    """Cota inferior del error de Bayes a partir del error de 1-NN `r1` con `k` clases."""
    r1 = min(float(r1), (k - 1) / k)
    return float((k - 1) / k * (1 - np.sqrt(max(0.0, 1 - k * r1 / (k - 1)))))


def ceiling_data(cfg: dict):
    """(X, y, lesion_id, partición, image_ids) de las imágenes con embeddings en cache."""
    from src.baselines import _read_cache

    splits = D.load_splits(cfg["paths"]["splits_csv"]).set_index("image_id")
    feats = {}
    for key in ("embeddings_cache", "embeddings_test_cache"):
        c = _read_cache(cfg["paths"][key])
        if c is not None:
            feats.update(zip(c[1], c[0]))
    if not feats:
        return Missing("no hay embeddings en cache (book/06-eda_imagenes.ipynb y `python -m src.baselines "
                       "--compute-embeddings`).")
    ids = np.array(sorted(i for i in feats if i in splits.index))
    classes = D.class_names(cfg)
    y = splits.loc[ids, "dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return np.stack([feats[i] for i in ids]).astype(np.float32), y, splits.loc[ids, "lesion_id"].to_numpy(), \
        splits.loc[ids, "split"].to_numpy(), ids


@tolerant()
def knn_ceiling(cfg: dict) -> pd.DataFrame:
    """Exactitud balanceada de k vecinos por k (todas las imágenes con embeddings, y solo test si
    los hay) y techo de Cover y Hart a partir de 1-NN."""
    from sklearn.metrics import balanced_accuracy_score

    data = ceiling_data(cfg)
    if is_missing(data):
        return data
    X, y, les, part, _ = data
    c = cfg["ceiling"]
    ks = sorted(set([1] + [int(k) for k in c["knn_k"]]))
    preds = knn_predictions(X, y, les, ks, c["metric"], bool(c["exclude_same_lesion"]))
    K = len(D.class_names(cfg))
    rows = []
    for subset, mask in (("todas", np.ones(len(y), bool)), ("test", part == "test")):
        if not mask.any():
            continue
        for k in ks:
            ba = balanced_accuracy_score(y[mask], preds[k][mask])
            row = {"subconjunto": subset, "k": k, "imagenes": int(mask.sum()), "balanced_accuracy_knn": float(ba),
                   "error_balanceado": float(1 - ba)}
            if k == 1:
                low = cover_hart_lower(1 - ba, K)
                row.update({"bayes_cota_inferior": low, "techo": 1 - low})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["subconjunto", "k"])


@tolerant()
def ceilings_table(cfg: dict, knn: pd.DataFrame) -> pd.DataFrame:
    """Techos para γ: Cover y Hart (1-NN, todas las imágenes) y la cota de lectores."""
    rows = []
    if not is_missing(knn) and ("todas", 1) in knn.index:
        rows.append({"techo_nombre": "Bayes (Cover y Hart, 1-NN, embeddings)", "techo": float(knn.loc[("todas", 1), "techo"]),
                     "metrica": "exactitud balanceada", "fuente": "estimado"})
    name = cfg["ceiling"]["readers_bound_name"]
    ref = [r for r in cfg["benchmark"].get("reference_bounds", []) if r["name"] == name]
    if ref:
        rows.append({"techo_nombre": name, "techo": float(ref[0]["value"]), "metrica": ref[0]["metric"],
                     "fuente": ref[0]["source"]})
    if not rows:
        return Missing("no hay techos disponibles.")
    return pd.DataFrame(rows).set_index("techo_nombre")
