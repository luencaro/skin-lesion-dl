"""Estadísticos ligeros (NumPy/pandas, sin SciPy) que se recalculan con los filtros del EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd


def epsilon2(values, groups) -> float:
    """ε² = H / (n − 1) de Kruskal–Wallis, con corrección por empates (como scipy.stats.kruskal)."""
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)}).dropna()
    n, k = len(df), df["g"].nunique()
    if k < 2 or n < 3:
        return float("nan")
    r = df["v"].rank(method="average")
    sums = r.groupby(df["g"]).agg(["sum", "count"])
    h = 12.0 / (n * (n + 1)) * float((sums["sum"] ** 2 / sums["count"]).sum()) - 3 * (n + 1)
    _, ties = np.unique(df["v"].to_numpy(), return_counts=True)
    corr = 1 - float((ties ** 3 - ties).sum()) / (n ** 3 - n)
    if corr <= 0:
        return float("nan")
    return float((h / corr) / (n - 1))


def cramers_v(a, b) -> float:
    """V de Cramér de la tabla de contingencia de dos variables categóricas."""
    table = pd.crosstab(pd.Series(np.asarray(a)), pd.Series(np.asarray(b))).to_numpy(dtype=float)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    n = table.sum()
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
