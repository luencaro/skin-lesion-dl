"""Tablas de contingencia y pruebas estadísticas para el EDA y el benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss


def value_counts_pct(s: pd.Series) -> pd.DataFrame:
    """Conteo y porcentaje de cada valor de `s`, de mayor a menor."""
    n = s.value_counts()
    return pd.DataFrame({"n": n, "porcentaje": n / n.sum() * 100})


def crosstab_with_pct(df: pd.DataFrame, row: str, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla `row` x `col` en conteos (con totales) y en porcentajes por fila. Excluye NaN."""
    sub = df[[row, col]].dropna()
    counts = pd.crosstab(sub[row], sub[col], margins=True, margins_name="Total")
    pct = pd.crosstab(sub[row], sub[col], normalize="index") * 100
    return counts, pct


def chi2_cramers_v(table) -> dict:
    """Chi-cuadrado de independencia (sin corrección de Yates) y V de Cramér.

    Se eliminan filas y columnas con total cero. `prop_esperadas_lt5` es la proporción
    de celdas con frecuencia esperada menor que 5.
    """
    t = np.asarray(table, dtype=float)
    t = t[t.sum(axis=1) > 0]
    t = t[:, t.sum(axis=0) > 0]
    r, k = t.shape
    if r < 2 or k < 2:
        return {"chi2": np.nan, "gl": 0, "p": np.nan, "v_cramer": np.nan, "n": int(t.sum()),
                "prop_esperadas_lt5": np.nan}
    chi2, p, dof, expected = ss.chi2_contingency(t, correction=False)
    n = t.sum()
    v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    return {
        "chi2": float(chi2),
        "gl": int(dof),
        "p": float(p),
        "v_cramer": float(v),
        "n": int(n),
        "prop_esperadas_lt5": float((expected < 5).mean()),
    }


def chi2_from_columns(df: pd.DataFrame, row: str, col: str) -> dict:
    """`chi2_cramers_v` sobre la tabla `row` x `col` de `df`, excluyendo NaN."""
    sub = df[[row, col]].dropna()
    return chi2_cramers_v(pd.crosstab(sub[row], sub[col]))


def chi2_table(df: pd.DataFrame, rows, col: str) -> pd.DataFrame:
    """Una fila de resultados de chi-cuadrado y V de Cramér por cada variable de `rows` frente a `col`."""
    return pd.DataFrame({r: chi2_from_columns(df, r, col) for r in rows}).T


def kruskal_epsilon2(values, groups) -> dict:
    """Kruskal-Wallis y tamaño de efecto épsilon cuadrado, ε² = H / (n - 1).

    Se excluyen los NaN. Si hay menos de dos grupos o todos los valores son iguales,
    devuelve NaN.
    """
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)}).dropna()
    samples = [g["v"].to_numpy() for _, g in df.groupby("g", observed=True)]
    n, k = len(df), len(samples)
    nan = {"H": np.nan, "gl": max(k - 1, 0), "p": np.nan, "epsilon2": np.nan, "n": n, "k": k}
    if k < 2:
        return nan
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            h, p = ss.kruskal(*samples)
    except ValueError:
        return nan
    if not np.isfinite(h):
        return nan
    return {"H": float(h), "gl": k - 1, "p": float(p), "epsilon2": float(h / (n - 1)), "n": n, "k": k}


def kruskal_table(df: pd.DataFrame, value_cols, group_col: str) -> pd.DataFrame:
    """Una fila de resultados de Kruskal-Wallis por cada variable de `value_cols` según `group_col`."""
    return pd.DataFrame({c: kruskal_epsilon2(df[c], df[group_col]) for c in value_cols}).T
