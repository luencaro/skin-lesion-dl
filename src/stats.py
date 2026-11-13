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


def _chi2_statistic(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Estadístico de Pearson sobre las dos últimas dimensiones (admite lotes de tablas)."""
    return ((observed - expected) ** 2 / expected).sum(axis=(-2, -1))


def chi2_monte_carlo_p(table, n_iter: int, seed: int, batch_size: int = 250) -> float:
    """p-valor de chi-cuadrado por simulación de Monte Carlo.

    Reconstruye las observaciones individuales de la tabla, permuta las etiquetas de
    columna `n_iter` veces (con lo que se conservan los dos márgenes) y devuelve
    (1 + número de estadísticos simulados >= observado) / (n_iter + 1).
    La tabla no debe tener filas ni columnas con total cero.
    """
    t = np.asarray(table, dtype=np.int64)
    r, k = t.shape
    rows = np.repeat(np.arange(r), t.sum(axis=1))
    cols = np.repeat(np.arange(k), t.sum(axis=0))
    n = rows.size
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    observed_stat = _chi2_statistic(t, expected)
    tol = 1e-9 * max(1.0, observed_stat)
    rng = np.random.default_rng(seed)
    exceed, done = 0, 0
    while done < n_iter:
        b = min(batch_size, n_iter - done)
        perm = rng.permuted(np.tile(cols, (b, 1)), axis=1)
        idx = rows[None, :] * k + perm + (np.arange(b) * r * k)[:, None]
        sims = np.bincount(idx.ravel(), minlength=b * r * k).reshape(b, r, k)
        exceed += int((_chi2_statistic(sims, expected) >= observed_stat - tol).sum())
        done += b
    return (exceed + 1) / (n_iter + 1)


def chi2_cramers_v(table, n_iter: int = 0, seed: int | None = None) -> dict:
    """Chi-cuadrado de independencia (sin corrección de Yates) y V de Cramér.

    Se eliminan filas y columnas con total cero. Devuelve el p-valor asintótico y, si
    `n_iter` > 0, el p-valor por simulación de Monte Carlo (`chi2_monte_carlo_p`).
    `prop_esperadas_lt5` es la proporción de celdas con frecuencia esperada menor que 5.
    """
    t = np.asarray(table, dtype=float)
    t = t[t.sum(axis=1) > 0]
    t = t[:, t.sum(axis=0) > 0]
    r, k = t.shape
    if r < 2 or k < 2:
        return {"chi2": np.nan, "gl": 0, "p_asintotico": np.nan, "p_simulado": np.nan,
                "v_cramer": np.nan, "n": int(t.sum()), "prop_esperadas_lt5": np.nan,
                "iteraciones_mc": n_iter}
    chi2, p, dof, expected = ss.chi2_contingency(t, correction=False)
    n = t.sum()
    v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    p_sim = chi2_monte_carlo_p(t, n_iter, seed) if n_iter > 0 else np.nan
    return {
        "chi2": float(chi2),
        "gl": int(dof),
        "p_asintotico": float(p),
        "p_simulado": float(p_sim),
        "v_cramer": float(v),
        "n": int(n),
        "prop_esperadas_lt5": float((expected < 5).mean()),
        "iteraciones_mc": n_iter,
    }


def chi2_from_columns(df: pd.DataFrame, row: str, col: str, n_iter: int = 0,
                      seed: int | None = None) -> dict:
    """`chi2_cramers_v` sobre la tabla `row` x `col` de `df`, excluyendo NaN."""
    sub = df[[row, col]].dropna()
    return chi2_cramers_v(pd.crosstab(sub[row], sub[col]), n_iter, seed)


def chi2_table(df: pd.DataFrame, rows, col: str, n_iter: int = 0, seed: int | None = None) -> pd.DataFrame:
    """Una fila de resultados de chi-cuadrado y V de Cramér por cada variable de `rows` frente a `col`."""
    return pd.DataFrame({r: chi2_from_columns(df, r, col, n_iter, seed) for r in rows}).T


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


def univariate_summary(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Estadísticos descriptivos por variable numérica: faltantes, describe() y asimetría."""
    d = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    out = d.describe().T
    out.insert(0, "faltantes", d.isna().sum())
    out["asimetria"] = d.skew()
    return out


def spearman_matrix(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Matriz de correlación de Spearman (observaciones completas por pares)."""
    return df[list(cols)].apply(pd.to_numeric, errors="coerce").corr(method="spearman")


def top_correlations(corr: pd.DataFrame, n: int) -> pd.DataFrame:
    """Los `n` pares de variables distintas con mayor |rho|, de mayor a menor."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().dropna().rename("rho").reset_index()
    pairs.columns = ["variable_1", "variable_2", "rho"]
    pairs["abs_rho"] = pairs["rho"].abs()
    return pairs.sort_values("abs_rho", ascending=False).head(n).reset_index(drop=True)
