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


def class_entropy(counts) -> dict:
    """Entropía de Shannon (bits) de una distribución de conteos, entropía máxima log2(k) y normalizada."""
    n = np.asarray(counts, dtype=float)
    n = n[n > 0]
    p = n / n.sum()
    h = float(-(p * np.log2(p)).sum())
    h_max = float(np.log2(len(n))) if len(n) > 1 else 0.0
    return {"clases": len(n), "entropia_bits": h, "entropia_maxima_bits": h_max,
            "entropia_normalizada": h / h_max if h_max > 0 else np.nan}


def two_sample_compare(a, b) -> dict:
    """Kolmogorov-Smirnov de dos muestras y diferencia de medias estandarizada.

    DME = (media_a - media_b) / sqrt((var_a + var_b) / 2), con varianzas muestrales
    (ddof = 1). Se excluyen los NaN.
    """
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy()
    out = {"n_a": len(a), "n_b": len(b), "media_a": np.nan, "media_b": np.nan,
           "ks": np.nan, "p_ks": np.nan, "dme": np.nan}
    if len(a) < 2 or len(b) < 2:
        return out
    ks = ss.ks_2samp(a, b)
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    out.update({"media_a": float(a.mean()), "media_b": float(b.mean()), "ks": float(ks.statistic),
                "p_ks": float(ks.pvalue), "dme": float((a.mean() - b.mean()) / sd) if sd > 0 else np.nan})
    return out


def compare_groups_vs_rest(df: pd.DataFrame, value: str, group: str) -> pd.DataFrame:
    """`two_sample_compare` de cada grupo (a) frente al resto de las filas (b)."""
    rows = {}
    for g in sorted(df[group].dropna().unique()):
        mask = df[group] == g
        rows[g] = two_sample_compare(df.loc[mask, value], df.loc[~mask, value])
    return pd.DataFrame(rows).T.rename_axis(f"{group} (frente al resto)")


def univariate_outliers(df: pd.DataFrame, cols, iqr_factor: float, robust_z: float):
    """Marca atípicos univariados por la regla de IQR y por el z robusto.

    IQR: valor < Q1 - iqr_factor * IQR o > Q3 + iqr_factor * IQR.
    z robusto: |0,6745 * (x - mediana) / MAD| > robust_z (Iglewicz y Hoaglin); si
    MAD = 0 no se marca ningún valor con esta regla. Los NaN no se marcan.
    Devuelve (marcas, resumen): `marcas` tiene una columna booleana por variable y regla
    (`<variable>__iqr`, `<variable>__z_robusto`), alineada con `df`; `resumen` tiene una
    fila por variable.
    """
    flags, rows = {}, {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        v = x.dropna()
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        lo, hi = q1 - iqr_factor * (q3 - q1), q3 + iqr_factor * (q3 - q1)
        med = v.median()
        mad = (v - med).abs().median()
        f_iqr = ((x < lo) | (x > hi)).fillna(False)
        if mad > 0:
            f_rz = ((0.6745 * (x - med) / mad).abs() > robust_z).fillna(False)
        else:
            f_rz = pd.Series(False, index=df.index)
        flags[f"{c}__iqr"] = f_iqr.astype(bool)
        flags[f"{c}__z_robusto"] = f_rz.astype(bool)
        n = len(v)
        rows[c] = {"n_validos": n, "limite_inferior_iqr": lo, "limite_superior_iqr": hi,
                   "atipicos_iqr": int(f_iqr.sum()), "pct_iqr": 100 * f_iqr.sum() / n if n else np.nan,
                   "mediana": med, "mad": mad, "mad_cero": bool(mad == 0),
                   "atipicos_z_robusto": int(f_rz.sum()),
                   "pct_z_robusto": 100 * f_rz.sum() / n if n else np.nan}
    return pd.DataFrame(flags, index=df.index), pd.DataFrame(rows).T


def outlier_counts_by_group(flags: pd.DataFrame, groups, order=None) -> dict:
    """Número de atípicos por grupo y variable, una tabla por regla (`iqr`, `z_robusto`)."""
    out = {}
    for regla in ("iqr", "z_robusto"):
        cols = [c for c in flags.columns if c.endswith(f"__{regla}")]
        t = flags[cols].groupby(np.asarray(groups)).sum()
        t.columns = [c.rsplit("__", 1)[0] for c in cols]
        if order is not None:
            t = t.loc[[o for o in order if o in t.index]]
        out[regla] = t
    return out


def robust_mahalanobis(df: pd.DataFrame, cols, quantile: float, seed: int):
    """Distancia de Mahalanobis robusta al cuadrado (MinCovDet) sobre variables estandarizadas.

    Se usan las variables con MAD > 0 (con MAD = 0 la covarianza robusta es singular) y
    solo las filas sin NaN en ellas; el resto queda con NaN. Umbral: cuantil `quantile`
    de la chi-cuadrado con tantos grados de libertad como variables usadas.
    Devuelve (resultado, info): `resultado` (alineado con `df`) tiene `mahal_d2` y
    `atipico_mahal`; `info` resume variables, filas y umbral.
    """
    from sklearn.covariance import MinCovDet
    from sklearn.preprocessing import StandardScaler

    X = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    usadas, excluidas = mahalanobis_variables(df, cols)
    completas = X[usadas].notna().all(axis=1)
    Z = StandardScaler().fit_transform(X.loc[completas, usadas])
    mcd = MinCovDet(random_state=seed).fit(Z)
    d2 = pd.Series(np.nan, index=df.index)
    d2[completas] = mcd.mahalanobis(Z)
    umbral = float(ss.chi2.ppf(quantile, df=len(usadas)))
    res = pd.DataFrame({"mahal_d2": d2, "atipico_mahal": (d2 > umbral)}, index=df.index)
    info = {"variables_usadas": ", ".join(usadas), "variables_excluidas_mad_cero": ", ".join(excluidas) or "ninguna",
            "filas_completas": int(completas.sum()), "filas_con_faltantes": int((~completas).sum()),
            "cuantil": quantile, "umbral_d2": umbral, "atipicos": int(res["atipico_mahal"].sum())}
    return res, info


def mahalanobis_variables(df: pd.DataFrame, cols) -> tuple[list, list]:
    """Variables que usa `robust_mahalanobis` (MAD > 0) y variables excluidas (MAD = 0)."""
    X = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    mad = (X - X.median()).abs().median()
    usadas = [c for c in cols if mad[c] > 0]
    return usadas, [c for c in cols if c not in usadas]


def vif_table(df: pd.DataFrame, cols) -> tuple[pd.DataFrame, dict]:
    """Factor de inflación de la varianza (statsmodels) de cada variable estandarizada, con
    constante, y número de condición de la matriz de correlación. Usa las filas sin NaN.

    Devuelve (tabla ordenada por VIF de mayor a menor, resumen). Un VIF infinito indica
    dependencia lineal exacta con otras variables.
    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df[list(cols)].apply(pd.to_numeric, errors="coerce").dropna()
    Z = (X - X.mean()) / X.std(ddof=0)
    Zc = sm.add_constant(Z.to_numpy(), has_constant="add")
    with np.errstate(divide="ignore", invalid="ignore"):
        vif = [float(variance_inflation_factor(Zc, i + 1)) for i in range(len(cols))]
    corr = np.corrcoef(Z.to_numpy(), rowvar=False)
    eig = np.linalg.eigvalsh(corr)
    resumen = {"variables": len(cols), "filas_completas": len(X),
               "numero_condicion": float(np.linalg.cond(corr)),
               "autovalor_minimo": float(eig.min()), "autovalor_maximo": float(eig.max())}
    tabla = pd.DataFrame({"vif": vif}, index=pd.Index(list(cols), name="variable"))
    return tabla.sort_values("vif", ascending=False), resumen


def mutual_info_table(df: pd.DataFrame, target: str, categorical, numeric, seed: int) -> pd.DataFrame:
    """Información mutua entre `target` y cada variable (`mutual_info_classif`).

    Categóricas: discretas, con los NaN como categoría propia ("<faltante>"). Numéricas:
    continuas, sin las filas con NaN. Resultado en nats y en bits, de mayor a menor.
    """
    from sklearn.feature_selection import mutual_info_classif

    y_ok = df[target].notna()
    rows = {}
    for c in categorical:
        x = df[c].astype("object").where(df[c].notna(), "<faltante>")
        codes = pd.factorize(x[y_ok])[0].reshape(-1, 1)
        mi = mutual_info_classif(codes, df.loc[y_ok, target], discrete_features=True, random_state=seed)[0]
        rows[c] = {"tipo": "categórica", "n": int(y_ok.sum()), "categorias": int(x[y_ok].nunique()), "mi_nats": mi}
    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce")
        ok = x.notna() & y_ok
        mi = mutual_info_classif(x[ok].to_numpy().reshape(-1, 1), df.loc[ok, target],
                                 discrete_features=False, random_state=seed)[0]
        rows[c] = {"tipo": "continua", "n": int(ok.sum()), "categorias": np.nan, "mi_nats": mi}
    out = pd.DataFrame(rows).T.rename_axis("variable")
    out["mi_nats"] = out["mi_nats"].astype(float)
    out["mi_bits"] = out["mi_nats"] / np.log(2)
    return out.sort_values("mi_nats", ascending=False)


def purity_table(df: pd.DataFrame, cols, target: str) -> pd.DataFrame:
    """Para cada categoría de cada variable de `cols`: imágenes, clase mayoritaria de
    `target` y su porcentaje. Excluye NaN."""
    rows = []
    for c in cols:
        sub = df[[c, target]].dropna()
        for cat, r in pd.crosstab(sub[c], sub[target]).iterrows():
            n = int(r.sum())
            rows.append({"variable": c, "categoria": cat, "n": n, "clase_mayoritaria": r.idxmax(),
                         "pct_mayoritaria": 100 * r.max() / n})
    return pd.DataFrame(rows).set_index(["variable", "categoria"])


def type_shares(meta: pd.DataFrame, type_col: str, class_col: str, lesion_col: str,
                type_order=None, class_order=None) -> dict:
    """Porcentaje de imágenes y de lesiones por `type_col`, global y por `class_col`.

    A nivel de lesión se usa el primer valor de cada lesión; `lesiones_con_mas_de_un_tipo`
    cuenta las lesiones con más de un valor de `type_col`.
    """
    les = meta.groupby(lesion_col).agg(**{class_col: (class_col, "first"), type_col: (type_col, "first"),
                                          "n_tipos": (type_col, "nunique")})
    out = {"lesiones_con_mas_de_un_tipo": int((les["n_tipos"] > 1).sum())}
    for unidad, d in (("imagenes", meta), ("lesiones", les)):
        glob = value_counts_pct(d[type_col])
        por = pd.crosstab(d[class_col], d[type_col], normalize="index") * 100
        if type_order is not None:
            extra = [t for t in glob.index if t not in type_order]
            glob = glob.reindex(list(type_order) + extra, fill_value=0)
            por = por.reindex(columns=list(type_order) + extra, fill_value=0)
        if class_order is not None:
            por = por.reindex([c for c in class_order if c in por.index])
        out[f"{unidad}_global"] = glob
        out[f"{unidad}_por_clase"] = por
    return out
