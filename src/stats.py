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


# ---------------------------------------------------------------------------
# Benchmark: tamaño muestral, intervalos y tamaños de efecto
# ---------------------------------------------------------------------------
# Las funciones reciben las diferencias pareadas d = a - b (una por corrida con la misma semilla
# y el mismo pliegue) o las dos muestras pareadas a y b. Todos los p-valores son bilaterales.


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def z_alpha_power(alpha: float, power: float) -> float:
    """z_{1 - alpha / 2} + z_{1 - beta}, con 1 - beta = power."""
    return float(ss.norm.ppf(1 - alpha / 2) + ss.norm.ppf(power))


def required_runs(sigma_d: float, delta: float, alpha: float, power: float) -> float:
    """R = (z_{1-alpha/2} + z_{1-beta})^2 sigma_d^2 / delta^2 (aproximación normal, sin redondear)."""
    return float(z_alpha_power(alpha, power) ** 2 * sigma_d ** 2 / delta ** 2)


def minimum_detectable_difference(sigma_d: float, n: int, alpha: float, power: float) -> float:
    """Diferencia mínima detectable con n corridas: (z_{1-alpha/2} + z_{1-beta}) sigma_d / sqrt(n)."""
    return float(z_alpha_power(alpha, power) * sigma_d / np.sqrt(n)) if n > 0 else float("nan")


def power_paired_t(sigma_d: float, n: int, delta: float, alpha: float) -> float:
    """Potencia de la t pareada bilateral para detectar una diferencia `delta` con n corridas
    (t no central con parámetro delta / (sigma_d / sqrt(n)) y n - 1 grados de libertad)."""
    if n < 2 or not np.isfinite(sigma_d) or sigma_d <= 0:
        return float("nan")
    df = n - 1
    nc = delta / (sigma_d / np.sqrt(n))
    crit = ss.t.ppf(1 - alpha / 2, df)
    return float(ss.nct.sf(crit, df, nc) + ss.nct.cdf(-crit, df, nc))


def shapiro_test(x) -> dict:
    """Shapiro-Wilk: W y p-valor. NaN si hay menos de 3 valores o todos son iguales."""
    x = _finite(x)
    if len(x) < 3 or np.ptp(x) == 0:
        return {"n": len(x), "W": np.nan, "p": np.nan}
    w, p = ss.shapiro(x)
    return {"n": len(x), "W": float(w), "p": float(p)}


def t_confidence_interval(x, alpha: float) -> tuple[float, float]:
    """IC 1 - alpha de la media por t de Student (ddof = 1)."""
    x = _finite(x)
    if len(x) < 2:
        return float("nan"), float("nan")
    half = ss.t.ppf(1 - alpha / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - half), float(x.mean() + half)


def bootstrap_confidence_interval(x, alpha: float, n_resamples: int, seed: int,
                                  method: str = "percentile") -> tuple[float, float]:
    """IC 1 - alpha de la media por bootstrap (scipy.stats.bootstrap con `method`). NaN si
    hay menos de 2 valores, todos iguales o el método no converge."""
    x = _finite(x)
    if len(x) < 2 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    try:
        with np.errstate(all="ignore"):
            res = ss.bootstrap((x,), np.mean, confidence_level=1 - alpha, n_resamples=n_resamples,
                               method=method, rng=np.random.default_rng(seed))
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except (ValueError, RuntimeError):
        return float("nan"), float("nan")


def cohens_dz(d) -> float:
    """d de Cohen pareada: media de las diferencias / su desviación estándar (ddof = 1)."""
    d = _finite(d)
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def _signed_ranks(d, zero_method: str = "wilcox") -> tuple[np.ndarray, np.ndarray]:
    """Rangos medios de |d| y signos. wilcox: descarta los ceros; pratt: los incluye al ordenar y
    luego los descarta."""
    d = _finite(d)
    if zero_method == "wilcox":
        d = d[d != 0]
        ranks = ss.rankdata(np.abs(d))
    elif zero_method == "pratt":
        ranks = ss.rankdata(np.abs(d))
        keep = d != 0
        d, ranks = d[keep], ranks[keep]
    else:
        raise ValueError(f"zero_method no soportado: {zero_method}")
    return ranks, np.sign(d)


def rank_biserial_paired(d, zero_method: str = "wilcox") -> float:
    """r de rango biserial para muestras pareadas: (W+ - W-) / (W+ + W-)."""
    ranks, sign = _signed_ranks(d, zero_method)
    total = ranks.sum()
    if total == 0:
        return float("nan")
    return float((ranks[sign > 0].sum() - ranks[sign < 0].sum()) / total)


def cliffs_delta(a, b) -> float:
    """Delta de Cliff: P(a > b) - P(a < b) sobre todos los pares (a_i, b_j)."""
    a, b = _finite(a), _finite(b)
    if not len(a) or not len(b):
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / diff.size)


def probability_of_superiority_paired(d) -> float:
    """Probabilidad de superioridad pareada: P(d > 0) + P(d = 0) / 2 (fracción de corridas
    pareadas en que a supera a b)."""
    d = _finite(d)
    if not len(d):
        return float("nan")
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


# ---------------------------------------------------------------------------
# Benchmark: pruebas pareadas
# ---------------------------------------------------------------------------


def paired_t_test(d) -> dict:
    """t pareada (t de una muestra sobre d frente a 0)."""
    d = _finite(d)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return {"n": len(d), "t": np.nan, "gl": max(len(d) - 1, 0), "p": np.nan}
    t, p = ss.ttest_1samp(d, 0.0)
    return {"n": len(d), "t": float(t), "gl": len(d) - 1, "p": float(p)}


def wilcoxon_exact(d, zero_method: str = "wilcox") -> dict:
    """Wilcoxon de rangos con signo con p-valor exacto.

    El p-valor se obtiene de la distribución de permutación exacta de W+ condicionada a los
    rangos observados (2^n asignaciones de signo, por programación dinámica sobre los rangos
    duplicados), así que sigue siendo exacto con empates. Sin empates coincide con
    scipy.stats.wilcoxon(method="exact"). p = min(1, 2 min(P(W+ <= w), P(W+ >= w))).
    """
    ranks, sign = _signed_ranks(d, zero_method)
    n = len(ranks)
    out = {"n_no_nulas": n, "W_mas": np.nan, "W_menos": np.nan, "p": np.nan,
           "empates": bool(len(np.unique(ranks)) < n) if n else False}
    if n == 0:
        return out
    r2 = np.rint(2 * ranks).astype(np.int64)  # rangos medios x 2 -> enteros
    total = int(r2.sum())
    dist = np.zeros(total + 1, dtype=np.float64)
    dist[0] = 1.0
    for r in r2:  # número de asignaciones de signo con suma (x 2) de rangos positivos = s
        shifted = np.zeros_like(dist)
        shifted[r:] = dist[:-r] if r > 0 else dist
        dist = dist + shifted
    dist /= dist.sum()
    w2 = int(r2[sign > 0].sum())
    cdf = dist[: w2 + 1].sum()
    sf = dist[w2:].sum()
    out.update({"W_mas": w2 / 2, "W_menos": (total - w2) / 2, "p": float(min(1.0, 2 * min(cdf, sf)))})
    return out


def nadeau_bengio_t(d, test_train_ratio: float) -> dict:
    """t corregida de Nadeau y Bengio (2003) para validación cruzada repetida:
    t = media(d) / sqrt((1 / n + n_val / n_train) s^2), con n - 1 grados de libertad."""
    d = _finite(d)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return {"n": n, "t": np.nan, "gl": max(n - 1, 0), "p": np.nan, "ee_corregido": np.nan}
    se = np.sqrt((1 / n + test_train_ratio) * d.var(ddof=1))
    t = d.mean() / se
    return {"n": n, "t": float(t), "gl": n - 1, "p": float(2 * ss.t.sf(abs(t), n - 1)),
            "ee_corregido": float(se), "ee_sin_corregir": float(d.std(ddof=1) / np.sqrt(n))}


def tost_paired(d, delta: float, alpha: float) -> dict:
    """Equivalencia por dos pruebas t unilaterales (TOST) con margen (-delta, delta).
    Equivalentes si p_tost = max(p_inferior, p_superior) < alpha, es decir, si el IC
    1 - 2 alpha de la media de d queda dentro del margen."""
    d = _finite(d)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return {"n": n, "p_inferior": np.nan, "p_superior": np.nan, "p_tost": np.nan,
                "ic_bajo": np.nan, "ic_alto": np.nan, "equivalentes": None}
    se = d.std(ddof=1) / np.sqrt(n)
    df = n - 1
    p_low = ss.t.sf((d.mean() + delta) / se, df)    # H0: media <= -delta
    p_up = ss.t.cdf((d.mean() - delta) / se, df)    # H0: media >= delta
    half = ss.t.ppf(1 - alpha, df) * se
    p = max(p_low, p_up)
    return {"n": n, "p_inferior": float(p_low), "p_superior": float(p_up), "p_tost": float(p),
            "ic_bajo": float(d.mean() - half), "ic_alto": float(d.mean() + half), "equivalentes": bool(p < alpha)}


def adjust_pvalues(pvals, method: str = "holm") -> np.ndarray:
    """p-valores ajustados (statsmodels.multipletests); los NaN se conservan."""
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method=method)[1]
    return out


# ---------------------------------------------------------------------------
# Benchmark: comparación conjunta (matriz corridas x modelos, sin faltantes)
# ---------------------------------------------------------------------------


def rm_anova(wide: pd.DataFrame) -> dict:
    """ANOVA de medidas repetidas de un factor (modelo; sujeto = corrida), con eta cuadrado
    parcial y corrección de Greenhouse-Geisser. `wide`: filas = corridas, columnas = modelos."""
    from statsmodels.stats.anova import AnovaRM

    x = wide.to_numpy(dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return {"F": np.nan, "gl_modelo": k - 1, "gl_error": np.nan, "p": np.nan}
    long = wide.reset_index(drop=True).rename_axis("corrida").reset_index().melt(
        id_vars="corrida", var_name="modelo", value_name="y")
    table = AnovaRM(long, "y", "corrida", within=["modelo"]).fit().anova_table
    f, df1, df2, p = (float(table.iloc[0][c]) for c in ("F Value", "Num DF", "Den DF", "Pr > F"))
    grand = x.mean()
    ss_model = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_subject = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_error = ((x - grand) ** 2).sum() - ss_model - ss_subject
    # Greenhouse-Geisser: épsilon a partir de la covarianza de contrastes ortonormales
    contrasts = np.linalg.qr(np.vstack([np.ones(k), np.eye(k)[:-1]]).T)[0][:, 1:].T  # (k-1) x k
    s = contrasts @ np.cov(x, rowvar=False) @ contrasts.T
    eps = float(np.trace(s) ** 2 / ((k - 1) * np.trace(s @ s))) if np.trace(s @ s) > 0 else np.nan
    p_gg = float(ss.f.sf(f, df1 * eps, df2 * eps)) if np.isfinite(eps) else np.nan
    return {"F": f, "gl_modelo": df1, "gl_error": df2, "p": p,
            "eta2_parcial": float(ss_model / (ss_model + ss_error)) if ss_model + ss_error > 0 else np.nan,
            "epsilon_gg": eps, "p_gg": p_gg}


def friedman_test(wide: pd.DataFrame) -> dict:
    """Friedman (corridas = bloques) y W de Kendall = chi2 / (n (k - 1)). Requiere k >= 3."""
    x = wide.to_numpy(dtype=float)
    n, k = x.shape
    if k < 3 or n < 2:
        return {"chi2": np.nan, "gl": k - 1, "p": np.nan, "W_kendall": np.nan, "n": n, "k": k}
    chi2, p = ss.friedmanchisquare(*[x[:, j] for j in range(k)])
    return {"chi2": float(chi2), "gl": k - 1, "p": float(p), "W_kendall": float(chi2 / (n * (k - 1))),
            "n": n, "k": k}


def mean_ranks(wide: pd.DataFrame) -> pd.Series:
    """Rango medio de cada modelo entre corridas (1 = peor, k = mejor; empates con rango medio)."""
    ranks = np.apply_along_axis(ss.rankdata, 1, wide.to_numpy(dtype=float))
    return pd.Series(ranks.mean(axis=0), index=wide.columns, name="rango_medio")


def nemenyi_test(wide: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc de Nemenyi tras Friedman: p-valor de cada par a partir de la distribución del
    rango studentizado, q = |R_i - R_j| / sqrt(k (k + 1) / (6 n)), p = P(Q_{k, inf} > q sqrt(2))."""
    n, k = wide.shape
    r = mean_ranks(wide)
    se = np.sqrt(k * (k + 1) / (6 * n))
    rows = []
    cols = list(wide.columns)
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(r.iloc[i] - r.iloc[j]) / se
            rows.append({"modelo_a": cols[i], "modelo_b": cols[j], "rango_a": r.iloc[i], "rango_b": r.iloc[j],
                         "q": float(q), "p": float(ss.studentized_range.sf(q * np.sqrt(2), k, np.inf))})
    return pd.DataFrame(rows)


def variance_decomposition(df: pd.DataFrame, value: str, model: str = "model", seed: str = "seed",
                           fold: str = "val_fold") -> pd.DataFrame:
    """Descomposición de la suma de cuadrados de `value` en un diseño equilibrado modelo x corrida
    (corrida = semilla y pliegue). Filas: modelo, corrida, residual (modelo x corrida) y total;
    si las corridas forman un producto completo semilla x pliegue con más de un nivel de cada uno,
    la corrida se separa además en semilla, pliegue y semilla x pliegue (filas con sangría « · »).
    `proporcion` = SC / SC total; F y p del modelo frente al residual."""
    d = df[[model, seed, fold, value]].copy()
    d["_run"] = d[seed].astype(str) + "_" + d[fold].astype(str)
    y = d[value].to_numpy(dtype=float)
    grand = y.mean()
    k, r = d[model].nunique(), d["_run"].nunique()
    if len(d) != k * r:
        raise ValueError("Diseño no equilibrado: cada modelo debe tener las mismas corridas")

    def ss_of(cols):
        g = d.groupby(cols)[value]
        return float((g.size() * (g.mean() - grand) ** 2).sum())

    ss_total = float(((y - grand) ** 2).sum())
    ss_model, ss_run = ss_of([model]), ss_of(["_run"])
    ss_res = ss_total - ss_model - ss_run
    df_model, df_run, df_res = k - 1, r - 1, (k - 1) * (r - 1)
    rows = [("modelo", ss_model, df_model), ("corrida (semilla y pliegue)", ss_run, df_run)]
    n_seed, n_fold = d[seed].nunique(), d[fold].nunique()
    if n_seed > 1 and n_fold > 1 and r == n_seed * n_fold:
        ss_seed, ss_fold = ss_of([seed]), ss_of([fold])
        rows += [(" · semilla", ss_seed, n_seed - 1), (" · pliegue", ss_fold, n_fold - 1),
                 (" · semilla x pliegue", ss_run - ss_seed - ss_fold, (n_seed - 1) * (n_fold - 1))]
    rows += [("residual (modelo x corrida)", ss_res, df_res), ("total", ss_total, len(d) - 1)]
    out = pd.DataFrame(rows, columns=["fuente", "SC", "gl"]).set_index("fuente")
    out["proporcion"] = out["SC"] / ss_total if ss_total > 0 else np.nan
    out["CM"] = out["SC"] / out["gl"].replace(0, np.nan)
    ms_res = ss_res / df_res if df_res > 0 else np.nan
    out["F"] = np.nan
    out["p"] = np.nan
    if ms_res and np.isfinite(ms_res) and ms_res > 0:
        for name in ("modelo", "corrida (semilla y pliegue)"):
            f = out.loc[name, "CM"] / ms_res
            out.loc[name, "F"] = f
            out.loc[name, "p"] = float(ss.f.sf(f, out.loc[name, "gl"], df_res))
    return out


# ---------------------------------------------------------------------------
# Benchmark: pruebas sobre las predicciones de test de una corrida pareada
# ---------------------------------------------------------------------------


def mcnemar_test(correct_a, correct_b, exact_below: int = 25) -> dict:
    """McNemar sobre aciertos pareados por imagen. b = a acierta y b falla, c = a falla y b
    acierta. Si b + c < exact_below, binomial exacta; si no, chi-cuadrado con corrección de
    continuidad."""
    from statsmodels.stats.contingency_tables import mcnemar

    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    table = np.array([[np.sum(a & b_), np.sum(a & ~b_)], [np.sum(~a & b_), np.sum(~a & ~b_)]])
    b, c = int(table[0, 1]), int(table[1, 0])
    exact = (b + c) < exact_below
    if b + c == 0:
        return {"b": b, "c": c, "estadistico": np.nan, "p": 1.0, "metodo": "sin discordancias"}
    res = mcnemar(table, exact=exact, correction=True)
    return {"b": b, "c": c, "estadistico": float(res.statistic), "p": float(res.pvalue),
            "metodo": "binomial exacta" if exact else "chi-cuadrado con corrección de continuidad"}


def delong_test(y_binary, score_a, score_b) -> dict:
    """DeLong para dos AUC correlacionadas sobre las mismas imágenes (algoritmo rápido de Sun y
    Xu, 2014). `y_binary`: 1 = clase positiva. Devuelve AUC de a y b, diferencia, z y p."""
    y = np.asarray(y_binary).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    if m < 2 or n < 2:
        return {"auc_a": np.nan, "auc_b": np.nan, "diferencia": np.nan, "z": np.nan, "p": np.nan,
                "positivos": m}
    scores = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    x, z = scores[:, y], scores[:, ~y]
    tx = np.apply_along_axis(ss.rankdata, 1, x)
    tz = np.apply_along_axis(ss.rankdata, 1, z)
    txz = np.apply_along_axis(ss.rankdata, 1, np.hstack([x, z]))
    aucs = txz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (txz[:, :m] - tx) / n
    v10 = 1.0 - (txz[:, m:] - tz) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        zstat, p = np.nan, (1.0 if diff == 0 else np.nan)
    else:
        zstat = diff / np.sqrt(var)
        p = float(2 * ss.norm.sf(abs(zstat)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diferencia": diff,
            "z": float(zstat) if np.isfinite(zstat) else np.nan, "p": p, "positivos": m}


def _macro_auc_ovr_batch(y: np.ndarray, proba: np.ndarray, idx: np.ndarray, classes) -> np.ndarray:
    """AUC-ROC macro uno contra el resto de cada remuestreo (filas de `idx`), por rangos."""
    aucs = []
    yb = y[idx]  # (B, N)
    for k in classes:
        s = proba[idx, k]
        ranks = ss.rankdata(s, axis=1)
        pos = yb == k
        m = pos.sum(axis=1)
        n = pos.shape[1] - m
        aucs.append(((ranks * pos).sum(axis=1) - m * (m + 1) / 2) / (m * n))
    return np.mean(aucs, axis=0)


def stratified_bootstrap_auc_diff(y, proba_a, proba_b, n_resamples: int, seed: int, alpha: float,
                                  batch: int = 200) -> dict:
    """Diferencia de AUC-ROC macro uno contra el resto (a - b) sobre las mismas imágenes, con IC
    1 - alpha por bootstrap percentil estratificado por clase (se remuestrea dentro de cada clase,
    conservando su tamaño, y se usan los mismos índices para los dos modelos)."""
    y = np.asarray(y)
    pa, pb = np.asarray(proba_a, float), np.asarray(proba_b, float)
    classes = np.unique(y)
    full = np.arange(len(y))[None, :]
    point = float(_macro_auc_ovr_batch(y, pa, full, classes)[0] - _macro_auc_ovr_batch(y, pb, full, classes)[0])
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in classes]
    diffs = []
    done = 0
    while done < n_resamples:
        b = min(batch, n_resamples - done)
        idx = np.hstack([g[rng.integers(0, len(g), size=(b, len(g)))] for g in groups])
        diffs.append(_macro_auc_ovr_batch(y, pa, idx, classes) - _macro_auc_ovr_batch(y, pb, idx, classes))
        done += b
    diffs = np.concatenate(diffs)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diferencia_auc_macro": point, "ic_bajo": float(lo), "ic_alto": float(hi),
            "p_bootstrap": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
            "remuestreos": int(n_resamples)}


def cluster_bootstrap_balanced_accuracy(y, groups, correct: dict, n_resamples: int, seed: int,
                                        batch: int = 500) -> dict:
    """Bootstrap por conglomerados de la exactitud balanceada media entre corridas.

    `y`: etiqueta entera de cada imagen (N). `groups`: conglomerado de cada imagen (p. ej.
    lesion_id); en cada remuestreo se sortean con reemplazo tantos conglomerados como hay, y
    cada imagen entra tantas veces como su conglomerado. `correct`: {clave: matriz booleana
    (corridas, N) de aciertos}. En cada remuestreo se recalcula la exactitud balanceada de cada
    corrida (media de la sensibilidad de las clases presentes en el remuestreo) y se promedia
    entre corridas. Todas las claves usan los mismos remuestreos, de modo que las diferencias
    entre claves quedan pareadas.

    Devuelve {clave: {"observado": float, "remuestreos": array (n_resamples,)}}.
    """
    y = np.asarray(y)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(g.max()) + 1
    classes = np.unique(y)
    masks = [y == k for k in classes]
    mats = {key: np.asarray(c, dtype=np.float64) for key, c in correct.items()}

    def mean_ba(w):  # w: (B, N) pesos por imagen -> {clave: (B,)}
        n = np.stack([w[:, m].sum(axis=1) for m in masks], axis=1)          # (B, K)
        out = {}
        for key, c in mats.items():
            hits = np.stack([w[:, m] @ c[:, m].T for m in masks], axis=2)    # (B, R, K)
            with np.errstate(invalid="ignore", divide="ignore"):
                recall = hits / n[:, None, :]
            out[key] = np.nanmean(recall, axis=2).mean(axis=1)               # clases presentes, luego corridas
        return out

    observed = mean_ba(np.ones((1, len(y))))
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in mats}
    done = 0
    while done < n_resamples:
        b = min(batch, n_resamples - done)
        draws = rng.integers(0, n_groups, size=(b, n_groups))
        counts = np.zeros((b, n_groups))
        np.add.at(counts, (np.arange(b)[:, None], draws), 1.0)
        for key, v in mean_ba(counts[:, g]).items():
            samples[key].append(v)
        done += b
    return {key: {"observado": float(observed[key][0]), "remuestreos": np.concatenate(samples[key])}
            for key in mats}
