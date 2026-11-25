"""Benchmark de los modelos base: carga de corridas pareadas y tablas de book/11-benchmark.ipynb.

Las observaciones pareadas son las corridas con la misma semilla y el mismo pliegue de
validación. Las comparaciones por pares usan las corridas que tienen los dos modelos del par;
las conjuntas (ANOVA de medidas repetidas, Friedman, descomposición de la varianza), las que
tienen todos los modelos. Para cada par (a, b), en el orden de benchmark.models, la diferencia
es d = a - b. Parámetros: sección `benchmark` de configs/base.yaml.

Todas las funciones son tolerantes (src/report.py): si faltan corridas devuelven un `Missing`
que el notebook muestra como mensaje.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import stats as S
from src.report import Missing, is_missing, tolerant
from src.results import GLOBAL_METRICS, load_run_info, load_run_metrics

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


@dataclass
class RunSet:
    """Corridas terminadas de varios modelos con una misma etiqueta (pilot o final)."""

    label: str
    models: list[str]                    # modelos con al menos una corrida terminada, en el orden de la config
    expected_models: list[str]
    expected_pairs: list[tuple[int, int]]
    runs: dict                           # {modelo: {(semilla, pliegue): carpeta}}
    metrics: pd.DataFrame                # largo: model, seed, val_fold, split, metric, value
    info: pd.DataFrame                   # por corrida: tiempos, parámetros y tamaños
    _preds: dict = field(default_factory=dict, repr=False)

    def predictions(self, model: str, seed: int, fold: int, split: str = "test") -> pd.DataFrame:
        """probabilities_<split>.csv de una corrida (en cache)."""
        key = (model, seed, fold, split)
        if key not in self._preds:
            self._preds[key] = pd.read_csv(self.runs[model][(seed, fold)] / f"probabilities_{split}.csv")
        return self._preds[key]


def bench_cfg(cfg: dict) -> dict:
    return cfg["benchmark"]


def load_runs(cfg: dict, label: str, pairs=None) -> RunSet:
    """Corridas terminadas de benchmark.models con etiqueta `label` y pares (semilla, pliegue)
    `pairs` (por defecto, los de final si label es la de final y los del piloto si es "pilot")."""
    from src.final import final_label, final_pairs
    from src.train import run_dir_for

    models = list(bench_cfg(cfg)["models"])
    if pairs is None:
        pairs = final_pairs(cfg) if label == final_label(cfg) else [tuple(p) for p in cfg["train"]["pilot"]["pairs"]]
    pairs = [(int(s), int(f)) for s, f in pairs]
    runs, mrows, irows = {}, [], []
    for m in models:
        found = {}
        for s, f in pairs:
            d = run_dir_for(cfg, m, label, s, f)
            if (d / "metrics.json").exists():
                found[(s, f)] = d
        if not found:
            continue
        runs[m] = found
        for (s, f), d in found.items():
            metrics = load_run_metrics(d)
            for split in SPLITS:
                for k in GLOBAL_METRICS + ["cross_entropy"]:
                    if split in metrics and k in metrics[split]:
                        mrows.append({"model": m, "seed": s, "val_fold": f, "split": split, "metric": k,
                                      "value": float(metrics[split][k])})
            info = load_run_info(d)
            inf = info.get("inference_s_per_image", {})
            irows.append({"model": m, "seed": s, "val_fold": f,
                          "train_time_s": info["train_time_s"]["total"],
                          "inference_ms_per_image_test": 1000 * inf["test"] if "test" in inf else np.nan,
                          "inference_ms_per_image_val": 1000 * inf["val"] if "val" in inf else np.nan,
                          "params_trainable": info["params"]["finetune"]["trainable"],
                          "params_total": info["params"]["finetune"]["total"],
                          "n_train": info["n_images"]["train"], "n_val": info["n_images"]["val"]})
    return RunSet(label=label, models=[m for m in models if m in runs], expected_models=models,
                  expected_pairs=pairs, runs=runs,
                  metrics=pd.DataFrame(mrows, columns=["model", "seed", "val_fold", "split", "metric", "value"]),
                  info=pd.DataFrame(irows))


def availability_table(rs: RunSet) -> pd.DataFrame:
    """Corridas previstas y terminadas por modelo, y corridas pareadas con todos los modelos."""
    rows = {m: {"previstas": len(rs.expected_pairs), "terminadas": len(rs.runs.get(m, {}))}
            for m in rs.expected_models}
    out = pd.DataFrame(rows).T
    common = set.intersection(*(set(r) for r in rs.runs.values())) if rs.runs else set()
    out["pareadas_con_todos"] = len(common) if len(rs.models) > 1 else 0
    return out.rename_axis(f"modelo ({rs.label})")


def _need_models(rs: RunSet, k: int, what: str) -> Missing | None:
    if len(rs.models) < k:
        have = ", ".join(rs.models) or "ninguno"
        return Missing(f"{what}: se necesitan corridas «{rs.label}» de al menos {k} modelo(s) "
                       f"(modelos con corridas: {have}).")
    return None


def wide(rs: RunSet, split: str, metric: str, models=None) -> pd.DataFrame:
    """Matriz corridas x modelos de `metric` en `split`, solo con las corridas completas (que
    tienen todos los `models`). Índice: (seed, val_fold)."""
    models = list(models or rs.models)
    sub = rs.metrics[(rs.metrics["split"] == split) & (rs.metrics["metric"] == metric)
                     & rs.metrics["model"].isin(models)]
    w = sub.pivot_table(index=["seed", "val_fold"], columns="model", values="value")
    return w.reindex(columns=models).dropna()


def model_pairs(rs: RunSet) -> list[tuple[str, str]]:
    return list(combinations(rs.models, 2))


def paired_diffs(rs: RunSet, split: str, metric: str) -> dict:
    """{(a, b): Series de d = a - b por corrida pareada} para cada par de modelos."""
    out = {}
    for a, b in model_pairs(rs):
        w = wide(rs, split, metric, [a, b])
        out[(a, b)] = (w[a] - w[b]).rename(f"{a} - {b}")
    return out


def _pair_index(pairs) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(list(pairs), names=["modelo_a", "modelo_b"])


# ---------------------------------------------------------------------------
# 2. Piloto conjunto: variabilidad, tamaño muestral y normalidad
# ---------------------------------------------------------------------------


@tolerant()
def pilot_model_summary(rs: RunSet, cfg: dict, split: str = "val") -> pd.DataFrame:
    """Por modelo: corridas, media, desviación estándar (ddof = 1) y CV de la métrica primaria."""
    m = _need_models(rs, 1, "El resumen del piloto")
    if m is not None:
        return m
    metric = bench_cfg(cfg)["primary_metric"]
    sub = rs.metrics[(rs.metrics["split"] == split) & (rs.metrics["metric"] == metric)]
    if sub.empty:
        return Missing(f"no hay {metric} de {split} en las corridas «{rs.label}».")
    g = sub.groupby("model")["value"]
    out = pd.DataFrame({"corridas": g.size(), "media": g.mean(), "std": g.std(ddof=1)})
    out["cv"] = out["std"] / out["media"]
    return out.reindex(rs.models).rename_axis(f"{metric} ({split})")


@tolerant()
def pilot_power_table(rs: RunSet, cfg: dict, split: str = "val") -> pd.DataFrame:
    """Por par de modelos: corridas pareadas, media y sigma_d de las diferencias, delta / sigma_d,
    R = (z_{1-a/2} + z_{1-b})^2 sigma_d^2 / delta^2 (y redondeado hacia arriba), diferencia
    mínima detectable con las corridas disponibles y Shapiro-Wilk de las diferencias."""
    m = _need_models(rs, 2, "El piloto conjunto")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    delta, alpha, power = float(b["delta"]), float(b["alpha"]), float(b["power"])
    rows = []
    for d in paired_diffs(rs, split, b["primary_metric"]).values():
        n = len(d)
        sd = float(d.std(ddof=1)) if n > 1 else np.nan
        r = S.required_runs(sd, delta, alpha, power) if np.isfinite(sd) else np.nan
        sh = S.shapiro_test(d)
        rows.append({"corridas_pareadas": n, "media_d": float(d.mean()) if n else np.nan, "sigma_d": sd,
                     "delta_sobre_sigma_d": delta / sd if sd else np.nan, "R": r,
                     "R_redondeado": int(np.ceil(r)) if np.isfinite(r) else np.nan,
                     "dif_minima_detectable": S.minimum_detectable_difference(sd, n, alpha, power),
                     "shapiro_W": sh["W"], "shapiro_p": sh["p"]})
    out = pd.DataFrame(rows, index=_pair_index(model_pairs(rs)))
    out.attrs["nota"] = f"delta = {delta}, alpha = {alpha}, potencia = {power}, métrica = {b['primary_metric']} ({split})"
    return out


@tolerant()
def pilot_differences(rs: RunSet, cfg: dict, split: str = "val") -> pd.DataFrame:
    """Diferencias pareadas por corrida (filas) y par de modelos (columnas)."""
    m = _need_models(rs, 2, "Las diferencias pareadas")
    if m is not None:
        return m
    return pd.concat(paired_diffs(rs, split, bench_cfg(cfg)["primary_metric"]).values(), axis=1)


# ---------------------------------------------------------------------------
# 3. Tabla comparativa
# ---------------------------------------------------------------------------


@tolerant()
def metric_intervals(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Por modelo y métrica: corridas, media, desviación estándar, IC 95 % por t de Student e
    IC 95 % por bootstrap de la media (método de benchmark.bootstrap.method)."""
    m = _need_models(rs, 1, "La tabla comparativa")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    split = split or b["split"]
    alpha, boot = float(b["alpha"]), b["bootstrap"]
    rows = []
    for model in rs.models:
        for k in b["metrics"]:
            x = rs.metrics.query("model == @model and split == @split and metric == @k")["value"].to_numpy()
            if not len(x):
                continue
            t_lo, t_hi = S.t_confidence_interval(x, alpha)
            b_lo, b_hi = S.bootstrap_confidence_interval(x, alpha, int(boot["n_resamples"]), int(boot["seed"]),
                                                         boot["method"])
            rows.append({"modelo": model, "metrica": k, "corridas": len(x), "media": x.mean(),
                         "std": x.std(ddof=1) if len(x) > 1 else np.nan,
                         "ic_t_bajo": t_lo, "ic_t_alto": t_hi, "ic_boot_bajo": b_lo, "ic_boot_alto": b_hi,
                         "metodo_boot": f"bootstrap {boot['method']} ({boot['n_resamples']} remuestreos)"})
    if not rows:
        return Missing(f"no hay métricas de {split} en las corridas «{rs.label}».")
    return pd.DataFrame(rows).set_index(["modelo", "metrica"])


@tolerant()
def cost_summary(rs: RunSet) -> pd.DataFrame:
    """Por modelo: tiempo de entrenamiento (h), inferencia por imagen en test (ms) y parámetros."""
    if rs.info.empty:
        return Missing(f"no hay corridas «{rs.label}».")
    g = rs.info.groupby("model")
    out = pd.DataFrame({
        "entrenamiento_h_media": g["train_time_s"].mean() / 3600,
        "entrenamiento_h_std": g["train_time_s"].std(ddof=1) / 3600,
        "inferencia_ms_imagen_media": g["inference_ms_per_image_test"].mean(),
        "inferencia_ms_imagen_std": g["inference_ms_per_image_test"].std(ddof=1),
        "parametros_entrenables": g["params_trainable"].first(),
        "parametros_totales": g["params_total"].first(),
    })
    return out.reindex(rs.models)


@tolerant()
def comparison_table(rs: RunSet, cfg: dict, split: str | None = None, digits: int = 4) -> pd.DataFrame:
    """Tabla 5.1: por modelo, cada métrica como «media ± std» y el costo computacional."""
    iv = metric_intervals(rs, cfg, split)
    if is_missing(iv):
        return iv
    txt = iv.apply(lambda r: f"{r['media']:.{digits}f} ± {r['std']:.{digits}f}", axis=1).unstack("metrica")
    txt = txt.reindex(columns=[k for k in bench_cfg(cfg)["metrics"] if k in txt.columns])
    cost = cost_summary(rs)
    txt["entrenamiento (h)"] = cost.apply(lambda r: f"{r['entrenamiento_h_media']:.2f} ± {r['entrenamiento_h_std']:.2f}", axis=1)
    txt["inferencia (ms/imagen)"] = cost["inferencia_ms_imagen_media"].map(lambda v: f"{v:.2f}")
    txt["parámetros entrenables"] = cost["parametros_entrenables"].map(lambda v: f"{int(v):,}".replace(",", " "))
    txt.insert(0, "corridas", iv.groupby(level="modelo")["corridas"].max())
    return txt.reindex(rs.models)


# ---------------------------------------------------------------------------
# 4. Ranking y trade-offs
# ---------------------------------------------------------------------------


@tolerant()
def ranking_table(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Modelos ordenados por la media de la métrica primaria, con su rango medio entre corridas
    pareadas (Friedman; k = mejor) si hay al menos dos modelos."""
    m = _need_models(rs, 1, "El ranking")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    split = split or b["split"]
    metric = b["primary_metric"]
    sub = rs.metrics.query("split == @split and metric == @metric and model in @rs.models")
    if sub.empty:
        return Missing(f"no hay {metric} de {split} en las corridas «{rs.label}».")
    g = sub.groupby("model")["value"]
    out = pd.DataFrame({"corridas": g.size(), "media": g.mean(), "std": g.std(ddof=1)})
    if len(rs.models) > 1:
        w = wide(rs, split, metric)
        if len(w):
            out["rango_medio"] = S.mean_ranks(w)
    out = out.sort_values("media", ascending=False)
    out.insert(0, "posicion", np.arange(1, len(out) + 1))
    return out.rename_axis(f"{metric} ({split})")


@tolerant()
def plot_tradeoffs(rs: RunSet, cfg: dict, split: str | None = None):
    """Métrica primaria (media ± std) frente a parámetros entrenables (escala log) y frente al
    tiempo de entrenamiento medio por corrida."""
    rank = ranking_table(rs, cfg, split)
    if is_missing(rank):
        return rank
    cost = cost_summary(rs)
    metric = bench_cfg(cfg)["primary_metric"]
    df = rank.join(cost)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (col, xerr, label, log) in zip(axes, (
            ("parametros_entrenables", None, "parámetros entrenables", True),
            ("entrenamiento_h_media", "entrenamiento_h_std", "tiempo de entrenamiento por corrida (h)", False))):
        for model, r in df.iterrows():
            ax.errorbar(r[col], r["media"], yerr=r["std"], xerr=r[xerr] if xerr else None, fmt="o", capsize=4)
            ax.annotate(model, (r[col], r["media"]), textcoords="offset points", xytext=(6, 6), fontsize=9)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"{metric} ({split or bench_cfg(cfg)['split']})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Baseline y margen
# ---------------------------------------------------------------------------


@tolerant()
def best_model_table(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Mejor y peor modelo por la media de la métrica primaria, con media, std e IC 95 % (t)."""
    rank = ranking_table(rs, cfg, split)
    iv = metric_intervals(rs, cfg, split)
    if is_missing(rank) or is_missing(iv):
        return rank if is_missing(rank) else iv
    metric = bench_cfg(cfg)["primary_metric"]
    rows = {}
    for role, model in (("mejor", rank.index[0]), ("peor", rank.index[-1])):
        r = iv.loc[(model, metric)]
        rows[role] = {"modelo": model, "corridas": r["corridas"], "media": r["media"], "std": r["std"],
                      "ic_t_bajo": r["ic_t_bajo"], "ic_t_alto": r["ic_t_alto"],
                      "ic_boot_bajo": r["ic_boot_bajo"], "ic_boot_alto": r["ic_boot_alto"]}
    return pd.DataFrame(rows).T


def _diff_row(d: pd.Series, mean_b: float, zero_method: str) -> dict:
    return {"corridas_pareadas": len(d), "diferencia": float(d.mean()),
            "diferencia_relativa_pct": 100 * float(d.mean()) / mean_b if mean_b else np.nan,
            "d_cohen_pareada": S.cohens_dz(d), "r_rango_biserial": S.rank_biserial_paired(d, zero_method)}


@tolerant()
def pairwise_differences(rs: RunSet, cfg: dict, split: str | None = None, reference: str | None = None) -> pd.DataFrame:
    """Diferencias absolutas (a - b) y relativas (% respecto de la media pareada de b) entre
    pares de modelos, con d de Cohen pareada y r de rango biserial.

    reference = None: todos los pares en el orden de la config. "mejor": mejor - cada otro.
    "peor": cada otro - peor.
    """
    m = _need_models(rs, 2, "Las diferencias entre modelos")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    split = split or b["split"]
    metric, zm = b["primary_metric"], b.get("wilcoxon_zero_method", "wilcox")
    if reference is None:
        pairs = model_pairs(rs)
    else:
        rank = ranking_table(rs, cfg, split)
        if is_missing(rank):
            return rank
        best, worst = rank.index[0], rank.index[-1]
        pairs = ([(best, o) for o in rank.index if o != best] if reference == "mejor"
                 else [(o, worst) for o in rank.index if o != worst])
    rows = []
    for a, c in pairs:
        w = wide(rs, split, metric, [a, c])
        rows.append(_diff_row(w[a] - w[c], float(w[c].mean()), zm))
    return pd.DataFrame(rows, index=_pair_index(pairs))


@tolerant()
def minimum_detectable_vs_best(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Mejora mínima detectable respecto del mejor modelo con las corridas pareadas disponibles,
    (z_{1-a/2} + z_{1-b}) sigma_d / sqrt(n), la potencia de la t pareada para delta y las
    corridas R necesarias para delta."""
    m = _need_models(rs, 2, "La mejora mínima detectable")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    split = split or b["split"]
    delta, alpha, power = float(b["delta"]), float(b["alpha"]), float(b["power"])
    rank = ranking_table(rs, cfg, split)
    if is_missing(rank):
        return rank
    best = rank.index[0]
    rows, pairs = [], []
    for o in rank.index[1:]:
        w = wide(rs, split, b["primary_metric"], [best, o])
        d = w[best] - w[o]
        sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
        pairs.append((best, o))
        rows.append({"corridas_pareadas": len(d), "sigma_d": sd,
                     "dif_minima_detectable": S.minimum_detectable_difference(sd, len(d), alpha, power),
                     "potencia_para_delta": S.power_paired_t(sd, len(d), delta, alpha),
                     "R_para_delta": S.required_runs(sd, delta, alpha, power) if np.isfinite(sd) else np.nan})
    return pd.DataFrame(rows, index=_pair_index(pairs))


@tolerant()
def reference_bounds_table(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Cotas de referencia de benchmark.reference_bounds: valor, métrica, nota y fuente (clave de
    references.bib), y la diferencia entre la media del mejor modelo y la cota, solo para las cotas
    cuya métrica es la primaria (benchmark.primary_metric_label); en las demás queda vacía, porque
    no son comparables."""
    b = bench_cfg(cfg)
    table = pd.DataFrame(b.get("reference_bounds") or [])
    if table.empty:
        return Missing("benchmark.reference_bounds está vacío en configs/base.yaml.")
    table = table.set_index("name").rename_axis("cota")
    table = table.reindex(columns=["value", "metric", "note", "source"]).rename(
        columns={"value": "valor", "metric": "metrica", "note": "nota", "source": "fuente"})
    rank = ranking_table(rs, cfg, split)
    primary = {str(b["primary_metric"]).lower(), str(b.get("primary_metric_label", "")).lower()}
    comparable = table["metrica"].astype(str).str.strip().str.lower().isin(primary) & table["valor"].notna()
    table["mejor_modelo_menos_cota"] = np.nan
    if not is_missing(rank):
        best_mean = float(rank["media"].iloc[0])
        table.loc[comparable, "mejor_modelo_menos_cota"] = best_mean - table.loc[comparable, "valor"].astype(float)
    return table


# ---------------------------------------------------------------------------
# 6.1 Supuestos
# ---------------------------------------------------------------------------


@tolerant()
def normality_table(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Shapiro-Wilk de las diferencias pareadas de cada par y criterio automático: paramétrico
    si ningún par rechaza la normalidad (p >= alpha), no paramétrico en otro caso."""
    m = _need_models(rs, 2, "La prueba de normalidad")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    split = split or b["split"]
    alpha = float(b["alpha"])
    rows = []
    for d in paired_diffs(rs, split, b["primary_metric"]).values():
        sh = S.shapiro_test(d)
        rows.append({"corridas_pareadas": sh["n"], "shapiro_W": sh["W"], "shapiro_p": sh["p"],
                     "rechaza_normalidad": bool(sh["p"] < alpha) if np.isfinite(sh["p"]) else None})
    out = pd.DataFrame(rows, index=_pair_index(model_pairs(rs)))
    out.attrs["criterio"] = test_family(out)
    return out


def test_family(normality: pd.DataFrame) -> str:
    """'paramétrico' si ningún par rechaza la normalidad y todos tienen p definido; si no,
    'no paramétrico'."""
    if is_missing(normality):
        return "no disponible"
    flags = normality["rechaza_normalidad"]
    return "paramétrico" if flags.notna().all() and not flags.astype(bool).any() else "no paramétrico"


@tolerant()
def criterion_table(normality: pd.DataFrame) -> pd.DataFrame:
    """Criterio elegido y pruebas principales que implica."""
    fam = test_family(normality)
    tests = {"paramétrico": ("ANOVA de medidas repetidas", "t pareada"),
             "no paramétrico": ("Friedman + Nemenyi", "Wilcoxon exacto")}[fam]
    return pd.Series({"criterio": fam, "comparación conjunta": tests[0], "comparaciones pareadas": tests[1],
                      "regla": "paramétrico si Shapiro-Wilk no rechaza la normalidad en ningún par (p >= alpha)"},
                     name="valor").to_frame()


@tolerant()
def plot_qq(rs: RunSet, cfg: dict, split: str | None = None):
    """Gráfico Q-Q normal de las diferencias pareadas de cada par."""
    m = _need_models(rs, 2, "Los gráficos Q-Q")
    if m is not None:
        return m
    from scipy import stats as ss

    b = bench_cfg(cfg)
    diffs = paired_diffs(rs, split or b["split"], b["primary_metric"])
    fig, axes = plt.subplots(1, len(diffs), figsize=(4 * len(diffs), 3.8), squeeze=False)
    for ax, ((a, c), d) in zip(axes[0], diffs.items()):
        if len(d) >= 3:
            ss.probplot(d.to_numpy(), dist="norm", plot=ax)
        ax.set_title(f"{a} - {c} (n = {len(d)})")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6.2 Comparación conjunta
# ---------------------------------------------------------------------------


@tolerant()
def joint_tests(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """ANOVA de medidas repetidas (con Greenhouse-Geisser) y Friedman (con W de Kendall) sobre
    las corridas que tienen todos los modelos."""
    m = _need_models(rs, 2, "La comparación conjunta")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    w = wide(rs, split or b["split"], b["primary_metric"])
    if len(w) < 2:
        return Missing("hay menos de dos corridas pareadas con todos los modelos.")
    rm = S.rm_anova(w)
    fr = S.friedman_test(w)
    out = pd.DataFrame({
        "ANOVA de medidas repetidas": {"estadistico": rm["F"], "gl": f"{rm['gl_modelo']:g}, {rm['gl_error']:g}",
                                       "p": rm["p"], "p_greenhouse_geisser": rm.get("p_gg"),
                                       "efecto": rm.get("eta2_parcial"), "medida_efecto": "eta² parcial",
                                       "corridas": len(w)},
        "Friedman": {"estadistico": fr["chi2"], "gl": f"{fr['gl']}", "p": fr["p"], "p_greenhouse_geisser": np.nan,
                     "efecto": fr["W_kendall"], "medida_efecto": "W de Kendall", "corridas": len(w)},
    }).T
    if not np.isfinite(fr["p"]):
        out.loc["Friedman", "medida_efecto"] = "requiere al menos 3 modelos"
    return out


@tolerant()
def nemenyi_table(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Post-hoc de Nemenyi (rangos medios por corrida) sobre las corridas con todos los modelos."""
    m = _need_models(rs, 3, "El post-hoc de Nemenyi")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    w = wide(rs, split or b["split"], b["primary_metric"])
    return S.nemenyi_test(w).set_index(["modelo_a", "modelo_b"])


# ---------------------------------------------------------------------------
# 6.3 Comparaciones pareadas
# ---------------------------------------------------------------------------


@tolerant()
def pairwise_tests(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Por par (a, b): diferencia media, IC 95 % bootstrap, t pareada y Wilcoxon exacto con
    p-valores sin ajustar y ajustados (benchmark.p_adjust entre los pares), d de Cohen pareada,
    delta de Cliff, probabilidad de superioridad pareada, potencia para delta y diferencia mínima
    detectable. `prueba_principal` sigue el criterio de 6.1."""
    m = _need_models(rs, 2, "Las comparaciones pareadas")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    split = split or b["split"]
    alpha, delta, power, boot = float(b["alpha"]), float(b["delta"]), float(b["power"]), b["bootstrap"]
    zm = b.get("wilcoxon_zero_method", "wilcox")
    family = test_family(normality_table(rs, cfg, split))
    rows = []
    for a, c in model_pairs(rs):
        w = wide(rs, split, b["primary_metric"], [a, c])
        d = (w[a] - w[c]).to_numpy()
        lo, hi = S.bootstrap_confidence_interval(d, alpha, int(boot["n_resamples"]), int(boot["seed"]), boot["method"])
        t = S.paired_t_test(d)
        wx = S.wilcoxon_exact(d, zm)
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
        rows.append({"corridas_pareadas": len(d), "diferencia_media": float(np.mean(d)) if len(d) else np.nan,
                     "ic_boot_bajo": lo, "ic_boot_alto": hi, "t": t["t"], "gl": t["gl"], "p_t": t["p"],
                     "W_mas": wx["W_mas"], "p_wilcoxon": wx["p"], "empates_wilcoxon": wx["empates"],
                     "d_cohen_pareada": S.cohens_dz(d), "delta_cliff": S.cliffs_delta(w[a], w[c]),
                     "prob_superioridad": S.probability_of_superiority_paired(d),
                     "potencia_para_delta": S.power_paired_t(sd, len(d), delta, alpha),
                     "dif_minima_detectable": S.minimum_detectable_difference(sd, len(d), alpha, power)})
    out = pd.DataFrame(rows, index=_pair_index(model_pairs(rs)))
    method = b.get("p_adjust", "holm")
    out.insert(out.columns.get_loc("p_t") + 1, f"p_t_{method}", S.adjust_pvalues(out["p_t"], method))
    out.insert(out.columns.get_loc("p_wilcoxon") + 1, f"p_wilcoxon_{method}", S.adjust_pvalues(out["p_wilcoxon"], method))
    main = "p_t" if family == "paramétrico" else "p_wilcoxon"
    out["prueba_principal"] = "t pareada" if family == "paramétrico" else "Wilcoxon exacto"
    out["significativa"] = out[f"{main}_{method}"] < alpha
    out.attrs["nota"] = (f"d = a - b; bootstrap {boot['method']} con {boot['n_resamples']} remuestreos; "
                         f"ajuste {method} entre {len(out)} pares; delta de Cliff sobre todos los pares (a_i, b_j); "
                         f"probabilidad de superioridad = P(d > 0) + P(d = 0) / 2")
    return out


# ---------------------------------------------------------------------------
# 6.4 Validación cruzada repetida: Nadeau y Bengio
# ---------------------------------------------------------------------------


@tolerant()
def nadeau_bengio_table(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """t corregida de Nadeau y Bengio por par y partición de benchmark.nadeau_bengio_splits, con
    la razón n_val / n_train media de las corridas, y p ajustado entre los pares."""
    m = _need_models(rs, 2, "La t de Nadeau y Bengio")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    ratio = float((rs.info["n_val"] / rs.info["n_train"]).mean())
    method = b.get("p_adjust", "holm")
    frames = []
    for split in b.get("nadeau_bengio_splits", ["val"]):
        rows, pairs = [], []
        for (a, c), d in paired_diffs(rs, split, b["primary_metric"]).items():
            if not len(d):
                continue
            pairs.append((split, a, c))
            rows.append({"diferencia_media": float(d.mean()), **S.nadeau_bengio_t(d, ratio)})
        if rows:
            t = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(pairs, names=["particion", "modelo_a", "modelo_b"]))
            t[f"p_{method}"] = S.adjust_pvalues(t["p"], method)
            frames.append(t)
    if not frames:
        return Missing("no hay diferencias pareadas en las particiones de nadeau_bengio_splits.")
    out = pd.concat(frames)
    out["razon_val_train"] = ratio
    return out


# ---------------------------------------------------------------------------
# 6.5 Equivalencia (TOST)
# ---------------------------------------------------------------------------


@tolerant()
def tost_table(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """TOST pareado con margen (-delta, delta) para cada par, con el IC 1 - 2 alpha de la diferencia."""
    m = _need_models(rs, 2, "La prueba de equivalencia")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    delta, alpha = float(b["delta"]), float(b["alpha"])
    rows = [{"diferencia_media": float(d.mean()) if len(d) else np.nan, **S.tost_paired(d, delta, alpha)}
            for d in paired_diffs(rs, split or b["split"], b["primary_metric"]).values()]
    out = pd.DataFrame(rows, index=_pair_index(model_pairs(rs)))
    out.attrs["nota"] = f"margen ±{delta}; IC al {100 * (1 - 2 * alpha):.0f} %"
    return out


# ---------------------------------------------------------------------------
# 6.6 McNemar por corrida pareada
# ---------------------------------------------------------------------------


def _common_runs(rs: RunSet, models) -> list[tuple[int, int]]:
    keys = set.intersection(*(set(rs.runs[m]) for m in models))
    return sorted(keys, key=lambda k: (k[1], k[0]))


def _aligned_test(rs: RunSet, a: str, c: str, seed: int, fold: int):
    pa = rs.predictions(a, seed, fold).set_index("image_id")
    pc = rs.predictions(c, seed, fold).set_index("image_id").reindex(pa.index)
    if pc["label"].isna().any() or (pa["label"] != pc["label"]).any():
        raise ValueError(f"Las predicciones de test de {a} y {c} (s{seed}_f{fold}) no tienen las mismas imágenes")
    return pa, pc


@tolerant()
def mcnemar_by_run(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """McNemar de cada par de modelos en cada corrida pareada sobre las predicciones de test,
    con p ajustado entre los pares dentro de cada corrida."""
    m = _need_models(rs, 2, "McNemar")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    method, exact_below = b.get("p_adjust", "holm"), int(b.get("mcnemar_exact_below", 25))
    rows = []
    for seed, fold in _common_runs(rs, rs.models):
        block = []
        for a, c in model_pairs(rs):
            pa, pc = _aligned_test(rs, a, c, seed, fold)
            res = S.mcnemar_test(pa["label"] == pa["pred"], pc["label"] == pc["pred"], exact_below)
            block.append({"run": f"s{seed}_f{fold}", "modelo_a": a, "modelo_b": c, **res})
        padj = S.adjust_pvalues([r["p"] for r in block], method)
        for r, p in zip(block, padj):
            r[f"p_{method}"] = p
        rows += block
    if not rows:
        return Missing("no hay corridas con predicciones de test de todos los modelos.")
    return pd.DataFrame(rows).set_index(["run", "modelo_a", "modelo_b"])


@tolerant()
def mcnemar_summary(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Resumen entre corridas por par: corridas, b y c medios, corridas con p ajustado < alpha,
    mediana y rango de p ajustado."""
    b = bench_cfg(cfg)
    alpha, method = float(b["alpha"]), b.get("p_adjust", "holm")
    g = table.groupby(level=["modelo_a", "modelo_b"], sort=False)
    col = f"p_{method}"
    return pd.DataFrame({"corridas": g.size(), "b_medio": g["b"].mean(), "c_medio": g["c"].mean(),
                         "corridas_significativas": g[col].apply(lambda p: int((p < alpha).sum())),
                         f"mediana_{col}": g[col].median(), f"min_{col}": g[col].min(), f"max_{col}": g[col].max()})


# ---------------------------------------------------------------------------
# 6.7 DeLong por clase y AUC macro por bootstrap estratificado
# ---------------------------------------------------------------------------


@tolerant()
def delong_by_run(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """DeLong uno contra el resto por clase, para cada par y corrida pareada (test), con p
    ajustado por separado para cada par y corrida, sobre sus contrastes por clase (uno por clase),
    como declara book/07-diseno-experimental.md."""
    m = _need_models(rs, 2, "DeLong")
    if m is not None:
        return m
    method = bench_cfg(cfg).get("p_adjust", "holm")
    rows = []
    for seed, fold in _common_runs(rs, rs.models):
        for a, c in model_pairs(rs):
            pa, pc = _aligned_test(rs, a, c, seed, fold)
            block = []
            for cls in [col[2:] for col in pa.columns if col.startswith("p_")]:
                res = S.delong_test(pa["label"] == cls, pa[f"p_{cls}"], pc[f"p_{cls}"])
                block.append({"run": f"s{seed}_f{fold}", "modelo_a": a, "modelo_b": c, "clase": cls, **res})
            padj = S.adjust_pvalues([r["p"] for r in block], method)
            for r, p in zip(block, padj):
                r[f"p_{method}"] = p
            rows += block
    if not rows:
        return Missing("no hay corridas con predicciones de test de todos los modelos.")
    return pd.DataFrame(rows).set_index(["run", "modelo_a", "modelo_b", "clase"])


@tolerant()
def delong_summary(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Por par y clase: AUC medias, diferencia media entre corridas y corridas con p ajustado < alpha."""
    b = bench_cfg(cfg)
    alpha, col = float(b["alpha"]), f"p_{b.get('p_adjust', 'holm')}"
    g = table.groupby(level=["modelo_a", "modelo_b", "clase"], sort=False)
    return pd.DataFrame({"corridas": g.size(), "auc_a_media": g["auc_a"].mean(), "auc_b_media": g["auc_b"].mean(),
                         "diferencia_media": g["diferencia"].mean(), "diferencia_std": g["diferencia"].std(ddof=1),
                         "corridas_significativas": g[col].apply(lambda p: int((p < alpha).sum())),
                         f"mediana_{col}": g[col].median()})


@tolerant()
def auc_bootstrap_by_run(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """Diferencia de AUC-ROC macro (a - b) en test por corrida pareada, con IC por bootstrap
    estratificado por clase (benchmark.bootstrap.n_resamples_auc remuestreos)."""
    m = _need_models(rs, 2, "El bootstrap de AUC")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    boot, alpha = b["bootstrap"], float(b["alpha"])
    rows = []
    for seed, fold in _common_runs(rs, rs.models):
        for a, c in model_pairs(rs):
            pa, pc = _aligned_test(rs, a, c, seed, fold)
            classes = [col[2:] for col in pa.columns if col.startswith("p_")]
            y = pa["label"].map({k: i for i, k in enumerate(classes)}).to_numpy()
            cols = [f"p_{k}" for k in classes]
            res = S.stratified_bootstrap_auc_diff(y, pa[cols].to_numpy(), pc[cols].to_numpy(),
                                                  int(boot["n_resamples_auc"]), int(boot["seed"]) + seed * 100 + fold,
                                                  alpha)
            rows.append({"run": f"s{seed}_f{fold}", "modelo_a": a, "modelo_b": c, **res})
    if not rows:
        return Missing("no hay corridas con predicciones de test de todos los modelos.")
    return pd.DataFrame(rows).set_index(["run", "modelo_a", "modelo_b"])


@tolerant()
def auc_bootstrap_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Por par: diferencia media de AUC macro entre corridas, su std y corridas cuyo IC excluye 0."""
    g = table.groupby(level=["modelo_a", "modelo_b"], sort=False)
    excl = (table["ic_bajo"] > 0) | (table["ic_alto"] < 0)
    return pd.DataFrame({"corridas": g.size(), "diferencia_media": g["diferencia_auc_macro"].mean(),
                         "diferencia_std": g["diferencia_auc_macro"].std(ddof=1),
                         "corridas_ic_excluye_0": excl.groupby(level=["modelo_a", "modelo_b"], sort=False).sum()})


# ---------------------------------------------------------------------------
# 6.8 Descomposición de la varianza
# ---------------------------------------------------------------------------


@tolerant()
def variance_table(rs: RunSet, cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Suma de cuadrados de la métrica primaria atribuible al modelo, a la corrida (semilla y
    pliegue, separadas si forman un producto completo) y al residual, con las corridas que tienen
    todos los modelos."""
    m = _need_models(rs, 2, "La descomposición de la varianza")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    w = wide(rs, split or b["split"], b["primary_metric"])
    if len(w) < 2:
        return Missing("hay menos de dos corridas pareadas con todos los modelos.")
    long = w.reset_index().melt(id_vars=["seed", "val_fold"], var_name="model", value_name="y")
    return S.variance_decomposition(long, "y")


# ---------------------------------------------------------------------------
# 7. Robustez comparada
# ---------------------------------------------------------------------------


@tolerant()
def robustness_by_group(rs: RunSet, cfg: dict, by: str = "dataset") -> pd.DataFrame:
    """Métricas de test por categoría de `by` (media y std entre corridas) de todos los modelos en
    una tabla: filas = categorías, columnas = (modelo, métrica)."""
    from src import evaluation as E

    m = _need_models(rs, 1, "La robustez comparada")
    if m is not None:
        return m
    meta = E.load_metadata_or_missing(cfg)
    if is_missing(meta):
        return meta
    ecfg = cfg["evaluation"]
    frames = {}
    for model in rs.models:
        preds = E.load_predictions([Path(d) for d in rs.runs[model].values()], "test")
        t = E.subgroup_metrics(preds, meta, by, ecfg["subgroup_metrics"], int(ecfg["subgroup_min_n"]))
        if not is_missing(t):
            frames[model] = t
    if not frames:
        return Missing("no hay predicciones de test.")
    out = pd.concat(frames, axis=1)
    return out


@tolerant()
def pairwise_adjusted_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Columnas de pairwise_tests con los p-valores sin ajustar y ajustados de la t pareada y de
    Wilcoxon, la prueba principal y si es significativa."""
    cols = [c for c in pairwise.columns if c.startswith("p_")] + ["prueba_principal", "significativa"]
    return pairwise[["corridas_pareadas", "diferencia_media", *cols]]


# ---------------------------------------------------------------------------
# 3. Bootstrap por lesiones sobre el conjunto de prueba
# ---------------------------------------------------------------------------


def _test_correct(rs: RunSet, model: str, keys, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """(imágenes con label, matriz de aciertos corridas x imágenes) de `model` en las corridas `keys`."""
    base, rows = None, []
    for seed, fold in keys:
        p = rs.predictions(model, seed, fold, split).set_index("image_id").sort_index()
        if base is None:
            base = p[["label"]]
        elif not p.index.equals(base.index):
            raise ValueError(f"Las predicciones de {split} de {model} (s{seed}_f{fold}) no tienen las mismas imágenes")
        rows.append((p["label"] == p["pred"]).to_numpy())
    return base, np.vstack(rows)


@tolerant()
def lesion_bootstrap(rs: RunSet, cfg: dict, split: str | None = None) -> dict:
    """IC 95 % (1 - alpha) por bootstrap percentil sobre `split` (test) remuestreando lesion_id:
    en cada remuestreo se sortean lesiones con reemplazo, se recalcula la exactitud balanceada de
    cada corrida definitiva sobre las imágenes de las lesiones sorteadas y se promedia entre
    corridas. Parámetros en benchmark.lesion_bootstrap. Todos los modelos y pares usan los
    mismos remuestreos.

    Devuelve {"Exactitud balanceada media por modelo": tabla, "Diferencias entre pares": tabla
    o Missing}. Las diferencias (a - b) usan las corridas que tienen los dos modelos.
    """
    from src import data as D

    m = _need_models(rs, 1, "El bootstrap por lesiones")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    split = split or b["split"]
    alpha = float(b["alpha"])
    lb = b["lesion_bootstrap"]
    n_res, seed = int(lb["n_resamples"]), int(lb["seed"])

    keys = {model: sorted(rs.runs[model]) for model in rs.models}
    correct, images = {}, None
    for model in rs.models:
        imgs, mat = _test_correct(rs, model, keys[model], split)
        if images is None:
            images = imgs
        elif not imgs.index.equals(images.index):
            raise ValueError(f"Las predicciones de {split} de {model} no tienen las mismas imágenes que las demás")
        correct[model] = mat
    for a, c in model_pairs(rs):
        common = sorted(set(keys[a]) & set(keys[c]))
        if common:
            for model in (a, c):
                idx = [keys[model].index(k) for k in common]
                correct[(a, c, model)] = correct[model][idx]

    splits = D.load_splits(cfg["paths"]["splits_csv"]).set_index("image_id")
    lesions = splits["lesion_id"].reindex(images.index)
    if lesions.isna().any():
        return Missing(f"{int(lesions.isna().sum())} imágenes de {split} no están en {cfg['paths']['splits_csv']}.")
    classes = sorted(images["label"].unique())
    y = images["label"].map({k: i for i, k in enumerate(classes)}).to_numpy()
    res = S.cluster_bootstrap_balanced_accuracy(y, lesions.to_numpy(), correct, n_res, seed)

    def interval(samples):
        lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi), float(samples.std(ddof=1))

    method = f"bootstrap percentil por lesion_id ({n_res} remuestreos, semilla {seed})"
    rows = []
    for model in rs.models:
        lo, hi, se = interval(res[model]["remuestreos"])
        rows.append({"modelo": model, "corridas": len(keys[model]), "media": res[model]["observado"],
                     "ic_bajo": lo, "ic_alto": hi, "ee_bootstrap": se})
    means = pd.DataFrame(rows).set_index("modelo")
    means["lesiones"] = lesions.nunique()
    means["imagenes"] = len(images)
    means["metodo"] = method

    rows, pairs = [], []
    for a, c in model_pairs(rs):
        if (a, c, a) not in res:
            continue
        d = res[(a, c, a)]["remuestreos"] - res[(a, c, c)]["remuestreos"]
        lo, hi, se = interval(d)
        pairs.append((a, c))
        rows.append({"corridas_pareadas": len(correct[(a, c, a)]),
                     "diferencia": res[(a, c, a)]["observado"] - res[(a, c, c)]["observado"],
                     "ic_bajo": lo, "ic_alto": hi, "ee_bootstrap": se, "metodo": method})
    if rows:
        diffs = pd.DataFrame(rows, index=_pair_index(pairs))
    else:
        diffs = Missing("Las diferencias por bootstrap de lesiones: se necesitan corridas pareadas de al menos "
                        f"2 modelos (modelos con corridas: {', '.join(rs.models)}).")
    return {"Exactitud balanceada media por modelo": means, "Diferencias entre pares (a - b)": diffs}
