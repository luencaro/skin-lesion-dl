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
from itertools import combinations, pairwise
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


def load_runs(cfg: dict, label: str, pairs=None, models=None) -> RunSet:
    """Corridas terminadas de `models` (por defecto benchmark.models) con etiqueta `label` y pares
    (semilla, pliegue) `pairs` (por defecto, los de final si label es la de final y los del piloto
    si es "pilot")."""
    from src.final import final_label, final_pairs
    from src.train import run_dir_for

    models = list(models or bench_cfg(cfg)["models"])
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
                          "n_train": info["n_images"]["train"], "n_val": info["n_images"]["val"],
                          "micro_batch_size": info.get("micro_batch_size", info.get("inference_batch_size")),
                          "gradient_accumulation_steps": info.get("gradient_accumulation_steps", 1)})
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
        "micro_lote": g["micro_batch_size"].first(),
        "pasos_de_acumulacion": g["gradient_accumulation_steps"].first(),
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
    txt["micro-lote x acumulación"] = [f"{int(a)} x {int(s)}" if pd.notna(a) else "" for a, s in
                                       zip(cost["micro_lote"], cost["pasos_de_acumulacion"])]
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
def mcnemar_by_run(rs: RunSet, cfg: dict, pairs=None) -> pd.DataFrame:
    """McNemar de cada par de modelos (`pairs`, por defecto todos) en cada corrida pareada sobre
    las predicciones de test, con p ajustado entre los pares dentro de cada corrida."""
    m = _need_models(rs, 2, "McNemar")
    if m is not None:
        return m
    b = bench_cfg(cfg)
    method, exact_below = b.get("p_adjust", "holm"), int(b.get("mcnemar_exact_below", 25))
    pairs = list(pairs or model_pairs(rs))
    rows = []
    for seed, fold in _common_runs(rs, sorted({x for p in pairs for x in p})):
        block = []
        for a, c in pairs:
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
def delong_by_run(rs: RunSet, cfg: dict, pairs=None) -> pd.DataFrame:
    """DeLong uno contra el resto por clase, para cada par (`pairs`, por defecto todos) y corrida
    pareada (test), con p ajustado por separado para cada par y corrida, sobre sus contrastes por
    clase (uno por clase), como declara book/07-diseno-experimental.md."""
    m = _need_models(rs, 2, "DeLong")
    if m is not None:
        return m
    method = bench_cfg(cfg).get("p_adjust", "holm")
    pairs = list(pairs or model_pairs(rs))
    rows = []
    for seed, fold in _common_runs(rs, sorted({x for p in pairs for x in p})):
        for a, c in pairs:
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


# ===========================================================================
# Entregable 2: benchmark de 8 modelos profundos y 2 baselines (book/20-benchmark-e2.ipynb)
# ===========================================================================
# Preguntas y reglas de decisión: preregistro_e2.md y book/12-preguntas-y-datos.md. Parámetros:
# secciones e2, e2_stats, replication y benchmark de base.yaml. Las corridas pareadas son las de
# final (misma semilla y mismo pliegue); el bootstrap por lesión remuestrea lesion_id del
# conjunto de prueba con la misma semilla para todos los modelos, de modo que las diferencias
# entre modelos quedan pareadas también en los remuestreos.

ROLES = {"base_models": "modelo base (E1)", "sota_models": "estado del arte", "baselines": "baseline"}


def e2_groups(cfg: dict) -> dict:
    """Listas de modelos del Entregable 2: base, estado del arte, baselines, profundos y todos."""
    e2 = cfg["e2"]
    base, sota, bl = list(e2["base_models"]), list(e2["sota_models"]), list(e2["baselines"])
    return {"base_models": base, "sota_models": sota, "baselines": bl, "deep_models": base + sota,
            "all_models": base + sota + bl, "cnn_models": list(e2.get("cnn_models", [])),
            "transformer_models": list(e2.get("transformer_models", []))}


def role_of(cfg: dict, model: str) -> str:
    groups = e2_groups(cfg)
    return next((ROLES[g] for g in ROLES if model in groups[g]), "otro")


def load_runs_e2(cfg: dict) -> RunSet:
    """Corridas definitivas de todos los modelos y baselines del Entregable 2, con las métricas
    de probabilidad (brier, ece_adaptativo, log_loss) de test añadidas a `metrics`."""
    from src import calibration as C
    from src import data as D
    from src.final import final_label

    rs = load_runs(cfg, final_label(cfg), models=e2_groups(cfg)["all_models"])
    extra = C.probability_metrics_by_run(rs, cfg, rs.models, D.class_names(cfg), "test")
    if not is_missing(extra):
        rs.metrics = pd.concat([rs.metrics, extra], ignore_index=True)
    return rs


def run_fingerprint(rs: RunSet) -> str:
    """Huella de las corridas cargadas (modelo, semilla, pliegue y fecha de metrics.json); cambia
    si se añade o se repite una corrida, lo que invalida las tablas en cache."""
    import hashlib

    parts = []
    for m in sorted(rs.runs):
        for key, d in sorted(rs.runs[m].items()):
            parts.append(f"{m}|{key}|{(Path(d) / 'metrics.json').stat().st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def cached(cfg: dict, rs: RunSet, name: str, fn, rerun: bool = False):
    """Resultado de `fn()` guardado en outputs/benchmark_e2/<name>.pkl con la huella de las
    corridas; se reutiliza si la huella coincide y `rerun` es False. Los Missing no se guardan."""
    import pickle

    path = cfg["paths"]["benchmark_e2_dir"] / f"{name}.pkl"
    fp = run_fingerprint(rs)
    if path.exists() and not rerun:
        with open(path, "rb") as f:
            stored = pickle.load(f)
        if stored.get("fingerprint") == fp:
            return stored["value"]
    value = fn()
    if not is_missing(value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"fingerprint": fp, "value": value}, f)
    return value


def cached_only(cfg: dict, rs: RunSet, name: str, message: str):
    """Resultado en cache de `name` si existe con la huella actual; si no, Missing(message)."""
    import pickle

    path = cfg["paths"]["benchmark_e2_dir"] / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            stored = pickle.load(f)
        if stored.get("fingerprint") == run_fingerprint(rs):
            return stored["value"]
        return Missing(f"{message} (la cache de {name} es de otras corridas)")
    return Missing(message)


@tolerant()
def availability_e2(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """Corridas previstas y terminadas por modelo con su rol, y corridas con test."""
    out = availability_table(rs)
    out.insert(0, "rol", [role_of(cfg, m) for m in out.index])
    out["con_test"] = [sum((Path(d) / "probabilities_test.csv").exists() for d in rs.runs.get(m, {}).values())
                       for m in out.index]
    return out


@tolerant()
def best_base(rs: RunSet, cfg: dict) -> str | Missing:
    """Mejor modelo base: e2.best_base si está fijado; si no, el modelo base con mayor media de la
    métrica primaria en benchmark.split."""
    fixed = cfg["e2"].get("best_base")
    base = [m for m in e2_groups(cfg)["base_models"] if m in rs.models]
    if fixed:
        return fixed if fixed in rs.models else Missing(f"e2.best_base = {fixed} no tiene corridas definitivas.")
    if not base:
        return Missing("no hay corridas definitivas de los modelos base.")
    b = bench_cfg(cfg)
    mt = rs.metrics
    sub = mt[(mt["split"] == b["split"]) & (mt["metric"] == b["primary_metric"]) & mt["model"].isin(base)]
    if sub.empty:
        return Missing("los modelos base no tienen la métrica primaria en test.")
    return str(sub.groupby("model")["value"].mean().idxmax())


@tolerant()
def divergent_runs(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """Corridas de los modelos profundos que cumplen la regla de exclusión del preregistro: pérdida
    NaN en history.csv o exactitud balanceada de validación (pesos restaurados) menor que
    e2.divergence. Los baselines no se entrenan por épocas y no se evalúan."""
    thr = float(cfg["e2"]["divergence"]["min_val_balanced_accuracy"])
    rows = []
    for m in [m for m in rs.models if m in e2_groups(cfg)["deep_models"]]:
        for (s, f), d in rs.runs[m].items():
            val = rs.metrics.query("model == @m and seed == @s and val_fold == @f and split == 'val' "
                                   "and metric == 'balanced_accuracy'")["value"]
            nan_loss = False
            h = Path(d) / "history.csv"
            if h.exists():
                hist = pd.read_csv(h)
                nan_loss = bool(hist["loss"].isna().any()) if "loss" in hist else False
            ba = float(val.iloc[0]) if len(val) else np.nan
            rows.append({"modelo": m, "run": f"s{s}_f{f}", "val_balanced_accuracy": ba, "perdida_nan": nan_loss,
                         "divergente": nan_loss or (np.isfinite(ba) and ba < thr)})
    t = pd.DataFrame(rows)
    if t.empty:
        return Missing("no hay corridas definitivas.")
    out = t.groupby("modelo", sort=False).agg(corridas=("run", "size"), divergentes=("divergente", "sum"),
                                              min_val_balanced_accuracy=("val_balanced_accuracy", "min"))
    out["corridas_divergentes"] = t[t["divergente"]].groupby("modelo")["run"].agg(", ".join).reindex(out.index)
    out.attrs["regla"] = f"pérdida NaN o exactitud balanceada de validación < {thr}"
    return out


# ---------------------------------------------------------------------------
# E2.1 Datos de test y bootstrap por lesión
# ---------------------------------------------------------------------------


@dataclass
class TestData:
    """Predicciones de test alineadas (mismas imágenes en el mismo orden) de todos los modelos."""

    image_ids: np.ndarray
    y: np.ndarray                     # (N,) etiqueta entera
    lesions: np.ndarray               # (N,) lesion_id
    classes: list[str]
    keys: dict                        # {modelo: [(semilla, pliegue), ...]}
    proba: dict                       # {modelo: (R, N, K)}
    meta: pd.DataFrame | None         # metadatos alineados con image_ids (o None)

    def correct(self, model: str) -> np.ndarray:
        return self.proba[model].argmax(axis=2) == self.y[None, :]

    def pred(self, model: str) -> np.ndarray:
        return self.proba[model].argmax(axis=2)

    def common(self, a: str, b: str) -> tuple[list[int], list[int]]:
        """Índices de las corridas comunes de a y b en self.keys[a] y self.keys[b]."""
        common = [k for k in self.keys[a] if k in self.keys[b]]
        return [self.keys[a].index(k) for k in common], [self.keys[b].index(k) for k in common]


@tolerant()
def test_data(rs: RunSet, cfg: dict) -> TestData | Missing:
    """TestData de las corridas con probabilities_test.csv (en cache dentro de `rs`)."""
    from src import data as D

    if "__test_data__" in rs._preds:
        return rs._preds["__test_data__"]
    classes = D.class_names(cfg)
    keys, proba, ids, labels = {}, {}, None, None
    for m in rs.models:
        ks = sorted(k for k, d in rs.runs[m].items() if (Path(d) / "probabilities_test.csv").exists())
        if not ks:
            continue
        mats = []
        for k in ks:
            p = rs.predictions(m, *k, "test").sort_values("image_id")
            if ids is None:
                ids, labels = p["image_id"].to_numpy(), p["label"].to_numpy()
            elif not np.array_equal(ids, p["image_id"].to_numpy()):
                raise ValueError(f"Las predicciones de test de {m} s{k[0]}_f{k[1]} no tienen las mismas imágenes")
            mats.append(p[[f"p_{c}" for c in classes]].to_numpy(np.float64))
        keys[m], proba[m] = ks, np.stack(mats)
    if not keys:
        return Missing("ninguna corrida definitiva tiene probabilities_test.csv.")
    splits = D.load_splits(cfg["paths"]["splits_csv"]).set_index("image_id")
    lesions = splits["lesion_id"].reindex(ids)
    if lesions.isna().any():
        return Missing(f"{int(lesions.isna().sum())} imágenes de test no están en data/splits.csv.")
    from src.evaluation import load_metadata_or_missing

    meta = load_metadata_or_missing(cfg)
    if is_missing(meta):
        meta = None
    else:
        meta = meta.reindex(ids)
        meta["age_group"] = age_groups(meta["age"], cfg["e2_stats"]["age_bins"])
    td = TestData(image_ids=ids, y=pd.Series(labels).map({c: i for i, c in enumerate(classes)}).to_numpy(),
                  lesions=lesions.to_numpy(), classes=classes, keys=keys, proba=proba, meta=meta)
    rs._preds["__test_data__"] = td
    return td


def age_groups(age: pd.Series, bins) -> pd.Series:
    """Intervalos de edad [a, b) como texto; "desconocida" si falta."""
    labels = [f"[{int(a)}, {int(b)})" for a, b in pairwise(bins)]
    out = pd.cut(age, bins=bins, right=False, labels=labels).astype(object)
    return out.where(age.notna(), "desconocida")


def lesion_ba_resamples(td: TestData, cfg: dict, masks: dict | None = None, n_resamples: int | None = None,
                        models=None) -> dict:
    """Bootstrap por lesión de la exactitud balanceada de cada corrida.

    Devuelve {(modelo, máscara): {"observado": (R,), "remuestreos": (B, R)}}; la máscara None son
    todas las imágenes. Los remuestreos (benchmark.lesion_bootstrap: semilla y número) son los
    mismos para todos los modelos y máscaras."""
    lb = bench_cfg(cfg)["lesion_bootstrap"]
    n = int(n_resamples or lb["n_resamples"])
    masks = {None: None, **(masks or {})}
    models = list(models or td.keys)
    corr = {m: td.correct(m) for m in models}
    out = {(m, k): {"observado": S.weighted_balanced_accuracy(np.ones((1, len(td.y))), td.y, corr[m], mk)[0],
                    "remuestreos": []} for m in models for k, mk in masks.items()}
    for w in S.cluster_weight_batches(td.lesions, n, int(lb["seed"])):
        for m in models:
            for k, mk in masks.items():
                out[(m, k)]["remuestreos"].append(S.weighted_balanced_accuracy(w, td.y, corr[m], mk))
    for v in out.values():
        v["remuestreos"] = np.vstack(v["remuestreos"])
    return out


def _boot(rs: RunSet, cfg: dict, td: TestData) -> dict:
    """lesion_ba_resamples sin máscaras con benchmark.lesion_bootstrap (en cache dentro de `rs`)."""
    if "__ba_boot__" not in rs._preds:
        rs._preds["__ba_boot__"] = lesion_ba_resamples(td, cfg)
    return rs._preds["__ba_boot__"]


# ---------------------------------------------------------------------------
# E2.2 Tabla principal
# ---------------------------------------------------------------------------


@tolerant()
def main_table(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """Tabla principal (formato largo): por modelo y métrica de e2_stats.table_metrics en test,
    corridas, media, desviación estándar e IC 95 % de la media por t de Student; para la métrica
    primaria, además, el IC 95 % por bootstrap por lesión (media entre corridas)."""
    b = bench_cfg(cfg)
    alpha = float(b["alpha"])
    split, primary = b["split"], b["primary_metric"]
    td = test_data(rs, cfg)
    boot = _boot(rs, cfg, td) if not is_missing(td) else {}
    rows = []
    for m in rs.models:
        for k in cfg["e2_stats"]["table_metrics"]:
            x = rs.metrics.query("model == @m and split == @split and metric == @k")["value"].to_numpy()
            if not len(x):
                continue
            lo, hi = S.t_confidence_interval(x, alpha)
            row = {"modelo": m, "rol": role_of(cfg, m), "metrica": k, "corridas": len(x), "media": float(x.mean()),
                   "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan, "ic_t_bajo": lo, "ic_t_alto": hi,
                   "ic_lesion_bajo": np.nan, "ic_lesion_alto": np.nan}
            if k == primary and (m, None) in boot:
                row["ic_lesion_bajo"], row["ic_lesion_alto"] = S.percentile_interval(
                    boot[(m, None)]["remuestreos"].mean(axis=1), alpha)
            rows.append(row)
    if not rows:
        return Missing(f"no hay métricas de {split}.")
    return pd.DataFrame(rows).set_index(["modelo", "metrica"])


def main_table_formatted(table, cfg: dict, cost=None, digits: int = 3) -> pd.DataFrame | Missing:
    """Tabla principal compacta: una fila por modelo con «media ± std» de cada métrica, el IC 95 %
    por lesión de la métrica primaria y, si se pasa `cost` (src.cost.cost_table), parámetros,
    GFLOPs y horas de entrenamiento por corrida."""
    if is_missing(table):
        return table
    primary = bench_cfg(cfg)["primary_metric"]
    fmt = table.apply(lambda r: f"{r['media']:.{digits}f} ± {r['std']:.{digits}f}", axis=1).unstack("metrica")
    fmt = fmt.reindex(columns=[k for k in cfg["e2_stats"]["table_metrics"] if k in fmt.columns])
    models = list(dict.fromkeys(table.index.get_level_values("modelo")))
    out = pd.DataFrame(index=models)
    out["rol"] = table.groupby(level="modelo")["rol"].first().reindex(models)
    out["corridas"] = table.groupby(level="modelo")["corridas"].max().reindex(models)
    out = out.join(fmt)
    if primary in table.index.get_level_values("metrica"):
        p = table.xs(primary, level="metrica")
        out[f"{primary} IC 95 % (lesión)"] = p.apply(
            lambda r: f"[{r['ic_lesion_bajo']:.{digits}f}; {r['ic_lesion_alto']:.{digits}f}]", axis=1)
    if isinstance(cost, pd.DataFrame):
        for col, name, f in (("params_total", "parámetros", lambda v: f"{v / 1e6:.1f} M"),
                             ("gflops", "GFLOPs", lambda v: f"{v:.2f}"),
                             ("entrenamiento_h_por_corrida", "horas por corrida", lambda v: f"{v:.2f}")):
            if col in cost.columns:
                out[name] = cost[col].reindex(out.index).map(lambda v, f=f: f(v) if pd.notna(v) else "")
    return out.rename_axis("modelo")


@tolerant()
def per_class_e2(rs: RunSet, cfg: dict, split: str | None = None, stat: str = "recall") -> pd.DataFrame:
    """Media entre corridas de `stat` (precision, recall o f1) por modelo (filas) y clase (columnas)."""
    split = split or bench_cfg(cfg)["split"]
    rows = []
    for m in rs.models:
        for d in rs.runs[m].values():
            met = load_run_metrics(d)
            if split in met:
                for c, v in met[split]["per_class"].items():
                    rows.append({"modelo": m, "clase": c, stat: v[stat]})
    if not rows:
        return Missing(f"no hay métricas por clase de {split}.")
    t = pd.DataFrame(rows).groupby(["modelo", "clase"], sort=False)[stat].mean().unstack("clase")
    return t.reindex(rs.models)


# ---------------------------------------------------------------------------
# E2.3 Contrastes confirmatorios (RQ1 y RQ2)
# ---------------------------------------------------------------------------


def _family_spec(rs: RunSet, cfg: dict, family: str) -> tuple[list[str], str] | Missing:
    spec = cfg["e2_stats"][family]
    ref = spec["reference"]
    if ref == "best_base":
        ref = best_base(rs, cfg)
        if is_missing(ref):
            return ref
    if ref not in rs.models:
        return Missing(f"la referencia {ref} de {family} no tiene corridas definitivas.")
    models = [m for m in e2_groups(cfg)[spec["models"]] if m != ref]
    have = [m for m in models if m in rs.models]
    if not have:
        return Missing(f"{family}: ningún modelo de {spec['models']} tiene corridas definitivas "
                       f"(previstos: {', '.join(models)}).")
    return have, ref


def _ratio(rs: RunSet, models) -> float:
    info = rs.info[rs.info["model"].isin(models)]
    return float((info["n_val"] / info["n_train"]).mean())


def decide(row: dict, cfg: dict) -> tuple[str, str]:
    """(decisión, prueba usada) según la regla del preregistro: si Shapiro-Wilk no rechaza la
    normalidad, la significación es la de e2_stats.decision_test_normal; si la rechaza, deben ser
    significativas todas las de decision_tests_non_normal. Después: superior y relevante si es
    significativa y la diferencia es al menos delta; superior sin relevancia si es significativa y
    menor; equivalente si el TOST (ajustado) es significativo; no concluyente en otro caso."""
    b, st = bench_cfg(cfg), cfg["e2_stats"]
    alpha, delta = float(b["alpha"]), float(b["delta"])
    col = {"nadeau_bengio": "p_nb_holm", "wilcoxon": "p_wilcoxon_holm", "bootstrap": "p_boot_holm"}
    normal = not (np.isfinite(row["shapiro_p"]) and row["shapiro_p"] < alpha)
    tests = [st["decision_test_normal"]] if normal else list(st["decision_tests_non_normal"])
    ps = [row[col[t]] for t in tests]
    sig = all(np.isfinite(p) and p < alpha for p in ps)
    used = " y ".join({"nadeau_bengio": "t de Nadeau-Bengio", "wilcoxon": "Wilcoxon",
                       "bootstrap": "bootstrap por lesión"}[t] for t in tests)
    if sig and row["diferencia_media"] >= delta:
        return "superior y relevante", used
    if sig:
        return "superior sin relevancia", used
    if np.isfinite(row["p_tost_holm"]) and row["p_tost_holm"] < alpha:
        return "equivalente", used
    return "no concluyente", used


@tolerant()
def contrast_family(rs: RunSet, cfg: dict, family: str) -> pd.DataFrame:
    """Contrastes de una familia confirmatoria (e2_stats.rq1 o rq2): cada modelo frente a la
    referencia, con d = modelo - referencia por corrida pareada (métrica primaria en test).

    Por contraste: Shapiro-Wilk de d; t pareada con la corrección de Nadeau y Bengio (unilateral,
    e2_stats.alternative) y su IC 95 % bilateral; Wilcoxon exacto unilateral; bootstrap pareado
    por lesión (benchmark.lesion_bootstrap) con IC 95 % y p unilateral; TOST con ±delta (error
    estándar de e2_stats.tost_se); Holm dentro de la familia para cada prueba; d de Cohen
    pareada, delta de Cliff, A12 de Vargha y Delaney con IC bootstrap sobre corridas y
    probabilidad de superioridad pareada; y la decisión de `decide`."""
    spec = _family_spec(rs, cfg, family)
    if is_missing(spec):
        return spec
    models, ref = spec
    b, st = bench_cfg(cfg), cfg["e2_stats"]
    alpha, delta, alt = float(b["alpha"]), float(b["delta"]), st["alternative"]
    split, metric, zm = b["split"], b["primary_metric"], b.get("wilcoxon_zero_method", "wilcox")
    td = test_data(rs, cfg)
    boot = _boot(rs, cfg, td) if not is_missing(td) else {}
    rows = []
    for m in models:
        w = wide(rs, split, metric, [m, ref])
        d = (w[m] - w[ref]).to_numpy()
        ratio = _ratio(rs, [m, ref])
        nb = S.nadeau_bengio_t(d, ratio, alt)
        half = (S.ss.t.ppf(1 - alpha / 2, max(len(d) - 1, 1)) * nb["ee_corregido"]) if np.isfinite(nb.get("ee_corregido", np.nan)) else np.nan
        wx = S.wilcoxon_exact(d, zm, alt)
        tost = S.tost_paired(d, delta, alpha, ratio if st.get("tost_se", "nadeau_bengio") == "nadeau_bengio" else None)
        row = {"familia": family, "modelo": m, "referencia": ref, "corridas_pareadas": len(d),
               "diferencia_media": float(np.mean(d)) if len(d) else np.nan,
               "ic_nb_bajo": float(np.mean(d) - half) if len(d) else np.nan,
               "ic_nb_alto": float(np.mean(d) + half) if len(d) else np.nan,
               "shapiro_p": S.shapiro_test(d)["p"], "t_nb": nb["t"], "p_nb": nb["p"],
               "W_mas": wx["W_mas"], "p_wilcoxon": wx["p"]}
        if (m, None) in boot and (ref, None) in boot:
            ia, ir = td.common(m, ref)
            diff = boot[(m, None)]["remuestreos"][:, ia].mean(axis=1) - boot[(ref, None)]["remuestreos"][:, ir].mean(axis=1)
            lo, hi = S.percentile_interval(diff, alpha)
            row.update({"diferencia_lesion": float(boot[(m, None)]["observado"][ia].mean()
                                                   - boot[(ref, None)]["observado"][ir].mean()),
                        "ic_lesion_bajo": lo, "ic_lesion_alto": hi, "p_boot": S.bootstrap_p(diff, 0.0, alt)})
        else:
            row.update({"diferencia_lesion": np.nan, "ic_lesion_bajo": np.nan, "ic_lesion_alto": np.nan,
                        "p_boot": np.nan})
        a12_lo, a12_hi = S.a12_bootstrap_ci(w[m].to_numpy(), w[ref].to_numpy(), int(st["a12_bootstrap"]),
                                            int(b["bootstrap"]["seed"]), alpha)
        row.update({"p_tost": tost["p_tost"], "ic_tost_bajo": tost["ic_bajo"], "ic_tost_alto": tost["ic_alto"],
                    "d_cohen": S.cohens_dz(d), "delta_cliff": S.cliffs_delta(w[m], w[ref]),
                    "A12": S.vargha_delaney_a12(w[m], w[ref]), "A12_ic_bajo": a12_lo, "A12_ic_alto": a12_hi,
                    "prob_superioridad": S.probability_of_superiority_paired(d)})
        rows.append(row)
    out = pd.DataFrame(rows)
    method = b.get("p_adjust", "holm")
    for p in ("p_nb", "p_wilcoxon", "p_boot", "p_tost"):
        out[f"{p}_holm"] = S.adjust_pvalues(out[p], method)
    dec = [decide(r, cfg) for r in out.to_dict("records")]
    out["prueba_de_decision"] = [u for _, u in dec]
    out["decision"] = [d_ for d_, _ in dec]
    out = out.set_index(["familia", "modelo"])
    out.attrs["nota"] = (f"d = modelo - {ref}; H1: d > 0; delta = {delta}; alpha = {alpha}; Holm entre los "
                         f"{len(out)} contrastes de la familia para cada prueba; razón n_val / n_train = "
                         f"{_ratio(rs, models + [ref]):.4f}")
    return out


A3_COLUMNS = ["referencia", "corridas_pareadas", "diferencia_media", "ic_nb_bajo", "ic_nb_alto", "shapiro_p",
              "p_nb_holm", "p_wilcoxon_holm", "p_boot_holm", "p_tost_holm", "d_cohen", "A12",
              "prueba_de_decision", "decision"]


@tolerant()
def contrasts_summary(*tables) -> pd.DataFrame:
    """Resumen de los contrastes (columnas de A3_COLUMNS) de una o varias familias."""
    frames = [t[[c for c in A3_COLUMNS if c in t.columns]] for t in tables if not is_missing(t)]
    if not frames:
        return Missing("no hay contrastes confirmatorios.")
    return pd.concat(frames)


def family_pairs(rs: RunSet, cfg: dict, families=("rq1", "rq2")) -> list[tuple[str, str]]:
    """Pares (modelo, referencia) de las familias confirmatorias con corridas."""
    pairs = []
    for fam in families:
        spec = _family_spec(rs, cfg, fam)
        if not is_missing(spec):
            pairs += [(m, spec[1]) for m in spec[0] if (m, spec[1]) not in pairs]
    return pairs


@tolerant()
def mcnemar_contrasts(rs: RunSet, cfg: dict, family: str) -> pd.DataFrame:
    """McNemar por corrida pareada para los contrastes de `family` (Holm entre ellos por corrida)."""
    pairs = family_pairs(rs, cfg, (family,))
    if not pairs:
        return Missing(f"{family}: no hay contrastes con corridas.")
    return mcnemar_by_run(rs, cfg, pairs)


@tolerant()
def delong_contrasts(rs: RunSet, cfg: dict, family: str) -> pd.DataFrame:
    """DeLong uno contra el resto por clase para los contrastes de `family` (Holm sobre las 7 clases por par y corrida)."""
    pairs = family_pairs(rs, cfg, (family,))
    if not pairs:
        return Missing(f"{family}: no hay contrastes con corridas.")
    return delong_by_run(rs, cfg, pairs)


# ---------------------------------------------------------------------------
# E2.4 Comparación global
# ---------------------------------------------------------------------------


def global_models(rs: RunSet, cfg: dict) -> list[str]:
    sel = cfg["e2_stats"].get("global_models", "all")
    return list(rs.models) if sel == "all" else [m for m in e2_groups(cfg).get(sel, sel) if m in rs.models]


@tolerant()
def global_wide(rs: RunSet, cfg: dict) -> pd.DataFrame:
    models = global_models(rs, cfg)
    if len(models) < 3:
        return Missing(f"la comparación global necesita al menos 3 modelos con corridas (hay {len(models)}).")
    b = bench_cfg(cfg)
    w = wide(rs, b["split"], b["primary_metric"], models)
    if len(w) < 2:
        return Missing("hay menos de dos corridas pareadas con todos los modelos.")
    return w


@tolerant()
def global_tests(w: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Friedman (con W de Kendall) y su corrección de Iman y Davenport sobre las corridas con todos los modelos."""
    n, k = w.shape
    fr = S.friedman_test(w)
    idv = S.iman_davenport(fr["chi2"], n, k)
    return pd.DataFrame({
        "Friedman": {"estadistico": fr["chi2"], "gl": f"{k - 1}", "p": fr["p"], "W_kendall": fr["W_kendall"],
                     "corridas": n, "modelos": k},
        "Iman-Davenport": {"estadistico": idv["F"], "gl": f"{idv['gl1']}, {idv['gl2']}", "p": idv["p"],
                           "W_kendall": np.nan, "corridas": n, "modelos": k},
    }).T


@tolerant()
def global_posthoc(w: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Post-hoc de todos los pares: Nemenyi y z de rangos con ajuste de Holm."""
    nem = S.nemenyi_test(w).set_index(["modelo_a", "modelo_b"])
    holm = S.friedman_posthoc_holm(w).set_index(["modelo_a", "modelo_b"])
    out = holm[["rango_a", "rango_b", "z", "p", "p_holm"]].join(nem[["q", "p"]].rename(columns={"p": "p_nemenyi"}))
    out["significativa_holm"] = out["p_holm"] < float(bench_cfg(cfg)["alpha"])
    return out


@tolerant()
def plot_cd_diagram(w: pd.DataFrame, cfg: dict):
    """Diagrama de diferencia crítica (Demšar, 2006): rango medio de cada modelo (1 = mejor) y
    barras que unen los modelos cuya diferencia de rangos es menor que la CD de Nemenyi."""
    n, k = w.shape
    alpha = float(bench_cfg(cfg)["alpha"])
    ranks = (k + 1 - S.mean_ranks(w)).sort_values()
    cd = S.critical_difference(k, n, alpha)
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.32 * k))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-(k / 2 + 3), 2)
    ax.hlines(0, 1, k, color="black")
    for r in range(1, k + 1):
        ax.vlines(r, -0.12, 0.12, color="black")
        ax.text(r, 0.3, str(r), ha="center", fontsize=8)
    ax.hlines(1.3, 1, 1 + cd, color="tab:red", lw=2)
    ax.text(1 + cd / 2, 1.5, f"CD = {cd:.2f}", ha="center", fontsize=8, color="tab:red")
    half = int(np.ceil(k / 2))
    for i, (m, r) in enumerate(ranks.items()):
        left = i < half
        y = -(i + 1) * 0.55 if left else -(k - i) * 0.55
        x_text = 0.6 if left else k + 0.4
        ax.plot([r, r, x_text], [0, y, y], color="gray", lw=0.8)
        ax.text(x_text, y, f"{m} ({r:.2f})", ha="left" if not left else "right", va="center", fontsize=8,
                transform=ax.transData)
    vals = ranks.to_numpy()
    cliques = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if vals[jj] - vals[i] < cd)
        if j > i and not any(a <= i and j <= b_ for a, b_ in cliques):
            cliques.append((i, j))
    for level, (i, j) in enumerate(cliques, start=1):
        ax.hlines(-0.25 - 0.18 * level, vals[i] - 0.05, vals[j] + 0.05, color="black", lw=3)
    ax.axis("off")
    ax.set_title(f"Diagrama de diferencia crítica (Nemenyi, alpha = {alpha}, {n} corridas)", fontsize=10)
    fig.tight_layout()
    return fig


@tolerant()
def cochran_by_run(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """Q de Cochran de los aciertos por imagen de test de todos los modelos globales, por corrida pareada."""
    td = test_data(rs, cfg)
    if is_missing(td):
        return td
    models = [m for m in global_models(rs, cfg) if m in td.keys]
    if len(models) < 2:
        return Missing("Cochran necesita al menos 2 modelos con predicciones de test.")
    keys = sorted(set.intersection(*(set(td.keys[m]) for m in models)), key=lambda k: (k[1], k[0]))
    rows = []
    for key in keys:
        hits = np.column_stack([td.correct(m)[td.keys[m].index(key)] for m in models])
        rows.append({"run": f"s{key[0]}_f{key[1]}", "modelos": len(models), **S.cochran_q(hits)})
    if not rows:
        return Missing("no hay corridas pareadas con todos los modelos.")
    return pd.DataFrame(rows).set_index("run")


@tolerant()
def cochran_summary(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    alpha = float(bench_cfg(cfg)["alpha"])
    return pd.Series({"corridas": len(table), "Q_mediana": table["Q"].median(), "p_mediana": table["p"].median(),
                      "p_maximo": table["p"].max(), "corridas_p_menor_alpha": int((table["p"] < alpha).sum())},
                     name="valor").to_frame()


# ---------------------------------------------------------------------------
# E2.5 Tabla A: bondad de ajuste
# ---------------------------------------------------------------------------


@tolerant()
def table_a(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """Por modelo (test, entre corridas): exactitud y binomial exacta unilateral frente a la
    proporción de la clase mayoritaria; AUC-ROC macro con IC por bootstrap por lesión y p frente a
    0,5; MCC con IC por lesión; pendiente e intercepto de calibración de la clase predicha;
    Hosmer-Lemeshow y Z de Spiegelhalter (medianas de p y corridas con p < alpha)."""
    td = test_data(rs, cfg)
    if is_missing(td):
        return td
    b, st = bench_cfg(cfg), cfg["e2_stats"]
    alpha = float(b["alpha"])
    k = len(td.classes)
    n = len(td.y)
    p0 = float(np.bincount(td.y, minlength=k).max() / n)
    models = list(td.keys)
    obs_auc = {m: np.array([S.weighted_macro_auc(np.ones((1, n)), td.y, p)[0] for p in td.proba[m]]) for m in models}
    obs_mcc = {m: np.array([S.weighted_mcc(np.ones((1, n)), td.y, pr, k)[0] for pr in td.pred(m)]) for m in models}
    boot_auc = {m: [] for m in models}
    boot_mcc = {m: [] for m in models}
    for w in S.cluster_weight_batches(td.lesions, int(st["auc_resamples"]), int(b["lesion_bootstrap"]["seed"])):
        for m in models:
            boot_auc[m].append(np.mean([S.weighted_macro_auc(w, td.y, p) for p in td.proba[m]], axis=0))
            boot_mcc[m].append(np.mean([S.weighted_mcc(w, td.y, pr, k) for pr in td.pred(m)], axis=0))
    rows = []
    for m in models:
        corr = td.correct(m)
        conf = td.proba[m].max(axis=2)
        acc = corr.mean(axis=1)
        p_bin = [S.binomial_vs_majority(int(c.sum()), n, p0) for c in corr]
        cal = pd.DataFrame([S.calibration_slope_intercept(c, cf) for c, cf in zip(corr, conf)])
        hl = pd.DataFrame([S.hosmer_lemeshow(c, cf, int(st["hosmer_lemeshow_groups"])) for c, cf in zip(corr, conf)])
        sp = pd.DataFrame([S.spiegelhalter_z(c, cf) for c, cf in zip(corr, conf)])
        auc_b, mcc_b = np.concatenate(boot_auc[m]), np.concatenate(boot_mcc[m])
        auc_lo, auc_hi = S.percentile_interval(auc_b, alpha)
        mcc_lo, mcc_hi = S.percentile_interval(mcc_b, alpha)
        rows.append({"modelo": m, "corridas": len(corr), "exactitud": float(acc.mean()), "clase_mayoritaria": p0,
                     "p_binomial_mediana": float(np.median(p_bin)), "corridas_binomial_sig": int(np.sum(np.asarray(p_bin) < alpha)),
                     "auc_macro": float(obs_auc[m].mean()), "auc_ic_bajo": auc_lo, "auc_ic_alto": auc_hi,
                     "p_auc_vs_0.5": S.bootstrap_p(auc_b - 0.5, 0.0, "greater"),
                     "mcc": float(obs_mcc[m].mean()), "mcc_ic_bajo": mcc_lo, "mcc_ic_alto": mcc_hi,
                     "pendiente_calibracion": float(cal["pendiente"].mean()),
                     "pendiente_std": float(cal["pendiente"].std(ddof=1)) if len(cal) > 1 else np.nan,
                     "intercepto_calibracion": float(cal["intercepto"].mean()),
                     "intercepto_en_promedio": float(cal["intercepto_en_promedio"].mean()),
                     "hl_H_mediana": float(hl["H"].median()), "hl_p_mediana": float(hl["p"].median()),
                     "corridas_hl_sig": int((hl["p"] < alpha).sum()),
                     "spiegelhalter_Z_media": float(sp["Z"].mean()), "spiegelhalter_p_mediana": float(sp["p"].median()),
                     "corridas_spiegelhalter_sig": int((sp["p"] < alpha).sum())})
    out = pd.DataFrame(rows).set_index("modelo")
    out.attrs["nota"] = (f"IC 95 % por bootstrap por lesión ({st['auc_resamples']} remuestreos) de la media entre "
                         f"corridas; calibración sobre la clase predicha (top-label); Hosmer-Lemeshow con "
                         f"{st['hosmer_lemeshow_groups']} grupos")
    return out


# ---------------------------------------------------------------------------
# E2.6 Tabla B: sobreajuste
# ---------------------------------------------------------------------------


def _train_pool(cfg: dict) -> pd.DataFrame:
    from src import data as D

    s = D.load_splits(cfg["paths"]["splits_csv"])
    return s[s["split"] == "train"].sort_values("image_id").reset_index(drop=True)


@tolerant()
def table_b(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """Por modelo: brecha train sin aumento - validación de la métrica primaria (media) con
    Wilcoxon exacto unilateral frente al margen e2.overfitting_margin (H1: brecha < δg y H1:
    brecha > δg); diferencia validación - test (media entre corridas) con IC por bootstrap por
    lesión (lesiones de train y de test remuestreadas por separado,
    e2_stats.exploratory_resamples); y coeficiente de variación de la métrica primaria de test
    entre semillas (media sobre pliegues) y entre todas las corridas."""
    td = test_data(rs, cfg)
    if is_missing(td):
        return td
    b, st = bench_cfg(cfg), cfg["e2_stats"]
    alpha, metric = float(b["alpha"]), b["primary_metric"]
    dg = float(cfg["e2"]["overfitting_margin"])
    zm = b.get("wilcoxon_zero_method", "wilcox")
    pool = _train_pool(cfg)
    idx = {i: j for j, i in enumerate(pool["image_id"])}
    y_tr = pool["dx"].map({c: i for i, c in enumerate(td.classes)}).to_numpy()
    models = list(td.keys)
    val_c, val_m = {}, {}
    for m in models:
        C_ = np.zeros((len(td.keys[m]), len(pool)))
        M_ = np.zeros_like(C_)
        for r, key in enumerate(td.keys[m]):
            p = rs.predictions(m, *key, "val")
            pos = p["image_id"].map(idx).to_numpy()
            M_[r, pos] = 1.0
            C_[r, pos] = (p["label"] == p["pred"]).to_numpy()
        val_c[m], val_m[m] = C_, M_
    n_res = int(st["exploratory_resamples"])
    seed = int(b["lesion_bootstrap"]["seed"])
    diffs = {m: [] for m in models}
    for w_tr, w_te in zip(S.cluster_weight_batches(pool["lesion_id"].to_numpy(), n_res, seed),
                          S.cluster_weight_batches(td.lesions, n_res, seed + 1)):
        for m in models:
            v = S.weighted_balanced_accuracy(w_tr, y_tr, val_c[m], val_m[m]).mean(axis=1)
            t = S.weighted_balanced_accuracy(w_te, td.y, td.correct(m)).mean(axis=1)
            diffs[m].append(v - t)
    rows = []
    for m in models:
        sub = rs.metrics.query("model == @m and metric == @metric")
        piv = sub.pivot_table(index=["seed", "val_fold"], columns="split", values="value")
        row = {"modelo": m, "corridas": len(piv), "delta_g": dg}
        if "train" in piv and piv["train"].notna().any():
            gap = (piv["train"] - piv["val"]).dropna().to_numpy()
            row.update({"brecha_train_val": float(gap.mean()), "brecha_std": float(gap.std(ddof=1)) if len(gap) > 1 else np.nan,
                        "p_brecha_menor_dg": S.wilcoxon_exact(gap - dg, zm, "less")["p"],
                        "p_brecha_mayor_dg": S.wilcoxon_exact(gap - dg, zm, "greater")["p"]})
            row["sobreajuste"] = ("brecha menor que δg" if row["p_brecha_menor_dg"] < alpha else
                                  "brecha mayor que δg" if row["p_brecha_mayor_dg"] < alpha else "no concluyente")
        else:
            row.update({"brecha_train_val": np.nan, "brecha_std": np.nan, "p_brecha_menor_dg": np.nan,
                        "p_brecha_mayor_dg": np.nan, "sobreajuste": "sin métricas de train"})
        vt = (piv["val"] - piv["test"]).dropna()
        lo, hi = S.percentile_interval(np.concatenate(diffs[m]), alpha)
        row.update({"val_menos_test": float(vt.mean()), "val_test_ic_bajo": lo, "val_test_ic_alto": hi})
        t = piv["test"].dropna().reset_index()
        cv_f = t.groupby("val_fold")["test"].agg(lambda s: s.std(ddof=1) / s.mean() if len(s) > 1 else np.nan)
        row["cv_entre_semillas"] = float(cv_f.mean())
        row["cv_todas_las_corridas"] = float(t["test"].std(ddof=1) / t["test"].mean()) if len(t) > 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).set_index("modelo")
    out.attrs["nota"] = (f"métrica {metric}; δg = {dg}; Wilcoxon exacto sobre brecha - δg; IC de val - test por "
                         f"bootstrap por lesión ({n_res} remuestreos)")
    return out


# ---------------------------------------------------------------------------
# E2.7 Modelos de efectos mixtos
# ---------------------------------------------------------------------------


@tolerant()
def mixed_effects_run_level(rs: RunSet, cfg: dict) -> dict:
    """LMM (statsmodels MixedLM, REML) de la métrica primaria de test por corrida: efecto fijo
    del modelo (referencia: mejor modelo base) e intercepto aleatorio de la corrida (semilla y
    pliegue). Devuelve efectos fijos y descomposición de la varianza (corrida y residual)."""
    import statsmodels.formula.api as smf

    b = bench_cfg(cfg)
    models = global_models(rs, cfg)
    ref = best_base(rs, cfg)
    if is_missing(ref):
        return ref
    w = wide(rs, b["split"], b["primary_metric"], models)
    if len(w) < 3 or len(models) < 2:
        return Missing("el modelo mixto por corrida necesita al menos 3 corridas pareadas y 2 modelos.")
    long = w.reset_index().melt(id_vars=["seed", "val_fold"], var_name="modelo", value_name="y")
    long["corrida"] = long["seed"].astype(str) + "_" + long["val_fold"].astype(str)
    fit = smf.mixedlm(f"y ~ C(modelo, Treatment('{ref}'))", long, groups=long["corrida"]).fit(reml=True)
    fe = pd.DataFrame({"coef": fit.fe_params, "ee": fit.bse_fe, "p": fit.pvalues[fit.fe_params.index]})
    ci = fit.conf_int().loc[fit.fe_params.index]
    fe["ic_bajo"], fe["ic_alto"] = ci[0], ci[1]
    fe.index = [i.replace(f"C(modelo, Treatment('{ref}'))[T.", "").rstrip("]") for i in fe.index]
    var_run, var_res = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    var = pd.DataFrame({"varianza": [var_run, var_res]}, index=["corrida (semilla y pliegue)", "residual"])
    var["proporcion"] = var["varianza"] / var["varianza"].sum()
    return {"Efectos fijos (diferencia frente a " + ref + ")": fe, "Componentes de varianza": var}


def _glmm_image_level(rs: RunSet, cfg: dict) -> dict | Missing:
    import scipy.sparse as sp
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    td = test_data(rs, cfg)
    if is_missing(td):
        return td
    ref = best_base(rs, cfg)
    if is_missing(ref):
        return ref
    models = [m for m in global_models(rs, cfg) if m in td.keys]
    keys = sorted(set.intersection(*(set(td.keys[m]) for m in models)))
    if len(keys) < 2 or len(models) < 2:
        return Missing("el GLMM necesita al menos 2 corridas pareadas y 2 modelos.")
    others = [m for m in models if m != ref]
    _, les = np.unique(td.lesions, return_inverse=True)
    n_img, n_run, n_les = len(td.y), len(keys), int(les.max()) + 1
    ys, mi, ri, li = [], [], [], []
    for j, m in enumerate(models):
        corr = td.correct(m)
        for r, key in enumerate(keys):
            ys.append(corr[td.keys[m].index(key)])
            mi.append(np.full(n_img, j))
            ri.append(np.full(n_img, r))
            li.append(les)
    y, mi, ri, li = (np.concatenate(v) for v in (ys, mi, ri, li))
    X = np.column_stack([np.ones(len(y))] + [(mi == models.index(m)).astype(float) for m in others])
    n = len(y)
    Z = sp.hstack([sp.csr_matrix((np.ones(n), (np.arange(n), ri)), shape=(n, n_run)),
                   sp.csr_matrix((np.ones(n), (np.arange(n), li)), shape=(n, n_les))]).tocsr()
    ident = np.r_[np.zeros(n_run, int), np.ones(n_les, int)]
    np.random.seed(int(cfg["e2_stats"]["mixed_effects"]["vb_seed"]))
    mod = BinomialBayesMixedGLM(y.astype(float), X, Z, ident, vcp_p=1.0, fe_p=2.0,
                                fep_names=["intercepto"] + others, vcp_names=["corrida", "lesion"])
    res = mod.fit_vb()
    fe = pd.DataFrame({"coef_logit": res.fe_mean, "ee_posterior": res.fe_sd}, index=["intercepto"] + others)
    z = float(S.ss.norm.ppf(1 - float(bench_cfg(cfg)["alpha"]) / 2))
    fe["ic_bajo"] = fe["coef_logit"] - z * fe["ee_posterior"]
    fe["ic_alto"] = fe["coef_logit"] + z * fe["ee_posterior"]
    fe["odds_ratio"] = np.exp(fe["coef_logit"])
    sd = np.exp(res.vcp_mean)
    var = pd.DataFrame({"varianza_latente": [sd[0] ** 2, sd[1] ** 2, np.pi ** 2 / 3]},
                       index=["corrida (semilla y pliegue)", "lesión", "residual (logística, π²/3)"])
    var["proporcion"] = var["varianza_latente"] / var["varianza_latente"].sum()
    return {"Efectos fijos (log-odds de acierto frente a " + ref + ")": fe, "Componentes de varianza (escala latente)": var,
            "Datos": pd.Series({"filas": n, "modelos": len(models), "corridas": n_run, "lesiones": n_les,
                                "imagenes": n_img}, name="valor").to_frame()}


def mixed_effects_image_level(rs: RunSet, cfg: dict, rerun: bool = False):
    """GLMM binomial de los aciertos por imagen de test (Bayes variacional, statsmodels
    BinomialBayesMixedGLM): efecto fijo del modelo (referencia: mejor modelo base) y efectos
    aleatorios cruzados de la corrida y de la lesión; descomposición de la varianza en la escala
    latente (residual logístico π²/3). Tarda decenas de minutos: solo se calcula con `rerun`
    (o con `python -m src.reproduce --glmm`); si no, se lee de outputs/benchmark_e2/glmm.pkl."""
    if rerun:
        return cached(cfg, rs, "glmm", lambda: _glmm_image_level(rs, cfg), rerun=True)
    return cached_only(cfg, rs, "glmm", "el GLMM por imagen no está calculado; pon RECALCULAR_GLMM = True "
                                         "o ejecuta `python -m src.reproduce --glmm`")


# ---------------------------------------------------------------------------
# E2.8 Familias exploratorias: RQ3 (fuente) y subgrupos (Benjamini-Hochberg)
# ---------------------------------------------------------------------------


@tolerant()
def source_gap(rs: RunSet, cfg: dict, models=None, n_resamples: int | None = None) -> pd.DataFrame:
    """Diferencia de exactitud balanceada en test entre las imágenes de e2_stats.source_target y
    las del resto de fuentes (media entre corridas), con IC 95 % y p bilateral por bootstrap por
    lesión y ajuste e2_stats.exploratory_p_adjust entre las filas. Si hay CNN y transformador,
    añade la fila «transformador - media CNN» (diferencia de las diferencias)."""
    td = test_data(rs, cfg)
    if is_missing(td):
        return td
    if td.meta is None:
        return Missing("faltan los metadatos (data/raw/HAM10000_metadata.csv).")
    st, b = cfg["e2_stats"], bench_cfg(cfg)
    alpha = float(b["alpha"])
    col, target = st["source_column"], st["source_target"]
    is_t = (td.meta[col] == target).to_numpy()
    models = [m for m in (models or td.keys) if m in td.keys]
    if not models:
        return Missing("ningún modelo pedido tiene predicciones de test.")
    res = lesion_ba_resamples(td, cfg, {"t": is_t, "r": ~is_t}, int(n_resamples or st["exploratory_resamples"]), models)
    rows, samples = [], {}
    for m in models:
        obs = res[(m, "t")]["observado"].mean() - res[(m, "r")]["observado"].mean()
        dist = res[(m, "t")]["remuestreos"].mean(axis=1) - res[(m, "r")]["remuestreos"].mean(axis=1)
        samples[m] = dist
        lo, hi = S.percentile_interval(dist, alpha)
        rows.append({"modelo": m, "familia": "transformador" if m in e2_groups(cfg)["transformer_models"] else
                     ("CNN" if m in e2_groups(cfg)["cnn_models"] else role_of(cfg, m)),
                     f"ba_{target}": float(res[(m, "t")]["observado"].mean()),
                     "ba_resto": float(res[(m, "r")]["observado"].mean()),
                     "diferencia": float(obs), "ic_bajo": lo, "ic_alto": hi, "p": S.bootstrap_p(dist, 0.0, "two-sided")})
    cnn = [m for m in models if m in e2_groups(cfg)["cnn_models"]]
    vit = [m for m in models if m in e2_groups(cfg)["transformer_models"]]
    if cnn and vit:
        dist = np.mean([samples[m] for m in vit], axis=0) - np.mean([samples[m] for m in cnn], axis=0)
        obs = np.mean([r["diferencia"] for r in rows if r["modelo"] in vit]) - \
            np.mean([r["diferencia"] for r in rows if r["modelo"] in cnn])
        lo, hi = S.percentile_interval(dist, alpha)
        rows.append({"modelo": "transformador - media CNN", "familia": "contraste", f"ba_{target}": np.nan,
                     "ba_resto": np.nan, "diferencia": float(obs), "ic_bajo": lo, "ic_alto": hi,
                     "p": S.bootstrap_p(dist, 0.0, "two-sided")})
    out = pd.DataFrame(rows).set_index("modelo")
    out[f"p_{st['exploratory_p_adjust']}"] = S.adjust_pvalues(out["p"], st["exploratory_p_adjust"])
    out["imagenes_fuente"] = int(is_t.sum())
    out["imagenes_resto"] = int((~is_t).sum())
    out.attrs["nota"] = (f"{target} - resto de fuentes; bootstrap por lesión de la media entre corridas; "
                         f"ajuste {st['exploratory_p_adjust']} entre las {len(out)} filas (exploratorio)")
    return out


@tolerant()
def subgroup_family(rs: RunSet, cfg: dict, models=None) -> pd.DataFrame:
    """Exactitud balanceada de test por categoría de cada variable de e2_stats.subgroups (fuente,
    sexo, grupo de edad, localización y dx_type), por modelo: media entre corridas con IC 95 % por
    bootstrap por lesión, diferencia frente al resto de las imágenes con p bilateral y ajuste
    e2_stats.exploratory_p_adjust sobre toda la familia. `pocas_imagenes` marca las categorías con
    menos de evaluation.subgroup_min_n imágenes."""
    td = test_data(rs, cfg)
    if is_missing(td):
        return td
    if td.meta is None:
        return Missing("faltan los metadatos (data/raw/HAM10000_metadata.csv).")
    st, b = cfg["e2_stats"], bench_cfg(cfg)
    alpha, min_n = float(b["alpha"]), int(cfg["evaluation"]["subgroup_min_n"])
    masks, info = {}, {}
    for var in st["subgroups"]:
        if var not in td.meta.columns:
            continue
        values = td.meta[var].astype(object).fillna("desconocido")
        for cat in sorted(values.unique(), key=str):
            m = (values == cat).to_numpy()
            if m.sum() == 0 or (~m).sum() == 0:
                continue
            masks[(var, cat, "in")], masks[(var, cat, "out")] = m, ~m
            info[(var, cat)] = int(m.sum())
    models = [m for m in (models or td.keys) if m in td.keys]
    res = lesion_ba_resamples(td, cfg, masks, int(st["exploratory_resamples"]), models)
    rows = []
    for m in models:
        for (var, cat), n in info.items():
            a, r = res[(m, (var, cat, "in"))], res[(m, (var, cat, "out"))]
            dist_a = a["remuestreos"].mean(axis=1)
            dist_d = dist_a - r["remuestreos"].mean(axis=1)
            lo, hi = S.percentile_interval(dist_a, alpha)
            dlo, dhi = S.percentile_interval(dist_d, alpha)
            rows.append({"variable": var, "categoria": str(cat), "modelo": m, "imagenes": n,
                         "balanced_accuracy": float(np.nanmean(a["observado"])), "ic_bajo": lo, "ic_alto": hi,
                         "diferencia_vs_resto": float(np.nanmean(a["observado"]) - np.nanmean(r["observado"])),
                         "dif_ic_bajo": dlo, "dif_ic_alto": dhi, "p": S.bootstrap_p(dist_d, 0.0, "two-sided"),
                         "pocas_imagenes": n < min_n})
    if not rows:
        return Missing("no hay categorías de subgrupo con imágenes.")
    out = pd.DataFrame(rows).set_index(["variable", "categoria", "modelo"])
    out[f"p_{st['exploratory_p_adjust']}"] = S.adjust_pvalues(out["p"], st["exploratory_p_adjust"])
    return out


# ---------------------------------------------------------------------------
# E2.9 Réplica de pinar2026
# ---------------------------------------------------------------------------


@tolerant()
def replication_table(rs: RunSet, cfg: dict) -> pd.DataFrame:
    """Réplica de ResNet-50 frente al valor publicado (replication): exitosa si el valor cae
    dentro del IC 95 % por t de la exactitud balanceada media de las corridas definitivas o si la
    diferencia absoluta es menor que delta (capítulo 14). Se informa también el IC por lesión."""
    rep, b = cfg["replication"], bench_cfg(cfg)
    m, value = rep["model"], float(rep["value"])
    alpha, delta = float(b["alpha"]), float(b["delta"])
    if m not in rs.models:
        return Missing(f"{m} no tiene corridas definitivas.")
    mt = rs.metrics
    x = mt[(mt["model"] == m) & (mt["split"] == b["split"]) & (mt["metric"] == b["primary_metric"])]["value"].to_numpy()
    lo, hi = S.t_confidence_interval(x, alpha)
    td = test_data(rs, cfg)
    llo = lhi = np.nan
    if not is_missing(td) and (m, None) in _boot(rs, cfg, td):
        llo, lhi = S.percentile_interval(_boot(rs, cfg, td)[(m, None)]["remuestreos"].mean(axis=1), alpha)
    inside = bool(lo <= value <= hi)
    close = bool(abs(x.mean() - value) < delta)
    return pd.Series({
        "modelo": m, "corridas": len(x), "media": float(x.mean()), "ic_t_bajo": lo, "ic_t_alto": hi,
        "ic_lesion_bajo": llo, "ic_lesion_alto": lhi, f"valor_{rep['source']}": value,
        f"ic_{rep['source']}": f"[{rep['ci'][0]}; {rep['ci'][1]}]", "diferencia": float(x.mean() - value),
        "valor_dentro_del_ic_t": inside, "diferencia_menor_que_delta": close,
        "ic_se_solapan": bool(lo <= rep["ci"][1] and rep["ci"][0] <= hi),
        "replica_exitosa": inside or close}, name="valor").to_frame()


# ---------------------------------------------------------------------------
# E2.10 Techo, costo, frontera de Pareto y eficiencia
# ---------------------------------------------------------------------------


@tolerant()
def model_scores(rs: RunSet, cfg: dict) -> pd.Series:
    b = bench_cfg(cfg)
    mt = rs.metrics
    sub = mt[(mt["split"] == b["split"]) & (mt["metric"] == b["primary_metric"])]
    if sub.empty:
        return Missing("no hay métrica primaria en test.")
    return sub.groupby("model")["value"].mean().reindex(rs.models).rename("score")


@tolerant()
def ranked_deep_models(rs: RunSet, cfg: dict) -> list[str] | Missing:
    """Modelos profundos con corridas, de mayor a menor media de la métrica primaria en test."""
    scores = model_scores(rs, cfg)
    if is_missing(scores):
        return scores
    deep = [m for m in e2_groups(cfg)["deep_models"] if m in scores.dropna().index]
    if not deep:
        return Missing("no hay modelos profundos con corridas definitivas.")
    return list(scores[deep].sort_values(ascending=False).index)


@tolerant()
def gamma_table(rs: RunSet, cfg: dict, ceilings: pd.DataFrame) -> pd.DataFrame:
    """γ = (s_techo - s_mejor) / (s_techo - s_ref) para cada techo de `ceilings` (columna
    `techo`), con s_mejor la mejor media de los modelos profundos y s_ref la del modelo de
    ceiling.gamma_reference (best_base, trivial o logreg)."""
    scores = model_scores(rs, cfg)
    if is_missing(scores):
        return scores
    deep = [m for m in e2_groups(cfg)["deep_models"] if m in scores.index]
    if not deep:
        return Missing("no hay modelos profundos con corridas.")
    best = scores[deep].idxmax()
    ref = cfg["ceiling"].get("gamma_reference", "best_base")
    ref = best_base(rs, cfg) if ref == "best_base" else ref
    if is_missing(ref) or ref not in scores.index:
        return Missing(f"la referencia de γ ({ref}) no tiene corridas.")
    out = ceilings.copy()
    out["mejor_modelo"], out["s_mejor"] = best, float(scores[best])
    out["referencia"], out["s_ref"] = ref, float(scores[ref])
    out["gamma"] = (out["techo"] - out["s_mejor"]) / (out["techo"] - out["s_ref"])
    return out


@tolerant()
def pareto_table(rs: RunSet, cfg: dict, cost: pd.DataFrame) -> pd.DataFrame:
    """Por modelo profundo con costo medido: métrica primaria media y, para cada costo de
    cost.pareto_costs, si está en la frontera de Pareto (ningún otro modelo tiene mayor o igual
    desempeño con menor o igual costo, siendo estrictamente mejor en alguno)."""
    scores = model_scores(rs, cfg)
    if is_missing(scores):
        return scores
    models = [m for m in e2_groups(cfg)["deep_models"] if m in scores.index and m in cost.index]
    if len(models) < 2:
        return Missing("la frontera de Pareto necesita al menos 2 modelos con corridas y costo medido.")
    costs = [c for c in cfg["cost"]["pareto_costs"] if c in cost.columns]
    if not costs:
        return Missing("no hay columnas de costo de cost.pareto_costs (python -m src.cost --all y corridas definitivas).")
    out = pd.DataFrame({"score": scores[models]})
    for c in costs:
        if c not in cost.columns:
            continue
        x = cost.loc[models, c].astype(float)
        out[c] = x
        front = []
        for m in models:
            dominated = any((scores[o] >= scores[m]) and (x[o] <= x[m]) and
                            ((scores[o] > scores[m]) or (x[o] < x[m])) for o in models if o != m)
            front.append(not dominated)
        out[f"pareto_{c}"] = front
    return out


@tolerant()
def plot_pareto(table: pd.DataFrame, cfg: dict):
    costs = [c for c in cfg["cost"]["pareto_costs"] if c in table.columns]
    fig, axes = plt.subplots(1, len(costs), figsize=(4.2 * len(costs), 3.8), squeeze=False)
    for ax, c in zip(axes[0], costs):
        front = table[table[f"pareto_{c}"]].sort_values(c)
        ax.scatter(table[c], table["score"], color="tab:blue")
        ax.plot(front[c], front["score"], color="tab:red", lw=1.5, marker="o", label="frontera")
        for m, r in table.iterrows():
            ax.annotate(m, (r[c], r["score"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if c in ("gflops", "params_total"):
            ax.set_xscale("log")
        ax.set_xlabel(c)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(bench_cfg(cfg)["primary_metric"])
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


@tolerant()
def efficiency_table(rs: RunSet, cfg: dict, cost: pd.DataFrame) -> pd.DataFrame:
    """Eficiencia relativa η_k = (s_k / C_k) / (s_base / C_base) respecto del mejor modelo base
    (desempeño por unidad de costo; > 1: más eficiente que la referencia), con C el costo
    cost.eta_cost e IC 95 % por bootstrap por lesión de s_k / s_base (corridas comunes). Se
    informa también la diferencia s_k - s_base."""
    td = test_data(rs, cfg)
    ref = best_base(rs, cfg)
    if is_missing(td) or is_missing(ref):
        return td if is_missing(td) else ref
    col = cfg["cost"]["eta_cost"]
    if col not in cost.columns or ref not in cost.index:
        return Missing(f"falta el costo {col} de {ref} (python -m src.cost --all).")
    alpha = float(bench_cfg(cfg)["alpha"])
    boot = _boot(rs, cfg, td)
    rows = []
    for m in e2_groups(cfg)["deep_models"]:
        if m == ref or m not in td.keys or m not in cost.index or (m, None) not in boot:
            continue
        ia, ir = td.common(m, ref)
        rel = float(cost.loc[m, col]) / float(cost.loc[ref, col])
        sk, sb = boot[(m, None)]["remuestreos"][:, ia].mean(axis=1), boot[(ref, None)]["remuestreos"][:, ir].mean(axis=1)
        ok, ob = boot[(m, None)]["observado"][ia].mean(), boot[(ref, None)]["observado"][ir].mean()
        lo, hi = S.percentile_interval(sk / sb / rel, alpha)
        rows.append({"modelo": m, "referencia": ref, "diferencia": float(ok - ob), f"{col}_relativo": rel,
                     "eta": float(ok / ob / rel), "eta_ic_bajo": lo, "eta_ic_alto": hi})
    if not rows:
        return Missing("no hay modelos con corridas y costo medido distintos de la referencia.")
    return pd.DataFrame(rows).set_index("modelo")
