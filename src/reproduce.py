"""Regenera todas las tablas y figuras del Entregable 2 desde los resultados guardados.

    python -m src.reproduce [--glmm] [--out outputs/reproduce]

No entrena ni ejecuta inferencia: lee las corridas de outputs/runs, las mediciones de
outputs/cost, outputs/robustness y outputs/interpretability, los embeddings en cache y los datos
de data/. Escribe cada tabla como CSV y cada figura como PNG en paths.reproduce_dir (o --out), y un
índice (indice.csv) con el estado de cada elemento: «generado» o el motivo por el que falta. Con
--glmm calcula además el GLMM por imagen (decenas de minutos) y lo deja en la cache del benchmark.
"""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import data as D
from src.report import Missing, is_missing

log = logging.getLogger(__name__)


class Writer:
    """Guarda tablas (CSV) y figuras (PNG) y lleva el índice."""

    def __init__(self, out: Path):
        self.out = out
        self.rows = []
        out.mkdir(parents=True, exist_ok=True)

    def __call__(self, name: str, obj) -> None:
        try:
            if callable(obj) and not isinstance(obj, (pd.DataFrame, pd.Series)):
                obj = obj()
        except Exception as e:  # noqa: BLE001 - un elemento que falla no detiene el resto
            self.rows.append({"elemento": name, "archivo": "", "estado": f"error: {e}"})
            log.warning(f"{name}: error\n{traceback.format_exc()}")
            return
        if isinstance(obj, tuple):
            obj = obj[0]
        if obj is None or isinstance(obj, Missing):
            self.rows.append({"elemento": name, "archivo": "", "estado": obj.message if obj is not None else "vacío"})
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                self(f"{name}__{_slug(k)}", v)
            return
        if hasattr(obj, "savefig"):
            path = self.out / f"{name}.png"
            obj.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(obj)
        else:
            path = self.out / f"{name}.csv"
            (obj.to_frame() if isinstance(obj, pd.Series) else obj).to_csv(path)
        self.rows.append({"elemento": name, "archivo": path.name, "estado": "generado"})

    def index(self) -> pd.DataFrame:
        t = pd.DataFrame(self.rows)
        t.to_csv(self.out / "indice.csv", index=False)
        return t


def _slug(text: str) -> str:
    import re
    import unicodedata

    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()[:60]


def reproduce(cfg: dict, out: Path, glmm: bool = False) -> pd.DataFrame:
    from src import audit as A
    from src import benchmark as B
    from src import calibration as C
    from src import ceiling as CE
    from src import cost as CO
    from src import errors as E
    from src import evaluation as EV
    from src import interpret as I
    from src import results as R
    from src import robustness as RB
    from src.final import finished_runs

    w = Writer(out)
    alpha = float(cfg["benchmark"]["alpha"])
    classes = D.class_names(cfg)
    groups = B.e2_groups(cfg)
    deep = groups["deep_models"]

    # Por modelo (book/08 a 10 y 15 a 19)
    for m in groups["all_models"]:
        runs = finished_runs(cfg, m)
        if is_missing(runs):
            w(f"modelo_{m}", runs)
            continue
        w(f"modelo_{m}_metricas_test", lambda runs=runs: R.add_summary_rows(R.run_metrics_table(runs, "test"), alpha=alpha))
        w(f"modelo_{m}_metricas_val", lambda runs=runs: R.add_summary_rows(R.run_metrics_table(runs, "val"), alpha=alpha))
        w(f"modelo_{m}_por_clase_test", lambda runs=runs: R.per_class_summary(R.per_class_table(runs, "test")))
        w(f"modelo_{m}_confusion_test", lambda runs=runs: R.plot_pooled_confusion(runs, "test"))
        w(f"modelo_{m}_sobreajuste", lambda runs=runs: EV.overfitting_table(runs, cfg["evaluation"]["overfitting_metrics"]))

    # Benchmark del Entregable 2 (book/20)
    rs = B.load_runs_e2(cfg)
    if glmm:
        B.mixed_effects_image_level(rs, cfg, rerun=True)
    td = B.test_data(rs, cfg)
    meta = EV.load_metadata_or_missing(cfg)
    measured = CO.load_cost(cfg, deep)
    times = CO.cost_from_runs(cfg, deep)
    cost = CO.cost_table(measured, times)
    gw = B.global_wide(rs, cfg)
    w("e2_disponibilidad", lambda: B.availability_e2(rs, cfg))
    w("e2_divergentes", lambda: B.divergent_runs(rs, cfg))
    main = B.main_table(rs, cfg)
    w("e2_tabla_principal_larga", main)
    w("e2_tabla_principal", lambda: B.main_table_formatted(main, cfg, cost if isinstance(cost, pd.DataFrame) else None))
    w("e2_sensibilidad_por_clase", lambda: B.per_class_e2(rs, cfg, stat="recall"))
    rq1, rq2 = B.contrast_family(rs, cfg, "rq1"), B.contrast_family(rs, cfg, "rq2")
    w("e2_rq1_contrastes", rq1)
    w("e2_rq2_contrastes", rq2)
    w("e2_contrastes_resumen", lambda: B.contrasts_summary(rq1, rq2))
    w("e2_global_friedman", lambda: B.global_tests(gw, cfg))
    w("e2_global_posthoc", lambda: B.global_posthoc(gw, cfg))
    w("e2_global_cd", lambda: B.plot_cd_diagram(gw, cfg))
    w("e2_cochran", lambda: B.cochran_by_run(rs, cfg))
    for fam in ("rq1", "rq2"):
        w(f"e2_mcnemar_{fam}", lambda fam=fam: B.mcnemar_summary(B.mcnemar_contrasts(rs, cfg, fam), cfg))
        w(f"e2_delong_{fam}", lambda fam=fam: B.delong_summary(B.delong_contrasts(rs, cfg, fam), cfg))
    w("e2_tabla_a", lambda: B.table_a(rs, cfg))
    w("e2_tabla_b", lambda: B.table_b(rs, cfg))
    w("e2_efectos_mixtos_corrida", lambda: B.mixed_effects_run_level(rs, cfg))
    w("e2_efectos_mixtos_imagen", lambda: B.mixed_effects_image_level(rs, cfg))
    temp = C.temperature_table(rs, cfg, rs.models, classes)
    w("e2_temperatura", lambda: C.temperature_summary(temp))
    w("e2_fiabilidad", lambda: C.plot_reliability_before_after(rs, cfg, rs.models, classes))
    w("e2_ensemble", lambda: C.ensemble_table(rs, cfg, deep, classes))
    conf = C.conformal_results(rs, cfg, deep, classes, meta)
    for g in ("global", "fuente", "clase"):
        w(f"e2_conforme_{g}", lambda g=g: C.conformal_summary(conf, g))
    w("e2_conforme_figura", lambda: C.plot_conformal(C.conformal_summary(conf, "global")))
    rob = RB.load_results(cfg, deep)
    w("e2_robustez_indice", lambda: RB.robustness_index(rob))
    w("e2_robustez_degradacion", lambda: RB.degradation_table(rob))
    w("e2_robustez_curvas", lambda: RB.plot_degradation(rob))
    top = I.top_models(cfg)
    if not is_missing(top):
        from src.gradcam import plot_gradcam_grid

        for m in top:
            maps = I.load_maps(cfg, m)
            w(f"e2_saliencia_{m}", lambda maps=maps: plot_gradcam_grid(maps))
        w("e2_borrado_insercion", lambda: I.deletion_insertion_summary(cfg, top))
        w("e2_estabilidad_mapas", lambda: I.stability_summary(cfg, top))
    frac = E.error_fraction(td, deep) if not is_missing(td) else td
    thr = float(cfg["errors"]["error_frac"])
    w("e2_errores_compartidos", lambda: E.shared_errors_table(frac, thr))
    w("e2_errores_por_modelo", lambda: E.errors_per_model(frac, thr))
    w("e2_correlacion_errores", lambda: E.error_correlation(td, deep))
    w("e2_correlacion_errores_figura", lambda: E.plot_error_correlation(E.error_correlation(td, deep)))
    scores = B.model_scores(rs, cfg)
    if not is_missing(scores):
        ranked = list(scores[[m for m in deep if m in scores.dropna().index]].sort_values(ascending=False).index)
        w("e2_ensemble_simple", lambda: E.ensemble_summary(E.ensemble_gain(td, ranked, int(cfg["errors"]["ensemble_top_k"]))))
    for split in cfg["errors"]["shortcut_splits"]:
        w(f"e2_atajos_{split}", lambda split=split: E.shortcut_table(rs, cfg, deep, split))
    w("e2_taxonomia", lambda: E.taxonomy_status(cfg))
    w("e2_rq3_fuente", lambda: B.source_gap(rs, cfg))
    w("e2_subgrupos", lambda: B.subgroup_family(rs, cfg))
    w("e2_limitacion_paper_guia", lambda: B.source_gap(rs, cfg, groups["base_models"]))
    knn = CE.knn_ceiling(cfg)
    w("e2_techo_knn", knn)
    w("e2_techo_gamma", lambda: B.gamma_table(rs, cfg, CE.ceilings_table(cfg, knn)))
    w("e2_costo", cost)
    if isinstance(cost, pd.DataFrame):
        par = B.pareto_table(rs, cfg, cost)
        w("e2_pareto", par)
        w("e2_pareto_figura", lambda: B.plot_pareto(par, cfg))
        w("e2_eficiencia", lambda: B.efficiency_table(rs, cfg, cost))
    w("e2_replica_pinar2026", lambda: B.replication_table(rs, cfg))

    # Auditoría (book/21)
    audit = A.run_audit(cfg, rs)
    w("auditoria_resumen", audit["resumen"])
    for name, detail in audit["detalle"].items():
        w(f"auditoria_{_slug(name)}", detail)
    return w.index()


def main(argv=None) -> None:
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Regenera tablas y figuras desde los resultados guardados.")
    parser.add_argument("--glmm", action="store_true", help="calcula también el GLMM por imagen (lento)")
    parser.add_argument("--out", default=None, help="carpeta de salida (por defecto paths.reproduce_dir)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = D.load_config(args.config)
    setup_logging(cfg["paths"]["logs_dir"] / "reproduce.log")
    out = Path(args.out) if args.out else cfg["paths"]["reproduce_dir"]
    idx = reproduce(cfg, out, args.glmm)
    log.info(f"{(idx['estado'] == 'generado').sum()} de {len(idx)} elementos generados -> {out}")
    log.info("Pendientes:\n" + idx[idx["estado"] != "generado"].to_string(index=False))


if __name__ == "__main__":
    main()
