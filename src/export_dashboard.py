"""Genera los datos precalculados del dashboard (dashboard/data/) desde data/, outputs/, artifacts/ y
los notebooks 05 y 06. El dashboard no importa TensorFlow ni src/: solo lee lo que escribe este script.

    python -m src.export_dashboard                # todo
    python -m src.export_dashboard --skip-eda     # solo la parte de modelos

Salida (configs/export.yaml, sección dashboard):

    manifest.json                     fecha, versiones y qué secciones tienen datos
    eda/images.json                   una fila por imagen: metadatos, partición y color (train)
    eda/summary.json                  conteos, ε² por clase y por fuente, calidad, ficha e interpretaciones
    eda/gallery/<image_id>.jpg        miniaturas aleatorias de train por clase
    examples/<image_id>.jpg + examples.json   galería de predicción (imágenes de test, 224 x 224)
    models/index.json                 estado ("pendiente" o "listo") y modelos con artefactos
    models/leaderboard.json           tabla de líderes (de la tabla maestra y outputs/cost)
    models/global.json                rangos medios y diferencia crítica (diagrama CD) y frontera de Pareto
    models/original.json              subsección del modelo original (procedencia, pseudocódigo, pruebas, robustez)
    models/<modelo>/                  metadata.json, ONNX servible, ood_reference.npz, diagnostics.json y,
                                      si supera el límite de tamaño, precomputed.json (galería)
    content/*.md                      MODEL_CARD.md y DATA_STATEMENT.md

La tabla maestra es src.benchmark.main_table sobre las corridas definitivas de todos los modelos
(incluidos el original y sus variantes), guardada en outputs/benchmark_e2/tabla_maestra.csv;
tests/test_dashboard_data.py comprueba que las cifras del dashboard coinciden con ella.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src.report import is_missing

log = logging.getLogger(__name__)

COLOR_VARS = {"media_r": "Media del canal R", "media_g": "Media del canal G", "media_b": "Media del canal B",
              "luminancia": "Luminancia", "contraste": "Contraste", "saturacion": "Saturación"}
META_COLUMNS = ["image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization", "dataset"]
LEADERBOARD_COLUMNS = ["modelo", "familia", "corridas", "ba_media", "ba_std", "ece_media", "params_total",
                       "gflops", "latencia_ms"]


# ---------------------------------------------------------------------------
# E/S
# ---------------------------------------------------------------------------


def _clean(obj):
    """Convierte NumPy/pandas a tipos JSON y NaN a None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, ensure_ascii=False, separators=(",", ":"))


def frame_records(df: pd.DataFrame, digits: int = 4) -> dict:
    """DataFrame -> {"columns": [...], "data": [[...], ...]} con floats redondeados (compacto)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(digits)
    return {"columns": list(out.columns), "data": out.astype(object).where(out.notna(), None).values.tolist()}


# ---------------------------------------------------------------------------
# Interpretaciones de los notebooks
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿(])")


def notebook_sentence(notebook: Path, anchor: str) -> str | None:
    """Oración de la primera celda markdown de interpretación de `notebook` que contiene `anchor`."""
    if not Path(notebook).exists():
        return None
    nb = json.loads(Path(notebook).read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        if anchor not in text:
            continue
        for paragraph in text.split("\n"):
            if anchor in paragraph:
                for sentence in _SENTENCE_END.split(paragraph.strip()):
                    if anchor in sentence:
                        return re.sub(r"`([^`]*)`", r"\1", sentence).strip()
    return None


def interpretations(ecfg: dict) -> dict:
    out = {}
    for key, spec in ecfg["dashboard"]["interpretations"].items():
        nb = D.PROJECT_ROOT / spec["notebook"]
        sentence = notebook_sentence(nb, spec["anchor"])
        out[key] = {"text": sentence, "source": Path(spec["notebook"]).name}
        if sentence is None:
            log.warning(f"Sin interpretación para {key}: no se encontró «{spec['anchor']}» en {spec['notebook']}")
    return out


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


def color_features_train(cfg: dict, ecfg: dict, train_ids) -> pd.DataFrame | None:
    """Características de color por imagen de train (src.eda.color_features sobre las imágenes
    originales, como el notebook 06), en cache en dashboard.color_features_cache."""
    from src import eda

    cache = D.PROJECT_ROOT / ecfg["dashboard"]["color_features_cache"]
    if cache.exists():
        df = pd.read_parquet(cache)
        if set(train_ids) <= set(df["image_id"]):
            return df
    paths = D.image_paths(cfg, train_ids)
    if not all(p.exists() for p in paths[:10]):
        log.warning("No hay imágenes en data/raw/images: el EDA del dashboard se genera sin el color por imagen")
        return None
    df, _ = eda.color_features(train_ids, paths, cfg["eda_luma_weights"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def epsilon2_table(df: pd.DataFrame, variables, cfg: dict) -> list[dict]:
    from src import stats as S

    src = cfg.get("source_column", "dataset")
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        by_class = S.kruskal_epsilon2(df[v], df["dx"])
        by_source = S.kruskal_epsilon2(df[v], df[src])
        rows.append({"variable": v, "etiqueta": COLOR_VARS.get(v, v), "epsilon2_clase": by_class["epsilon2"],
                     "p_clase": by_class["p"], "epsilon2_fuente": by_source["epsilon2"], "p_fuente": by_source["p"],
                     "n": by_class["n"]})
    return rows


def make_thumbnail(src: Path, dst: Path, size: int, quality: int) -> bool:
    from PIL import Image

    if not src.exists():
        return False
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=int(quality), optimize=True)
    return True


def gallery(cfg: dict, ecfg: dict, train: pd.DataFrame, out: Path) -> dict:
    """Miniaturas aleatorias (semilla fija) de train por clase."""
    t = ecfg["dashboard"]["thumbnails"]
    rng = np.random.default_rng(int(t["seed"]))
    folder = out / "eda" / "gallery"
    if folder.exists():
        shutil.rmtree(folder)
    result = {}
    for dx, g in train.groupby("dx"):
        ids = g["image_id"].to_numpy()
        pick = np.sort(rng.choice(ids, size=min(int(t["per_class"]), len(ids)), replace=False))
        done = [i for i in pick if make_thumbnail(D.image_path(cfg, i), folder / f"{i}.jpg", int(t["size"]),
                                                   int(t["quality"]))]
        result[str(dx)] = done
    return result


def lesion_overlap(splits: pd.DataFrame) -> dict:
    """Lesiones en más de una partición (train/test) y en más de un pliegue de train."""
    by_split = splits.groupby("lesion_id")["split"].nunique()
    tr = splits[splits["split"] == "train"]
    by_fold = tr.groupby("lesion_id")["fold"].nunique()
    return {"lesiones_en_varias_particiones": int((by_split > 1).sum()),
            "lesiones_en_varios_pliegues": int((by_fold > 1).sum())}


def export_eda(cfg: dict, ecfg: dict, out: Path) -> dict:
    from src import stats as S

    meta = D.load_metadata(cfg)
    splits = D.load_splits(cfg["paths"]["splits_csv"])
    images = meta[META_COLUMNS].merge(splits[["image_id", "split", "fold"]], on="image_id", how="left")
    images["age"] = pd.to_numeric(images["age"], errors="coerce")
    train = images[images["split"] == "train"]
    color = color_features_train(cfg, ecfg, train["image_id"].to_numpy())
    if color is not None:
        images = images.merge(color, on="image_id", how="left")
    write_json(frame_records(images), out / "eda" / "images.json")

    classes = D.class_names(cfg)
    counts = images["dx"].value_counts().reindex(classes).fillna(0).astype(int)
    tr_color = images[images["split"] == "train"]
    eps = epsilon2_table(tr_color, list(COLOR_VARS), cfg) if color is not None else []
    eps += epsilon2_table(tr_color, ["age"], cfg)

    cand = pd.read_csv(cfg["paths"]["duplicates_candidates_csv"]) if cfg["paths"]["duplicates_candidates_csv"].exists() else None
    conf = D.load_confirmed_duplicates(cfg["paths"]["duplicates_confirmed_csv"])
    folds = splits[splits["split"] == "train"]["fold"].value_counts().sort_index()
    quality = {
        "duplicados_candidatos": len(cand) if cand is not None else None,
        "duplicados_confirmados": len(conf) if conf is not None else 0,
        "umbral_phash": int(cfg["phash_threshold"]),
        **lesion_overlap(splits),
        "n_train": int((splits["split"] == "train").sum()),
        "n_test": int((splits["split"] == "test").sum()),
        "imagenes_por_pliegue": {str(int(k)): int(v) for k, v in folds.items()},
        "n_lesiones": int(meta["lesion_id"].nunique()),
    }
    summary = {
        "n_images": len(images), "n_lesions": int(meta["lesion_id"].nunique()),
        "classes": classes, "sources": list(cfg["valid_categories"]["dataset"]),
        "class_counts": counts.to_dict(),
        "imbalance_ratio": float(counts.max() / counts[counts > 0].min()),
        "entropy": S.class_entropy(counts.to_numpy()),
        "epsilon2": eps,
        "color_variables": COLOR_VARS if color is not None else {},
        "quality": quality,
        "interpretations": interpretations(ecfg),
        "gallery": gallery(cfg, ecfg, train, out),
        "data_card": data_card(cfg, images),
    }
    write_json(summary, out / "eda" / "summary.json")
    log.info(f"EDA: {len(images)} imágenes, {sum(len(v) for v in summary['gallery'].values())} miniaturas -> {out / 'eda'}")
    return summary


def data_card(cfg: dict, images: pd.DataFrame) -> list[dict]:
    """Ficha del conjunto de datos (capítulo 12), con las cifras recalculadas."""
    dx_type = images["dx_type"].value_counts(normalize=True) * 100
    fmt = lambda k: f"{dx_type.get(k, 0):.1f} %".replace(".", ",")
    return [
        {"campo": "Nombre y versión", "valor": "HAM10000, versión 4 en Harvard Dataverse (doi: 10.7910/DVN/DBW86T)"},
        {"campo": "Tamaño", "valor": f"{len(images)} imágenes de {images['lesion_id'].nunique()} lesiones"},
        {"campo": "Tipo de datos", "valor": "Imágenes dermatoscópicas JPEG, RGB, 8 bits por canal, 600 × 450 píxeles"},
        {"campo": "Variable objetivo", "valor": "Diagnóstico en 7 clases: " + ", ".join(D.class_names(cfg))},
        {"campo": "Etiquetado", "valor": f"Histopatología ({fmt('histo')}), seguimiento ({fmt('follow_up')}), "
                                         f"consenso ({fmt('consensus')}) y confocal ({fmt('confocal')})"},
        {"campo": "Recolección", "valor": "Dos centros (Viena y Queensland) y cuatro fuentes de adquisición"},
        {"campo": "Cobertura", "valor": "Poblaciones mayoritariamente de piel clara; fototipo no registrado"},
        {"campo": "Unidad de independencia", "valor": "La lesión (lesion_id); no hay identificador de paciente"},
        {"campo": "Licencia", "valor": "CC BY-NC 4.0 (Tschandl, Rosendahl y Kittler, 2018)"},
    ]


def export_examples(cfg: dict, ecfg: dict, out: Path) -> list[dict]:
    """Galería de predicción: imágenes de test (HAM10000) por clase, a 224 x 224."""
    e = ecfg["dashboard"]["examples"]
    splits = D.load_splits(cfg["paths"]["splits_csv"])
    test = splits[splits["split"] == "test"]
    rng = np.random.default_rng(int(e["seed"]))
    folder = out / "examples"
    if folder.exists():
        shutil.rmtree(folder)
    rows = []
    for dx in D.class_names(cfg):
        ids = np.sort(test[test["dx"] == dx]["image_id"].to_numpy())
        for i in np.sort(rng.choice(ids, size=min(int(e["per_class"]), len(ids)), replace=False)):
            src = D.image_path(cfg, i)
            if not src.exists():
                continue
            from PIL import Image

            with Image.open(src) as im:
                im = im.convert("RGB").resize((int(e["size"]), int(e["size"])), Image.Resampling.BILINEAR)
                folder.mkdir(parents=True, exist_ok=True)
                im.save(folder / f"{i}.jpg", "JPEG", quality=int(e["quality"]))
            rows.append({"image_id": str(i), "label": dx, "file": f"{i}.jpg"})
    write_json(rows, out / "examples" / "examples.json")
    return rows


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


def all_models(cfg: dict) -> list[str]:
    from src.benchmark import e2_groups
    from src.models import REGISTRY

    e3 = cfg.get("e3", {})
    extra = [e3.get("original_model")] + list(e3.get("ablation_models", []))
    return [m for m in e2_groups(cfg)["all_models"] + [x for x in extra if x] if m in REGISTRY or m in
            cfg["e2"].get("baselines", [])]


def load_master_table(cfg: dict, ecfg: dict):
    """Tabla maestra (formato largo de src.benchmark.main_table). Se recalcula si hay corridas
    definitivas y se guarda en dashboard.master_table; si no hay corridas, se lee la guardada."""
    from src import benchmark as B
    from src import calibration as C
    from src.final import final_label

    path = D.PROJECT_ROOT / ecfg["dashboard"]["master_table"]
    rs = B.load_runs(cfg, final_label(cfg), models=all_models(cfg))
    if rs.models:
        extra = C.probability_metrics_by_run(rs, cfg, rs.models, D.class_names(cfg), "test")
        if not is_missing(extra):
            rs.metrics = pd.concat([rs.metrics, extra], ignore_index=True)
        table = B.main_table(rs, cfg)
        if not is_missing(table):
            path.parent.mkdir(parents=True, exist_ok=True)
            table.reset_index().to_csv(path, index=False)
            return table.reset_index(), rs
    if path.exists():
        return pd.read_csv(path), rs
    return None, rs


def leaderboard_from_master(master: pd.DataFrame, cfg: dict, cost: pd.DataFrame | None = None,
                            latency: dict | None = None) -> pd.DataFrame:
    """Una fila por modelo con las cifras de la tabla maestra (sin redondear)."""
    from src.export import family_of

    primary = cfg["benchmark"]["primary_metric"]
    rows = []
    for m, g in master.groupby("modelo", sort=False):
        g = g.set_index("metrica")
        if primary not in g.index:
            continue
        row = {"modelo": m, "familia": family_of(cfg, m), "corridas": int(g.loc[primary, "corridas"]),
               "ba_media": float(g.loc[primary, "media"]), "ba_std": float(g.loc[primary, "std"]),
               "ece_media": float(g.loc["ece", "media"]) if "ece" in g.index else np.nan,
               "params_total": np.nan, "gflops": np.nan, "latencia_ms": np.nan}
        if cost is not None and m in cost.index:
            for col, key in (("params_total", "params_total"), ("gflops", "gflops"), ("latency_ms_per_image", "latencia_ms")):
                if col in cost.columns:
                    row[key] = float(cost.loc[m, col])
        if latency and m in latency:
            row["latencia_ms"] = float(latency[m])  # ONNX Runtime en CPU (mediana), como en el despliegue
        rows.append(row)
    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS).sort_values("ba_media", ascending=False)


def pareto_front(df: pd.DataFrame, score: str, cost: str) -> list[bool]:
    """Ningún otro punto tiene desempeño >= con costo <= (y uno de los dos estrictamente)."""
    front = []
    for i, r in df.iterrows():
        dominated = any((o[score] >= r[score]) and (o[cost] <= r[cost]) and ((o[score] > r[score]) or (o[cost] < r[cost]))
                        for j, o in df.iterrows() if j != i)
        front.append(not dominated)
    return front


def global_comparison(rs, cfg: dict, board: pd.DataFrame) -> dict:
    from src import benchmark as B
    from src import stats as S

    out = {"cd": None, "pareto": None}
    b = cfg["benchmark"]
    if len(rs.models) >= 3:
        w = B.wide(rs, b["split"], b["primary_metric"])
        if len(w) >= 2:
            ranks = S.mean_ranks(w)
            n, k = w.shape
            out["cd"] = {"ranks": ranks.to_dict(), "cd": S.critical_difference(k, n, float(b["alpha"])),
                         "n_runs": n, "k": k, "alpha": float(b["alpha"]), "friedman": S.friedman_test(w),
                         "note": "rango 1 = peor, k = mejor (S.mean_ranks)"}
    for c in ("gflops", "latencia_ms", "params_total"):
        sub = board.dropna(subset=["ba_media", c]).reset_index(drop=True)
        if len(sub) >= 2:
            sub = sub.assign(frontera=pareto_front(sub, "ba_media", c))
            out.setdefault("pareto_by", {})[c] = sub[["modelo", "familia", "ba_media", c, "frontera"]].to_dict("records")
    out["pareto"] = out.get("pareto_by")
    return out


def diagnostics(rs, model: str, classes, n_bins: int = 15, n_points: int = 60) -> dict | None:
    """Confusión normalizada, ROC y PR por clase (uno contra el resto) y fiabilidad, con las
    predicciones de test de todas las corridas definitivas del modelo (agrupadas)."""
    from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

    frames = [rs.predictions(model, s, f, "test") for (s, f), d in rs.runs.get(model, {}).items()
              if (Path(d) / "probabilities_test.csv").exists()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    p = df[[f"p_{c}" for c in classes]].to_numpy(np.float64)
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    roc, pr = {}, {}
    for k, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y == k, p[:, k])
        prec, rec, _ = precision_recall_curve(y == k, p[:, k])
        idx = np.unique(np.linspace(0, len(fpr) - 1, min(n_points, len(fpr))).astype(int))
        jdx = np.unique(np.linspace(0, len(rec) - 1, min(n_points, len(rec))).astype(int))
        roc[c] = {"fpr": fpr[idx], "tpr": tpr[idx], "auc": auc(fpr, tpr)}
        pr[c] = {"recall": rec[jdx], "precision": prec[jdx], "ap": float(np.mean(prec[:-1] * -np.diff(rec))) if len(rec) > 1 else None}
    conf = p.max(1)
    correct = pred == y
    bins = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    rel = [{"bin": int(b), "confianza": float(conf[which == b].mean()), "exactitud": float(correct[which == b].mean()),
            "n": int((which == b).sum())} for b in range(n_bins) if (which == b).any()]
    return {"classes": classes, "n_predictions": len(y), "n_runs": len(frames),
            "confusion": cm.tolist(), "confusion_norm": (cm / np.maximum(cm.sum(1, keepdims=True), 1)).tolist(),
            "roc": roc, "pr": pr, "reliability": rel}


def precompute_examples(model_dir: Path, onnx_file: Path, meta: dict, examples: list[dict], out: Path) -> dict:
    """Predicciones de la galería para los modelos que no se sirven en vivo (live: false)."""
    from PIL import Image

    from src.export import onnx_predict, onnx_session

    sess = onnx_session(onnx_file)
    res = {}
    for ex in examples:
        img = np.asarray(Image.open(out / "examples" / ex["file"]).convert("RGB"), dtype=np.float32)[None]
        o = onnx_predict(sess, img)
        entry = {"logits": o["logits"][0]}
        if "feature_map" in o:
            entry["cam"] = np.einsum("hwc,ck->khw", o["feature_map"][0], o["head_weights"]).round(4)
        if "embedding" in o:
            entry["embedding"] = o["embedding"][0].astype(np.float32)
        res[ex["image_id"]] = entry
    return res


def export_models(cfg: dict, ecfg: dict, out: Path, examples: list[dict]) -> dict:
    from src.cost import load_cost
    from src.export import METADATA_FILE, OOD_FILE, artifacts_dir, family_of

    classes = D.class_names(cfg)
    models_out = out / "models"
    if models_out.exists():
        shutil.rmtree(models_out)
    models_out.mkdir(parents=True)
    master, rs = load_master_table(cfg, ecfg)
    cost = load_cost(cfg, all_models(cfg))
    cost = None if is_missing(cost) else cost

    # Artefactos ONNX
    artifacts, latency = [], {}
    prefer16 = bool(ecfg["dashboard"].get("prefer_fp16", True))
    for m in all_models(cfg):
        adir = artifacts_dir(ecfg, m)
        if not (adir / METADATA_FILE).exists():
            continue
        meta = D.load_json(adir / METADATA_FILE)
        serving = meta.get("serving_file", "model.onnx") if prefer16 else "model.onnx"
        dst = models_out / m
        dst.mkdir(parents=True, exist_ok=True)
        if meta.get("live", False):
            shutil.copy2(adir / serving, dst / "model.onnx")
        elif examples:
            write_json(precompute_examples(dst, adir / serving, meta, examples, out), dst / "precomputed.json")
        if (adir / OOD_FILE).exists():
            shutil.copy2(adir / OOD_FILE, dst / OOD_FILE)
        bench = (meta.get("fp16", {}).get("benchmark") if serving != "model.onnx" else None) or meta.get("benchmark", {})
        meta["dashboard"] = {"serving_file": serving, "latency_ms_median": bench.get("latency_ms_median"),
                             "latency_ms_p95": bench.get("latency_ms_p95"), "peak_rss_mb": bench.get("peak_rss_mb")}
        latency[m] = bench.get("latency_ms_median")
        write_json(meta, dst / METADATA_FILE)
        artifacts.append({"model": m, "family": family_of(cfg, m), "live": bool(meta.get("live")),
                          "cam": bool(meta.get("cam", {}).get("supported")), "size_mb": meta.get("size_mb")})

    board = leaderboard_from_master(master, cfg, cost, latency) if master is not None else None
    if board is not None:
        write_json({"rows": board.to_dict("records"), "primary_metric": cfg["benchmark"]["primary_metric"],
                    "split": cfg["benchmark"]["split"], "source": ecfg["dashboard"]["master_table"]},
                   models_out / "leaderboard.json")
        write_json(global_comparison(rs, cfg, board), models_out / "global.json")
    for m in rs.models:
        diag = diagnostics(rs, m, classes, int(cfg["calibration"]["n_bins"]))
        if diag:
            write_json(diag, models_out / m / "diagnostics.json")
    write_json(original_section(cfg, rs), models_out / "original.json")
    status = "listo" if (board is not None and len(board)) or artifacts else "pendiente"
    index = {"status": status, "models_with_runs": rs.models, "artifacts": artifacts,
             "generated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    write_json(index, models_out / "index.json")
    log.info(f"Modelos: estado {status}; {len(rs.models)} con corridas, {len(artifacts)} con artefactos ONNX")
    return index


def _chapter_pseudocode() -> str | None:
    path = D.PROJECT_ROOT / "book" / "24-modelo-original-diseno.md"
    if not path.exists():
        return None
    m = re.search(r"```text\n(Algoritmo: capa CCI.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)
    return m.group(1).strip() if m else None


def _test_inventory() -> list[dict]:
    """Pruebas de tests/test_original_model.py (nombre y primera línea del docstring) y su
    resultado en outputs/tests/junit.xml si existe (pytest --junitxml)."""
    import ast
    import xml.etree.ElementTree as ET

    path = D.PROJECT_ROOT / "tests" / "test_original_model.py"
    if not path.exists():
        return []
    status = {}
    junit = D.PROJECT_ROOT / "outputs" / "tests" / "junit.xml"
    if junit.exists():
        for case in ET.parse(junit).getroot().iter("testcase"):
            name = case.get("name", "").split("[")[0]
            failed = any(child.tag in ("failure", "error") for child in case)
            status[name] = status.get(name, True) and not failed
    rows = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            doc = (ast.get_docstring(node) or "").split("\n")[0]
            rows.append({"prueba": node.name, "descripcion": doc,
                         "resultado": None if node.name not in status else ("aprobada" if status[node.name] else "fallida")})
    return rows


def original_section(cfg: dict, rs) -> dict:
    """Subsección del modelo original: procedencia, pseudocódigo, demostración, pruebas, robustez y veredicto."""
    import yaml

    with open(D.PROJECT_ROOT / "configs" / "cci.yaml", encoding="utf-8") as f:
        cci = yaml.safe_load(f)
    prov = D.PROJECT_ROOT / "experiments" / "procedencia.md"
    robustness = None
    try:
        from src.robustness import load_results

        res = load_results(cfg, [m for m in rs.models if m.startswith("cci")] + [cci.get("backbone") or ""])
        if not is_missing(res):
            robustness = frame_records(res.reset_index() if not isinstance(res.index, pd.RangeIndex) else res)
    except Exception as e:  # noqa: BLE001 - la robustez es opcional
        log.warning(f"Robustez del modelo original no disponible: {e}")
    return {"backbone": cci.get("backbone"), "variant": cci.get("variant"), "layer": cci.get("cci"),
            "provenance_md": prov.read_text(encoding="utf-8") if prov.exists() else None,
            "pseudocode": _chapter_pseudocode(), "tests": _test_inventory(), "robustness": robustness,
            "has_runs": any(m.startswith("cci") for m in rs.models),
            "verdict": None}


def export_content(out: Path) -> None:
    for name in ("MODEL_CARD.md", "DATA_STATEMENT.md"):
        src = D.PROJECT_ROOT / name
        if src.exists():
            (out / "content").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / "content" / name)


def main(argv=None) -> None:
    from src.export import load_export_config
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Genera dashboard/data/ desde outputs/, artifacts/ y los notebooks.")
    parser.add_argument("--skip-eda", action="store_true", help="no regenera el EDA (más rápido)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--export-config", default=None)
    parser.add_argument("--out", default=None, help="carpeta de salida (por defecto dashboard.out_dir)")
    args = parser.parse_args(argv)
    cfg = D.load_config(args.config)
    ecfg = load_export_config(args.export_config)
    out = Path(args.out) if args.out else D.PROJECT_ROOT / ecfg["dashboard"]["out_dir"]
    setup_logging(cfg["paths"]["logs_dir"] / "export_dashboard.log")
    out.mkdir(parents=True, exist_ok=True)
    sections = {}
    if not args.skip_eda:
        sections["eda"] = bool(export_eda(cfg, ecfg, out))
    examples = export_examples(cfg, ecfg, out)
    sections["examples"] = len(examples)
    sections["models"] = export_models(cfg, ecfg, out, examples)["status"]
    export_content(out)
    write_json({"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "sections": sections,
                "classes": D.class_names(cfg), "sources": list(cfg["valid_categories"]["dataset"])},
               out / "manifest.json")
    log.info(f"Datos del dashboard -> {out}")


if __name__ == "__main__":
    main()
