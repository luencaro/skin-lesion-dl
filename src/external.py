"""Evaluación externa en PAD-UFES-20, Derm7pt y BCN20000 (desviación D5; protocolo, sección 5).

    python -m src.external --model cci [--model ...]     # o --all
    python -m src.external --hashes                       # solo registra o comprueba los SHA-256

Solo inferencia, sin reentrenar ni ajustar: cada corrida definitiva terminada de cada modelo
(outputs/runs/<modelo>/<final.label>/s<semilla>_f<pliegue>/model.weights.h5) predice las imágenes de
cada conjunto de configs/external.yaml, redimensionadas como HAM10000 (bilineal con antialias a
train.image_size, sin conservar la relación de aspecto). Las etiquetas se mapean a las 7 clases de
HAM10000 y se excluyen las imágenes de clases sin equivalente; una etiqueta no declarada o un
archivo de metadatos con otras columnas es un error (ExternalFormatError), no se adivina.

Resultados en outputs/external/<modelo>/<conjunto>/s<semilla>_f<pliegue>.csv (probabilidades por
imagen) y tablas:
- métricas por corrida y conjunto: exactitud balanceada sobre las clases presentes (media de la
  sensibilidad de las clases con al menos una imagen; predecir una clase ausente cuenta como error)
  y secundarias (exactitud, F1 macro y AUC sobre las clases presentes, MCC, ECE, entropía cruzada);
- IC 95 % de la media entre corridas por bootstrap por lesión (PAD-UFES-20, BCN20000) o por caso
  (Derm7pt), con benchmark.lesion_bootstrap;
- Friedman con la corrección de Iman y Davenport sobre los cuatro conjuntos (HAM10000 y los tres
  externos) como bloques, y el modelo original frente a cada modelo con z de rangos y Holm.
Las sumas SHA-256 de los archivos descargados se registran en data/external_hashes.json (la
primera vez) y se comprueban en las ejecuciones siguientes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import data as D
from src import stats as S
from src.report import Missing

log = logging.getLogger(__name__)

DEFAULT_CONFIG = D.PROJECT_ROOT / "configs" / "external.yaml"
HAM = "HAM10000"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ExternalFormatError(ValueError):
    """Los archivos descargados no tienen el formato que declara configs/external.yaml."""


@dataclass
class ExternalSet:
    """Imágenes evaluables de un conjunto externo, tras el mapeo y las exclusiones."""

    key: str                 # clave en configs/external.yaml (p. ej. pad_ufes_20)
    name: str                # nombre para las tablas (p. ej. PAD-UFES-20)
    image_ids: np.ndarray    # identificador de cada imagen (valor de image_column)
    paths: np.ndarray        # ruta absoluta de cada imagen
    labels: np.ndarray       # índice de clase de HAM10000
    groups: np.ndarray       # lesión o caso (unidad del bootstrap)
    excluded: dict           # {etiqueta original: imágenes excluidas}


# ---------------------------------------------------------------------------
# Configuración y metadatos
# ---------------------------------------------------------------------------


def load_external_config(path: str | Path | None = None) -> dict:
    """configs/external.yaml con las rutas resueltas desde la raíz del proyecto."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        ecfg = yaml.safe_load(f)
    for key in ("hashes_json", "results_dir", "cache_dir"):
        ecfg[key] = (D.PROJECT_ROOT / ecfg[key]).resolve()
    for dcfg in ecfg["datasets"].values():
        dcfg["root"] = (D.PROJECT_ROOT / dcfg["root"]).resolve()
    return ecfg


def read_metadata(dcfg: dict) -> pd.DataFrame:
    """Archivo de metadatos del conjunto con las columnas imagen, etiqueta y grupo de la config.
    ExternalFormatError si falta el archivo o alguna columna (con las columnas encontradas)."""
    path = Path(dcfg["root"]) / dcfg["metadata"]
    if not path.exists():
        raise ExternalFormatError(f"{dcfg['name']}: no existe {path}. Descarga el conjunto en {dcfg['root']} "
                                  f"(README.md) o corrige `metadata` en configs/external.yaml.")
    meta = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = [dcfg["image_column"], dcfg["label_column"], dcfg["group_column"]]
    missing = [c for c in need if c not in meta.columns]
    if missing:
        raise ExternalFormatError(f"{dcfg['name']}: {path.name} no tiene las columnas {missing}; columnas "
                                  f"encontradas: {list(meta.columns)}. Corrige configs/external.yaml.")
    empty = [c for c in need if (meta[c].str.strip() == "").any()]
    if empty:
        raise ExternalFormatError(f"{dcfg['name']}: valores vacíos en las columnas {empty} de {path.name}")
    out = pd.DataFrame({"image_id": meta[dcfg["image_column"]].str.strip(),
                        "label": meta[dcfg["label_column"]].str.strip(),
                        "group": meta[dcfg["group_column"]].str.strip()})
    if out["image_id"].duplicated().any():
        dup = out.loc[out["image_id"].duplicated(), "image_id"].iloc[0]
        raise ExternalFormatError(f"{dcfg['name']}: image_id repetido en {path.name}, p. ej. {dup}")
    return out


def apply_mapping(meta: pd.DataFrame, dcfg: dict, classes: list[str]) -> tuple[pd.DataFrame, dict]:
    """(filas evaluables con la columna dx, {etiqueta excluida: imágenes}). ExternalFormatError si
    hay etiquetas que no están en `mapping` ni en `exclude`, o si el mapeo apunta a una clase que no
    es de HAM10000."""
    mapping, exclude = dict(dcfg["mapping"]), set(dcfg.get("exclude") or [])
    bad_targets = sorted(set(mapping.values()) - set(classes))
    if bad_targets:
        raise ExternalFormatError(f"{dcfg['name']}: el mapeo apunta a clases que no son de HAM10000: {bad_targets}")
    overlap = sorted(set(mapping) & exclude)
    if overlap:
        raise ExternalFormatError(f"{dcfg['name']}: etiquetas a la vez mapeadas y excluidas: {overlap}")
    unknown = sorted(set(meta["label"]) - set(mapping) - exclude)
    if unknown:
        raise ExternalFormatError(f"{dcfg['name']}: etiquetas sin mapeo ni exclusión declarados: {unknown}. "
                                  f"Etiquetas encontradas: {sorted(set(meta['label']))}")
    is_excluded = meta["label"].isin(exclude)
    excluded = meta.loc[is_excluded, "label"].value_counts().sort_index().to_dict()
    kept = meta[~is_excluded].copy()
    kept["dx"] = kept["label"].map(mapping)
    return kept.reset_index(drop=True), {k: int(v) for k, v in excluded.items()}


def resolve_image_paths(image_ids, dcfg: dict) -> np.ndarray:
    """Ruta de cada imagen: `image_dir`/<image_id> (image_search: direct) o el archivo con ese nombre
    en `image_dir` y sus subcarpetas (recursive; se añade la extensión si el id no la trae).
    ExternalFormatError si falta alguna o si un nombre aparece en varias carpetas."""
    base = Path(dcfg["root"]) / dcfg.get("image_dir", ".")
    ids = [str(i) for i in image_ids]
    if dcfg.get("image_search", "direct") == "direct":
        paths = [base / i for i in ids]
    else:
        index: dict[str, list[Path]] = {}
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(p.name, []).append(p)
                index.setdefault(p.stem, []).append(p)
        repeated = [i for i in ids if len(index.get(i, [])) > 1]
        if repeated:
            raise ExternalFormatError(f"{dcfg['name']}: {len(repeated)} imágenes con el mismo nombre en varias "
                                      f"carpetas de {base}, p. ej. {repeated[0]}: {index[repeated[0]]}")
        paths = [index[i][0] if i in index else base / i for i in ids]
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise ExternalFormatError(f"{dcfg['name']}: faltan {len(missing)} de {len(ids)} imágenes, p. ej. "
                                  f"{missing[0]}. Comprueba la descarga y `image_dir` en configs/external.yaml.")
    return np.array([str(p) for p in paths])


def load_external_set(key: str, ecfg: dict, classes: list[str]) -> ExternalSet:
    dcfg = ecfg["datasets"][key]
    kept, excluded = apply_mapping(read_metadata(dcfg), dcfg, classes)
    paths = resolve_image_paths(kept["image_id"], dcfg)
    labels = kept["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    return ExternalSet(key, dcfg["name"], kept["image_id"].to_numpy(), paths, labels,
                       kept["group"].to_numpy(), excluded)


def composition_table(sets: list[ExternalSet], classes: list[str]) -> pd.DataFrame:
    """Imágenes evaluables por clase, grupos (lesiones o casos) y excluidas de cada conjunto."""
    rows = {}
    for s in sets:
        counts = np.bincount(s.labels, minlength=len(classes))
        rows[s.name] = {**dict(zip(classes, counts.tolist())), "imagenes": len(s.labels),
                        "grupos": len(np.unique(s.groups)), "excluidas": sum(s.excluded.values()),
                        "excluidas_por_etiqueta": s.excluded}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Hashes de los archivos descargados
# ---------------------------------------------------------------------------


def dataset_file_hashes(dcfg: dict) -> dict:
    """{ruta relativa a la raíz del proyecto: SHA-256} de todos los archivos de `root`."""
    root = Path(dcfg["root"])
    if not root.exists():
        raise ExternalFormatError(f"{dcfg['name']}: no existe {root} (descarga el conjunto, README.md)")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    hashes = D._hash_files(files)
    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(D.PROJECT_ROOT))
        except ValueError:  # fuera del proyecto (p. ej. en las pruebas)
            return str(p)

    return {rel(p): h for p, h in zip(files, hashes)}


def record_or_check_hashes(ecfg: dict, keys) -> dict:
    """Registra en hashes_json los SHA-256 de los conjuntos `keys` que aún no estén y comprueba los
    ya registrados. RuntimeError si un archivo cambió, falta o aparece uno nuevo."""
    path = Path(ecfg["hashes_json"])
    registry = D.load_json(path) if path.exists() else {}
    for key in keys:
        current = dataset_file_hashes(ecfg["datasets"][key])
        if key not in registry:
            registry[key] = current
            log.info(f"{ecfg['datasets'][key]['name']}: {len(current)} archivos registrados en {path.name}")
            continue
        old = registry[key]
        changed = sorted(f for f in set(old) & set(current) if old[f] != current[f])
        gone, new = sorted(set(old) - set(current)), sorted(set(current) - set(old))
        if changed or gone or new:
            raise RuntimeError(f"{ecfg['datasets'][key]['name']}: los archivos no coinciden con {path}: "
                               f"{len(changed)} cambiados, {len(gone)} faltantes y {len(new)} nuevos "
                               f"(p. ej. {(changed or gone or new)[0]})")
    D.save_json(registry, path)
    return registry


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def balanced_accuracy_present(y_true, y_pred) -> float:
    """Media de la sensibilidad de las clases presentes en y_true; una predicción de una clase
    ausente cuenta como error de la clase real."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean([(y_pred[y_true == k] == k).mean() for k in np.unique(y_true)]))


def external_metrics(y_true, proba, classes: list[str], ece_bins: int) -> dict:
    """Métricas de un conjunto externo: exactitud balanceada sobre las clases presentes, y F1 macro,
    AUC-ROC macro uno contra el resto y AUC-PR macro sobre las clases presentes; exactitud, MCC,
    ECE y entropía cruzada de src.metrics.classification_metrics (sobre las 7 clases)."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    from src.metrics import classification_metrics

    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=np.float64)
    pred = p.argmax(axis=1)
    present = np.unique(y)
    base = classification_metrics(y, p, classes, ece_bins)
    out = {k: base[k] for k in ("n", "accuracy", "mcc", "ece", "cross_entropy")}
    out["balanced_accuracy"] = balanced_accuracy_present(y, pred)
    out["clases_presentes"] = len(present)
    out["f1_macro"] = float(f1_score(y, pred, labels=present, average="macro", zero_division=0))
    onehot = (y[:, None] == present[None, :]).astype(int)
    scores = p[:, present]
    out["auc_roc_macro_ovr"] = (float(np.mean([roc_auc_score(onehot[:, j], scores[:, j])
                                               for j in range(len(present))])) if len(present) > 1 else np.nan)
    out["auc_pr_macro"] = float(average_precision_score(onehot, scores, average="macro")) if len(present) > 1 else np.nan
    out["prediccion_clase_ausente"] = float(np.isin(pred, present, invert=True).mean())
    return out


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------


def _cache_key(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__")


def cached_images(cfg: dict, ecfg: dict, s: ExternalSet) -> list[str]:
    """Rutas de la caché en float32 de las imágenes de `s`, redimensionadas como HAM10000."""
    keys = [_cache_key(i) for i in s.image_ids]
    return D.ensure_resized_cache(cfg, keys, verify=True, sources=s.paths, out_dir=Path(ecfg["cache_dir"]) / s.key)


def prediction_path(ecfg: dict, model_name: str, key: str, seed: int, fold: int) -> Path:
    return Path(ecfg["results_dir"]) / model_name / key / f"s{seed}_f{fold}.csv"


def predict_run(cfg: dict, model_name: str, run_dir: Path, files: dict, sets: dict) -> dict:
    """{clave del conjunto: probabilidades (N, 7)} con los pesos de `run_dir`."""
    import keras
    import tensorflow as tf

    from src.gradcam import load_trained_model

    model = load_trained_model(cfg, model_name, run_dir)
    read = D._raw_image_loader(int(cfg["train"]["image_size"]))
    bs = int(cfg["train"]["batch_size"])
    out = {}
    for key, paths in files.items():
        ds = tf.data.Dataset.from_tensor_slices(paths).map(read, num_parallel_calls=tf.data.AUTOTUNE).batch(bs)
        out[key] = np.asarray(model.predict(ds.prefetch(tf.data.AUTOTUNE), verbose=0), dtype=np.float64)
        if len(out[key]) != len(sets[key].labels):
            raise RuntimeError(f"{model_name}: {len(out[key])} predicciones para {len(sets[key].labels)} imágenes")
    del model
    keras.backend.clear_session()
    return out


def run_external(cfg: dict, ecfg: dict, model_name: str, keys=None, overwrite: bool = False) -> pd.DataFrame | Missing:
    """Predice los conjuntos `keys` (todos por defecto) con cada corrida definitiva terminada de
    `model_name` que tenga pesos; reutiliza las predicciones guardadas. Devuelve las métricas."""
    from src.final import finished_runs
    from src.gradcam import WEIGHTS_FILE

    keys = list(keys or ecfg["datasets"])
    runs = finished_runs(cfg, model_name)
    if isinstance(runs, Missing):
        return runs
    runs = [d for d in runs if (Path(d) / WEIGHTS_FILE).exists()]
    if not runs:
        return Missing(f"las corridas definitivas de {model_name} no tienen {WEIGHTS_FILE}")
    classes = D.class_names(cfg)
    sets = {k: load_external_set(k, ecfg, classes) for k in keys}
    record_or_check_hashes(ecfg, keys)
    files = {k: cached_images(cfg, ecfg, sets[k]) for k in keys}
    for run_dir in runs:
        info = D.load_json(Path(run_dir) / "run_info.json")
        seed, fold = int(info["seed"]), int(info["val_fold"])
        todo = [k for k in keys if overwrite or not prediction_path(ecfg, model_name, k, seed, fold).exists()]
        if not todo:
            continue
        probas = predict_run(cfg, model_name, Path(run_dir), {k: files[k] for k in todo}, sets)
        for k, p in probas.items():
            s = sets[k]
            df = pd.DataFrame({"image_id": s.image_ids, "group": s.groups, "label": [classes[i] for i in s.labels],
                               "pred": [classes[i] for i in p.argmax(axis=1)]})
            for i, c in enumerate(classes):
                df[f"p_{c}"] = p[:, i]
            path = prediction_path(ecfg, model_name, k, seed, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            log.info(f"{model_name} s{seed}_f{fold} | {s.name}: bal_acc "
                     f"{balanced_accuracy_present(s.labels, p.argmax(axis=1)):.4f} ({len(s.labels)} imágenes)")
    return metrics_table(cfg, ecfg, [model_name], keys)


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------


def load_predictions(cfg: dict, ecfg: dict, model_name: str, key: str) -> dict:
    """{(semilla, pliegue): DataFrame de probabilidades} guardadas de un modelo y un conjunto."""
    out = {}
    for path in sorted((Path(ecfg["results_dir"]) / model_name / key).glob("s*_f*.csv")):
        seed, fold = (int(x[1:]) for x in path.stem.split("_"))
        out[(seed, fold)] = pd.read_csv(path, dtype={"image_id": str, "group": str})
    return out


def _arrays(df: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {c: i for i, c in enumerate(classes)}
    return df["label"].map(idx).to_numpy(), df[[f"p_{c}" for c in classes]].to_numpy(dtype=np.float64)


def metrics_table(cfg: dict, ecfg: dict, models, keys=None) -> pd.DataFrame | Missing:
    """Una fila por modelo, conjunto y corrida con external_metrics."""
    classes, bins = D.class_names(cfg), int(cfg["train"]["ece_bins"])
    rows = []
    for m in models:
        for k in keys or ecfg["datasets"]:
            for (seed, fold), df in load_predictions(cfg, ecfg, m, k).items():
                y, p = _arrays(df, classes)
                rows.append({"modelo": m, "conjunto": ecfg["datasets"][k]["name"], "seed": seed, "val_fold": fold,
                             **external_metrics(y, p, classes, bins)})
    if not rows:
        return Missing("no hay predicciones externas guardadas (python -m src.external --model ...).")
    return pd.DataFrame(rows)


def bootstrap_table(cfg: dict, ecfg: dict, models, keys=None) -> pd.DataFrame | Missing:
    """Por modelo y conjunto: media entre corridas de la exactitud balanceada (clases presentes) e
    IC 95 % por bootstrap por lesión o por caso (`group_column`), con benchmark.lesion_bootstrap.
    Todos los modelos usan los mismos remuestreos de cada conjunto."""
    classes = D.class_names(cfg)
    lb = cfg["benchmark"]["lesion_bootstrap"]
    alpha = float(cfg["benchmark"]["alpha"])
    rows = []
    for k in keys or ecfg["datasets"]:
        correct, ref = {}, None
        for m in models:
            preds = load_predictions(cfg, ecfg, m, k)
            if not preds:
                continue
            frames = list(preds.values())
            if ref is None:
                ref = frames[0][["image_id", "group", "label"]]
            for df in frames:
                if not df["image_id"].equals(ref["image_id"]):
                    raise RuntimeError(f"{m} / {k}: las predicciones no tienen las mismas imágenes en el mismo orden")
            correct[m] = np.stack([_arrays(df, classes)[1].argmax(axis=1) == _arrays(df, classes)[0] for df in frames])
        if not correct:
            continue
        y = ref["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        boot = S.cluster_bootstrap_balanced_accuracy(y, ref["group"].to_numpy(), correct, int(lb["n_resamples"]),
                                                     int(lb["seed"]))
        for m, r in boot.items():
            lo, hi = S.percentile_interval(r["remuestreos"], alpha)
            rows.append({"modelo": m, "conjunto": ecfg["datasets"][k]["name"], "corridas": len(correct[m]),
                         "balanced_accuracy": r["observado"], "ic95_bajo": lo, "ic95_alto": hi,
                         "unidad_bootstrap": ecfg["datasets"][k]["group_column"], "grupos": ref["group"].nunique()})
    if not rows:
        return Missing("no hay predicciones externas guardadas para el bootstrap.")
    return pd.DataFrame(rows)


def ham10000_scores(cfg: dict, models, split: str = "test") -> pd.Series:
    """Media entre corridas definitivas de la exactitud balanceada de HAM10000 en `split`."""
    from src.final import finished_runs

    out = {}
    for m in models:
        runs = finished_runs(cfg, m)
        if isinstance(runs, Missing):
            continue
        vals = [D.load_json(Path(d) / "metrics.json").get(split, {}).get("balanced_accuracy") for d in runs]
        vals = [v for v in vals if v is not None]
        if vals:
            out[m] = float(np.mean(vals))
    return pd.Series(out, name=HAM, dtype=float)


def friedman_control(scores: pd.DataFrame, control: str, alpha: float = 0.05) -> dict:
    """Friedman con Iman y Davenport sobre `scores` (filas = conjuntos como bloques, columnas =
    modelos; mayor es mejor) y el modelo `control` frente a cada uno de los demás:
    z = (R_j - R_control) / sqrt(k (k + 1) / (6 n)) con los rangos medios (1 = mejor), p bilateral
    y Holm sobre los k - 1 contrastes (Demšar, 2006). Solo los conjuntos con todos los modelos."""
    import scipy.stats as ss

    w = scores.dropna()
    if control not in w.columns:
        raise ValueError(f"{control} no está entre los modelos: {list(w.columns)}")
    n, k = w.shape
    fr = S.friedman_test(w)
    idv = S.iman_davenport(fr["chi2"], n, k)
    ranks = w.rank(axis=1, ascending=False).mean()  # 1 = mejor exactitud balanceada
    se = np.sqrt(k * (k + 1) / (6 * n)) if n else np.nan
    rows = []
    for m in w.columns:
        if m == control:
            continue
        z = (ranks[m] - ranks[control]) / se
        rows.append({"modelo": m, "rango_medio": float(ranks[m]), "rango_medio_control": float(ranks[control]),
                     "z": float(z), "p": float(2 * ss.norm.sf(abs(z)))})
    comp = pd.DataFrame(rows)
    if len(comp):
        comp["p_holm"] = S.adjust_pvalues(comp["p"], "holm")
        comp["significativo"] = comp["p_holm"] < alpha
    return {"conjuntos": list(w.index), "n_conjuntos": n, "k_modelos": k, "friedman": fr,
            "iman_davenport": idv, "comparaciones": comp}


def friedman_table(cfg: dict, ecfg: dict, models, control: str | None = None) -> dict | Missing:
    """Exactitud balanceada media por conjunto (HAM10000 en ecfg.ham10000_split y los externos) y
    modelo, y friedman_control con el modelo original (e3.original_model) como control."""
    control = control or cfg.get("e3", {}).get("original_model")
    ext = metrics_table(cfg, ecfg, models)
    if isinstance(ext, Missing):
        return ext
    wide = ext.pivot_table(index="conjunto", columns="modelo", values="balanced_accuracy", aggfunc="mean")
    ham = ham10000_scores(cfg, models, ecfg.get("ham10000_split", "test"))
    scores = pd.concat([ham.to_frame().T, wide]).reindex(columns=list(models))
    if control not in scores.columns or scores[control].isna().all():
        return Missing(f"no hay resultados del modelo original ({control}) en los conjuntos.")
    return {"exactitud_balanceada": scores, **friedman_control(scores, control, float(cfg["benchmark"]["alpha"]))}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_models(cfg: dict) -> list[str]:
    """Modelos profundos del Entregable 2 y el modelo original del Entregable 3."""
    from src.benchmark import e2_groups

    models = list(e2_groups(cfg)["deep_models"])
    original = cfg.get("e3", {}).get("original_model")
    return models + ([original] if original and original not in models else [])


def main(argv=None) -> None:
    from src.models import REGISTRY
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Evaluación externa (solo inferencia, desviación D5).")
    parser.add_argument("--model", action="append", choices=sorted(REGISTRY), default=None)
    parser.add_argument("--all", action="store_true", help="modelos profundos del E2 y el modelo original")
    parser.add_argument("--dataset", action="append", default=None, help="clave de configs/external.yaml")
    parser.add_argument("--hashes", action="store_true", help="solo registra o comprueba los SHA-256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default=None, help="YAML base (por defecto configs/base.yaml)")
    parser.add_argument("--external-config", default=None, help="por defecto configs/external.yaml")
    args = parser.parse_args(argv)
    cfg = D.load_config(args.config)
    ecfg = load_external_config(args.external_config)
    keys = args.dataset or list(ecfg["datasets"])
    unknown = sorted(set(keys) - set(ecfg["datasets"]))
    if unknown:
        sys.exit(f"Conjuntos desconocidos: {unknown}; disponibles: {list(ecfg['datasets'])}")
    setup_logging(cfg["paths"]["logs_dir"] / "external.log")
    try:
        if args.hashes:
            record_or_check_hashes(ecfg, keys)
            sets = [load_external_set(k, ecfg, D.class_names(cfg)) for k in keys]
            log.info("Composición:\n" + composition_table(sets, D.class_names(cfg)).to_string())
            return
        models = list(args.model or []) + (default_models(cfg) if args.all else [])
        if not models:
            sys.exit("Indica --model, --all o --hashes")
        for m in dict.fromkeys(models):
            res = run_external(cfg, ecfg, m, keys, args.overwrite)
            log.info(f"{m}: {res.message if isinstance(res, Missing) else f'{len(res)} filas de métricas'}")
    except ExternalFormatError as e:
        sys.exit(f"Error de formato: {e}")


if __name__ == "__main__":
    main()
