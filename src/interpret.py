"""Interpretabilidad de los mejores modelos: mapas de saliencia, borrado e inserción y estabilidad.

    python -m src.interpret [--model ...]     # por defecto, los interpretability.top_k mejores

- Mapas: Grad-CAM (src/gradcam.py) para las CNN, sobre gradcam_layer y respecto de la clase
  predicha; attention rollout (Abnar y Zuidema, 2020) para ViT: producto por bloques de
  0,5 A + 0,5 I (A = atención fusionada entre cabezas con interpretability.rollout_head_fusion,
  filas normalizadas), y la fila del token de clase sobre los 14 x 14 parches. El rollout no
  depende de la clase. Saliencia por modelo según `saliency` de configs/<modelo>.yaml.
- Borrado e inserción (Petsiuk et al., 2018): se ordenan los píxeles por saliencia y, en
  interpretability.steps pasos, se sustituye la fracción más saliente por la imagen de referencia
  (borrado) o se parte de la referencia y se insertan los píxeles originales (inserción). La
  referencia es la imagen desenfocada (interpretability.baseline = blur) o su media. Se registra la
  probabilidad de la clase predicha sin perturbar y su área bajo la curva (AUC); un orden aleatorio
  sirve de control. Un buen mapa tiene AUC de borrado baja y de inserción alta.
- Estabilidad entre semillas: correlación de Pearson y de Spearman entre los mapas (reducidos a
  interpretability.stability_size) de las corridas de interpretability.stability_runs (mismo pliegue, distinta semilla).

Muestra: la de evaluation.gradcam (aciertos y errores por clase de la corrida interpretability.run
del mejor modelo), común a todos los modelos. Resultados en outputs/interpretability/<modelo>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src.report import Missing, is_missing, tolerant

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention rollout (ViT de keras_hub)
# ---------------------------------------------------------------------------


def vit_attention_maps(model, images: np.ndarray):
    """Tensores de atención (por bloque, (B, cabezas, T, T)) y salida del encoder, recorriendo a
    mano los bloques de ViTEncoder con return_attention_scores=True."""
    import tensorflow as tf

    from src.models.common import apply_pre_backbone, get_backbone
    from src.models.vit_b16 import EMBEDDING_LAYER, ENCODER_LAYER

    backbone = get_backbone(model)
    embed = backbone.get_layer(EMBEDDING_LAYER)
    encoder = backbone.get_layer(ENCODER_LAYER)
    x = embed(apply_pre_backbone(model, tf.convert_to_tensor(images, tf.float32)))
    scores = []
    for block in encoder.encoder_layers:
        h = block.layer_norm_1(x)
        att, s = block.mha(h, h, return_attention_scores=True, training=False)
        x = att + x
        x = x + block.mlp(block.layer_norm_2(x), training=False)
        scores.append(s)
    return scores, encoder.layer_norm(x)


def attention_rollout(model, images: np.ndarray, head_fusion: str = "mean", discard_ratio: float = 0.0,
                      batch_size: int = 16) -> np.ndarray:
    """Mapas (N, H, W) en [0, 1] del attention rollout del token de clase, redimensionados a la entrada."""
    import tensorflow as tf

    size = images.shape[1:3]
    maps = []
    for start in range(0, len(images), batch_size):
        scores, _ = vit_attention_maps(model, images[start:start + batch_size])
        rollout = None
        for s in scores:
            a = tf.cast(s, tf.float32)
            a = {"mean": tf.reduce_mean, "max": tf.reduce_max, "min": tf.reduce_min}[head_fusion](a, axis=1)
            a = a.numpy()
            if discard_ratio > 0:
                flat = a.reshape(a.shape[0], -1)
                k = int(flat.shape[1] * discard_ratio)
                idx = np.argsort(flat, axis=1)[:, :k]
                for i in range(len(flat)):
                    keep_cls = idx[i] != 0  # no se descarta la autoatención del token de clase
                    flat[i, idx[i][keep_cls]] = 0
                a = flat.reshape(a.shape)
            t = a.shape[-1]
            a = 0.5 * a + 0.5 * np.eye(t)[None]
            a = a / a.sum(axis=-1, keepdims=True)
            rollout = a if rollout is None else a @ rollout
        cls = rollout[:, 0, 1:]
        g = round(np.sqrt(cls.shape[1]))
        m = cls.reshape(-1, g, g)
        m = m / np.maximum(m.max(axis=(1, 2), keepdims=True), 1e-12)
        maps.append(tf.image.resize(m[..., None], size, method="bilinear")[..., 0].numpy())
    return np.concatenate(maps)


# ---------------------------------------------------------------------------
# Mapas de saliencia de cualquier modelo
# ---------------------------------------------------------------------------


def saliency_maps(cfg: dict, model_name: str, model, images: np.ndarray, class_idx) -> tuple[np.ndarray, str]:
    """(mapas (N, H, W) en [0, 1], método) según `saliency` de configs/<modelo>.yaml."""
    from src.gradcam import gradcam_heatmaps
    from src.train import load_model_config

    mcfg = load_model_config(model_name)
    icfg = cfg["interpretability"]
    if mcfg.get("saliency", "gradcam") == "rollout":
        return attention_rollout(model, images, icfg["rollout_head_fusion"], float(icfg["rollout_discard_ratio"]),
                                 int(icfg["batch_size"])), "attention rollout"
    layer = mcfg.get("gradcam_layer")
    if not layer:
        raise ValueError(f"configs/{model_name}.yaml no define gradcam_layer")
    return gradcam_heatmaps(model, images, layer, class_idx=class_idx, batch_size=int(icfg["batch_size"])), \
        f"Grad-CAM ({layer})"


@tolerant()
def compute_saliency(cfg: dict, model_name: str, run_dir: Path, sample: pd.DataFrame) -> dict:
    """Como gradcam.compute_gradcam, pero con el método de saliencia del modelo (Grad-CAM o
    attention rollout): imágenes de `sample`, mapas respecto de la clase predicha y método.
    Restaura la política de precisión de train.mixed_precision al terminar."""
    import keras

    from src.gradcam import load_model_float32
    from src.train import configure_runtime

    classes = D.class_names(cfg)
    images = D.load_resized_images(cfg, sample["image_id"])
    try:
        model = load_model_float32(cfg, model_name, run_dir)
        maps, method = saliency_maps(cfg, model_name, model, images, [classes.index(p) for p in sample["pred"]])
    finally:
        keras.backend.clear_session()
        configure_runtime(cfg)
    return {"images": images, "heatmaps": maps, "layer": method, "sample": sample.reset_index(drop=True)}


# ---------------------------------------------------------------------------
# Borrado e inserción
# ---------------------------------------------------------------------------


def reference_image(img: np.ndarray, kind: str, sigma: float) -> np.ndarray:
    if kind == "mean":
        return np.broadcast_to(img.mean(axis=(0, 1), keepdims=True), img.shape).astype(np.float32)
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(img.astype(np.float32), sigma=(sigma, sigma, 0))


def deletion_insertion_curves(model, image: np.ndarray, saliency: np.ndarray, target: int, steps: int,
                              baseline: np.ndarray, order: np.ndarray | None = None) -> dict:
    """Probabilidad de `target` al borrar e insertar la fracción t / steps de píxeles más salientes
    (o en `order`), t = 0..steps. Devuelve las curvas y sus AUC (regla del trapecio en [0, 1])."""
    h, w_ = saliency.shape
    order = np.argsort(-saliency.ravel(), kind="mergesort") if order is None else order
    n = h * w_
    batch_del, batch_ins = [], []
    for t in range(steps + 1):
        k = round(n * t / steps)
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        mask = mask.reshape(h, w_)[..., None]
        batch_del.append(np.where(mask, baseline, image))
        batch_ins.append(np.where(mask, image, baseline))
    p_del = np.asarray(model.predict_on_batch(np.stack(batch_del).astype(np.float32)))[:, target]
    p_ins = np.asarray(model.predict_on_batch(np.stack(batch_ins).astype(np.float32)))[:, target]
    x = np.linspace(0, 1, steps + 1)
    return {"borrado": p_del, "insercion": p_ins, "auc_borrado": float(np.trapezoid(p_del, x)),
            "auc_insercion": float(np.trapezoid(p_ins, x))}


def deletion_insertion_table(cfg: dict, model, images: np.ndarray, maps: np.ndarray, targets, ids) -> pd.DataFrame:
    """AUC de borrado e inserción por imagen con el mapa y con un orden aleatorio (control)."""
    icfg = cfg["interpretability"]
    steps, kind, sigma = int(icfg["steps"]), icfg["baseline"], float(icfg["blur_sigma"])
    rng = np.random.default_rng(int(cfg["evaluation"]["gradcam"]["sample_seed"]))
    rows = []
    for img, sal, t, iid in zip(images, maps, targets, ids):
        base = reference_image(img, kind, sigma)
        res = deletion_insertion_curves(model, img, sal, int(t), steps, base)
        rnd = deletion_insertion_curves(model, img, sal, int(t), steps, base, rng.permutation(sal.size))
        rows.append({"image_id": iid, "auc_borrado": res["auc_borrado"], "auc_insercion": res["auc_insercion"],
                     "auc_borrado_aleatorio": rnd["auc_borrado"], "auc_insercion_aleatorio": rnd["auc_insercion"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Estabilidad entre semillas
# ---------------------------------------------------------------------------


def map_correlations(maps_by_run: dict, size: int = 56) -> pd.DataFrame:
    """Correlación de Pearson y Spearman de los mapas de cada imagen entre cada par de corridas
    (mapas reducidos a size x size por promedio de bloques)."""
    from scipy.stats import pearsonr, spearmanr

    def reduce(m):
        h = m.shape[1]
        f = h // size
        return m[:, : f * size, : f * size].reshape(len(m), size, f, size, f).mean(axis=(2, 4))

    small = {k: reduce(v) for k, v in maps_by_run.items()}
    rows = []
    for a, b in combinations(sorted(small), 2):
        for i in range(len(small[a])):
            x, y = small[a][i].ravel(), small[b][i].ravel()
            ok = np.ptp(x) > 0 and np.ptp(y) > 0
            rows.append({"corrida_a": a, "corrida_b": b, "imagen": i,
                         "pearson": float(pearsonr(x, y)[0]) if ok else np.nan,
                         "spearman": float(spearmanr(x, y)[0]) if ok else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------


def out_dir(cfg: dict, model_name: str) -> Path:
    return cfg["paths"]["interpretability_dir"] / model_name


def top_models(cfg: dict) -> list[str] | Missing:
    """Los interpretability.top_k modelos profundos con mayor media de la métrica primaria en test."""
    from src import benchmark as B
    from src.report import is_missing

    rs = B.load_runs_e2(cfg)
    scores = B.model_scores(rs, cfg)
    if is_missing(scores):
        return scores
    deep = [m for m in B.e2_groups(cfg)["deep_models"] if m in scores.dropna().index]
    if not deep:
        return Missing("no hay modelos profundos con corridas definitivas.")
    return list(scores[deep].sort_values(ascending=False).index[: int(cfg["interpretability"]["top_k"])])


def common_sample(cfg: dict, models) -> pd.DataFrame | Missing:
    """Muestra de evaluation.gradcam a partir de la corrida interpretability.run del primer modelo."""
    from src.final import final_label
    from src.gradcam import sample_images
    from src.train import run_dir_for

    s, f = (int(v) for v in cfg["interpretability"]["run"])
    d = run_dir_for(cfg, models[0], final_label(cfg), s, f)
    if not (d / "probabilities_test.csv").exists():
        return Missing(f"{models[0]} no tiene probabilities_test.csv en s{s}_f{f}.")
    g = cfg["evaluation"]["gradcam"]
    return sample_images(d, g["per_class_correct"], g["per_class_wrong"], g["sample_seed"])


def run_interpretability(cfg: dict, models) -> None:
    """Calcula y guarda mapas, borrado e inserción y estabilidad de `models` (solo inferencia)."""
    import keras

    from src.final import final_label
    from src.gradcam import WEIGHTS_FILE, load_model_float32
    from src.train import configure_runtime, run_dir_for

    sample = common_sample(cfg, models)
    if is_missing(sample):
        raise ValueError(sample.message)
    root = cfg["paths"]["interpretability_dir"]
    root.mkdir(parents=True, exist_ok=True)
    sample.to_csv(root / "muestra.csv", index=False)
    images = D.load_resized_images(cfg, sample["image_id"])
    icfg = cfg["interpretability"]
    main_run = tuple(int(v) for v in icfg["run"])
    runs = [main_run] + [tuple(map(int, r)) for r in icfg["stability_runs"] if tuple(map(int, r)) != main_run]
    try:
        for m in models:
            od = out_dir(cfg, m)
            od.mkdir(parents=True, exist_ok=True)
            maps_by_run = {}
            for s, f in runs:
                d = run_dir_for(cfg, m, final_label(cfg), s, f)
                if not (d / WEIGHTS_FILE).exists():
                    log.info(f"{m} s{s}_f{f}: sin {WEIGHTS_FILE}; se omite")
                    continue
                model = load_model_float32(cfg, m, d)
                probs = np.asarray(model.predict(images, batch_size=int(icfg["batch_size"]), verbose=0))
                targets = probs.argmax(axis=1)
                maps, method = saliency_maps(cfg, m, model, images, targets)
                maps_by_run[f"s{s}_f{f}"] = maps
                np.savez_compressed(od / f"mapas_s{s}_f{f}.npz", maps=maps.astype(np.float16), targets=targets,
                                    image_ids=sample["image_id"].to_numpy().astype(str), method=method)
                if (s, f) == main_run:
                    di = deletion_insertion_table(cfg, model, images, maps, targets, sample["image_id"])
                    di.to_csv(od / "borrado_insercion.csv", index=False)
                del model
                keras.backend.clear_session()
            if len(maps_by_run) > 1:
                map_correlations(maps_by_run, int(icfg["stability_size"])).to_csv(od / "estabilidad.csv", index=False)
            log.info(f"{m}: mapas de {len(maps_by_run)} corridas -> {od}")
    finally:
        keras.backend.clear_session()
        configure_runtime(cfg)


# ---------------------------------------------------------------------------
# Tablas y gráficos para el notebook
# ---------------------------------------------------------------------------


def load_maps(cfg: dict, model_name: str) -> dict | Missing:
    """Resultado compatible con gradcam.plot_gradcam_grid de la corrida interpretability.run."""
    s, f = (int(v) for v in cfg["interpretability"]["run"])
    p = out_dir(cfg, model_name) / f"mapas_s{s}_f{f}.npz"
    sample_csv = cfg["paths"]["interpretability_dir"] / "muestra.csv"
    if not p.exists() or not sample_csv.exists():
        return Missing(f"no hay mapas de {model_name}; ejecuta `python -m src.interpret`.")
    z = np.load(p, allow_pickle=False)
    sample = pd.read_csv(sample_csv)
    classes = D.class_names(cfg)
    sample = sample.assign(pred=[classes[i] for i in z["targets"]])
    sample["resultado"] = np.where(sample["pred"] == sample["label"], "acierto", "error")
    return {"images": D.load_resized_images(cfg, sample["image_id"]), "heatmaps": z["maps"].astype(np.float32),
            "layer": str(z["method"]), "sample": sample}


@tolerant()
def deletion_insertion_summary(cfg: dict, models) -> pd.DataFrame:
    """Media y desviación estándar de las AUC de borrado e inserción (mapa y control aleatorio) por modelo."""
    rows = {}
    for m in models:
        p = out_dir(cfg, m) / "borrado_insercion.csv"
        if p.exists():
            t = pd.read_csv(p)
            cols = ["auc_borrado", "auc_borrado_aleatorio", "auc_insercion", "auc_insercion_aleatorio"]
            rows[m] = {**{f"{c}_media": t[c].mean() for c in cols}, **{f"{c}_std": t[c].std(ddof=1) for c in cols},
                       "imagenes": len(t)}
    if not rows:
        return Missing("no hay resultados de borrado e inserción; ejecuta `python -m src.interpret`.")
    return pd.DataFrame(rows).T


@tolerant()
def stability_summary(cfg: dict, models) -> pd.DataFrame:
    """Correlación media de Pearson y Spearman entre los mapas de distintas semillas por modelo."""
    rows = {}
    for m in models:
        p = out_dir(cfg, m) / "estabilidad.csv"
        if p.exists():
            t = pd.read_csv(p)
            rows[m] = {"pares_de_corridas": t[["corrida_a", "corrida_b"]].drop_duplicates().shape[0],
                       "imagenes": t["imagen"].nunique(), "pearson_media": t["pearson"].mean(),
                       "pearson_std": t["pearson"].std(ddof=1), "spearman_media": t["spearman"].mean(),
                       "spearman_std": t["spearman"].std(ddof=1)}
    if not rows:
        return Missing("no hay resultados de estabilidad; ejecuta `python -m src.interpret` (requiere los pesos "
                       "de interpretability.stability_runs).")
    return pd.DataFrame(rows).T


def main(argv=None) -> None:
    from src.models import REGISTRY
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Mapas de saliencia, borrado e inserción y estabilidad.")
    parser.add_argument("--model", action="append", choices=sorted(REGISTRY), default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = D.load_config(args.config)
    setup_logging(cfg["paths"]["logs_dir"] / "interpret.log")
    models = args.model or top_models(cfg)
    if isinstance(models, Missing):
        sys.exit(f"Error: {models.message}")
    try:
        run_interpretability(cfg, models)
    except ValueError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
