"""Grad-CAM sobre los pesos guardados de una corrida definitiva.

Los modelos de src/models tienen la estructura Input -> Lambda(preprocess_input) -> backbone
(submodelo anidado) -> cabeza lineal -> Dense softmax. La capa objetivo (`gradcam_layer` de
configs/<modelo>.yaml: out_relu, conv5_block3_out o block5_conv3) está dentro del backbone, así
que se construye un submodelo del backbone con dos salidas (la capa objetivo y la salida del
backbone) y se recorre a mano el resto: preprocess_input antes, la cabeza después. El gradiente
se toma del logit de la clase (antes de la softmax), como en Selvaraju et al. (2017).

El modelo se reconstruye en float32 (sin precisión mixta, para que los gradientes no pierdan
precisión) y se cargan los pesos de model.weights.h5; los pesos de ImageNet no se descargan.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data as D
from src.report import Missing, tolerant

WEIGHTS_FILE = "model.weights.h5"


def gradcam_run_dir(cfg: dict, model_name: str) -> Path | Missing:
    """Carpeta de la corrida definitiva de evaluation.gradcam.run, o `Missing` si no tiene pesos."""
    from src.final import final_label
    from src.train import run_dir_for

    seed, fold = (int(v) for v in cfg["evaluation"]["gradcam"]["run"])
    d = run_dir_for(cfg, model_name, final_label(cfg), seed, fold)
    if not (d / WEIGHTS_FILE).exists() or not (d / "probabilities_test.csv").exists():
        return Missing(f"la corrida definitiva s{seed}_f{fold} de {model_name} no tiene {WEIGHTS_FILE} y "
                       f"probabilities_test.csv; ejecuta `python -m src.final --model {model_name}`.")
    return d


@tolerant()
def sample_images(run_dir: Path, per_class_correct: int, per_class_wrong: int, seed: int) -> pd.DataFrame:
    """Muestra fija de imágenes de test de la corrida: por clase real, `per_class_correct`
    aciertos y `per_class_wrong` errores (menos si no hay suficientes), elegidos con `seed`."""
    preds = pd.read_csv(Path(run_dir) / "probabilities_test.csv")
    classes = [c[2:] for c in preds.columns if c.startswith("p_")]
    preds["correct"] = preds["label"] == preds["pred"]
    preds["confidence"] = preds[[f"p_{c}" for c in classes]].max(axis=1)
    rng = np.random.default_rng(seed)
    parts = []
    for c in classes:
        for ok, k in ((True, per_class_correct), (False, per_class_wrong)):
            pool = preds[(preds["label"] == c) & (preds["correct"] == ok)].sort_values("image_id")
            if len(pool) and k > 0:
                idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
                parts.append(pool.iloc[np.sort(idx)])
    sample = pd.concat(parts, ignore_index=True)
    sample["resultado"] = np.where(sample["correct"], "acierto", "error")
    return sample[["image_id", "label", "pred", "resultado", "confidence"]]


def load_model_float32(cfg: dict, model_name: str, run_dir: Path):
    """Modelo de `model_name` en float32 con los pesos de `run_dir`/model.weights.h5."""
    import keras

    from src.models import REGISTRY
    from src.train import load_model_config, resolve_hparams

    model_cfg = load_model_config(model_name)
    info = Path(run_dir) / "run_info.json"
    hp = D.load_json(info)["hyperparameters"] if info.exists() else resolve_hparams(model_cfg)
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("float32")
    model = REGISTRY[model_name](dict(model_cfg, weights=None), len(D.class_names(cfg)), hp["dropout"])
    model.load_weights(Path(run_dir) / WEIGHTS_FILE)
    return model


def gradcam_heatmaps(model, images: np.ndarray, layer_name: str, class_idx=None, batch_size: int = 8) -> np.ndarray:
    """Mapas Grad-CAM (N, H, W) en [0, 1], redimensionados al tamaño de entrada.

    `images`: (N, H, W, 3) float32 en [0, 255] (la entrada del modelo). `class_idx`: clase de
    cada imagen respecto de la que se calcula el mapa (por defecto, la predicha).
    """
    import keras
    import tensorflow as tf

    from src.models.common import PREPROCESS_NAME, get_backbone, head_layers

    backbone = get_backbone(model)
    feature_model = keras.Model(backbone.inputs, [backbone.get_layer(layer_name).output, backbone.output])
    preprocess = model.get_layer(PREPROCESS_NAME)
    head = head_layers(model)
    *hidden, output = head
    size = images.shape[1:3]
    maps = []
    for start in range(0, len(images), batch_size):
        x = tf.convert_to_tensor(images[start:start + batch_size], dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv, y = feature_model(preprocess(x), training=False)
            for layer in hidden:
                y = layer(y, training=False)
            logits = tf.matmul(tf.cast(y, tf.float32), output.kernel) + output.bias
            if class_idx is None:
                target = tf.argmax(logits, axis=1, output_type=tf.int32)
            else:
                target = tf.convert_to_tensor(np.asarray(class_idx)[start:start + batch_size], dtype=tf.int32)
            score = tf.gather(logits, target, axis=1, batch_dims=1)
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(tf.cast(grads, tf.float32), axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * tf.cast(conv, tf.float32), axis=-1))
        cam = tf.math.divide_no_nan(cam, tf.reduce_max(cam, axis=(1, 2), keepdims=True))
        cam = tf.image.resize(cam[..., None], size, method="bilinear")[..., 0]
        maps.append(cam.numpy())
    return np.concatenate(maps, axis=0)


@tolerant()
def compute_gradcam(cfg: dict, model_name: str, run_dir: Path, sample: pd.DataFrame) -> dict:
    """Imágenes de `sample`, sus mapas Grad-CAM respecto de la clase predicha y la capa usada.
    Restaura la política de precisión de train.mixed_precision al terminar."""
    import keras

    from src.train import configure_runtime, load_model_config

    layer = load_model_config(model_name).get("gradcam_layer")
    if not layer:
        return Missing(f"configs/{model_name}.yaml no define gradcam_layer.")
    classes = D.class_names(cfg)
    images = D.load_resized_images(cfg, sample["image_id"])
    try:
        model = load_model_float32(cfg, model_name, run_dir)
        pred_idx = [classes.index(p) for p in sample["pred"]]
        heatmaps = gradcam_heatmaps(model, images, layer, class_idx=pred_idx)
    finally:
        keras.backend.clear_session()
        configure_runtime(cfg)
    return {"images": images, "heatmaps": heatmaps, "layer": layer, "sample": sample.reset_index(drop=True)}


@tolerant()
def plot_gradcam_grid(result: dict, alpha: float = 0.45, ncols: int = 7, cmap: str = "jet"):
    """Cuadrícula de pares (imagen, imagen con el mapa superpuesto), con image_id, clase real,
    predicha y resultado (acierto o error)."""
    sample = result["sample"]
    n = len(sample)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(2.0 * ncols, 4.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in sample.iterrows():
        i, j = divmod(k, ncols)
        img = result["images"][k] / 255.0
        axes[2 * i][j].imshow(img)
        axes[2 * i][j].set_title(f"{r['image_id']}\nreal: {r['label']} | pred: {r['pred']}\n{r['resultado']}",
                                 fontsize=7, color="tab:green" if r["resultado"] == "acierto" else "tab:red")
        axes[2 * i + 1][j].imshow(img)
        axes[2 * i + 1][j].imshow(result["heatmaps"][k], cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    fig.suptitle(f"Grad-CAM ({result['layer']}) respecto de la clase predicha")
    fig.tight_layout()
    return fig
