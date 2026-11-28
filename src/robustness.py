"""Robustez frente a perturbaciones de la imagen (solo inferencia con los pesos guardados).

    python -m src.robustness --model vit_b16 [--model ...]     # o --all

Para cada corrida de robustness.runs de un modelo (con model.weights.h5), predice las imágenes de
test sin perturbar y con cada perturbación de robustness.perturbations en sus tres niveles de
severidad: ruido gaussiano (desviación en [0, 255]), desenfoque gaussiano (sigma en píxeles),
compresión JPEG (calidad) y brillo (delta aditivo sobre [0, 1]). Las perturbaciones se aplican a la
imagen ya redimensionada a train.image_size, como entra al modelo, y son deterministas
(robustness.seed). El modelo se carga en la política de precisión del entrenamiento.

Resultado por corrida: outputs/robustness/<modelo>/s<semilla>_f<pliegue>.csv con perturbación,
severidad (0 = sin perturbar), nivel, exactitud balanceada y exactitud. Índice de robustez: área
bajo la curva de exactitud balanceada frente a la severidad (0 a 3, trapecios) dividida por la del
rectángulo de la exactitud sin perturbar (3 x BA_0); vale 1 si no hay degradación.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data as D
from src.report import Missing, tolerant

log = logging.getLogger(__name__)
CLEAN = "ninguna"


# ---------------------------------------------------------------------------
# Perturbaciones (imágenes uint8 o float en [0, 255], (N, H, W, 3))
# ---------------------------------------------------------------------------


def gaussian_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(x.astype(np.float32) + rng.normal(0.0, sigma, size=x.shape), 0, 255).astype(np.float32)


def gaussian_blur(x: np.ndarray, sigma: float, rng=None) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return np.stack([gaussian_filter(img.astype(np.float32), sigma=(sigma, sigma, 0)) for img in x])


def jpeg(x: np.ndarray, quality: int, rng=None) -> np.ndarray:
    from PIL import Image

    out = []
    for img in x:
        buf = io.BytesIO()
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        with Image.open(buf) as im:
            out.append(np.asarray(im.convert("RGB"), dtype=np.float32))
    return np.stack(out)


def brightness(x: np.ndarray, delta: float, rng=None) -> np.ndarray:
    return np.clip(x.astype(np.float32) / 255.0 + delta, 0.0, 1.0) * 255.0


PERTURBATIONS = {"gaussian_noise": gaussian_noise, "gaussian_blur": gaussian_blur, "jpeg": jpeg,
                 "brightness": brightness}


def conditions(cfg: dict) -> list[tuple[str, int, float | None]]:
    """(perturbación, severidad 0..3, nivel) en orden; la primera es la imagen sin perturbar."""
    out = [(CLEAN, 0, None)]
    for name, levels in cfg["robustness"]["perturbations"].items():
        if name not in PERTURBATIONS:
            raise ValueError(f"Perturbación desconocida: {name}")
        out += [(name, i + 1, float(v)) for i, v in enumerate(levels)]
    return out


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------


def load_test_images(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(image_ids, etiquetas, imágenes uint8 (N, S, S, 3)) de test, redimensionadas como en el entrenamiento."""
    splits = D.load_splits(cfg["paths"]["splits_csv"])
    test = D.select_split(splits, "test", 0)
    load = D._image_loader(cfg, cache_uint8=True)
    imgs = np.stack([load(str(D.image_path(cfg, i))).numpy() for i in test["image_id"]])
    y = test["dx"].map({c: i for i, c in enumerate(D.class_names(cfg))}).to_numpy()
    return test["image_id"].to_numpy(), y, imgs


def result_path(cfg: dict, model_name: str, seed: int, fold: int) -> Path:
    return cfg["paths"]["robustness_dir"] / model_name / f"s{seed}_f{fold}.csv"


def evaluate_run(cfg: dict, model_name: str, run_dir: Path, y: np.ndarray, images: np.ndarray, seed: int) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    from src.gradcam import load_trained_model

    model = load_trained_model(cfg, model_name, run_dir)
    bs = int(cfg["train"]["batch_size"])
    rows = []
    for name, sev, level in conditions(cfg):
        rng = np.random.default_rng([int(cfg["robustness"]["seed"]), zlib.crc32(name.encode()), sev])
        preds = []
        for start in range(0, len(images), bs):
            batch = images[start:start + bs].astype(np.float32)
            if name != CLEAN:
                batch = PERTURBATIONS[name](batch, level, rng)
            preds.append(np.asarray(model.predict_on_batch(batch.astype(np.float32))).argmax(axis=1))
        pred = np.concatenate(preds)
        rows.append({"perturbacion": name, "severidad": sev, "nivel": level,
                     "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                     "accuracy": float(accuracy_score(y, pred)), "n": len(y)})
        log.info(f"{model_name} s{seed} | {name} {level}: bal_acc {rows[-1]['balanced_accuracy']:.4f}")
    import keras

    del model
    keras.backend.clear_session()
    return pd.DataFrame(rows)


def run_robustness(cfg: dict, model_name: str, overwrite: bool = False) -> pd.DataFrame | Missing:
    """Evalúa las corridas de robustness.runs con pesos guardados (reutiliza las terminadas)."""
    from src.final import final_label
    from src.gradcam import WEIGHTS_FILE
    from src.train import run_dir_for

    pending = []
    for s, f in (tuple(map(int, p)) for p in cfg["robustness"]["runs"]):
        d = run_dir_for(cfg, model_name, final_label(cfg), s, f)
        if (d / WEIGHTS_FILE).exists() and (overwrite or not result_path(cfg, model_name, s, f).exists()):
            pending.append((s, f, d))
    if pending:
        _, y, images = load_test_images(cfg)
        for s, f, d in pending:
            out = evaluate_run(cfg, model_name, d, y, images, s)
            path = result_path(cfg, model_name, s, f)
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
    return load_results(cfg, [model_name])


# ---------------------------------------------------------------------------
# Tablas y gráficos
# ---------------------------------------------------------------------------


def load_results(cfg: dict, models) -> pd.DataFrame | Missing:
    """Resultados guardados de robustez (formato largo, con columnas modelo y run)."""
    frames = []
    for m in models:
        for p in sorted((cfg["paths"]["robustness_dir"] / m).glob("s*_f*.csv")):
            df = pd.read_csv(p)
            df.insert(0, "run", p.stem)
            df.insert(0, "modelo", m)
            frames.append(df)
    if not frames:
        return Missing("no hay resultados de robustez en outputs/robustness/; ejecuta `python -m src.robustness --all` "
                       "(necesita model.weights.h5 de las corridas de robustness.runs).")
    return pd.concat(frames, ignore_index=True)


def _curve(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    clean = df[df["perturbacion"] == CLEAN]
    out = []
    for name in df["perturbacion"].unique():
        if name == CLEAN:
            continue
        sub = pd.concat([clean.assign(perturbacion=name), df[df["perturbacion"] == name]])
        out.append(sub)
    return pd.concat(out) if out else df.iloc[:0]


@tolerant()
def robustness_index(results: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.DataFrame:
    """Índice de robustez por modelo y perturbación: media y desviación estándar entre corridas
    de trapz(metric, severidad) / (severidad_max x metric sin perturbar), y su media sobre las
    perturbaciones (columna «media»)."""
    curves = _curve(results, metric)
    rows = []
    for (m, run, name), g in curves.groupby(["modelo", "run", "perturbacion"]):
        g = g.sort_values("severidad")
        x, v = g["severidad"].to_numpy(float), g[metric].to_numpy(float)
        rows.append({"modelo": m, "run": run, "perturbacion": name,
                     "indice": float(np.trapezoid(v, x) / (x.max() * v[0])) if v[0] > 0 else np.nan})
    t = pd.DataFrame(rows)
    per = t.groupby(["modelo", "perturbacion"])["indice"].agg(["mean", "std"]).unstack("perturbacion")
    per.columns = [f"{name}_{s}" for s, name in per.columns]
    per["media"] = t.groupby(["modelo", "run"])["indice"].mean().groupby("modelo").mean()
    per["corridas"] = t.groupby("modelo")["run"].nunique()
    return per


@tolerant()
def degradation_table(results: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.DataFrame:
    """Media entre corridas de `metric` por modelo (filas) y perturbación y severidad (columnas)."""
    g = results.groupby(["modelo", "perturbacion", "severidad"])[metric].mean().unstack(["perturbacion", "severidad"])
    return g


@tolerant()
def plot_degradation(results: pd.DataFrame, metric: str = "balanced_accuracy"):
    """Curvas de degradación: `metric` (media entre corridas) frente a la severidad, un panel por perturbación."""
    curves = _curve(results, metric)
    names = list(dict.fromkeys(curves["perturbacion"]))
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.8), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        sub = curves[curves["perturbacion"] == name]
        for m, g in sub.groupby("modelo", sort=False):
            mean = g.groupby("severidad")[metric].mean()
            ax.plot(mean.index, mean.to_numpy(), marker="o", label=m)
        ax.set_title(name)
        ax.set_xlabel("severidad")
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(metric)
    axes[0][-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def main(argv=None) -> None:
    from src.benchmark import e2_groups
    from src.models import REGISTRY
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Robustez frente a perturbaciones (solo inferencia).")
    parser.add_argument("--model", action="append", choices=sorted(REGISTRY), default=None)
    parser.add_argument("--all", action="store_true", help="todos los modelos profundos del Entregable 2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = D.load_config(args.config)
    models = list(args.model or []) + (e2_groups(cfg)["deep_models"] if args.all else [])
    if not models:
        sys.exit("Indica --model o --all")
    setup_logging(cfg["paths"]["logs_dir"] / "robustness.log")
    for m in dict.fromkeys(models):
        res = run_robustness(cfg, m, args.overwrite)
        done = res.message if isinstance(res, Missing) else f"{res['run'].nunique()} corridas"
        log.info(f"{m}: {done}")


if __name__ == "__main__":
    main()
