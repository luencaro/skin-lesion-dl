"""Funciones de apoyo para el EDA de imágenes: caché, muestreo, figuras, color, artefactos,
máscaras de lesión, validación adversarial y embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

_PIL_FORMAT = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}


# ---------------------------------------------------------------------------
# Lectura, caché y muestreo
# ---------------------------------------------------------------------------


def load_rgb(path: str | Path) -> Image.Image:
    """Abre una imagen y la devuelve en RGB, ya cargada en memoria (el archivo queda cerrado)."""
    with Image.open(path) as im:
        return im.convert("RGB")


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Redimensiona conservando la relación de aspecto para que el lado mayor sea `max_side`.

    No amplía imágenes más pequeñas.
    """
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return img.copy()
    return img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


def cache_path(cache_dir: str | Path, image_id: str, max_side: int, fmt: str) -> Path:
    """Ruta en caché; cada `max_side` tiene su propia subcarpeta."""
    return Path(cache_dir) / str(max_side) / f"{image_id}.{fmt}"


def build_cache(
    image_ids, src_paths, cache_dir, max_side: int, fmt: str, jpeg_quality: int
) -> pd.Series:
    """Crea (o reutiliza) versiones reducidas de las imágenes. Devuelve image_id -> ruta en caché."""
    pil_format = _PIL_FORMAT[fmt.lower()]
    save_kwargs = {"quality": jpeg_quality} if pil_format == "JPEG" else {}
    out = {}
    for iid, src in tqdm(zip(image_ids, src_paths), total=len(src_paths), desc="Caché EDA"):
        dst = cache_path(cache_dir, iid, max_side, fmt)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            small = resize_max_side(load_rgb(src), max_side)
            tmp = dst.with_name(dst.name + ".tmp")
            small.save(tmp, format=pil_format, **save_kwargs)
            tmp.replace(dst)
        out[iid] = dst
    return pd.Series(out, name="cache_path")


def sample_per_class(df: pd.DataFrame, n: int, seed: int, col: str = "dx") -> pd.DataFrame:
    """Muestra aleatoria de hasta `n` filas por clase (todas si la clase tiene menos)."""
    shuffled = df.sample(frac=1, random_state=seed)
    return shuffled.groupby(col, sort=False).head(n).sort_values([col, "image_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def plot_image_grid(paths, titles, ncols: int, cell_size: float = 2.0, suptitle: str | None = None):
    """Cuadrícula de imágenes con un título por imagen."""
    paths, titles = list(paths), list(titles)
    n = len(paths)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size, nrows * cell_size * 1.1), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p, t in zip(axes.ravel(), paths, titles):
        ax.imshow(load_rgb(p))
        ax.set_title(t, fontsize=7)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def _draw_pairs(axes, paths_a, paths_b, titles_a, titles_b, headers) -> None:
    """Dibuja pares en una matriz de ejes (n_filas, 2); los ejes sobrantes quedan vacíos."""
    for ax in axes.ravel():
        ax.axis("off")
    for row, (pa, pb, ta, tb, hdr) in enumerate(zip(paths_a, paths_b, titles_a, titles_b, headers)):
        for ax, p, t in ((axes[row, 0], pa, ta), (axes[row, 1], pb, tb)):
            ax.imshow(load_rgb(p))
            ax.set_title(t, fontsize=7)
        axes[row, 0].text(-0.05, 0.5, hdr, transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=7)


def plot_pairs(paths_a, paths_b, titles_a, titles_b, headers, cell_size: float = 2.5):
    """Pares de imágenes lado a lado, una fila por par."""
    paths_a = list(paths_a)
    fig, axes = plt.subplots(len(paths_a), 2, figsize=(2 * cell_size, len(paths_a) * cell_size * 1.15),
                             squeeze=False)
    _draw_pairs(axes, paths_a, paths_b, titles_a, titles_b, headers)
    fig.tight_layout()
    return fig


def plot_pair_pages(paths_a, paths_b, titles_a, titles_b, headers, page_size: int,
                    pages_per_figure: int, cell_size: float = 2.5) -> list:
    """Pares de imágenes en páginas de `page_size` pares, con `pages_per_figure` páginas una al
    lado de la otra en cada figura. Devuelve la lista de figuras."""
    items = list(zip(paths_a, paths_b, titles_a, titles_b, headers))
    pages = [items[i:i + page_size] for i in range(0, len(items), page_size)]
    figs = []
    for first in range(0, len(pages), pages_per_figure):
        group = pages[first:first + pages_per_figure]
        nrows = max(len(p) for p in group)
        fig = plt.figure(figsize=(2 * len(group) * cell_size, nrows * cell_size * 1.15 + 0.4),
                         layout="constrained")
        subfigs = np.atleast_1d(fig.subfigures(1, len(group)))
        for k, (sf, page) in enumerate(zip(subfigs, group)):
            _draw_pairs(sf.subplots(nrows, 2, squeeze=False), *map(list, zip(*page)))
            sf.suptitle(f"Página {first + k + 1} de {len(pages)}")
        figs.append(fig)
    return figs


def _test_label(test: dict | pd.Series | None) -> str:
    if test is None:
        return ""
    return f"\nH = {test['H']:.2f}, p = {test['p']:.2g}, ε² = {test['epsilon2']:.3f}"


def boxplot_by_class(df, value: str, group: str = "dx", order=None, title: str | None = None,
                     test=None, ax=None):
    """Boxplot de `value` por `group`; añade al título H, p y ε² si se pasa `test`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x=group, y=value, order=order, ax=ax, fliersize=1)
    ax.set_title((title or value) + _test_label(test), fontsize=9)
    ax.set_xlabel(group)
    return ax


def boxplot_grid(df, values, group: str, order=None, tests: pd.DataFrame | None = None,
                 ncols: int = 3, titles: dict | None = None, suptitle: str | None = None):
    """Un boxplot por variable de `values`, con el resultado de Kruskal-Wallis en el título."""
    values = list(values)
    nrows = int(np.ceil(len(values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.6), squeeze=False)
    for ax in axes.ravel()[len(values):]:
        ax.axis("off")
    for ax, v in zip(axes.ravel(), values):
        test = tests.loc[v] if tests is not None else None
        boxplot_by_class(df, v, group, order, (titles or {}).get(v, v), test, ax)
        ax.tick_params(axis="x", labelrotation=45)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_before_after(paths, transform, titles, cell_size: float = 2.5, labels=("Original", "Corregida")):
    """Cada imagen de `paths` junto a `transform(imagen)`, una fila por imagen."""
    paths, titles = list(paths), list(titles)
    fig, axes = plt.subplots(len(paths), 2, figsize=(2 * cell_size, len(paths) * cell_size), squeeze=False)
    for row, (p, t) in enumerate(zip(paths, titles)):
        img = load_rgb(p)
        for ax, im, lab in ((axes[row, 0], img, labels[0]), (axes[row, 1], transform(img), labels[1])):
            ax.imshow(im)
            ax.set_title(f"{t} · {lab}", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Anotación de artefactos y viñeteado
# ---------------------------------------------------------------------------


def annotation_template(sample: pd.DataFrame, artifact_cols) -> pd.DataFrame:
    """Plantilla image_id, dx y una columna vacía por artefacto."""
    out = sample[["image_id", "dx"]].reset_index(drop=True)
    for c in artifact_cols:
        out[c] = pd.NA
    return out


def annotation_status(df: pd.DataFrame, artifact_cols) -> dict:
    """Filas completas (todas las columnas de artefacto en 0/1) y valores no válidos."""
    vals = df[list(artifact_cols)].apply(pd.to_numeric, errors="coerce")
    present = df[list(artifact_cols)].notna()
    invalid = present & ~vals.isin([0, 1])
    complete = vals.isin([0, 1]).all(axis=1)
    return {
        "n_filas": len(df),
        "n_completas": int(complete.sum()),
        "n_celdas_no_validas": int(invalid.to_numpy().sum()),
        "completo": bool(len(df) > 0 and complete.all()),
    }


def artifact_prevalence(df: pd.DataFrame, artifact_cols, col: str = "dx") -> pd.DataFrame:
    """Porcentaje de imágenes con cada artefacto (valor 1), por clase."""
    vals = df[list(artifact_cols)].apply(pd.to_numeric, errors="coerce")
    vals[col] = df[col]
    return vals.groupby(col).mean() * 100


def vignetting_index(img: Image.Image, corner_frac: float, center_frac: float) -> dict:
    """Intensidad media (escala de grises) de las cuatro esquinas y de una ventana central.

    Esquinas: cuadrados de lado `corner_frac` * lado menor. Centro: ventana de
    `center_frac` * alto por `center_frac` * ancho.
    """
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    c = max(1, round(min(h, w) * corner_frac))
    corners = np.concatenate([g[:c, :c].ravel(), g[:c, -c:].ravel(), g[-c:, :c].ravel(), g[-c:, -c:].ravel()])
    ch, cw = max(1, round(h * center_frac)), max(1, round(w * center_frac))
    top, left = (h - ch) // 2, (w - cw) // 2
    center = g[top:top + ch, left:left + cw]
    esq, cen = float(corners.mean()), float(center.mean())
    return {"media_esquinas": esq, "media_centro": cen, "razon_esquinas_centro": esq / cen if cen > 0 else np.nan}


def vignetting_table(image_ids, paths, corner_frac: float, center_frac: float) -> pd.DataFrame:
    rows = []
    for iid, p in tqdm(zip(image_ids, paths), total=len(paths), desc="Viñeteado"):
        rows.append({"image_id": iid, **vignetting_index(load_rgb(p), corner_frac, center_frac)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dimensiones y color
# ---------------------------------------------------------------------------


def file_info_table(image_ids, paths) -> pd.DataFrame:
    """Ancho, alto, relación de aspecto, modo, formato y tamaño de archivo (sin decodificar píxeles)."""
    rows = []
    for iid, p in tqdm(zip(image_ids, paths), total=len(paths), desc="Dimensiones"):
        p = Path(p)
        with Image.open(p) as im:
            w, h = im.size
            rows.append({"image_id": iid, "ancho": w, "alto": h, "relacion_aspecto": w / h,
                         "modo": im.mode, "formato": im.format, "tamano_kb": p.stat().st_size / 1024})
    return pd.DataFrame(rows)


def color_features(image_ids, paths, luma_weights) -> tuple[pd.DataFrame, dict]:
    """Características de color por imagen y media/desviación global por canal RGB.

    Por imagen (escala [0, 1]): media de R, G y B; luminancia (media de la combinación
    lineal `luma_weights` de RGB); contraste (desviación estándar de esa luminancia);
    saturación (media del canal S de HSV). Las estadísticas globales acumulan todos los
    píxeles de todas las imágenes.
    """
    w = np.asarray(luma_weights, dtype=np.float64)
    s = np.zeros(3)
    s2 = np.zeros(3)
    n_px = 0
    rows = []
    for iid, p in tqdm(zip(image_ids, paths), total=len(paths), desc="Color"):
        img = load_rgb(p)
        x = np.asarray(img, dtype=np.float64).reshape(-1, 3) / 255.0
        sat = np.asarray(img.convert("HSV"), dtype=np.float64)[..., 1] / 255.0
        s += x.sum(axis=0)
        s2 += np.square(x).sum(axis=0)
        n_px += x.shape[0]
        lum = x @ w
        mean = x.mean(axis=0)
        rows.append({"image_id": iid, "media_r": mean[0], "media_g": mean[1], "media_b": mean[2],
                     "luminancia": lum.mean(), "contraste": lum.std(), "saturacion": sat.mean()})
    media = s / n_px
    std = np.sqrt(np.maximum(s2 / n_px - media**2, 0))
    globales = {
        "escala": "[0, 1]",
        "canales": ["R", "G", "B"],
        "media": media.tolist(),
        "std": std.tolist(),
        "media_255": (media * 255).tolist(),
        "std_255": (std * 255).tolist(),
        "n_imagenes": len(rows),
        "n_pixeles": int(n_px),
    }
    return pd.DataFrame(rows), globales


def shades_of_gray(img: Image.Image, p: float) -> Image.Image:
    """Constancia de color Shades of Gray (Finlayson y Trezzi, 2004) con norma de Minkowski `p`."""
    x = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    illum = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)
    norm = np.linalg.norm(illum)
    if norm == 0:
        return img.copy()
    illum = illum / norm
    out = np.clip(x / (illum * np.sqrt(3)), 0, 1)
    return Image.fromarray(np.round(out * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Máscaras de lesión
# ---------------------------------------------------------------------------


def lesion_features(img_path, mask_path, mask_threshold: int, luma_weights) -> dict:
    """Área relativa, centroide, contacto con el borde y color medio de lesión y fondo.

    `dist_centroide` es la distancia del centroide al centro de la imagen dividida por la
    semidiagonal (0 = centro, 1 = esquina).
    """
    with Image.open(mask_path) as m:
        mask_img = m.convert("L")
    img = load_rgb(img_path)
    if mask_img.size != img.size:
        mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_img) > mask_threshold
    x = np.asarray(img, dtype=np.float64) / 255.0
    w = np.asarray(luma_weights, dtype=np.float64)
    h, wd = mask.shape
    out = {"fraccion_area": float(mask.mean())}
    if mask.any():
        ys, xs = np.nonzero(mask)
        cy, cx = (h - 1) / 2, (wd - 1) / 2
        out["dist_centroide"] = float(np.hypot(ys.mean() - cy, xs.mean() - cx) / np.hypot(cy, cx))
        out["toca_borde"] = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    else:
        out["dist_centroide"] = np.nan
        out["toca_borde"] = np.nan
    for nombre, region in (("lesion", mask), ("fondo", ~mask)):
        if region.any():
            mean = x[region].mean(axis=0)
            out.update({f"{nombre}_r": mean[0], f"{nombre}_g": mean[1], f"{nombre}_b": mean[2],
                        f"{nombre}_luminancia": float(mean @ w)})
        else:
            out.update({f"{nombre}_{k}": np.nan for k in ("r", "g", "b", "luminancia")})
    return out


def lesion_features_table(image_ids, img_paths, mask_paths, mask_threshold: int, luma_weights) -> pd.DataFrame:
    rows = []
    for iid, ip, mp in tqdm(zip(image_ids, img_paths, mask_paths), total=len(img_paths), desc="Máscaras"):
        if not Path(mp).exists():
            rows.append({"image_id": iid, "mascara_existe": False})
            continue
        rows.append({"image_id": iid, "mascara_existe": True,
                     **lesion_features(ip, mp, mask_threshold, luma_weights)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validación adversarial
# ---------------------------------------------------------------------------


def color_histogram(img: Image.Image, bins: int) -> np.ndarray:
    """Histograma normalizado de `bins` intervalos por canal RGB, concatenado (3 * bins)."""
    x = np.asarray(img.convert("RGB"))
    n = x.shape[0] * x.shape[1]
    return np.concatenate([np.histogram(x[..., c], bins=bins, range=(0, 256))[0] / n for c in range(3)])


def color_histograms(paths, bins: int) -> np.ndarray:
    return np.stack([color_histogram(load_rgb(p), bins) for p in tqdm(paths, desc="Histogramas")])


def adversarial_validation(X, y, n_splits: int, seed: int, max_iter: int) -> dict:
    """AUC de una regresión logística (con estandarización) que predice `y` con CV estratificada."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return {"auc_media": float(aucs.mean()), "auc_std": float(aucs.std()), "auc_por_pliegue": aucs.tolist()}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def mobilenet_embeddings(paths, input_size: int, batch_size: int) -> np.ndarray:
    """Vector de características (pooling promedio) de MobileNetV2 preentrenada en ImageNet.

    Las imágenes se redimensionan a `input_size` x `input_size` y pasan por
    `mobilenet_v2.preprocess_input`. La primera llamada descarga los pesos de Keras.
    """
    from tensorflow import keras

    model = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", pooling="avg", input_shape=(input_size, input_size, 3)
    )
    preprocess = keras.applications.mobilenet_v2.preprocess_input
    paths = list(paths)
    feats = []
    for start in tqdm(range(0, len(paths), batch_size), desc="Embeddings"):
        batch = np.stack([
            np.asarray(load_rgb(p).resize((input_size, input_size), Image.Resampling.BILINEAR), dtype=np.float32)
            for p in paths[start:start + batch_size]
        ])
        feats.append(np.asarray(model.predict_on_batch(preprocess(batch))))
    return np.concatenate(feats)


def tsne_2d(X, seed: int, perplexity: float) -> np.ndarray:
    from sklearn.manifold import TSNE

    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)
