"""Funciones de apoyo para el EDA de imágenes: caché, muestreo, figuras, color, artefactos,
máscaras de lesión, mapas medios y validación adversarial."""

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


def load_mask(mask_path, size, mask_threshold: int) -> np.ndarray:
    """Máscara binaria (alto, ancho) redimensionada con vecino más cercano a `size` = (ancho, alto)."""
    with Image.open(mask_path) as m:
        mask = m.convert("L")
    if mask.size != tuple(size):
        mask = mask.resize(tuple(size), Image.Resampling.NEAREST)
    return np.asarray(mask) > mask_threshold


def corner_mask(shape, corner_frac: float) -> np.ndarray:
    """Máscara booleana de las cuatro esquinas: cuadrados de lado `corner_frac` * lado menor."""
    h, w = shape
    c = max(1, round(min(h, w) * corner_frac))
    out = np.zeros((h, w), dtype=bool)
    out[:c, :c] = out[:c, -c:] = out[-c:, :c] = out[-c:, -c:] = True
    return out


def vignetting_index(img: Image.Image, lesion: np.ndarray, corner_frac: float, luma_weights) -> dict:
    """Índice de viñeteado con la máscara de lesión.

    `lum_esquinas`: luminancia media (escala [0, 1]) de las cuatro esquinas sin píxeles de
    lesión. `mediana_piel`: mediana de la luminancia del fondo de piel (ni lesión ni
    esquina). `indice_vineteado` = lum_esquinas / mediana_piel.
    """
    lum = (np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0) @ np.asarray(luma_weights, dtype=np.float64)
    corners = corner_mask(lum.shape, corner_frac)
    esq = corners & ~lesion
    piel = ~corners & ~lesion
    lum_esq = float(lum[esq].mean()) if esq.any() else np.nan
    med_piel = float(np.median(lum[piel])) if piel.any() else np.nan
    indice = lum_esq / med_piel if med_piel > 0 else np.nan
    return {"lum_esquinas": lum_esq, "mediana_piel": med_piel, "indice_vineteado": indice}


def vignetting_table(image_ids, img_paths, mask_paths, corner_frac: float, mask_threshold: int,
                     luma_weights) -> pd.DataFrame:
    """`vignetting_index` por imagen; NaN si la imagen no tiene máscara."""
    rows = []
    for iid, ip, mp in tqdm(zip(image_ids, img_paths, mask_paths), total=len(img_paths), desc="Viñeteado"):
        if not Path(mp).exists():
            rows.append({"image_id": iid, "lum_esquinas": np.nan, "mediana_piel": np.nan,
                         "indice_vineteado": np.nan})
            continue
        img = load_rgb(ip)
        lesion = load_mask(mp, img.size, mask_threshold)
        rows.append({"image_id": iid, **vignetting_index(img, lesion, corner_frac, luma_weights)})
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
# Univariado y correlaciones
# ---------------------------------------------------------------------------


def plot_histograms(df: pd.DataFrame, cols, bins, ncols: int = 3, titles: dict | None = None,
                    suptitle: str | None = None):
    """Un histograma por variable de `cols` (se excluyen NaN)."""
    cols = list(cols)
    ncols = min(ncols, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2), squeeze=False)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    for ax, c in zip(axes.ravel(), cols):
        vals = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
        ax.hist(vals, bins=bins)
        ax.set_title((titles or {}).get(c, c), fontsize=9)
        ax.set_ylabel("Frecuencia")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_counts(conteo: pd.DataFrame, title: str, label: str, value_label: str = "Imágenes",
                horizontal: bool = False):
    """Barras con la columna `n` de una tabla de `stats.value_counts_pct`."""
    labels = conteo.index.astype(str).tolist()
    values = conteo["n"].to_numpy()
    if horizontal:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(labels))))
        ax.barh(labels[::-1], values[::-1])
        ax.set_xlabel(value_label)
        ax.set_ylabel(label)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values)
        ax.set_xlabel(label)
        ax.set_ylabel(value_label)
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_corr_heatmap(corr: pd.DataFrame, title: str):
    """Heatmap de correlaciones con valores anotados y escala divergente centrada en 0."""
    size = max(6, 0.7 * len(corr))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, annot_kws={"fontsize": 7}, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Control de calidad de las máscaras
# ---------------------------------------------------------------------------


def mask_qc_flags(les: pd.DataFrame, min_frac: float, max_frac: float) -> pd.DataFrame:
    """Marca máscaras casi vacías (fracción < `min_frac`) y casi completas (fracción > `max_frac`)."""
    out = les[["image_id", "dx", "fraccion_area"]].copy()
    out["casi_vacia"] = out["fraccion_area"] < min_frac
    out["casi_completa"] = out["fraccion_area"] > max_frac
    return out


def mask_qc_table(flags: pd.DataFrame, order=None, col: str = "dx") -> pd.DataFrame:
    """Conteo de máscaras casi vacías y casi completas por `col`, con una fila Total."""
    t = flags.groupby(col).agg(n=("image_id", "size"), casi_vacia=("casi_vacia", "sum"),
                               casi_completa=("casi_completa", "sum"))
    if order is not None:
        t = t.loc[[o for o in order if o in t.index]]
    t.loc["Total"] = t.sum()
    return t.astype(int)


def plot_mask_overlays(img_paths, mask_paths, titles, ncols: int, mask_threshold: int,
                       cell_size: float = 2.2, suptitle: str | None = None):
    """Cuadrícula de imágenes con el contorno de la máscara de lesión superpuesto."""
    img_paths, mask_paths, titles = list(img_paths), list(mask_paths), list(titles)
    n = len(img_paths)
    ncols = max(1, min(ncols, n))
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size, nrows * cell_size * 1.05), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, ip, mp, t in zip(axes.ravel(), img_paths, mask_paths, titles):
        img = load_rgb(ip)
        mask = load_mask(mp, img.size, mask_threshold)
        ax.imshow(img)
        if mask.any() and not mask.all():
            ax.contour(mask.astype(float), levels=[0.5], colors="yellow", linewidths=1)
        ax.set_title(t, fontsize=7)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Mapas medios
# ---------------------------------------------------------------------------


def load_masks_resized(mask_paths, size, mask_threshold: int) -> np.ndarray:
    """Máscaras binarizadas y redimensionadas a `size` = (ancho, alto) por promedio de área.

    Devuelve un arreglo (N, alto, ancho) con valores en [0, 1].
    """
    out = []
    for mp in tqdm(list(mask_paths), desc="Máscaras redimensionadas"):
        with Image.open(mp) as m:
            binary = np.asarray(m.convert("L")) > mask_threshold
        small = Image.fromarray(binary.astype(np.uint8) * 255).resize(tuple(size), Image.Resampling.BOX)
        out.append(np.asarray(small, dtype=np.float32) / 255.0)
    return np.stack(out)


def group_means(arr: np.ndarray, labels, order=None) -> dict:
    """Media de `arr` a lo largo del primer eje para cada valor de `labels` (se omiten NaN)."""
    labels = pd.Series(list(labels))
    keys = order if order is not None else labels.dropna().unique().tolist()
    return {k: arr[(labels == k).to_numpy()].mean(axis=0) for k in keys if (labels == k).any()}


def plot_maps(maps: dict, ncols: int, cmap: str, vmin: float, vmax: float, cbar_label: str,
              suptitle: str | None = None, cell_size: float = 2.8):
    """Mapas 2D con una escala común (`vmin`, `vmax`) y una sola barra de color."""
    n = len(maps)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size + 1, nrows * cell_size * 0.85),
                             squeeze=False, layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    im = None
    for ax, (title, m) in zip(axes.ravel(), maps.items()):
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(str(title), fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label=cbar_label, shrink=0.8)
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def plot_rgb_grid(images: dict, ncols: int, suptitle: str | None = None, cell_size: float = 2.8):
    """Imágenes RGB con valores en [0, 1], una por panel."""
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size, nrows * cell_size * 0.85),
                             squeeze=False, layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, img) in zip(axes.ravel(), images.items()):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(str(title), fontsize=9)
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def symmetric_limit(maps: dict) -> float:
    """Máximo valor absoluto entre todos los mapas (para una escala divergente común)."""
    return float(max(np.nanmax(np.abs(m)) for m in maps.values()))


def max_value(maps: dict) -> float:
    """Máximo valor entre todos los mapas (para una escala común que empieza en 0)."""
    return float(max(np.nanmax(m) for m in maps.values()))


# ---------------------------------------------------------------------------
# Capacidad predictiva de la lesión y del fondo
# ---------------------------------------------------------------------------


def _region_histogram(pixels: np.ndarray, bins: int) -> np.ndarray:
    """Histograma normalizado de `bins` intervalos por canal de píxeles (m, 3); NaN si m = 0."""
    if len(pixels) == 0:
        return np.full(3 * bins, np.nan)
    return np.concatenate([np.histogram(pixels[:, c], bins=bins, range=(0, 256))[0] / len(pixels)
                           for c in range(3)])


def region_color_histograms(img_paths, mask_paths, bins: int, mask_threshold: int) -> dict:
    """Histogramas de color del fondo (máscara = 0) y de la lesión (máscara = 1) por imagen.

    Devuelve `fondo` y `lesion` (N, 3 * bins) y `frac_fondo` y `frac_lesion` (N,). Las
    imágenes sin máscara tienen NaN.
    """
    fondo, lesion, frac_f, frac_l = [], [], [], []
    for ip, mp in tqdm(zip(img_paths, mask_paths), total=len(img_paths), desc="Histogramas por región"):
        if not Path(mp).exists():
            fondo.append(np.full(3 * bins, np.nan))
            lesion.append(np.full(3 * bins, np.nan))
            frac_f.append(np.nan)
            frac_l.append(np.nan)
            continue
        img = load_rgb(ip)
        x = np.asarray(img)
        m = load_mask(mp, img.size, mask_threshold)
        fondo.append(_region_histogram(x[~m], bins))
        lesion.append(_region_histogram(x[m], bins))
        frac_f.append(float((~m).mean()))
        frac_l.append(float(m.mean()))
    return {"fondo": np.stack(fondo), "lesion": np.stack(lesion),
            "frac_fondo": np.asarray(frac_f), "frac_lesion": np.asarray(frac_l)}


def fold_cv_classification(X, y, folds, max_iter: int) -> dict:
    """Regresión logística multinomial (estandarización previa, class_weight="balanced")
    evaluada con los pliegues predefinidos de `folds`.

    Devuelve la exactitud balanceada por pliegue, su media y desviación estándar, el nivel
    de azar (1 / número de clases) y las predicciones fuera de pliegue.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, folds = np.asarray(X), np.asarray(y), np.asarray(folds)
    oof = np.empty(len(y), dtype=object)
    scores = []
    for f in np.unique(folds):
        te = folds == f
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=max_iter, class_weight="balanced"))
        model.fit(X[~te], y[~te])
        oof[te] = model.predict(X[te])
        scores.append(balanced_accuracy_score(y[te], oof[te].astype(y.dtype)))
    scores = np.asarray(scores)
    return {"exactitud_balanceada_media": float(scores.mean()),
            "exactitud_balanceada_std": float(scores.std()),
            "azar": 1 / len(np.unique(y)), "n": len(y), "n_clases": len(np.unique(y)),
            "por_pliegue": scores.tolist(), "oof": oof}


def plot_confusion(y_true, y_pred, labels, title: str):
    """Matriz de confusión normalizada por fila (clase real)."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object),
                          labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(1 + 0.9 * len(labels), 0.8 + 0.8 * len(labels)))
    sns.heatmap(pd.DataFrame(cm, index=labels, columns=labels), annot=True, fmt=".2f",
                cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Imágenes media y de desviación
# ---------------------------------------------------------------------------


def mean_std_images(paths, groupings: dict, size, luma_weights) -> dict:
    """Imagen media (RGB) y media y desviación estándar de la luminancia por píxel.

    Cada imagen se redimensiona a `size` = (ancho, alto) por promedio de área, con
    valores en [0, 1]. Se acumulan en una sola pasada para todas las imágenes
    (`resultado["global"]["Todo"]`) y para cada valor de cada agrupación de
    `groupings` = {nombre: etiquetas por imagen} (`resultado[nombre][valor]`). Cada
    entrada tiene `media_rgb` (alto, ancho, 3), `media_lum` y `std_lum` (alto, ancho) y `n`.
    """
    w = np.asarray(luma_weights, dtype=np.float64)
    labels = {g: list(v) for g, v in groupings.items()}
    acc: dict = {}

    def add(key, x, lum):
        if key not in acc:
            acc[key] = [np.zeros_like(x), np.zeros_like(lum), np.zeros_like(lum), 0]
        a = acc[key]
        a[0] += x
        a[1] += lum
        a[2] += lum**2
        a[3] += 1

    for i, p in enumerate(tqdm(list(paths), desc="Imágenes media")):
        x = np.asarray(load_rgb(p).resize(tuple(size), Image.Resampling.BOX), dtype=np.float64) / 255.0
        lum = x @ w
        add(("global", "Todo"), x, lum)
        for g, lab in labels.items():
            if not pd.isna(lab[i]):
                add((g, lab[i]), x, lum)

    out: dict = {}
    for (g, k), (s, sl, sl2, n) in acc.items():
        media_lum = sl / n
        out.setdefault(g, {})[k] = {"media_rgb": s / n, "media_lum": media_lum,
                                    "std_lum": np.sqrt(np.maximum(sl2 / n - media_lum**2, 0)), "n": n}
    return out
