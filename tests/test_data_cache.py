"""Pruebas de la caché de imágenes redimensionadas (src/data.py), solo en CPU y con JPEG sintéticos.

Comprueban que la caché en disco devuelve exactamente las mismas imágenes que el redimensionado
del entrenamiento, que la caché uint8 difiere a lo sumo 0,5 por el redondeo, que barajar rutas en
lugar de imágenes conserva la permutación y que la verificación por SHA-256 detecta archivos
alterados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image

from src import data as D

CLASSES = ["a", "b", "c"]
N_IMAGES = 12
H, W = 45, 60          # misma relación de aspecto que 450 x 600
SIZE = 16


@pytest.fixture
def toy(tmp_path):
    """Imágenes JPEG sintéticas, su DataFrame y una configuración mínima."""
    rng = np.random.default_rng(0)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    ids = [f"ISIC_{i:07d}" for i in range(N_IMAGES)]
    for i in ids:
        arr = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        Image.fromarray(arr).save(images_dir / f"{i}.jpg", quality=90)
    df = pd.DataFrame({"image_id": ids, "dx": [CLASSES[k % len(CLASSES)] for k in range(N_IMAGES)]})
    cfg = {
        "paths": {"images_dir": images_dir, "resized_cache_dir": tmp_path / "resized_cache"},
        "image_ext": ".jpg",
        "valid_categories": {"dx": CLASSES},
        "train": {"image_size": SIZE, "resize_method": "bilinear", "resize_antialias": True, "batch_size": 4,
                  "cache": "disk", "cache_dtype": "float32", "resized_cache_verify": True, "augmentation": {}},
    }
    return cfg, df


def _with(cfg: dict, **train) -> dict:
    return {**cfg, "train": {**cfg["train"], **train}}


def _reference_images(cfg: dict, df: pd.DataFrame) -> np.ndarray:
    """Redimensionado del entrenamiento dentro de tf.data (como la caché en memoria anterior)."""
    load = D._image_loader(cfg)
    paths = [str(p) for p in D.image_paths(cfg, df["image_id"])]
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load)
    return np.stack([x.numpy() for x in ds])


def _images(source: D.SplitSource) -> np.ndarray:
    return np.stack([x.numpy() for x, _ in source.base])


def _train_epochs(cfg: dict, sources: dict, seed: int, epochs: int = 2) -> list[np.ndarray]:
    """Imágenes de train (sin aumento) en el orden barajado de `epochs` épocas."""
    ds = D.make_datasets(cfg, val_fold=0, seed=seed, sources=sources).train
    it = iter(ds.repeat(epochs))
    n_batches = -(-N_IMAGES // cfg["train"]["batch_size"])
    return [np.concatenate([next(it)[0].numpy() for _ in range(n_batches)]) for _ in range(epochs)]


def test_disk_cache_identical_to_memory_cache(toy):
    cfg, df = toy
    ref = _reference_images(cfg, df)
    disk = _images(D._split_source(cfg, df, CLASSES))
    memory = _images(D._split_source(_with(cfg, cache="memory"), df, CLASSES))
    assert disk.dtype == np.float32 and disk.shape == (N_IMAGES, SIZE, SIZE, 3)
    np.testing.assert_array_equal(disk, ref)      # bit a bit
    np.testing.assert_array_equal(memory, ref)


def test_disk_cache_is_computed_once(toy):
    cfg, df = toy
    D.ensure_resized_cache(cfg, df["image_id"])
    files = sorted(D.resized_cache_dir(cfg).glob(f"*{D.RESIZED_CACHE_SUFFIX}"))
    assert len(files) == N_IMAGES
    mtimes = [f.stat().st_mtime_ns for f in files]
    D.ensure_resized_cache(cfg, df["image_id"])
    assert [f.stat().st_mtime_ns for f in files] == mtimes


def test_uint8_cache_rounding_at_most_half(toy):
    cfg, df = toy
    ref = _reference_images(cfg, df)
    u8 = _images(D._split_source(_with(cfg, cache="memory", cache_dtype="uint8"), df, CLASSES))
    diff = np.abs(u8 - ref)
    assert diff.max() <= 0.5 + 1e-4
    assert np.all(u8 == np.round(u8))


@pytest.mark.parametrize("seed", [0, 3])
def test_shuffle_order_independent_of_cache(toy, seed):
    """Barajar rutas (disk) o imágenes (memory) con la misma semilla da la misma permutación."""
    cfg, df = toy
    epochs = {}
    for name, c in {"memory": _with(cfg, cache="memory"), "disk": cfg,
                    "uint8": _with(cfg, cache="memory", cache_dtype="uint8")}.items():
        src = D._split_source(c, df, CLASSES)
        tf.random.set_seed(1234)
        epochs[name] = _train_epochs(c, {"train": src, "val": src}, seed)
    for e_mem, e_disk, e_u8 in zip(epochs["memory"], epochs["disk"], epochs["uint8"]):
        np.testing.assert_array_equal(e_disk, e_mem)
        assert np.abs(e_u8 - e_mem).max() <= 0.5 + 1e-4
    assert not np.array_equal(epochs["memory"][0], epochs["memory"][1])  # se rebaraja cada época


def test_disk_cache_detects_corruption(toy):
    cfg, df = toy
    files = D.ensure_resized_cache(cfg, df["image_id"])
    with open(files[3], "r+b") as f:
        f.seek(100)
        b = f.read(1)
        f.seek(100)
        f.write(bytes([b[0] ^ 0xFF]))
    with pytest.raises(RuntimeError, match="SHA-256"):
        D.ensure_resized_cache(cfg, df["image_id"])
    D.ensure_resized_cache(cfg, df["image_id"], verify=False)  # sin verificar no se comprueba


def test_disk_cache_rejects_other_resize_params(toy):
    cfg, df = toy
    D.ensure_resized_cache(cfg, df["image_id"])
    manifest = D.resized_cache_dir(cfg) / "manifest.json"
    data = D.load_json(manifest)
    data["params"]["resize_antialias"] = False
    D.save_json(data, manifest)
    with pytest.raises(ValueError, match="parámetros"):
        D.ensure_resized_cache(cfg, df["image_id"])
    # Otro redimensionado usa otra subcarpeta
    other = _with(cfg, resize_antialias=False)
    assert D.resized_cache_dir(other) != D.resized_cache_dir(cfg)


def test_cache_mode_backwards_compatible():
    assert D.cache_mode({"cache_in_memory": True}) == "memory"
    assert D.cache_mode({"cache_in_memory": False}) == "none"
    assert D.cache_mode({"cache": "disk", "cache_in_memory": True}) == "disk"
    with pytest.raises(ValueError):
        D.cache_mode({"cache": "ssd"})
