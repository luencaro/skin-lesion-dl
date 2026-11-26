"""Carga de datos, integridad, duplicados por pHash, partición agrupada y tf.data de HAM10000."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"

DUPLICATE_COLUMNS = [
    "image_id_a",
    "image_id_b",
    "lesion_id_a",
    "lesion_id_b",
    "dx_a",
    "dx_b",
    "distancia",
    "confirmado",
]
SPLIT_COLUMNS = ["image_id", "lesion_id", "group_id", "dx", "split", "fold"]


# ---------------------------------------------------------------------------
# Configuración y E/S
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict:
    """Lee el YAML de configuración y resuelve `paths` como rutas absolutas desde la raíz del proyecto."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg.get("paths", {}).items()}
    return cfg


def save_json(obj, path: str | Path) -> None:
    """Guarda `obj` como JSON con sangría, creando la carpeta si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metadata(cfg: dict) -> pd.DataFrame:
    """Lee el CSV de metadatos. La columna de fuente (`source_column`) es opcional."""
    return pd.read_csv(cfg["paths"]["metadata_csv"])


def load_metadata_validated(cfg: dict) -> pd.DataFrame:
    """load_metadata validado con el contrato de datos (paths.data_contract, src.audit). Lanza
    ValueError con las reglas que fallan."""
    from src.audit import FALLIDO, load_contract, validate_metadata

    meta = load_metadata(cfg)
    report = validate_metadata(meta, load_contract(cfg))
    bad = report[report["resultado"] == FALLIDO]
    if len(bad):
        rules = "; ".join(f"{r.columna}: {r.regla} ({r.infracciones})" for r in bad.itertuples())
        raise ValueError(f"Los metadatos no cumplen el contrato de datos: {rules}")
    return meta


def has_source_column(df: pd.DataFrame, cfg: dict) -> bool:
    col = cfg.get("source_column")
    return bool(col) and col in df.columns


def image_path(cfg: dict, image_id: str) -> Path:
    return cfg["paths"]["images_dir"] / f"{image_id}{cfg['image_ext']}"


def image_paths(cfg: dict, image_ids) -> list[Path]:
    return [image_path(cfg, i) for i in image_ids]


def mask_path(cfg: dict, image_id: str) -> Path:
    return cfg["paths"]["segmentations_dir"] / f"{image_id}{cfg['mask_suffix']}"


def mask_paths(cfg: dict, image_ids) -> list[Path]:
    return [mask_path(cfg, i) for i in image_ids]


# ---------------------------------------------------------------------------
# Integridad
# ---------------------------------------------------------------------------


def check_integrity(meta: pd.DataFrame, cfg: dict) -> dict:
    """Compara los image_id del CSV con los archivos de imagen presentes en disco."""
    ext = cfg["image_ext"]
    files = {p.stem for p in cfg["paths"]["images_dir"].glob(f"*{ext}")}
    ids = set(meta["image_id"])
    return {
        "n_filas": len(meta),
        "n_image_id_unicos": int(meta["image_id"].nunique()),
        "n_lesion_id_unicos": int(meta["lesion_id"].nunique()),
        "n_archivos_imagen": len(files),
        "image_id_repetidos": sorted(meta.loc[meta["image_id"].duplicated(), "image_id"]),
        "image_id_sin_archivo": sorted(ids - files),
        "archivos_sin_image_id": sorted(files - ids),
    }


def find_unreadable_images(paths) -> pd.DataFrame:
    """Devuelve las imágenes que PIL no puede verificar (`Image.verify`)."""
    rows = []
    for p in tqdm(paths, desc="Verificando imágenes"):
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001 - se registra cualquier fallo de lectura
            rows.append({"ruta": str(p), "error": repr(e)})
    return pd.DataFrame(rows, columns=["ruta", "error"])


def mark_missing(df: pd.DataFrame, tokens) -> pd.DataFrame:
    """Copia de `df` donde los valores de `tokens` (p. ej. "unknown", "") pasan a NaN."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].where(~out[col].isin(tokens))
    return out


def missing_summary(df: pd.DataFrame, tokens) -> pd.DataFrame:
    """Conteo y porcentaje de faltantes por columna, contando `tokens` como faltantes."""
    na = mark_missing(df, tokens).isna()
    return pd.DataFrame({"faltantes": na.sum(), "porcentaje": na.mean() * 100})


# ---------------------------------------------------------------------------
# Hashes de archivos
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_file_hashes(cfg: dict) -> dict:
    """SHA-256 de los .zip del primer nivel de `raw_dir` y del CSV de metadatos.

    La búsqueda no es recursiva: no entra en subcarpetas de `raw_dir`.
    """
    raw_dir = cfg["paths"]["raw_dir"]
    files = sorted(raw_dir.glob("*.zip")) + [cfg["paths"]["metadata_csv"]]
    return {
        str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
        for p in tqdm(files, desc="SHA-256")
    }


def record_hashes(hashes: dict, path: str | Path) -> pd.DataFrame:
    """Compara `hashes` con el registro en `path`.

    Si el registro no existe, lo crea con `hashes`. Si existe, no lo modifica.
    """
    path = Path(path)
    if path.exists():
        registrados = load_json(path)
    else:
        save_json(hashes, path)
        registrados = dict(hashes)
    nombres = sorted(set(hashes) | set(registrados))
    tabla = pd.DataFrame(
        {
            "sha256_actual": [hashes.get(n) for n in nombres],
            "sha256_registrado": [registrados.get(n) for n in nombres],
        },
        index=pd.Index(nombres, name="archivo"),
    )
    tabla["coincide"] = tabla["sha256_actual"] == tabla["sha256_registrado"]
    return tabla


# ---------------------------------------------------------------------------
# pHash y distancias de Hamming
# ---------------------------------------------------------------------------


def compute_phash(image_ids, paths, hash_size: int) -> pd.DataFrame:
    """pHash (imagehash) de cada imagen como cadena hexadecimal."""
    rows = []
    for iid, p in tqdm(zip(image_ids, paths), total=len(paths), desc="pHash"):
        with Image.open(p) as im:
            rows.append({"image_id": iid, "phash": str(imagehash.phash(im, hash_size=hash_size))})
    return pd.DataFrame(rows, columns=["image_id", "phash"])


def load_or_compute_phash(cfg: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """Reutiliza `phash_csv` si cubre todos los image_id con el tamaño de hash actual; si no, lo recalcula."""
    path = cfg["paths"]["phash_csv"]
    hash_size = cfg["phash_hash_size"]
    expected_len = hash_size * hash_size // 4
    if path.exists():
        cached = pd.read_csv(path, dtype={"image_id": str, "phash": str})
        if set(meta["image_id"]) <= set(cached["image_id"]) and (
            cached["phash"].str.len() == expected_len
        ).all():
            print(f"pHash reutilizado desde {path}")
            return cached[cached["image_id"].isin(meta["image_id"])].reset_index(drop=True)
    df = compute_phash(meta["image_id"], image_paths(cfg, meta["image_id"]), hash_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"pHash guardado en {path}")
    return df


def _hex_to_bytes(hex_hashes) -> np.ndarray:
    """Matriz (N, n_bytes) uint8 a partir de hashes hexadecimales de igual longitud."""
    raw = b"".join(bytes.fromhex(h) for h in hex_hashes)
    n = len(hex_hashes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, -1)


def hamming_pairs(hex_hashes, threshold: int, block_size: int = 512) -> pd.DataFrame:
    """Pares (i, j), i < j, con distancia de Hamming <= `threshold`.

    Compara por bloques de filas contra todas las columnas posteriores con operaciones
    vectorizadas de NumPy (XOR + conteo de bits), sin bucles dobles sobre las imágenes.
    """
    packed = _hex_to_bytes(list(hex_hashes))
    n = len(packed)
    out_i, out_j, out_d = [], [], []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = packed[start:stop]
        cols = packed[start:]
        dist = np.bitwise_count(block[:, None, :] ^ cols[None, :, :]).sum(axis=2, dtype=np.int32)
        ii, jj = np.nonzero(dist <= threshold)
        gi, gj = ii + start, jj + start
        keep = gj > gi
        out_i.append(gi[keep])
        out_j.append(gj[keep])
        out_d.append(dist[ii[keep], jj[keep]])
    if not out_i:
        return pd.DataFrame({"i": [], "j": [], "distancia": []}, dtype=int)
    return pd.DataFrame(
        {"i": np.concatenate(out_i), "j": np.concatenate(out_j), "distancia": np.concatenate(out_d)}
    )


def distance_distribution(
    meta: pd.DataFrame, phash_df: pd.DataFrame, max_distance: int, block_size: int = 512
) -> pd.DataFrame:
    """Conteo de todos los pares de imágenes por distancia de Hamming, de 0 a `max_distance`.

    Columnas: pares con el mismo lesion_id, pares con lesion_id distinto y, de estos
    últimos, cuántos tienen además dx distinto. Recorre todos los pares i < j por bloques
    de filas con operaciones vectorizadas (sin bucles dobles sobre las imágenes).
    """
    df = phash_df.merge(meta[["image_id", "lesion_id", "dx"]], on="image_id").reset_index(drop=True)
    packed = _hex_to_bytes(df["phash"].tolist())
    lesion = pd.factorize(df["lesion_id"])[0]
    dx = pd.factorize(df["dx"])[0]
    n_bins = packed.shape[1] * 8 + 1
    same = np.zeros(n_bins, dtype=np.int64)
    diff = np.zeros(n_bins, dtype=np.int64)
    diff_dx = np.zeros(n_bins, dtype=np.int64)
    n = len(df)
    for start in range(0, n - 1, block_size):
        stop = min(start + block_size, n)
        dist = np.bitwise_count(packed[start:stop, None, :] ^ packed[None, start:, :]).sum(axis=2, dtype=np.int64)
        upper = np.arange(start, n)[None, :] > np.arange(start, stop)[:, None]
        same_les = lesion[start:stop, None] == lesion[None, start:]
        other_dx = dx[start:stop, None] != dx[None, start:]
        same += np.bincount(dist[upper & same_les], minlength=n_bins)
        diff += np.bincount(dist[upper & ~same_les], minlength=n_bins)
        diff_dx += np.bincount(dist[upper & ~same_les & other_dx], minlength=n_bins)
    out = pd.DataFrame(
        {
            "pares_mismo_lesion_id": same,
            "pares_lesion_id_distinto": diff,
            "pares_lesion_id_distinto_dx_distinto": diff_dx,
        },
        index=pd.RangeIndex(n_bins, name="distancia"),
    )
    return out.loc[: max_distance]


def within_lesion_distances(meta: pd.DataFrame, phash_df: pd.DataFrame) -> pd.DataFrame:
    """Distancia de Hamming entre cada par de imágenes que comparten lesion_id."""
    df = meta[["image_id", "lesion_id", "dx"]].merge(phash_df, on="image_id").reset_index(drop=True)
    df["pos"] = np.arange(len(df))
    packed = _hex_to_bytes(df["phash"].tolist())
    pairs = df[["pos", "image_id", "lesion_id", "dx"]].merge(
        df[["pos", "image_id", "lesion_id"]], on="lesion_id", suffixes=("_a", "_b")
    )
    pairs = pairs[pairs["pos_a"] < pairs["pos_b"]].reset_index(drop=True)
    pairs["distancia"] = np.bitwise_count(
        packed[pairs["pos_a"].to_numpy()] ^ packed[pairs["pos_b"].to_numpy()]
    ).sum(axis=1)
    return pairs[["lesion_id", "dx", "image_id_a", "image_id_b", "distancia"]]


def duplicate_candidates(meta: pd.DataFrame, phash_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Pares con distancia <= `threshold` y lesion_id distinto. La columna `confirmado` queda vacía."""
    df = (
        phash_df.merge(meta[["image_id", "lesion_id", "dx"]], on="image_id")
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    pairs = hamming_pairs(df["phash"], threshold)
    a = df.iloc[pairs["i"].to_numpy()].reset_index(drop=True)
    b = df.iloc[pairs["j"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "image_id_a": a["image_id"],
            "image_id_b": b["image_id"],
            "lesion_id_a": a["lesion_id"],
            "lesion_id_b": b["lesion_id"],
            "dx_a": a["dx"],
            "dx_b": b["dx"],
            "distancia": pairs["distancia"].to_numpy(),
            "confirmado": pd.NA,
        }
    )
    out = out[out["lesion_id_a"] != out["lesion_id_b"]]
    return out.sort_values(["distancia", "image_id_a", "image_id_b"]).reset_index(drop=True)


def load_confirmed_duplicates(path: str | Path) -> pd.DataFrame | None:
    """Pares con `confirmado == 1` del archivo revisado; None si el archivo no existe.

    Lanza ValueError si faltan columnas o si alguna fila de `confirmado` no es 0 ni 1.
    """
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    faltan = [c for c in DUPLICATE_COLUMNS if c not in df.columns]
    if faltan:
        raise ValueError(f"{path} no tiene las columnas {faltan}")
    conf = pd.to_numeric(df["confirmado"], errors="coerce")
    invalidas = ~conf.isin([0, 1])
    if invalidas.any():
        raise ValueError(
            f"{path}: {int(invalidas.sum())} filas con 'confirmado' distinto de 0 o 1"
        )
    return df[conf == 1].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Grupos (union-find)
# ---------------------------------------------------------------------------


class UnionFind:
    """Union-find con compresión de caminos y unión por rango."""

    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = dict.fromkeys(self.parent, 0)

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_groups(meta: pd.DataFrame, confirmed_pairs: pd.DataFrame | None = None) -> pd.Series:
    """group_id por imagen: une las imágenes de un mismo lesion_id y los pares confirmados.

    El identificador de cada grupo es el menor lesion_id del grupo. El resultado está
    alineado con el índice de `meta`.
    """
    uf = UnionFind(meta["image_id"])
    for _, ids in meta.groupby("lesion_id")["image_id"]:
        members = ids.tolist()
        for other in members[1:]:
            uf.union(members[0], other)
    if confirmed_pairs is not None and len(confirmed_pairs):
        known = set(uf.parent)
        desconocidos = (
            set(confirmed_pairs["image_id_a"]) | set(confirmed_pairs["image_id_b"])
        ) - known
        if desconocidos:
            raise ValueError(f"Pares con image_id que no están en los metadatos: {sorted(desconocidos)[:10]}")
        for a, b in zip(confirmed_pairs["image_id_a"], confirmed_pairs["image_id_b"]):
            uf.union(a, b)
    roots = meta["image_id"].map(uf.find)
    return meta.groupby(roots)["lesion_id"].transform("min").rename("group_id")


# ---------------------------------------------------------------------------
# Partición
# ---------------------------------------------------------------------------


def closest_n_splits(test_size: float) -> int:
    """Número de pliegues k (>= 2) cuyo 1/k es el más cercano a `test_size`."""
    upper = max(3, int(np.ceil(2 / test_size)))
    return min(range(2, upper + 1), key=lambda k: (abs(1 / k - test_size), k))


def make_splits(
    meta: pd.DataFrame, group_id: pd.Series, test_size: float, n_folds: int, seed: int
) -> pd.DataFrame:
    """Partición train/test y pliegues de train con StratifiedGroupKFold (estrato dx, grupo group_id).

    Test es el primer pliegue de un StratifiedGroupKFold con `closest_n_splits(test_size)`
    pliegues. Train se reparte en `n_folds` pliegues; en test, fold = -1.
    """
    df = meta[["image_id", "lesion_id", "dx"]].reset_index(drop=True)
    df["group_id"] = group_id.to_numpy()
    n = len(df)

    k_test = closest_n_splits(test_size)
    outer = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    _, test_idx = next(outer.split(df, df["dx"], groups=df["group_id"]))

    split = np.full(n, "train", dtype=object)
    split[test_idx] = "test"
    fold = np.full(n, -1, dtype=int)

    train_pos = np.flatnonzero(split == "train")
    tr = df.iloc[train_pos]
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, val_idx) in enumerate(inner.split(tr, tr["dx"], groups=tr["group_id"])):
        fold[train_pos[val_idx]] = f

    df["split"] = split
    df["fold"] = fold
    return df[SPLIT_COLUMNS]


def save_splits(splits: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits[SPLIT_COLUMNS].to_csv(path, index=False, lineterminator="\n")


def load_splits(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str, "lesion_id": str, "group_id": str, "dx": str, "split": str})
    faltan = [c for c in SPLIT_COLUMNS if c not in df.columns]
    if faltan:
        raise ValueError(f"{path} no tiene las columnas {faltan}")
    return df[SPLIT_COLUMNS]


def validate_splits(splits: pd.DataFrame, n_folds: int | None = None) -> None:
    """Asserts de la partición: imágenes únicas y ningún grupo en dos particiones ni en dos pliegues."""
    assert splits["image_id"].is_unique, "Hay image_id repetidos en la partición"
    assert splits["split"].isin(["train", "test"]).all(), "Valores de split distintos de train/test"
    assert splits.groupby("group_id")["split"].nunique().max() == 1, "Un group_id aparece en train y en test"
    train = splits[splits["split"] == "train"]
    test = splits[splits["split"] == "test"]
    assert (test["fold"] == -1).all(), "Hay imágenes de test con fold distinto de -1"
    assert (train["fold"] >= 0).all(), "Hay imágenes de train sin pliegue"
    assert train.groupby("group_id")["fold"].nunique().max() == 1, "Un group_id aparece en dos pliegues"
    if n_folds is not None:
        assert set(train["fold"].unique()) == set(range(n_folds)), "Los pliegues no son 0..n_folds-1"


def check_splits_match(splits: pd.DataFrame, meta: pd.DataFrame, group_id: pd.Series) -> None:
    """Asserts de coherencia entre una partición guardada y los metadatos/grupos actuales."""
    assert set(splits["image_id"]) == set(meta["image_id"]), (
        "La partición y los metadatos no tienen los mismos image_id"
    )
    actual = pd.DataFrame({"image_id": meta["image_id"].to_numpy(), "grupo_actual": group_id.to_numpy()})
    m = splits.merge(actual, on="image_id")
    assert m.groupby("grupo_actual")["split"].nunique().max() == 1, (
        "Un grupo actual (lesion_id + duplicados confirmados) cruza train y test: "
        "regenera la partición con overwrite_splits: true"
    )
    tr = m[m["split"] == "train"]
    assert tr.groupby("grupo_actual")["fold"].nunique().max() == 1, (
        "Un grupo actual cruza dos pliegues: regenera la partición con overwrite_splits: true"
    )


def splits_hashes(path: str | Path, splits: pd.DataFrame) -> dict:
    """SHA-256 del archivo de partición y de la lista ordenada de image_id de test."""
    test_ids = sorted(splits.loc[splits["split"] == "test", "image_id"])
    return {
        "splits_csv_sha256": sha256_file(path),
        "test_image_ids_sha256": sha256_text("\n".join(test_ids)),
        "n_train": int((splits["split"] == "train").sum()),
        "n_test": len(test_ids),
    }


# ---------------------------------------------------------------------------
# tf.data para el entrenamiento
# ---------------------------------------------------------------------------
# TensorFlow se importa dentro de las funciones para que el EDA no lo cargue.


def class_names(cfg: dict) -> list[str]:
    """Clases en el orden que define el índice de la etiqueta (valid_categories.dx)."""
    return list(cfg["valid_categories"]["dx"])


@dataclass
class SplitSource:
    """Imágenes de una partición: image_id y etiquetas en orden, y el dataset (imagen, etiqueta)
    sin lotes, ya decodificado, redimensionado y (si `cache_in_memory`) en cache."""

    image_ids: np.ndarray
    labels: np.ndarray
    base: object  # tf.data.Dataset


@dataclass
class Datasets:
    train: object  # tf.data.Dataset (aumentado, barajado, en lotes)
    train_eval: object  # train sin aumento ni barajado, en lotes (orden de train_ids)
    val: object
    test: object | None
    train_ids: np.ndarray
    train_labels: np.ndarray
    val_ids: np.ndarray
    val_labels: np.ndarray
    test_ids: np.ndarray | None
    test_labels: np.ndarray | None
    class_names: list[str]
    sources: dict  # {"train": SplitSource, "val": ..., "test": ...}; reutilizable entre corridas


def select_split(splits: pd.DataFrame, which: str, val_fold: int) -> pd.DataFrame:
    """Filas de train (split == train y fold != val_fold), val (fold == val_fold) o test, ordenadas por image_id."""
    if which == "train":
        m = (splits["split"] == "train") & (splits["fold"] != val_fold)
    elif which == "val":
        m = (splits["split"] == "train") & (splits["fold"] == val_fold)
    elif which == "test":
        m = splits["split"] == "test"
    else:
        raise ValueError(f"Partición desconocida: {which}")
    return splits[m].sort_values("image_id").reset_index(drop=True)


def _image_loader(cfg: dict, cache_uint8: bool = False):
    """Función ruta -> imagen redimensionada a image_size x image_size (float32 en [0, 255], o
    uint8 si `cache_uint8`), con el mismo redimensionado que el entrenamiento."""
    import tensorflow as tf

    tcfg = cfg["train"]
    size = int(tcfg["image_size"])
    method = tcfg.get("resize_method", "bilinear")
    antialias = bool(tcfg.get("resize_antialias", True))

    def load(path):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, [size, size], method=method, antialias=antialias)  # float32
        img = tf.clip_by_value(img, 0.0, 255.0)
        if cache_uint8:
            img = tf.cast(tf.round(img), tf.uint8)
        return img

    return load


def load_resized_images(cfg: dict, image_ids) -> np.ndarray:
    """Imágenes (N, image_size, image_size, 3) float32 en [0, 255], redimensionadas como en el
    entrenamiento y sin aumento (para Grad-CAM y visualizaciones)."""
    load = _image_loader(cfg)
    return np.stack([load(str(image_path(cfg, i))).numpy() for i in image_ids]).astype(np.float32)


def _split_source(cfg: dict, df: pd.DataFrame, classes: list[str]):
    import tensorflow as tf

    tcfg = cfg["train"]
    cache_uint8 = tcfg.get("cache_dtype", "float32") == "uint8"

    unknown = sorted(set(df["dx"]) - set(classes))
    if unknown:
        raise ValueError(f"Etiquetas fuera de valid_categories.dx: {unknown}")
    labels = df["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    paths = [str(p) for p in image_paths(cfg, df["image_id"])]
    faltan = [p for p in paths if not Path(p).exists()]
    if faltan:
        raise FileNotFoundError(f"{len(faltan)} imágenes no encontradas, p. ej. {faltan[0]}")

    load_image = _image_loader(cfg, cache_uint8)

    def load(path, label):
        return load_image(path), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if tcfg.get("cache_in_memory", True):
        ds = ds.cache()
    if cache_uint8:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return SplitSource(image_ids=df["image_id"].to_numpy(), labels=labels, base=ds)


def load_split_sources(cfg: dict, val_fold: int, include_test: bool = False) -> dict:
    """SplitSource de train, val y (opcional) test para un pliegue de validación."""
    splits = load_splits(cfg["paths"]["splits_csv"])
    n_folds = cfg.get("n_folds")
    if n_folds is not None and not 0 <= val_fold < n_folds:
        raise ValueError(f"val_fold={val_fold} fuera de 0..{n_folds - 1}")
    classes = class_names(cfg)
    names = ["train", "val"] + (["test"] if include_test else [])
    return {n: _split_source(cfg, select_split(splits, n, val_fold), classes) for n in names}


def make_augment_fn(aug: dict, seed: int):
    """Función (imagen, etiqueta) -> (imagen aumentada, etiqueta) sobre imágenes float32 en [0, 255].

    Usa operaciones aleatorias sin estado alimentadas por un tf.random.Generator con `seed`.
    """
    import tensorflow as tf

    rng = tf.random.Generator.from_seed(seed, alg="philox")
    n_seeds = 8

    def augment(img, label):
        seeds = rng.make_seeds(n_seeds)
        s = [seeds[:, i] for i in range(n_seeds)]
        static_shape = img.shape
        size = tf.shape(img)[:2]
        if aug.get("flip_horizontal"):
            img = tf.image.stateless_random_flip_left_right(img, s[0])
        if aug.get("flip_vertical"):
            img = tf.image.stateless_random_flip_up_down(img, s[1])
        if aug.get("rotate_90"):
            k = tf.random.stateless_uniform([], s[2], minval=0, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k)
        zoom = float(aug.get("zoom_max") or 0.0)
        if zoom > 0:
            u = tf.random.stateless_uniform([3], s[3])
            frac = 1.0 / (1.0 + zoom * u[0])          # lado relativo del recorte
            y0, x0 = (1.0 - frac) * u[1], (1.0 - frac) * u[2]
            box = tf.stack([y0, x0, y0 + frac, x0 + frac])[None, :]
            img = tf.image.crop_and_resize(img[None], box, [0], size, method="bilinear")[0]
        b, c = float(aug.get("brightness") or 0), float(aug.get("contrast") or 0)
        sat, hue = float(aug.get("saturation") or 0), float(aug.get("hue") or 0)
        if b or c or sat or hue:
            x = img / 255.0
            if b:
                x = tf.image.stateless_random_brightness(x, b, s[4])
            if c:
                x = tf.image.stateless_random_contrast(x, 1.0 - c, 1.0 + c, s[5])
            if sat:
                x = tf.image.stateless_random_saturation(x, 1.0 - sat, 1.0 + sat, s[6])
            if hue:
                x = tf.image.stateless_random_hue(x, hue, s[7])
            img = tf.clip_by_value(x, 0.0, 1.0) * 255.0
        return tf.ensure_shape(tf.clip_by_value(img, 0.0, 255.0), static_shape), label

    return augment


def micro_batch_size(batch_size: int, accumulation_steps: int) -> int:
    """Micro-lote del tf.data con acumulación de gradientes: batch_size / accumulation_steps."""
    if accumulation_steps < 1 or batch_size % accumulation_steps:
        raise ValueError(f"train.batch_size ({batch_size}) debe ser múltiplo de gradient_accumulation_steps "
                         f"({accumulation_steps})")
    return batch_size // accumulation_steps


def check_accumulation(n_train: int, batch_size: int, accumulation_steps: int) -> None:
    """Error si el último lote de la época tiene menos imágenes que `accumulation_steps` (no podría
    partirse en ese número de micro-lotes no vacíos)."""
    last = n_train % batch_size or batch_size
    if accumulation_steps > 1 and last < accumulation_steps:
        raise ValueError(f"El último lote de train tiene {last} imágenes, menos que gradient_accumulation_steps "
                         f"({accumulation_steps})")


def split_into_micro_batches(accumulation_steps: int):
    """Función para flat_map: parte un lote (x, y) de tamaño b en `accumulation_steps` micro-lotes
    consecutivos de tamaños floor((i + 1) b / a) - floor(i b / a), todos no vacíos si b >= a. Cada
    actualización acumula exactamente los micro-lotes de un lote efectivo, así que las
    actualizaciones no se desalinean entre épocas aunque el último lote esté incompleto."""
    import tensorflow as tf

    a = int(accumulation_steps)

    def split(x, y):
        b = tf.shape(x)[0]
        bounds = (tf.range(a + 1) * b) // a
        return tf.data.Dataset.range(a).map(
            lambda i: (x[bounds[i]:bounds[i + 1]], y[bounds[i]:bounds[i + 1]]))

    return split


def make_datasets(
    cfg: dict, val_fold: int, seed: int, include_test: bool = False, sources: dict | None = None,
    accumulation_steps: int = 1,
) -> Datasets:
    """tf.data de train (aumentado y barajado con `seed`), train sin aumento, val y test (opcional).

    Train: split == "train" y fold != val_fold; val: fold == val_fold; test: split == "test".
    Las imágenes originales se redimensionan a image_size x image_size sin conservar la
    relación de aspecto, como float32 en [0, 255]; la cache va tras el redimensionado y el
    aumento después de la cache. Val y test conservan el orden de `val_ids` / `test_ids`.
    `sources` (de un Datasets anterior con el mismo val_fold) reutiliza la cache. Con
    `accumulation_steps` > 1, cada lote de train.batch_size de train se parte en ese número de
    micro-lotes (split_into_micro_batches) y val, test y train sin aumento usan lotes de
    batch_size / accumulation_steps.
    """
    import tensorflow as tf

    if sources is None:
        sources = load_split_sources(cfg, val_fold, include_test)
    elif include_test and "test" not in sources:
        sources = dict(sources, test=load_split_sources(cfg, val_fold, True)["test"])

    tcfg = cfg["train"]
    bs = int(tcfg["batch_size"])
    micro = micro_batch_size(bs, accumulation_steps)
    tr = sources["train"]
    train = (
        tr.base.shuffle(len(tr.labels), seed=seed, reshuffle_each_iteration=True)
        .map(make_augment_fn(tcfg.get("augmentation") or {}, seed), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(bs)
    )
    if accumulation_steps > 1:
        check_accumulation(len(tr.labels), bs, accumulation_steps)
        train = train.flat_map(split_into_micro_batches(accumulation_steps))
    train = train.prefetch(tf.data.AUTOTUNE)

    def evaluation(src):
        return src.base.batch(micro).prefetch(tf.data.AUTOTUNE)

    te = sources.get("test") if include_test else None
    return Datasets(
        train=train,
        train_eval=evaluation(tr),
        val=evaluation(sources["val"]),
        test=evaluation(te) if te is not None else None,
        train_ids=tr.image_ids,
        train_labels=tr.labels,
        val_ids=sources["val"].image_ids,
        val_labels=sources["val"].labels,
        test_ids=te.image_ids if te is not None else None,
        test_labels=te.labels if te is not None else None,
        class_names=class_names(cfg),
        sources=sources,
    )


def class_weights(labels, n_classes: int) -> dict[int, float]:
    """Pesos de frecuencia inversa w_k = N / (K * N_k)."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes)
    if (counts == 0).any():
        raise ValueError(f"Clases sin ejemplos en train: {np.flatnonzero(counts == 0).tolist()}")
    n = counts.sum()
    return {k: float(n / (n_classes * counts[k])) for k in range(n_classes)}
