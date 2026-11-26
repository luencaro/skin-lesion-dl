"""Evaluación externa (src/external.py) con conjuntos sintéticos pequeños, solo en CPU y sin red."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image

from src import data as D
from src import external as E

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SIZE = 16


def _cfg(tmp_path):
    return {"paths": {"images_dir": tmp_path / "ham", "resized_cache_dir": tmp_path / "rc"}, "image_ext": ".jpg",
            "valid_categories": {"dx": CLASSES},
            "train": {"image_size": SIZE, "resize_method": "bilinear", "resize_antialias": True, "batch_size": 4,
                      "ece_bins": 15, "cache": "disk", "resized_cache_verify": True}}


@pytest.fixture
def pad(tmp_path):
    """Conjunto tipo PAD-UFES-20: metadata.csv e imágenes PNG en subcarpetas."""
    root = tmp_path / "pad"
    rng = np.random.default_rng(0)
    rows = []
    for i, (dx, lesion) in enumerate([("ACK", 1), ("ACK", 1), ("BCC", 2), ("MEL", 3), ("NEV", 4), ("SEK", 5),
                                      ("SCC", 6), ("SCC", 7)]):
        part = root / f"imgs_part_{1 + i % 2}"
        part.mkdir(parents=True, exist_ok=True)
        name = f"PAT_{i}_{lesion}.png"
        Image.fromarray(rng.integers(0, 256, (30 + i, 40, 3), dtype=np.uint8)).save(part / name)
        rows.append({"patient_id": f"PAT_{i}", "lesion_id": str(lesion), "img_id": name, "diagnostic": dx})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    dcfg = {"name": "PAD-UFES-20", "root": root, "metadata": "metadata.csv", "image_column": "img_id",
            "image_dir": ".", "image_search": "recursive", "label_column": "diagnostic", "group_column": "lesion_id",
            "mapping": {"ACK": "akiec", "BCC": "bcc", "MEL": "mel", "NEV": "nv", "SEK": "bkl"}, "exclude": ["SCC"]}
    ecfg = {"datasets": {"pad": dcfg}, "hashes_json": tmp_path / "external_hashes.json",
            "cache_dir": tmp_path / "ext_cache", "results_dir": tmp_path / "out"}
    return ecfg, dcfg


def test_mapping_and_exclusions(pad):
    ecfg, _ = pad
    s = E.load_external_set("pad", ecfg, CLASSES)
    assert len(s.labels) == 6 and s.excluded == {"SCC": 2}
    assert [CLASSES[i] for i in s.labels] == ["akiec", "akiec", "bcc", "mel", "nv", "bkl"]
    assert list(s.groups) == ["1", "1", "2", "3", "4", "5"]
    assert all(p.endswith(".png") for p in s.paths)   # encontradas en las subcarpetas


def test_project_config_matches_protocol():
    ecfg = E.load_external_config()
    d = ecfg["datasets"]
    assert d["pad_ufes_20"]["mapping"] == {"ACK": "akiec", "BCC": "bcc", "MEL": "mel", "NEV": "nv", "SEK": "bkl"}
    assert d["pad_ufes_20"]["exclude"] == ["SCC"] and d["bcn20000"]["exclude"] == ["SCC"]
    assert d["bcn20000"]["mapping"] == {"NV": "nv", "MEL": "mel", "BCC": "bcc", "BKL": "bkl", "AK": "akiec",
                                        "DF": "df", "VASC": "vasc"}
    assert d["derm7pt"]["exclude"] == ["melanosis", "miscellaneous"] and d["derm7pt"]["image_column"] == "derm"
    assert set(d["derm7pt"]["mapping"].values()) == set(CLASSES) - {"akiec"}
    for dcfg in d.values():
        assert str(dcfg["root"]).endswith(f"data/external/{dcfg['root'].name}")


def test_unknown_label_is_an_error(pad):
    _, dcfg = pad
    meta = E.read_metadata(dcfg)
    meta.loc[0, "label"] = "LENTIGO"
    with pytest.raises(E.ExternalFormatError, match="sin mapeo ni exclusión"):
        E.apply_mapping(meta, dcfg, CLASSES)


def test_unexpected_metadata_columns_are_an_error(pad):
    _, dcfg = pad
    meta = pd.read_csv(dcfg["root"] / "metadata.csv").rename(columns={"diagnostic": "diagnosis"})
    meta.to_csv(dcfg["root"] / "metadata.csv", index=False)
    with pytest.raises(E.ExternalFormatError, match=r"no tiene las columnas \['diagnostic'\].*diagnosis"):
        E.read_metadata(dcfg)


def test_missing_metadata_file_and_images_are_errors(pad):
    ecfg, dcfg = pad
    with pytest.raises(E.ExternalFormatError, match="no existe"):
        E.read_metadata({**dcfg, "metadata": "otro.csv"})
    next(dcfg["root"].rglob("PAT_2_2.png")).unlink()
    with pytest.raises(E.ExternalFormatError, match="faltan 1 de 6"):
        E.load_external_set("pad", ecfg, CLASSES)


def test_balanced_accuracy_on_present_classes():
    y = np.array([0, 0, 1, 1, 1, 1])
    pred = np.array([0, 6, 1, 1, 1, 6])           # 6 es una clase ausente: cuenta como error
    assert E.balanced_accuracy_present(y, pred) == pytest.approx((0.5 + 0.75) / 2)
    proba = np.eye(7)[pred] * 0.9 + 0.1 / 7
    m = E.external_metrics(y, proba, CLASSES, 15)
    assert m["balanced_accuracy"] == pytest.approx(0.625) and m["clases_presentes"] == 2
    assert m["prediccion_clase_ausente"] == pytest.approx(2 / 6)
    assert np.isfinite(m["auc_roc_macro_ovr"]) and 0 <= m["f1_macro"] <= 1


def test_resize_same_as_ham10000(pad, tmp_path):
    ecfg, _ = pad
    cfg = _cfg(tmp_path)
    s = E.load_external_set("pad", ecfg, CLASSES)
    files = E.cached_images(cfg, ecfg, s)
    read = D._raw_image_loader(SIZE)
    cached = np.stack([read(f).numpy() for f in files])
    load = D._image_loader(cfg, any_format=True)
    ref = np.stack([x.numpy() for x in tf.data.Dataset.from_tensor_slices(list(s.paths)).map(load)])
    np.testing.assert_array_equal(cached, ref)
    assert cached.shape == (6, SIZE, SIZE, 3)
    # JPEG: decode_image da lo mismo que el decode_jpeg de HAM10000
    jpg = tmp_path / "x.jpg"
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (45, 60, 3), dtype=np.uint8)).save(jpg, quality=90)
    a = tf.data.Dataset.from_tensors(str(jpg)).map(D._image_loader(cfg)).get_single_element().numpy()
    b = tf.data.Dataset.from_tensors(str(jpg)).map(load).get_single_element().numpy()
    np.testing.assert_array_equal(a, b)


def test_hashes_recorded_then_checked(pad):
    ecfg, dcfg = pad
    reg = E.record_or_check_hashes(ecfg, ["pad"])
    assert len(reg["pad"]) == 9                     # 8 imágenes y metadata.csv
    E.record_or_check_hashes(ecfg, ["pad"])         # sin cambios: no falla
    with open(next(dcfg["root"].rglob("PAT_0_1.png")), "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="1 cambiados"):
        E.record_or_check_hashes(ecfg, ["pad"])


def test_friedman_control_with_holm():
    scores = pd.DataFrame({"cci": [0.80, 0.60, 0.55, 0.50], "a": [0.78, 0.58, 0.50, 0.52],
                           "b": [0.70, 0.40, 0.45, 0.30]}, index=["HAM10000", "PAD", "Derm7pt", "BCN"])
    out = E.friedman_control(scores, "cci")
    comp = out["comparaciones"].set_index("modelo")
    assert out["n_conjuntos"] == 4 and out["k_modelos"] == 3 and len(comp) == 2
    assert comp.loc["a", "rango_medio"] == 1.75 and comp.loc["b", "rango_medio"] == 3.0
    se = np.sqrt(3 * 4 / (6 * 4))
    assert comp.loc["b", "z"] == pytest.approx((3.0 - 1.25) / se)
    assert (comp["p_holm"] >= comp["p"]).all() and np.isfinite(out["iman_davenport"]["F"])


def test_derm7pt_direct_paths_with_subfolders(tmp_path):
    root = tmp_path / "derm7pt"
    (root / "meta").mkdir(parents=True)
    rows = []
    for i, dx in enumerate(["clark nevus", "melanoma (in situ)", "melanosis", "dermatofibroma"]):
        rel = f"NEL/Nel{i:03d}.jpg"
        (root / "images" / "NEL").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((20, 30, 3), 40 * i, np.uint8)).save(root / "images" / rel)
        rows.append({"case_num": str(i + 1), "diagnosis": dx, "derm": rel, "clinic": f"NEL/Nel{i:03d}c.jpg"})
    pd.DataFrame(rows).to_csv(root / "meta" / "meta.csv", index=False)
    dcfg = {**E.load_external_config()["datasets"]["derm7pt"], "root": root}
    s = E.load_external_set("d", {"datasets": {"d": dcfg}}, CLASSES)
    assert [CLASSES[i] for i in s.labels] == ["nv", "mel", "df"] and s.excluded == {"melanosis": 1}
    assert list(s.groups) == ["1", "2", "4"] and E._cache_key(s.image_ids[0]) == "NEL__Nel000.jpg"


def test_tables_from_saved_predictions(pad, tmp_path):
    ecfg, _ = pad
    cfg = {**_cfg(tmp_path), "benchmark": {"alpha": 0.05, "lesion_bootstrap": {"n_resamples": 200, "seed": 0}}}
    s = E.load_external_set("pad", ecfg, CLASSES)
    rng = np.random.default_rng(0)
    for model in ("cci", "a"):
        for fold in range(2):
            p = rng.dirichlet(np.ones(7), size=len(s.labels))
            p[np.arange(len(s.labels)), s.labels] += 1.0 if model == "cci" else 0.0
            p /= p.sum(axis=1, keepdims=True)
            df = pd.DataFrame({"image_id": s.image_ids, "group": s.groups, "label": [CLASSES[i] for i in s.labels],
                               "pred": [CLASSES[i] for i in p.argmax(axis=1)]})
            for i, c in enumerate(CLASSES):
                df[f"p_{c}"] = p[:, i]
            path = E.prediction_path(ecfg, model, "pad", 0, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
    m = E.metrics_table(cfg, ecfg, ["cci", "a"])
    assert len(m) == 4 and (m.loc[m.modelo == "cci", "balanced_accuracy"] == 1.0).all()
    b = E.bootstrap_table(cfg, ecfg, ["cci", "a"]).set_index("modelo")
    assert b.loc["cci", "balanced_accuracy"] == 1.0 and b.loc["cci", "corridas"] == 2
    assert b.loc["a", "ic95_bajo"] <= b.loc["a", "balanced_accuracy"] <= b.loc["a", "ic95_alto"]
    assert b.loc["a", "grupos"] == 5
