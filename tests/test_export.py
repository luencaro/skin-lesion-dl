"""Exportación a ONNX (src/export.py): CAM sin gradientes, FP16 con TOST, latencia y fuera de dominio."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from src.export import (
    cam_head,
    convert_fp16,
    export_keras_model,
    fp16_equivalence,
    knn_distance,
    l2_normalize,
    onnx_predict,
    onnx_session,
    roundtrip_check,
)
from tests.conftest import (
    positive_images,
    randomize_weight_net,
    tiny_cci,
    tiny_classifier,
)


def test_layernorm_head_cam_is_exact_up_to_affine(tmp_path):
    """Con LayerNorm en la cabeza (como ConvNeXt), head_weights = γ ⊙ W y el CAM normalizado coincide
    con el de la definición exacta salvo una transformación afín positiva por imagen."""
    keras.utils.set_random_seed(0)
    model = tiny_classifier(layernorm=True)
    ln = model.get_layer("head_layernorm")
    ln.gamma.assign(np.random.default_rng(0).uniform(0.5, 2.0, ln.gamma.shape))
    head = cam_head(model)
    assert head["kind"] == "gap_layernorm_linear"
    graph = export_keras_model(model, tmp_path / "m.onnx")
    x = positive_images(2)
    out = onnx_predict(onnx_session(tmp_path / "m.onnx"), x)
    w = model.get_layer("predictions").kernel.numpy()
    fmap = out["feature_map"]
    for n in range(2):
        flat = fmap[n].reshape(-1, fmap.shape[-1])
        mu, sd = flat.mean(), None
        # logit exacto: W^T (γ (GAP - μ) / σ + β) + b, con μ y σ de GAP(A) (escalares por imagen)
        g = flat.mean(0)
        mu, sd = g.mean(), np.sqrt(g.var() + ln.epsilon)
        exact = ((flat - mu) / sd * ln.gamma.numpy()) @ w             # CAM "exacto" por posición
        cam = flat @ out["head_weights"]
        for k in range(w.shape[1]):
            a, b = np.polyfit(cam[:, k], exact[:, k], 1)
            assert a > 0
            np.testing.assert_allclose(a * cam[:, k] + b, exact[:, k], rtol=1e-4, atol=1e-4)
    assert graph["outputs"][-2:] == ["feature_map", "head_weights"]


def test_head_without_gap_has_no_cam():
    inputs = keras.Input((8, 8, 3))
    from src.models.common import BACKBONE_NAME, OUTPUT_NAME, PREPROCESS_NAME

    inner = keras.Input((8, 8, 3))
    bb = keras.Model(inner, keras.layers.Conv2D(2, 3)(inner), name=BACKBONE_NAME)
    x = keras.layers.Rescaling(1.0, name=PREPROCESS_NAME)(inputs)
    x = keras.layers.Flatten()(bb(x))
    model = keras.Model(inputs, keras.layers.Dense(3, activation="softmax", name=OUTPUT_NAME)(x))
    assert cam_head(model) is None


def test_roundtrip_and_fp16(tmp_path):
    keras.utils.set_random_seed(0)
    model = tiny_cci("full")
    randomize_weight_net(model)
    export_keras_model(model, tmp_path / "m.onnx")
    x = positive_images(8, low=1, high=254)
    rt = roundtrip_check(model, onnx_session(tmp_path / "m.onnx"), x, 1e-4)
    assert rt["passed"] and rt["max_abs_error"] < 1e-4
    convert_fp16(tmp_path / "m.onnx", tmp_path / "m16.onnx")
    p16 = onnx_predict(onnx_session(tmp_path / "m16.onnx"), x)["probabilities"]
    assert p16.dtype == np.float32                                     # entradas y salidas en float32
    assert np.abs(p16 - model(x).numpy()).max() < 5e-2


def test_fp16_tost_decision():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    groups = np.repeat(np.arange(200), 2)
    pred = np.where(rng.random(400) < 0.7, y, rng.integers(0, 4, 400))
    same = fp16_equivalence(y, groups, pred, pred.copy(), delta=0.02, alpha=0.05, n_resamples=300, seed=0)
    assert same["equivalentes"] and same["diferencia"] == 0.0
    worse = np.where(rng.random(400) < 0.5, rng.integers(0, 4, 400), pred)
    diff = fp16_equivalence(y, groups, pred, worse, delta=0.02, alpha=0.05, n_resamples=300, seed=0)
    assert not diff["equivalentes"] and diff["diferencia"] < -0.02


def test_knn_distance_and_ood():
    rng = np.random.default_rng(0)
    ref = l2_normalize(rng.normal(size=(50, 8)) + 5)
    near = l2_normalize(ref[:3] + 0.01)
    far = l2_normalize(-ref[:3])
    assert knn_distance(near, ref, 5).max() < knn_distance(far, ref, 5).min()


def test_latency_benchmark_subprocess(tmp_path):
    from src.export import benchmark_onnx

    keras.utils.set_random_seed(0)
    export_keras_model(tiny_cci("shades_of_gray"), tmp_path / "m.onnx")
    res = benchmark_onnx(tmp_path / "m.onnx", {"repeats": 5, "warmup": 1, "intra_op_threads": 1})
    assert res["latency_ms_p95"] >= res["latency_ms_median"] > 0
    assert res["peak_rss_mb"] > 0 and res["file_size_mb"] > 0


@pytest.mark.parametrize("bad", [np.zeros((1, 32, 32, 4), np.float32)])
def test_onnx_rejects_wrong_shape(tmp_path, bad):
    keras.utils.set_random_seed(0)
    export_keras_model(tiny_cci("full"), tmp_path / "m.onnx")
    with pytest.raises(Exception, match="(?i)dimension|shape|index"):
        onnx_predict(onnx_session(tmp_path / "m.onnx"), bad)
