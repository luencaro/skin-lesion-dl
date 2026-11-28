"""Pruebas de la capa CCI y del modelo original (src/original_model.py), solo en CPU."""

from __future__ import annotations

import math

import keras
import numpy as np
import pytest
import tensorflow as tf

from src.original_model import (
    CCI_NAME,
    THETA_NAME,
    VARIANTS,
    CCILayer,
    area_pool_matrix,
    build_cci_model,
    inverse_softplus,
)
from tests.conftest import (
    LAYER_CFG,
    N_CLASSES,
    REDUCED,
    SIZE,
    log_uniform_gains,
    pattern_dataset,
    positive_images,
    randomize_weight_net,
    sharpen_head,
    structured_images,
    tiny_cci,
    tiny_classifier,
)

# ---------------------------------------------------------------------------
# Construcción, formas y validación de la entrada (iii)
# ---------------------------------------------------------------------------


def test_parameter_count_matches_chapter_24():
    """g_φ con 16 filtros sobre 56 x 56: 2785 parámetros más ϑ (capítulo 24, sección Complejidad)."""
    layer = CCILayer(reduced_size=56, filters=16)
    layer.build((None, 224, 224, 3))
    assert layer.count_params() == 2785 + 1
    assert layer.trainable_weights[0].name == THETA_NAME


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_output_shapes(variant):
    model = tiny_cci(variant)
    x = positive_images(3)
    assert model(x).shape == (3, N_CLASSES)
    layer = model.get_layer(CCI_NAME)
    assert layer(x).shape == x.shape
    info = layer.explain(x)
    assert info["weights"].shape == (3, REDUCED, REDUCED)
    np.testing.assert_allclose(info["weights"].sum(axis=(1, 2)), 1.0, atol=1e-5)
    assert info["illuminant"].shape == (3, 3)
    assert info["corrected"].shape == x.shape
    assert info["p"] == pytest.approx(6.0, abs=1e-5)


def test_initial_state_is_shades_of_gray():
    """Con ϑ = softplus⁻¹(5) y la última capa de g_φ en cero, la capa empieza como Shades of Gray con p = 6."""
    layer = tiny_cci("full").get_layer(CCI_NAME)
    x = positive_images(2)
    info = layer.explain(x)
    np.testing.assert_allclose(info["weights"], 1.0 / REDUCED ** 2, rtol=1e-5)
    xbar = info["reduced"].reshape(2, -1, 3).astype(np.float64)
    sog = np.mean(xbar ** 6, axis=1) ** (1 / 6)
    np.testing.assert_allclose(info["illuminant"], sog, rtol=1e-4)


@pytest.mark.parametrize("bad, error, match", [
    (np.full((1, SIZE, SIZE, 4), 50.0, np.float32), ValueError, "3 canales"),
    (np.full((1, SIZE, SIZE, 1), 50.0, np.float32), ValueError, "3 canales"),
    (np.full((SIZE, SIZE, 3), 50.0, np.float32), ValueError, "B, H, W, 3"),
    (np.full((1, SIZE, SIZE, 3), 50, np.int32), TypeError, "punto flotante"),
    (np.full((1, SIZE, SIZE, 3), 50, np.uint8), TypeError, "punto flotante"),
    (np.full((1, SIZE, SIZE, 3), 300.0, np.float32), ValueError, "fuera de"),
    (np.full((1, SIZE, SIZE, 3), -1.0, np.float32), ValueError, "fuera de"),
    (np.full((1, SIZE, SIZE, 3), np.nan, np.float32), ValueError, "no finitos"),
])
def test_invalid_inputs_raise_clear_errors(bad, error, match):
    layer = tiny_cci("full").get_layer(CCI_NAME)
    with pytest.raises(error, match=match):
        layer(tf.convert_to_tensor(bad))


def test_wrong_spatial_size_after_build():
    layer = tiny_cci("full").get_layer(CCI_NAME)
    with pytest.raises(ValueError, match="se construyó"):
        layer(positive_images(1, size=2 * SIZE))


def test_invalid_variant_and_backbone():
    with pytest.raises(ValueError, match="Variante desconocida"):
        build_cci_model(tiny_classifier(), "otra", layer_cfg=LAYER_CFG)
    with pytest.raises(TypeError):
        build_cci_model(3, "full", layer_cfg=LAYER_CFG)


def test_backbone_null_explains_rule():
    from src.original_model import build_from_config

    with pytest.raises(ValueError, match="capítulo 24"):
        build_from_config({"name": "cci", "backbone": None}, 7, 0.0)


def test_classifier_is_not_modified_and_structure_is_kept():
    from src.models.common import (
        get_backbone,
        head_layers,
        pre_backbone_layers,
        set_backbone_trainable,
    )

    base = tiny_classifier()
    x = positive_images(2)
    before = base(x).numpy()
    model = build_cci_model(base, "full", layer_cfg=LAYER_CFG)
    np.testing.assert_array_equal(base(x).numpy(), before)          # el clasificador original no cambia
    assert [lyr.name for lyr in pre_backbone_layers(model)] == [CCI_NAME, "preprocess"]
    assert [lyr.name for lyr in head_layers(model)] == ["gap", "dropout", "predictions"]
    assert get_backbone(model) is get_backbone(base)                 # capas compartidas, no copiadas
    set_backbone_trainable(model, False)                            # fase A: capa CCI y cabeza entrenables
    names = {w.path for w in model.trainable_weights}
    assert any(THETA_NAME in n for n in names) and any("predictions" in n for n in names)
    assert not any(n.startswith("backbone") for n in names)


def test_area_pool_matrix_matches_average_pooling():
    m = area_pool_matrix(12, 3)
    np.testing.assert_allclose(m.sum(axis=1), 1.0)
    np.testing.assert_allclose(m[0, :4], 0.25)
    m = area_pool_matrix(10, 4)                                      # no divisible: pesos fraccionarios
    np.testing.assert_allclose(m.sum(axis=1), 1.0)
    assert m[0, 2] == pytest.approx(0.5 / 2.5)
    layer = CCILayer(**{**LAYER_CFG, "reduced_size": 7})
    x = positive_images(2, size=30)
    assert layer(x).shape == x.shape


def test_inverse_softplus():
    for y in (0.1, 1.0, 5.0, 20.0):
        assert math.log1p(math.exp(inverse_softplus(y))) == pytest.approx(y)


# ---------------------------------------------------------------------------
# Invarianza a las ganancias de canal (v)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["full", "fixed_p", "uniform_weights", "shades_of_gray"])
def test_gain_invariance(variant):
    """max |f(Dx) − f(x)| < 1e-4 para ganancias log-uniformes en [0,5; 2] (teorema del capítulo 24)."""
    keras.utils.set_random_seed(0)
    model = tiny_cci(variant)
    randomize_weight_net(model, scale=10.0)
    sharpen_head(model)
    x = np.concatenate([positive_images(3), structured_images(3)])
    worst = 0.0
    for seed in range(5):
        d = log_uniform_gains(len(x), seed=seed)
        dx = x * d
        assert dx.min() > 1.0 and dx.max() < 255.0                  # sin ceros ni saturación
        worst = max(worst, float(np.abs(model(dx).numpy() - model(x).numpy()).max()))
    assert worst < 1e-4


def test_variant_without_invariance_is_not_invariant():
    keras.utils.set_random_seed(0)
    model = tiny_cci("no_invariance")
    randomize_weight_net(model, scale=10.0)
    sharpen_head(model)
    x = structured_images(6)
    d = log_uniform_gains(len(x), seed=3)
    diff = float(np.abs(model(x * d).numpy() - model(x).numpy()).max())
    assert diff > 1e-4
    # y el modelo completo con los mismos pesos sí lo es
    full = tiny_cci("full")
    full.set_weights(model.get_weights())
    assert float(np.abs(full(x * d).numpy() - full(x).numpy()).max()) < 1e-4


def test_backbone_alone_is_not_invariant():
    keras.utils.set_random_seed(0)
    base = tiny_classifier()
    sharpen_head(base)
    x = structured_images(6)
    d = log_uniform_gains(len(x), seed=3)
    assert float(np.abs(base(x * d).numpy() - base(x).numpy()).max()) > 1e-4


def test_illuminant_is_homogeneous_and_weights_invariant():
    keras.utils.set_random_seed(0)
    model = tiny_cci("full")
    randomize_weight_net(model)
    layer = model.get_layer(CCI_NAME)
    x = structured_images(3)
    d = log_uniform_gains(3, seed=7)
    a, b = layer.explain(x), layer.explain(x * d)
    np.testing.assert_allclose(b["weights"], a["weights"], atol=1e-6)
    np.testing.assert_allclose(b["illuminant"], a["illuminant"] * d[:, 0, 0, :], rtol=1e-5)


# ---------------------------------------------------------------------------
# Gradientes (iv) y (vi)
# ---------------------------------------------------------------------------


def _loss_grads(model, x, y):
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x, training=True))
    return model.trainable_weights, tape.gradient(loss, model.trainable_weights)


@pytest.mark.parametrize("variant", ["full", "no_invariance", "uniform_weights", "fixed_p"])
def test_gradients_finite_and_nonzero(variant):
    keras.utils.set_random_seed(0)
    model = tiny_cci(variant)
    randomize_weight_net(model)       # en el inicio exacto, la última capa de g_φ en cero anula el gradiente de las anteriores
    x, y = pattern_dataset(2)
    weights, grads = _loss_grads(model, x, y)
    for w, g in zip(weights, grads):
        assert g is not None, w.path
        g = g.numpy()
        assert np.all(np.isfinite(g)), w.path
        assert np.abs(g).max() > 0, w.path
    if variant == "fixed_p":
        assert all(THETA_NAME not in w.path for w in weights)


def test_gradient_at_initialization_flows_to_last_layer_only():
    """En el inicio (pesos uniformes), solo la última convolución de g_φ y ϑ reciben gradiente."""
    keras.utils.set_random_seed(0)
    model = tiny_cci("full")
    x, y = pattern_dataset(2)
    weights, grads = _loss_grads(model, x, y)
    by_path = {w.path: g.numpy() for w, g in zip(weights, grads)}
    g_out = [v for k, v in by_path.items() if "g_out" in k]
    g_first = [v for k, v in by_path.items() if "g_conv1" in k or "g_conv2" in k]
    assert all(np.abs(v).max() > 0 for v in g_out)
    assert all(np.abs(v).max() == 0 for v in g_first)
    assert np.abs(next(v for k, v in by_path.items() if THETA_NAME in k)) > 0


def test_illuminant_gradient_wrt_p_matches_finite_differences():
    """∂e_c/∂p por diferenciación automática, por la fórmula cerrada del capítulo 24 y por
    diferencias finitas centrales (en float64)."""
    rng = np.random.default_rng(0)
    n = 64
    xbar = rng.uniform(5.0, 250.0, size=(2, n, 3))
    scores = rng.normal(0.0, 1.0, size=(2, n))
    log_w = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
    log_xbar = tf.constant(np.log(xbar))
    log_w_t = tf.constant(log_w)

    def e_of(p):
        return tf.exp(CCILayer.log_illuminant(log_w_t, log_xbar, p))

    for p0 in (1.5, 3.0, 6.0, 9.0):
        p = tf.Variable(p0, dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            e = e_of(p)
            e_list = [e[b, c] for b in range(2) for c in range(3)]
        auto = np.array([tape.gradient(ec, p).numpy() for ec in e_list])
        h = 1e-5
        fd = ((e_of(tf.constant(p0 + h, tf.float64)) - e_of(tf.constant(p0 - h, tf.float64))) / (2 * h)).numpy().ravel()
        w = np.exp(log_w)[:, :, None]
        S = (w * xbar ** p0).sum(axis=1)
        e_np = S ** (1 / p0)
        closed = (e_np * ((w * xbar ** p0 * np.log(xbar)).sum(axis=1) / (p0 * S) - np.log(S) / p0 ** 2)).ravel()
        np.testing.assert_allclose(auto, fd, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(auto, closed, rtol=1e-9)


def test_theta_gradient_chain_rule():
    """∂e/∂ϑ = ∂e/∂p · σ(ϑ) a través de la capa (float64)."""
    layer = CCILayer(**{**LAYER_CFG, "learn_weights": False}, dtype="float64")
    x = tf.constant(positive_images(1).astype(np.float64))
    layer.build(x.shape)
    with tf.GradientTape() as tape:
        e = tf.exp(layer._forward(x)[1]["log_illuminant"])[0, 0]
    g_theta = tape.gradient(e, layer.theta).numpy()
    theta = float(layer.theta.numpy())
    h = 1e-6
    vals = []
    for t in (theta + h, theta - h):
        layer.theta.assign(t)
        vals.append(float(tf.exp(layer._forward(x)[1]["log_illuminant"])[0, 0]))
    assert g_theta == pytest.approx((vals[0] - vals[1]) / (2 * h), rel=1e-6)


# ---------------------------------------------------------------------------
# Sobreajuste (i) y reproducibilidad (ii)
# ---------------------------------------------------------------------------


def _fit(model, x, y, epochs: int, lr: float = 1e-2, seed: int = 0):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model.fit(x, y, epochs=epochs, batch_size=len(x), shuffle=False, verbose=0)


def test_overfits_small_synthetic_batch():
    keras.utils.set_random_seed(0)
    model = tiny_cci("full")
    x, y = pattern_dataset(4)
    hist = _fit(model, x, y, epochs=150)
    assert hist.history["loss"][-1] < 0.1
    assert (model.predict(x, verbose=0).argmax(1) == y).mean() == 1.0
    p = model.get_layer(CCI_NAME).explain(x[:1])["p"]
    assert np.isfinite(p) and p >= 1.0


def _train_once(seed: int):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = tiny_cci("full", dropout=0.2)
    x, y = pattern_dataset(2)
    _fit(model, x, y, epochs=3)
    return [w.copy() for w in model.get_weights()], model.predict(x, verbose=0)


def test_reproducible_with_same_seed():
    w1, p1 = _train_once(123)
    w2, p2 = _train_once(123)
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(p1, p2)
    w3, _ = _train_once(124)
    assert any(not np.array_equal(a, b) for a, b in zip(w1, w3))


# ---------------------------------------------------------------------------
# Guardado, carga y ONNX (vii)
# ---------------------------------------------------------------------------


def test_weights_roundtrip(tmp_path):
    keras.utils.set_random_seed(0)
    model = tiny_cci("full")
    randomize_weight_net(model)
    model.get_layer(CCI_NAME).theta.assign(1.3)
    x = positive_images(3)
    path = tmp_path / "model.weights.h5"
    model.save_weights(path)
    other = tiny_cci("full")
    other.load_weights(path)
    np.testing.assert_allclose(other(x).numpy(), model(x).numpy(), atol=1e-6)


def test_keras_file_roundtrip(tmp_path):
    keras.utils.set_random_seed(0)
    model = tiny_cci("no_invariance")
    randomize_weight_net(model)
    x = positive_images(3)
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    np.testing.assert_allclose(loaded(x).numpy(), model(x).numpy(), atol=1e-6)
    cfg = loaded.get_layer(CCI_NAME).get_config()
    assert cfg["invariant"] is False and cfg["reduced_size"] == REDUCED


@pytest.mark.parametrize("variant", ["full", "uniform_weights"])
def test_onnx_export_matches_keras(tmp_path, variant):
    from src.export import cam_head, export_keras_model, onnx_predict, onnx_session

    keras.utils.set_random_seed(0)
    model = tiny_cci(variant)
    randomize_weight_net(model)
    x = positive_images(5, low=1.0, high=254.0)
    graph = export_keras_model(model, tmp_path / "model.onnx")
    assert graph["outputs"] == ["logits", "probabilities", "embedding", "feature_map", "head_weights"]
    out = onnx_predict(onnx_session(tmp_path / "model.onnx"), x)
    ref = model(x).numpy()
    assert np.abs(out["probabilities"] - ref).max() < 1e-4
    assert (out["probabilities"].argmax(1) == ref.argmax(1)).all()
    # CAM sin gradientes: la media espacial del CAM más el sesgo reproduce los logits
    head = cam_head(model)
    bias = model.get_layer("predictions").bias.numpy()
    cam = np.einsum("nhwc,ck->nhwk", out["feature_map"], out["head_weights"])
    np.testing.assert_allclose(cam.mean(axis=(1, 2)) + bias, out["logits"], atol=1e-4)
    np.testing.assert_allclose(out["head_weights"], head["weights"], atol=1e-7)
