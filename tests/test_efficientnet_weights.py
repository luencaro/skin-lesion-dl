"""EfficientNet-B0: el archivo de pesos de ImageNet no depende del nombre del backbone (sin red)."""

from __future__ import annotations

from src.models import efficientnetb0 as E
from src.models.common import BACKBONE_NAME


def test_weights_file_is_b0():
    name, url, file_hash = E.weights_file(include_top=False)
    assert name == "efficientnetb0_notop.h5"
    assert url.endswith("/keras-applications/efficientnetb0_notop.h5")
    assert len(file_hash) == 32


def test_backbone_name_does_not_change_weights_file(monkeypatch):
    calls = []

    def fake_get_file(fname, origin, **kwargs):
        calls.append((fname, origin, kwargs.get("file_hash")))
        raise RuntimeError("sin red")

    monkeypatch.setattr(E.keras.utils, "get_file", fake_get_file)
    try:
        E.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(32, 32, 3), name=BACKBONE_NAME)
    except RuntimeError:
        pass
    assert calls and calls[0][:2] == E.weights_file(False)[:2]
    model = E.EfficientNetB0(include_top=False, weights=None, input_shape=(32, 32, 3), name=BACKBONE_NAME)
    assert model.name == BACKBONE_NAME
