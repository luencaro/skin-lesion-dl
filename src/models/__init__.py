"""Registro de modelos.

Cada módulo de este paquete que defina `build(model_cfg, n_classes, dropout) -> keras.Model`
se registra con el nombre del módulo (p. ej. src/models/vgg16.py -> "vgg16"). Para añadir un
modelo basta con su archivo aquí y configs/<nombre>.yaml.
"""

from __future__ import annotations

import importlib
import pkgutil

from src.models.common import count_params, get_backbone, set_backbone_trainable

_SKIP = {"common"}


def _discover() -> dict:
    registry = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name in _SKIP or info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        build = getattr(module, "build", None)
        if callable(build):
            registry[info.name] = build
    return registry


REGISTRY = _discover()

__all__ = ["REGISTRY", "count_params", "get_backbone", "set_backbone_trainable"]
