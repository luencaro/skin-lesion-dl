"""Modelo original (capa CCI + backbone del REGISTRY), registrado como "cci".

El backbone, la variante y los parámetros de la capa se leen de configs/cci.yaml (y de
configs/cci_<variante>.yaml, que lo extienden, para la ablación del capítulo 24). La
implementación está en src/original_model.py; aquí solo se adapta a la interfaz del REGISTRY,
`build(model_cfg, n_classes, dropout)`. La importación es diferida para evitar el ciclo
src.models -> src.original_model -> src.models.
"""

from __future__ import annotations

# Nombres adicionales con el mismo constructor (configs/<alias>.yaml con extends: cci).
ALIASES = ("cci_fixed_p", "cci_uniform_weights", "cci_no_invariance", "cci_shades_of_gray")


def build(model_cfg: dict, n_classes: int, dropout: float):
    from src.original_model import build_from_config

    return build_from_config(model_cfg, n_classes, dropout)
