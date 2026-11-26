"""Verificación de la arquitectura de los modelos registrados (sin entrenar ni descargar pesos).

    python -m src.verify_models [--model densenet201 ...]

Construye cada modelo con weights=None (la misma arquitectura que con los pesos de ImageNet) y
el número de clases de base.yaml, y compara su número total de parámetros con `expected_params`
de configs/<modelo>.yaml. Comprueba también la estructura que asume el pipeline: una capa
`preprocess`, un backbone anidado con nombre `backbone` y la salida `predictions` en float32 con
n_classes unidades. Termina con código 1 si algo falla.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import data as D


def verify_model(cfg: dict, model_name: str) -> dict:
    """Parámetros (total, entrenables con el backbone congelado y descongelado) y comprobaciones."""
    import keras

    from src.models import REGISTRY, count_params, set_backbone_trainable
    from src.models.common import BACKBONE_NAME, OUTPUT_NAME, PREPROCESS_NAME
    from src.train import load_model_config

    model_cfg = load_model_config(model_name)
    n_classes = len(D.class_names(cfg))
    keras.backend.clear_session()
    model = REGISTRY[model_name](dict(model_cfg, weights=None), n_classes, 0.0)
    names = {layer.name for layer in model.layers}
    set_backbone_trainable(model, False)
    frozen = count_params(model)
    set_backbone_trainable(model, True)
    full = count_params(model)
    out = model.get_layer(OUTPUT_NAME)
    expected = model_cfg.get("expected_params")
    checks = {
        "capa preprocess": PREPROCESS_NAME in names,
        "backbone anidado": BACKBONE_NAME in names,
        "salida float32": out.dtype_policy.compute_dtype == "float32",
        f"salida de {n_classes} clases": int(out.units) == n_classes,
        "entrada 224": tuple(model.input_shape[1:]) == (int(model_cfg["input_size"]),) * 2 + (3,),
        "parámetros esperados": expected is not None and int(expected) == full["total"],
    }
    keras.backend.clear_session()
    return {"modelo": model_name, "parametros_totales": full["total"], "esperados": expected,
            "entrenables_fase_A": frozen["trainable"], "entrenables_fase_B": full["trainable"],
            "no_entrenables_fase_B": full["non_trainable"], **checks, "ok": all(checks.values())}


def verify_models(cfg: dict, models=None) -> pd.DataFrame:
    from src.models import REGISTRY

    return pd.DataFrame([verify_model(cfg, m) for m in (models or sorted(REGISTRY))]).set_index("modelo")


def main(argv=None) -> None:
    from src.models import REGISTRY

    parser = argparse.ArgumentParser(description="Verifica la arquitectura y los parámetros de los modelos.")
    parser.add_argument("--model", nargs="*", choices=sorted(REGISTRY), default=None)
    parser.add_argument("--config", default=None, help="YAML base (por defecto configs/base.yaml)")
    args = parser.parse_args(argv)
    table = verify_models(D.load_config(args.config), args.model)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.T.to_string())
    sys.exit(0 if table["ok"].all() else 1)


if __name__ == "__main__":
    main()
