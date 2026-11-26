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
    if model_cfg.get("backbone") and expected is None:   # modelo original: backbone sin modificar + capa CCI
        from src.original_model import CCI_NAME

        base = load_model_config(model_cfg["backbone"]).get("expected_params")
        expected = None if base is None else int(base) + model.get_layer(CCI_NAME).count_params()
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


def buildable(model_name: str) -> bool:
    """False para los modelos CCI cuyo backbone aún es null en configs/cci.yaml."""
    from src.train import load_model_config

    cfg = load_model_config(model_name)
    return not ("backbone" in cfg and not cfg["backbone"])


def verify_models(cfg: dict, models=None) -> pd.DataFrame:
    """Verifica `models` (por defecto, todos los del REGISTRY que se pueden construir; los modelos
    CCI se omiten mientras su backbone sea null)."""
    from src.models import REGISTRY

    if models is None:
        skipped = [m for m in sorted(REGISTRY) if not buildable(m)]
        if skipped:
            print(f"Se omiten (backbone null en configs/cci.yaml): {', '.join(skipped)}")
        models = [m for m in sorted(REGISTRY) if m not in skipped]
    return pd.DataFrame([verify_model(cfg, m) for m in models]).set_index("modelo")


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
