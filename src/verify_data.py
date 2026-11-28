"""Verificación de los datos crudos de HAM10000 antes de ejecutar los notebooks.

    python -m src.verify_data

Comprueba que existan los archivos registrados en data/hashes.json y que su SHA-256
coincida, y que las imágenes y máscaras descomprimidas estén donde las espera src/data.py
(paths.images_dir y paths.segmentations_dir de configs/base.yaml) para todos los image_id del
CSV de metadatos. Termina con código 1 si algo falla. No modifica ningún archivo.
"""

from __future__ import annotations

import argparse
import sys

from src import data as D


def check_hashes(cfg: dict) -> list[dict]:
    """Una fila por archivo de hashes_json: existe, SHA-256 actual y si coincide con el registrado."""
    registered = D.load_json(cfg["paths"]["hashes_json"])
    rows = []
    for rel, expected in registered.items():
        path = D.PROJECT_ROOT / rel
        actual = D.sha256_file(path) if path.exists() else None
        rows.append({"archivo": rel, "existe": path.exists(), "coincide": actual == expected,
                     "sha256": actual, "registrado": expected})
    return rows


def check_extracted(cfg: dict) -> dict:
    """Imágenes y máscaras que faltan para los image_id del CSV de metadatos."""
    meta = D.load_metadata(cfg)
    ids = list(meta["image_id"])
    missing_images = [i for i in ids if not D.image_path(cfg, i).exists()]
    missing_masks = [i for i in ids if not D.mask_path(cfg, i).exists()]
    return {"n_metadatos": len(ids), "imagenes_faltantes": missing_images, "mascaras_faltantes": missing_masks}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Verifica los archivos de HAM10000 y sus SHA-256.")
    parser.add_argument("--config", default=None, help="YAML base (por defecto configs/base.yaml)")
    parser.add_argument("--skip-extracted", action="store_true",
                        help="no comprueba las imágenes y máscaras descomprimidas")
    args = parser.parse_args(argv)
    cfg = D.load_config(args.config)
    ok = True

    print(f"SHA-256 frente a {cfg['paths']['hashes_json'].relative_to(D.PROJECT_ROOT)}:")
    for r in check_hashes(cfg):
        if not r["existe"]:
            estado = "FALTA"
        elif r["coincide"]:
            estado = "ok"
        else:
            estado = f"NO COINCIDE (actual {r['sha256']}, registrado {r['registrado']})"
        ok &= r["existe"] and r["coincide"]
        print(f"  {r['archivo']}: {estado}")

    if not args.skip_extracted:
        if not cfg["paths"]["metadata_csv"].exists():
            print("No se comprueban las imágenes descomprimidas: falta el CSV de metadatos.")
            ok = False
        else:
            ext = check_extracted(cfg)
            for key, folder in (("imagenes_faltantes", "images_dir"), ("mascaras_faltantes", "segmentations_dir")):
                faltan = ext[key]
                rel = cfg["paths"][folder].relative_to(D.PROJECT_ROOT)
                if faltan:
                    ok = False
                    print(f"  {rel}/: faltan {len(faltan)} de {ext['n_metadatos']} (p. ej. {faltan[0]})")
                else:
                    print(f"  {rel}/: {ext['n_metadatos']} de {ext['n_metadatos']} ok")

    print("Datos verificados." if ok else "La verificación falló: revisa la sección «Obtención de los datos» del README.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
