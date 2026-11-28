"""Datos de dashboard sintéticos con modelos exportados de verdad (clasificador diminuto + capa
CCI en ONNX), para probar la página de modelos y la predicción sin haber entrenado."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def master_table(models=("mobilenetv2", "resnet50", "cci")) -> pd.DataFrame:
    """Tabla maestra sintética con el formato de src.benchmark.main_table (largo)."""
    rng = np.random.default_rng(0)
    rows = []
    for m in models:
        for k in ("balanced_accuracy", "f1_macro", "accuracy", "ece"):
            mean = rng.uniform(0.6, 0.8) if k != "ece" else rng.uniform(0.02, 0.08)
            rows.append({"modelo": m, "rol": "x", "metrica": k, "corridas": 25, "media": mean, "std": rng.uniform(0.01, 0.03),
                         "ic_t_bajo": mean - 0.01, "ic_t_alto": mean + 0.01, "ic_lesion_bajo": np.nan, "ic_lesion_alto": np.nan})
    return pd.DataFrame(rows)


def _write(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def build(out: Path, real_data: Path | None = None, size: int = 224) -> Path:
    """Crea `out` con EDA (copiado de `real_data` si existe), ejemplos, dos modelos ONNX (uno en
    vivo y otro precalculado), tabla de líderes, diagnósticos y comparación global."""
    import keras
    from PIL import Image

    from src.export import (
        cam_head,
        export_keras_model,
        knn_distance,
        l2_normalize,
        onnx_predict,
        onnx_session,
    )
    from src.export_dashboard import leaderboard_from_master, precompute_examples
    from src.original_model import build_cci_model
    from tests.conftest import tiny_classifier

    out = Path(out)
    if real_data is not None and (real_data / "eda" / "summary.json").exists():
        shutil.copytree(real_data / "eda", out / "eda", dirs_exist_ok=True)
    rng = np.random.default_rng(0)
    examples = []
    (out / "examples").mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(CLASSES):
        img = rng.uniform(30, 220, (size, size, 3)).astype(np.uint8)
        name = f"ISIC_{9000000 + i}"
        Image.fromarray(img).save(out / "examples" / f"{name}.jpg", quality=90)
        examples.append({"image_id": name, "label": c, "file": f"{name}.jpg"})
    _write(examples, out / "examples" / "examples.json")

    keras.mixed_precision.set_global_policy("float32")
    artifacts = []
    for name, live in (("cci", True), ("resnet50", False)):
        keras.utils.set_random_seed(0)
        base = tiny_classifier(n_classes=7, size=size)
        model = build_cci_model(base, "full", layer_cfg={"reduced_size": 56, "filters": 4}) if name == "cci" else base
        d = out / "models" / name
        graph = export_keras_model(model, d / "model.onnx")
        sess = onnx_session(d / "model.onnx")
        ref = l2_normalize(onnx_predict(sess, rng.uniform(0, 255, (20, size, size, 3)).astype(np.float32))["embedding"])
        np.savez_compressed(d / "ood_reference.npz", embeddings=ref.astype(np.float16))
        thr = float(np.quantile(knn_distance(ref, ref, 5), 0.99))
        meta = {"model": name, "family": "Original (CCI)" if name == "cci" else "CNN", "classes": CLASSES,
                "resolution": [size, size], "temperature": 1.3, "live": live, "size_mb": 0.1,
                "conformal": {"quantiles": {"0.1": 0.8, "0.05": 0.9}}, "cam": {"supported": cam_head(model) is not None},
                "ood": {"k": 5, "threshold": thr, "quantile": 0.99}, "outputs": {n: {} for n in graph["outputs"]}}
        _write(meta, d / "metadata.json")
        if not live:
            pre = precompute_examples(d, d / "model.onnx", meta, examples, out)
            (d / "model.onnx").unlink()
            _write({k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()} for k, v in pre.items()},
                   d / "precomputed.json")
        artifacts.append({"model": name, "family": meta["family"], "live": live, "cam": True, "size_mb": 0.1})
        p = rng.dirichlet(np.ones(7), 200)
        y = rng.integers(0, 7, 200)
        _write({"classes": CLASSES, "n_predictions": 200, "n_runs": 1,
                "confusion_norm": (np.eye(7) * 0.6 + 0.4 / 7).tolist(), "confusion": np.eye(7).tolist(),
                "roc": {c: {"fpr": [0, 0.5, 1], "tpr": [0, 0.8, 1], "auc": 0.7} for c in CLASSES},
                "pr": {c: {"recall": [1, 0.5, 0], "precision": [0.2, 0.6, 1], "ap": 0.5} for c in CLASSES},
                "reliability": [{"bin": b, "confianza": (b + 0.5) / 15, "exactitud": (b + 0.5) / 16, "n": 10} for b in range(15)]},
               d / "diagnostics.json")
        del p, y
    cfg = {"benchmark": {"primary_metric": "balanced_accuracy"}, "e2": {}, "e3": {"original_model": "cci"}}
    board = leaderboard_from_master(master_table(), cfg)
    _write({"rows": board.where(board.notna(), None).to_dict("records"), "split": "test", "source": "sintética"},
           out / "models" / "leaderboard.json")
    _write({"cd": {"ranks": {"mobilenetv2": 1.4, "resnet50": 2.0, "cci": 2.6}, "cd": 0.9, "n_runs": 25, "k": 3, "alpha": 0.05},
            "pareto": {"gflops": [{"modelo": "mobilenetv2", "familia": "CNN", "ba_media": 0.65, "gflops": 0.6, "frontera": True},
                                  {"modelo": "cci", "familia": "Original (CCI)", "ba_media": 0.72, "gflops": 4.5, "frontera": True}]}},
           out / "models" / "global.json")
    _write({"status": "listo", "models_with_runs": ["cci", "resnet50"], "artifacts": artifacts}, out / "models" / "index.json")
    _write({"backbone": None, "tests": [{"prueba": "test_x", "descripcion": "d", "resultado": "aprobada"}]},
           out / "models" / "original.json")
    return out


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
