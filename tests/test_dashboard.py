"""Dashboard: las cifras de la tabla de líderes coinciden con la tabla maestra, y la prueba de humo
(dashboard/smoke_test.py) pasa con datos sintéticos que incluyen modelos ONNX reales."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"
CFG = {"benchmark": {"primary_metric": "balanced_accuracy"}, "e2": {"transformer_models": ["vit_b16"]},
       "e3": {"original_model": "cci", "ablation_models": ["cci_fixed_p"]}}


def _check_board(board: pd.DataFrame, master: pd.DataFrame) -> None:
    for _, row in board.iterrows():
        g = master[master["modelo"] == row["modelo"]].set_index("metrica")
        assert row["ba_media"] == pytest.approx(float(g.loc["balanced_accuracy", "media"]), abs=1e-12)
        assert row["ba_std"] == pytest.approx(float(g.loc["balanced_accuracy", "std"]), abs=1e-12)
        assert row["corridas"] == int(g.loc["balanced_accuracy", "corridas"])
        if "ece" in g.index:
            assert row["ece_media"] == pytest.approx(float(g.loc["ece", "media"]), abs=1e-12)
        if "f1_macro" in g.index:
            assert row["f1_media"] == pytest.approx(float(g.loc["f1_macro", "media"]), abs=1e-12)
            assert row["f1_std"] == pytest.approx(float(g.loc["f1_macro", "std"]), abs=1e-12)
        else:
            assert pd.isna(row.get("f1_media")) and pd.isna(row.get("f1_std"))
    assert set(board["modelo"]) == set(master["modelo"])


def test_leaderboard_matches_master_table():
    from src.export_dashboard import leaderboard_from_master
    from tests.fake_dashboard import master_table

    master = master_table(("mobilenetv2", "vit_b16", "cci", "cci_fixed_p"))
    board = leaderboard_from_master(master, CFG)
    _check_board(board, master)
    assert dict(zip(board["modelo"], board["familia"])) == {"mobilenetv2": "CNN", "vit_b16": "Transformer",
                                                            "cci": "Original (CCI)", "cci_fixed_p": "Ablación (CCI)"}
    assert list(board["ba_media"]) == sorted(board["ba_media"], reverse=True)


def test_leaderboard_f1_missing_is_nan():
    """Sin la métrica f1_macro en la tabla maestra, f1_media y f1_std quedan en NaN."""
    from src.export_dashboard import LEADERBOARD_COLUMNS, leaderboard_from_master
    from tests.fake_dashboard import master_table

    master = master_table(("mobilenetv2", "resnet50"))
    board = leaderboard_from_master(master[master["metrica"] != "f1_macro"], CFG)
    assert {"f1_media", "f1_std"} <= set(LEADERBOARD_COLUMNS) and list(board.columns) == LEADERBOARD_COLUMNS
    assert board["f1_media"].isna().all() and board["f1_std"].isna().all()
    assert board["ba_media"].notna().all()


def test_dashboard_reads_f1_from_leaderboard(tmp_path, monkeypatch):
    """La tabla de líderes del dashboard toma el F1 macro de leaderboard.json (no de diagnostics.json)."""
    models = tmp_path / "models"
    (models / "resnet50").mkdir(parents=True)
    rows = [{"modelo": "mobilenetv2", "familia": "CNN", "ba_media": 0.7, "f1_media": 0.61, "f1_std": 0.02},
            {"modelo": "resnet50", "familia": "CNN", "ba_media": 0.72, "f1_media": None, "f1_std": None}]
    (models / "leaderboard.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    cm = np.eye(7).tolist()   # F1 = 1 si se calculara de la matriz de confusión
    (models / "resnet50" / "diagnostics.json").write_text(json.dumps({"classes": list("abcdefg"), "confusion": cm}),
                                                          encoding="utf-8")
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(tmp_path))
    code = ("import json; from core import data; "
            "print(json.dumps({m: data.model_row(m)['f1'] for m in ('mobilenetv2', 'resnet50', 'vgg16')}))")
    res = subprocess.run([sys.executable, "-c", code], cwd=DASH, capture_output=True, text=True, timeout=120, check=True)
    assert json.loads(res.stdout.strip().splitlines()[-1]) == {"mobilenetv2": 0.61, "resnet50": None, "vgg16": None}


def test_published_dashboard_matches_master_table():
    """Si ya se generaron dashboard/data/models/leaderboard.json y la tabla maestra, deben coincidir."""
    import yaml

    ecfg = yaml.safe_load((ROOT / "configs" / "export.yaml").read_text(encoding="utf-8"))
    master_path = ROOT / ecfg["dashboard"]["master_table"]
    board_path = ROOT / ecfg["dashboard"]["out_dir"] / "models" / "leaderboard.json"
    if not (master_path.exists() and board_path.exists()):
        pytest.skip("aún no hay tabla maestra ni tabla de líderes publicadas (resultados pendientes)")
    rows = json.loads(board_path.read_text(encoding="utf-8"))["rows"]
    _check_board(pd.DataFrame(rows), pd.read_csv(master_path))


def test_notebook_sentence_extraction(tmp_path):
    from src.export_dashboard import notebook_sentence

    nb = {"cells": [{"cell_type": "markdown", "source": [("**Interpretación**\n\nPrimera frase con 0,5 y `código`. "
                                                          "Segunda frase ancla aquí. Tercera.")]}]}
    path = tmp_path / "x.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    assert notebook_sentence(path, "ancla") == "Segunda frase ancla aquí."
    assert notebook_sentence(path, "código") == "Primera frase con 0,5 y código."
    assert notebook_sentence(path, "no está") is None


def test_dashboard_stats_match_scipy():
    from scipy import stats as ss

    sys.path.insert(0, str(DASH))
    from core.stats import epsilon2

    rng = np.random.default_rng(0)
    v = np.round(rng.normal(size=300), 1)                              # con empates
    g = rng.integers(0, 4, 300)
    h, _ = ss.kruskal(*[v[g == k] for k in range(4)])
    assert epsilon2(v, g) == pytest.approx(h / 299, rel=1e-10)


def test_dashboard_does_not_import_tensorflow_or_src():
    for path in DASH.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import tensorflow" not in text and "from src" not in text and "import src" not in text, path


def test_smoke_test_with_fake_models(tmp_path):
    from tests.fake_dashboard import build

    data = build(tmp_path / "data", ROOT / "dashboard" / "data")
    env = {**os.environ, "DASHBOARD_DATA_DIR": str(data)}
    res = subprocess.run([sys.executable, "smoke_test.py"], cwd=DASH, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "predicción cci (carga)" in res.stdout and "predicción resnet50 (ejemplo)" in res.stdout


def test_smoke_test_pending_state(tmp_path):
    """Sin resultados de entrenamiento: el dashboard funciona y Modelos muestra el estado pendiente."""
    import shutil

    data = tmp_path / "data"
    if (ROOT / "dashboard" / "data" / "eda").exists():
        shutil.copytree(ROOT / "dashboard" / "data" / "eda", data / "eda")
    env = {**os.environ, "DASHBOARD_DATA_DIR": str(data)}
    res = subprocess.run([sys.executable, "smoke_test.py"], cwd=DASH, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "sin modelos exportados" in res.stdout
