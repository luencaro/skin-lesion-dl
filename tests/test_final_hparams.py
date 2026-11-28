"""Hiperparámetros de la ejecución definitiva: YAML del modelo o best.json del ajuste (t00 a t05)."""

from __future__ import annotations

import pytest

from src import data as D
from src import final as F
from src.tune import sample_configs

YAML_HP = {"learning_rate": 1e-4, "weight_decay": 1e-5, "dropout": 0.2}


@pytest.fixture
def cfg(tmp_path):
    cfg = D.load_config()
    for key in ("runs_dir", "tuning_dir", "logs_dir", "final_dir"):
        cfg["paths"][key] = tmp_path / key
    cfg["train"]["tuning"]["n_trials"] = 6
    return cfg


@pytest.fixture
def null_yaml(monkeypatch):
    real = F.load_model_config

    def fake(name, hp=None):
        return {**real(name), "hyperparameters": dict(hp or dict.fromkeys(YAML_HP))}

    monkeypatch.setattr(F, "load_model_config", fake)
    return fake


def _write_best(cfg, trial, n_trials=6, hp=None):
    tun = cfg["train"]["tuning"]
    hp = hp or sample_configs(tun["space"], max(n_trials, trial + 1), tun["sampler_seed"])[trial]
    path = cfg["paths"]["tuning_dir"] / "mobilenetv2" / "best.json"
    D.save_json({"model": "mobilenetv2", "trial": trial, "hyperparameters": hp, "val_balanced_accuracy": 0.7,
                 "n_trials": n_trials}, path)
    return hp


def test_yaml_values_take_precedence(cfg, null_yaml, monkeypatch):
    monkeypatch.setattr(F, "load_model_config", lambda name: null_yaml(name, YAML_HP))
    _write_best(cfg, 3)
    hp, source = F.final_hparams(cfg, "mobilenetv2")
    assert hp == YAML_HP and source == {"source": "configs/mobilenetv2.yaml"}


def test_null_yaml_reads_best_json(cfg, null_yaml):
    expected = _write_best(cfg, 3)
    hp, source = F.final_hparams(cfg, "mobilenetv2")
    assert hp == {k: float(v) for k, v in expected.items()}
    assert source["source"] == "best.json" and source["trial"] == 3 and source["label"] == "tuning_t03"
    assert source["path"].endswith("mobilenetv2/best.json")


def test_missing_best_json_is_a_clear_error(cfg, null_yaml):
    with pytest.raises(ValueError, match=r"en null y no existe .*best\.json.*python -m src\.tune --model mobilenetv2"):
        F.final_hparams(cfg, "mobilenetv2")


@pytest.mark.parametrize("trial,n_trials", [(6, 7), (2, 12)])
def test_best_json_outside_t00_t05_is_rejected(cfg, null_yaml, trial, n_trials):
    _write_best(cfg, trial, n_trials)
    with pytest.raises(ValueError, match="no se seleccionó entre t00 y t05"):
        F.final_hparams(cfg, "mobilenetv2")


def test_best_json_not_matching_sampler_is_rejected(cfg, null_yaml):
    hp = _write_best(cfg, 1)
    _write_best(cfg, 1, hp={**hp, "dropout": 0.0 if hp["dropout"] == 0.4 else 0.4})
    with pytest.raises(ValueError, match="no coinciden con el muestreador"):
        F.final_hparams(cfg, "mobilenetv2")


def test_partially_null_yaml_is_rejected(cfg, null_yaml, monkeypatch):
    monkeypatch.setattr(F, "load_model_config", lambda name: null_yaml(name, {**YAML_HP, "dropout": None}))
    with pytest.raises(ValueError, match="otros en null"):
        F.final_hparams(cfg, "mobilenetv2")


def test_run_final_passes_hparams_and_source_to_each_run(cfg, null_yaml, monkeypatch):
    cfg["final"]["seeds"], cfg["final"]["folds"] = [0], [0, 1]
    cfg["train"]["isolate_runs"] = True
    expected = _write_best(cfg, 4)
    calls = []

    def fake_run(cfg_, model, seed, fold, log_file=None, **kwargs):
        calls.append(kwargs)
        return {"seed": seed, "val_fold": fold, "val_balanced_accuracy": 0.5, "test_balanced_accuracy": 0.5,
                "train_balanced_accuracy": 0.6, "best_epoch": 1, "epochs_warmup": 5, "epochs_finetune": 2,
                "train_time_s": 1.0, "run_dir": f"s{seed}_f{fold}"}

    monkeypatch.setattr(F, "run_training_isolated", fake_run)
    F.run_final(cfg, "mobilenetv2")
    assert len(calls) == 2
    for kw in calls:
        assert kw["hparams"] == {k: float(v) for k, v in expected.items()}
        assert kw["hparams_source"]["source"] == "best.json" and kw["hparams_source"]["trial"] == 4


def test_run_training_records_source_in_config_yaml(cfg, monkeypatch):
    import types

    import numpy as np
    import yaml

    from src import train as T

    class Stop(Exception):
        pass

    classes = D.class_names(cfg)
    stub = types.SimpleNamespace(class_names=classes, train_labels=np.repeat(np.arange(len(classes)), 3))
    monkeypatch.setattr(T.D, "make_datasets", lambda *a, **k: stub)
    monkeypatch.setattr(T, "configure_runtime", lambda cfg: None)
    monkeypatch.setattr(T, "build_model", lambda *a, **k: (_ for _ in ()).throw(Stop()))
    source = {"source": "best.json", "path": "outputs/tuning/mobilenetv2/best.json", "trial": 4,
              "label": "tuning_t04", "n_trials": 6, "val_balanced_accuracy": 0.7}
    with pytest.raises(Stop):
        T.run_training(cfg, "mobilenetv2", 0, 1, hparams=YAML_HP, label="final", hparams_source=source)
    path = T.run_dir_for(cfg, "mobilenetv2", "final", 0, 1) / "config.yaml"
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["run"]["hyperparameters_source"] == source
    assert saved["model"]["hyperparameters"] == YAML_HP
