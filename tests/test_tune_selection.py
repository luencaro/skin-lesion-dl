"""Selección del ajuste con n_trials menor que los ensayos ya evaluados (desviación D1), sin entrenar."""

from __future__ import annotations

import json

import pytest

from src import data as D
from src.train import run_dir_for
from src.tune import run_tuning, sample_configs


def _fake_run(cfg, model, i, hp, bal):
    run_dir = run_dir_for(cfg, model, f"tuning_t{i:02d}", 0, 0)
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({"val": {"balanced_accuracy": bal}}))
    (run_dir / "run_info.json").write_text(json.dumps({
        "seed": 0, "val_fold": 0, "label": f"tuning_t{i:02d}", "hyperparameters": hp,
        "epochs": {"finetune_best": 3, "warmup": 5, "finetune_run": 8}, "train_time_s": {"total": 1.0}}))


@pytest.fixture
def cfg(tmp_path):
    cfg = D.load_config()
    cfg["train"]["tuning"]["n_trials"] = 6
    cfg["train"]["isolate_runs"] = True   # el padre no configura la GPU
    for key in ("runs_dir", "tuning_dir", "logs_dir"):
        cfg["paths"][key] = tmp_path / key
    return cfg


def test_selection_ignores_trials_beyond_n_trials(cfg):
    tun = cfg["train"]["tuning"]
    configs = sample_configs(tun["space"], 7, tun["sampler_seed"])
    for i, hp in enumerate(configs):
        _fake_run(cfg, "mobilenetv2", i, hp, bal=0.9 if i == 6 else 0.5 + 0.01 * i)   # t06 sería la mejor
    trials = run_tuning(cfg, "mobilenetv2")
    best = D.load_json(cfg["paths"]["tuning_dir"] / "mobilenetv2" / "best.json")
    assert best["trial"] == 5 and best["n_trials"] == 6 and best["excluded_trials"] == [6]
    assert trials.set_index("trial")["excluded"].to_dict() == {**{i: False for i in range(6)}, 6: True}
    assert (cfg["paths"]["runs_dir"] / "mobilenetv2" / "tuning_t06").exists()   # no se borra


def test_mismatch_with_sampler_is_an_error(cfg):
    tun = cfg["train"]["tuning"]
    configs = sample_configs(tun["space"], 6, tun["sampler_seed"])
    k = next(i for i, hp in enumerate(configs) if hp["dropout"] != 0.4)
    for i, hp in enumerate(configs):
        _fake_run(cfg, "mobilenetv2", i, {**hp, "dropout": 0.4 if i == k else hp["dropout"]}, 0.5)
    with pytest.raises(RuntimeError, match="otros hiperparámetros"):
        run_tuning(cfg, "mobilenetv2")
