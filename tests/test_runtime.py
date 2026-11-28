"""Pruebas de la configuración del entorno de ejecución (src/train.py), sin GPU."""

from __future__ import annotations

import os

import pytest

from src import train as T


def _cfg(tmp_path, **train):
    return {"paths": {"xla_cache_dir": tmp_path / "dir con espacios" / "xla"}, "train": train}


def test_xla_autotune_cache_flag_quotes_path_with_spaces(tmp_path, monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=4")
    T.configure_xla_env(_cfg(tmp_path, xla_autotune_cache=True))
    flags = os.environ["XLA_FLAGS"]
    cache = tmp_path / "dir con espacios" / "xla" / "autotune"
    assert flags == f'--xla_gpu_autotune_level=4 --xla_gpu_per_fusion_autotune_cache_dir="{cache}"'
    assert cache.is_dir()
    T.configure_xla_env(_cfg(tmp_path, xla_autotune_cache=True))  # no se duplica
    assert os.environ["XLA_FLAGS"] == flags


def test_xla_env_untouched_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("TF_DEVICE_MIN_SYS_MEMORY_IN_MB", raising=False)
    T.configure_xla_env(_cfg(tmp_path, xla_autotune_cache=False))

    assert "XLA_FLAGS" not in os.environ and "TF_DEVICE_MIN_SYS_MEMORY_IN_MB" not in os.environ


def test_gpu_min_system_memory(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_DEVICE_MIN_SYS_MEMORY_IN_MB", raising=False)
    T.configure_xla_env(_cfg(tmp_path, gpu_min_system_memory_mb=512))

    assert os.environ["TF_DEVICE_MIN_SYS_MEMORY_IN_MB"] == "512"


def test_isolated_run_rejects_shared_sources(tmp_path):
    with pytest.raises(ValueError, match="sources"):
        T.run_training_isolated({}, "mobilenetv2", 0, 0, sources={"train": None})


def test_gpu_allocator(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_GPU_ALLOCATOR", raising=False)
    T.configure_xla_env(_cfg(tmp_path, gpu_allocator="cuda_malloc_async"))
    assert os.environ["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
