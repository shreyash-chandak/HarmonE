"""Tests for core/device.py — CV device resolution (2026-10-03)."""

from __future__ import annotations

import pytest
import torch

import core.device as cd


@pytest.fixture(autouse=True)
def _reset():
    cd._DEVICE = None
    yield
    cd._DEVICE = None


def _fake_cuda(monkeypatch, available: bool, count: int = 1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count if available else 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_a, **_k: "FakeGPU")


def test_cpu_is_cpu():
    assert cd.resolve_device("cpu").type == "cpu"


def test_auto_without_cuda_is_cpu(monkeypatch):
    _fake_cuda(monkeypatch, False)
    assert cd.resolve_device("auto").type == "cpu"
    assert cd.resolve_device(None).type == "cpu"


def test_auto_with_cuda_is_explicit_index(monkeypatch):
    _fake_cuda(monkeypatch, True)
    assert str(cd.resolve_device("auto")) == "cuda:0"


def test_cuda_required_but_missing_fails_fast(monkeypatch):
    _fake_cuda(monkeypatch, False)
    with pytest.raises(RuntimeError, match="Refusing to\\s+fall back to CPU"):
        cd.resolve_device("cuda")


def test_cuda_index_out_of_range(monkeypatch):
    _fake_cuda(monkeypatch, True, count=1)
    with pytest.raises(RuntimeError, match="CUDA device"):
        cd.resolve_device("cuda:3")


def test_unknown_setting_rejected():
    with pytest.raises(ValueError):
        cd.resolve_device("tpu")


def test_configure_then_get(monkeypatch):
    _fake_cuda(monkeypatch, True)
    cd.configure_device({"device": "cuda"})
    assert str(cd.get_device()) == "cuda:0"
    assert cd.device_info() == {"device": "cuda:0", "gpu_name": "FakeGPU"}


def test_yolo_device_never_bare_cuda(monkeypatch):
    # ultralytics strips "cuda:" and treats the rest as an index — a bare
    # "cuda" would be parsed as an invalid device id.
    _fake_cuda(monkeypatch, True)
    cd.configure_device({"device": "cuda"})
    assert cd.yolo_device() == "cuda:0"
    cd.configure_device({"device": "cpu"})
    assert cd.yolo_device() == "cpu"


def test_cv_configs_require_gpu():
    import json
    from pathlib import Path
    cfg_dir = Path(__file__).resolve().parent.parent / "configs" / "datasets"
    for name in ("acdc", "bdd100k", "imagenet", "imagenet_c", "iwildcam"):
        cfg = json.loads((cfg_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert cfg.get("device") == "cuda", name
