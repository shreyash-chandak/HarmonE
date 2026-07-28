"""
tests/test_phase3_vmr.py — Tests for core/vmr.py (Versioned Model Repository).

Tests cover:
  - store(): creates version directory, copies weights, writes meta and distribution
  - list_versions(): returns versions sorted newest-first
  - best_match strategy="best_score": returns highest proxy_score version
  - best_match strategy="closest_distribution": histogram matching
  - restore(): returns correct weights path; raises on missing file
  - best_match() returns None when no versions stored
  - Multiple models are isolated
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.vmr import VMR, VMRVersion


@pytest.fixture
def fake_weights(tmp_path):
    """Create a dummy .pt file to use as weights source."""
    f = tmp_path / "dummy.pt"
    f.write_bytes(b"fake weights content")
    return str(f)


@pytest.fixture
def vmr(tmp_path):
    return VMR(base_dir=str(tmp_path / "vmr"))


class TestVMRStore:
    def test_store_creates_version_dir(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [1.0, 2.0, 3.0]}
        version = vmr.store("lstm", fake_weights, dist, tag="test", proxy_score=0.85)
        assert os.path.isdir(version.version_dir)

    def test_store_copies_weights(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [1.0]}
        version = vmr.store("lstm", fake_weights, dist)
        assert os.path.isfile(version.weights_path)
        assert Path(version.weights_path).read_bytes() == b"fake weights content"

    def test_store_writes_meta(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [1.0]}
        version = vmr.store("lstm", fake_weights, dist, proxy_score=0.9, tag="mytag")
        meta_path = os.path.join(version.version_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["model"] == "lstm"
        assert meta["tag"] == "mytag"
        assert meta["proxy_score"] == pytest.approx(0.9)

    def test_store_writes_distribution(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [10.0, 20.0, 30.0]}
        version = vmr.store("lstm", fake_weights, dist)
        dist_path = os.path.join(version.version_dir, "distribution.json")
        with open(dist_path) as f:
            loaded = json.load(f)
        assert loaded["data"] == pytest.approx([10.0, 20.0, 30.0])

    def test_store_returns_vmrversion(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": []}
        version = vmr.store("yolo_n", fake_weights, dist)
        assert isinstance(version, VMRVersion)
        assert version.model == "yolo_n"


class TestVMRListVersions:
    def test_empty_returns_empty_list(self, vmr):
        assert vmr.list_versions("nonexistent") == []

    def test_multiple_versions_sorted_newest_first(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [1.0]}
        import time
        v1 = vmr.store("lstm", fake_weights, dist, tag="first")
        time.sleep(0.01)
        v2 = vmr.store("lstm", fake_weights, dist, tag="second")
        versions = vmr.list_versions("lstm")
        assert len(versions) == 2
        # newest first means v2 should come before v1
        assert versions[0].tag == "second"
        assert versions[1].tag == "first"

    def test_models_are_isolated(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [1.0]}
        vmr.store("model_a", fake_weights, dist)
        vmr.store("model_b", fake_weights, dist)
        assert len(vmr.list_versions("model_a")) == 1
        assert len(vmr.list_versions("model_b")) == 1


class TestVMRBestMatch:
    def test_no_versions_returns_none(self, vmr):
        assert vmr.best_match("nonexistent") is None

    def test_best_score_picks_highest_proxy(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [1.0]}
        import time
        vmr.store("lstm", fake_weights, dist, proxy_score=0.6, tag="bad")
        time.sleep(0.01)
        vmr.store("lstm", fake_weights, dist, proxy_score=0.9, tag="good")
        time.sleep(0.01)
        vmr.store("lstm", fake_weights, dist, proxy_score=0.7, tag="ok")

        best = vmr.best_match("lstm", strategy="best_score")
        assert best is not None
        assert best.tag == "good"

    def test_best_score_falls_back_to_newest_when_no_scores(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [1.0]}
        import time
        vmr.store("lstm", fake_weights, dist, tag="older")
        time.sleep(0.002)  # micro-second precision timestamps distinguish these
        vmr.store("lstm", fake_weights, dist, tag="newer")

        best = vmr.best_match("lstm", strategy="best_score")
        assert best is not None
        # "newer" timestamp is later → sorts first in reverse order → versions[0]
        assert best.tag == "newer"

    def test_closest_distribution_selects_minimal_kl(self, vmr, fake_weights):
        # Reference dist: uniform over 3 bins
        ref_dist = {"type": "histogram", "data": [100.0, 100.0, 100.0]}
        # Close version: nearly uniform
        close_dist = {"type": "histogram", "data": [101.0, 99.0, 100.0]}
        # Far version: heavily skewed
        far_dist = {"type": "histogram", "data": [300.0, 1.0, 1.0]}

        import time
        vmr.store("lstm", fake_weights, close_dist, tag="close")
        time.sleep(0.01)
        vmr.store("lstm", fake_weights, far_dist, tag="far")

        best = vmr.best_match("lstm", current_distribution=ref_dist, strategy="closest_distribution")
        assert best is not None
        assert best.tag == "close"


class TestVMRRestore:
    def test_restore_returns_weights_path(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [1.0]}
        version = vmr.store("lstm", fake_weights, dist)
        path = vmr.restore(version)
        assert os.path.isfile(path)

    def test_restore_raises_on_missing_weights(self, vmr, fake_weights):
        dist = {"type": "histogram", "data": [1.0]}
        version = vmr.store("lstm", fake_weights, dist)
        # Delete the weights file
        os.remove(version.weights_path)
        with pytest.raises(FileNotFoundError):
            vmr.restore(version)
