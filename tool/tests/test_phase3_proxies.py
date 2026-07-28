"""
tests/test_phase3_proxies.py — Tests for Phase 3 accuracy proxy layer.

Tests cover:
  - AccuracyProxy ABC cannot be instantiated directly
  - ConfidenceProxy: empty, single, multi-value
  - CalibratedConfidenceProxy: T=1.0 (no calibration) == ConfidenceProxy
  - CalibratedConfidenceProxy: T>1 reduces scores (over-confident model)
  - CalibratedConfidenceProxy: save/load calibration file round-trips
  - AgreementProxy: full agreement, zero agreement, fallback to confidence
  - AgreementProxy IoU helper edge cases
"""

import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.proxies.base import AccuracyProxy
from core.proxies.confidence import ConfidenceProxy
from core.proxies.calibrated_confidence import CalibratedConfidenceProxy, _sigmoid
from core.proxies.agreement import AgreementProxy


# ── helpers ────────────────────────────────────────────────────────────────────

def _record(confs, model="m1", **kw):
    return {"model": model, "confidences": list(confs), **kw}


# ── AccuracyProxy ABC ──────────────────────────────────────────────────────────

class TestAccuracyProxyABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            AccuracyProxy()

    def test_setup_noop_by_default(self):
        p = ConfidenceProxy()
        p.setup({}, None)  # must not raise


# ── ConfidenceProxy ────────────────────────────────────────────────────────────

class TestConfidenceProxy:
    def test_empty_confs_returns_zero(self):
        p = ConfidenceProxy()
        assert p.score(_record([])) == 0.0

    def test_single_conf_returned_directly(self):
        p = ConfidenceProxy()
        assert p.score(_record([0.75])) == pytest.approx(0.75)

    def test_mean_of_multiple(self):
        p = ConfidenceProxy()
        assert p.score(_record([0.4, 0.6])) == pytest.approx(0.5)

    def test_all_high_confs(self):
        p = ConfidenceProxy()
        assert p.score(_record([0.9, 0.95, 0.85])) == pytest.approx((0.9 + 0.95 + 0.85) / 3)

    def test_name_attribute(self):
        assert ConfidenceProxy.name == "confidence"


# ── CalibratedConfidenceProxy ──────────────────────────────────────────────────

class TestCalibratedConfidenceProxy:
    def test_t1_matches_sigmoid_of_logit(self):
        """T=1.0 should recover: sigmoid(log(c/(1-c)) / 1) = c for mid-range c."""
        proxy = CalibratedConfidenceProxy.__new__(CalibratedConfidenceProxy)
        proxy._temperatures = {"m1": 1.0}
        proxy._calibration_path = ""
        # For c=0.7: log(0.7/0.3)/1 → sigmoid → should be ≈ 0.7
        c = 0.7
        expected = _sigmoid(math.log(c / (1 - c)))
        result = proxy.score(_record([c]))
        assert result == pytest.approx(expected, abs=1e-6)

    def test_high_temperature_reduces_score(self):
        """T>1 (overconfident model) should pull scores toward 0.5."""
        proxy_low_T = CalibratedConfidenceProxy.__new__(CalibratedConfidenceProxy)
        proxy_low_T._temperatures = {"m1": 1.0}
        proxy_low_T._calibration_path = ""

        proxy_high_T = CalibratedConfidenceProxy.__new__(CalibratedConfidenceProxy)
        proxy_high_T._temperatures = {"m1": 3.0}
        proxy_high_T._calibration_path = ""

        record = _record([0.9, 0.8, 0.85])
        score_low = proxy_low_T.score(record)
        score_high = proxy_high_T.score(record)
        assert score_high < score_low, "Higher T should reduce overconfident scores"

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "calibration.json")
            proxy = CalibratedConfidenceProxy.__new__(CalibratedConfidenceProxy)
            proxy._calibration_path = path
            proxy._temperatures = {"yolo_n": 1.5, "yolo_s": 0.8}
            proxy._save_calibration()

            proxy2 = CalibratedConfidenceProxy.__new__(CalibratedConfidenceProxy)
            proxy2._calibration_path = path
            proxy2._temperatures = {}
            proxy2._load_calibration()
            assert proxy2._temperatures == {"yolo_n": 1.5, "yolo_s": 0.8}

    def test_missing_model_defaults_to_t1(self):
        proxy = CalibratedConfidenceProxy.__new__(CalibratedConfidenceProxy)
        proxy._temperatures = {"other_model": 2.0}
        proxy._calibration_path = ""
        # "unknown_model" not in _temperatures → T=1.0
        c = 0.6
        expected = _sigmoid(math.log(c / (1 - c)))
        assert proxy.score(_record([c], model="unknown_model")) == pytest.approx(expected, abs=1e-4)

    def test_empty_confs_returns_zero(self):
        proxy = CalibratedConfidenceProxy.__new__(CalibratedConfidenceProxy)
        proxy._temperatures = {}
        proxy._calibration_path = ""
        assert proxy.score(_record([])) == 0.0

    def test_name_attribute(self):
        assert CalibratedConfidenceProxy.name == "calibrated_confidence"


# ── AgreementProxy ─────────────────────────────────────────────────────────────

class TestAgreementProxy:
    def _box(self, x1, y1, x2, y2, cls=0):
        return {"bbox": [x1, y1, x2, y2], "class": cls, "conf": 0.9}

    def test_full_agreement_returns_one(self):
        proxy = AgreementProxy(iou_threshold=0.5)
        primary = [[self._box(0, 0, 10, 10)]]
        companion = [[self._box(0, 0, 10, 10)]]
        assert proxy.score(_record([], primary_predictions=primary, companion_predictions=companion)) == pytest.approx(1.0)

    def test_zero_agreement_class_mismatch(self):
        proxy = AgreementProxy(iou_threshold=0.5)
        primary = [[self._box(0, 0, 10, 10, cls=0)]]
        companion = [[self._box(0, 0, 10, 10, cls=1)]]  # different class
        score = proxy.score(_record([], primary_predictions=primary, companion_predictions=companion))
        assert score == pytest.approx(0.0)

    def test_partial_agreement(self):
        proxy = AgreementProxy(iou_threshold=0.5)
        primary = [[self._box(0, 0, 10, 10), self._box(20, 20, 30, 30)]]
        companion = [[self._box(0, 0, 10, 10)]]  # only first box matches
        score = proxy.score(_record([], primary_predictions=primary, companion_predictions=companion))
        assert score == pytest.approx(0.5)

    def test_no_companion_falls_back_to_confidence(self):
        proxy = AgreementProxy()
        record = _record([0.6, 0.8])
        assert proxy.score(record) == pytest.approx(0.7)

    def test_empty_primary_returns_zero(self):
        proxy = AgreementProxy(iou_threshold=0.5)
        score = proxy.score(_record([], primary_predictions=[[]], companion_predictions=[[]]))
        assert score == pytest.approx(0.0)

    def test_iou_perfect_overlap(self):
        iou = AgreementProxy._iou([0, 0, 10, 10], [0, 0, 10, 10])
        assert iou == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        iou = AgreementProxy._iou([0, 0, 5, 5], [10, 10, 20, 20])
        assert iou == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        # Two 10x10 boxes overlapping by 5x5
        iou = AgreementProxy._iou([0, 0, 10, 10], [5, 5, 15, 15])
        expected = 25.0 / (100 + 100 - 25)
        assert iou == pytest.approx(expected, abs=1e-6)

    def test_name_attribute(self):
        assert AgreementProxy.name == "agreement"
