"""
core/proxies/confidence.py — Raw mean detection confidence proxy.

The simplest proxy: mean confidence of all detections in the interval.
This is the current CV monitor behavior (avg_conf) and serves as the
uncalibrated baseline for proxy validation (RQ3).

No calibration needed; setup() is a no-op.
"""

from __future__ import annotations

from .base import AccuracyProxy


class ConfidenceProxy(AccuracyProxy):
    """S_proxy = mean(confidence_scores) over the monitoring interval."""

    name = "confidence"

    def score(self, inference_record: dict) -> float:
        confs = inference_record.get("confidences", [])
        if not confs:
            return 0.0
        return float(sum(confs) / len(confs))
