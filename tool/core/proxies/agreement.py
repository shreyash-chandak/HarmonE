"""
core/proxies/agreement.py — Multi-model agreement accuracy proxy.

Runs the smallest available companion model on a config-fraction subsample
of inputs and measures detection-level agreement (matched boxes IoU > iou_threshold,
same class). Agreement rate is used as the proxy A_i.

Higher agreement → current model is likely making accurate predictions.
Overhead: extra inference energy logged under event type "proxy_agreement_energy".

Config keys (in thresholds.json or dataset config):
  agreement_subsample_frac: float (default 0.1) — fraction of interval images to check
  agreement_iou_threshold:  float (default 0.5) — IoU threshold for box match
  agreement_companion:      str | None — companion model name; None → smallest available
"""

from __future__ import annotations

from typing import Any

from .base import AccuracyProxy


class AgreementProxy(AccuracyProxy):
    """Proxy based on detection-level agreement with a companion model.

    Note: Full implementation requires access to the model objects at score() time.
    In the runtime path, monitor.py passes the loaded companion model via
    inference_record["companion_predictions"]. If absent, falls back to
    raw confidence (graceful degradation).
    """

    name = "agreement"

    def __init__(
        self,
        iou_threshold: float = 0.5,
        subsample_frac: float = 0.1,
        companion_model: str | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.subsample_frac = subsample_frac
        self.companion_model = companion_model

    def score(self, inference_record: dict) -> float:
        """Compute agreement rate.

        Expects inference_record to contain:
          "confidences": list[float]
          "companion_predictions": list[list[dict]] — boxes from companion model
            Each box: {"bbox": [x1,y1,x2,y2], "class": int, "conf": float}
          "primary_predictions": same structure for current model

        Falls back to mean confidence if companion predictions are unavailable.
        """
        primary = inference_record.get("primary_predictions")
        companion = inference_record.get("companion_predictions")

        if not primary or not companion:
            # Graceful fallback: use raw confidence
            confs = inference_record.get("confidences", [])
            return float(sum(confs) / len(confs)) if confs else 0.0

        return self._compute_agreement(primary, companion)

    def _compute_agreement(
        self,
        primary: list[list[dict]],
        companion: list[list[dict]],
    ) -> float:
        """Compute detection-level agreement across the sampled frame set."""
        total_frames = min(len(primary), len(companion))
        if total_frames == 0:
            return 0.0

        agreements = 0
        total_boxes = 0

        for p_boxes, c_boxes in zip(primary, companion):
            for p_box in p_boxes:
                total_boxes += 1
                # Check if any companion box matches this primary box
                matched = any(
                    self._boxes_agree(p_box, c_box)
                    for c_box in c_boxes
                )
                if matched:
                    agreements += 1

        return agreements / max(total_boxes, 1)

    def _boxes_agree(self, a: dict, b: dict) -> bool:
        """Return True if two boxes agree (same class, IoU > threshold)."""
        if a.get("class") != b.get("class"):
            return False
        iou = self._iou(a["bbox"], b["bbox"])
        return iou >= self.iou_threshold

    @staticmethod
    def _iou(ba: list[float], bb: list[float]) -> float:
        """Compute IoU of two [x1, y1, x2, y2] boxes."""
        ix1 = max(ba[0], bb[0])
        iy1 = max(ba[1], bb[1])
        ix2 = min(ba[2], bb[2])
        iy2 = min(ba[3], bb[3])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = (ba[2] - ba[0]) * (ba[3] - ba[1])
        area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-9)
