"""tests/test_offline_eval.py — Offline eval, 3-way split (2026-08-30 audit).

Covers experiments/offline_eval_{detection,classification,segmentation}.py
via the fast path (saved predictions/*.{npz,txt,png}) — the path every real
run_experiment.py CV run now populates — using small synthetic fixtures with
hand-computed expected metrics. Also covers offline_eval_common.py's
class_mapping.csv lookup and offline_eval.py's dispatcher/manifest-merge
behavior via monkeypatch (no real model weights needed for any of this).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.offline_eval_common import build_stream_index, load_class_names
from experiments.offline_eval_classification import evaluate_classification
from experiments.offline_eval_segmentation import (
    _class_confusion_counts,
    _miou_from_counts,
    evaluate_segmentation,
)

# evaluate_detection's _compute_map needs torch + torchvision.ops.box_iou at
# CALL time (not import time — the import inside offline_eval_detection.py
# is lazy). This dev environment has torch but not torchvision (see
# tests/test_cv_prt_vmr.py's docstring for the same constraint) — skip only
# the detection test class rather than importorskip-ing the whole module,
# which would otherwise silently skip the classification/segmentation/common
# tests below too (they need neither).
try:
    import torchvision  # noqa: F401
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

if _HAS_TORCHVISION:
    from experiments.offline_eval_detection import evaluate_detection


def _predictions_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── offline_eval_common ─────────────────────────────────────────────────────

class TestBuildStreamIndex:
    def test_inline_label_and_label_path_columns_both_supported(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.jpg").write_bytes(b"")
        (tmp_path / "b_mask.png").write_bytes(b"")
        manifest = tmp_path / "m.csv"
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["input_path", "label", "label_path"])
            w.writeheader()
            w.writerow({"input_path": "a.jpg", "label": "3", "label_path": ""})
            w.writerow({"input_path": "b.jpg", "label": "", "label_path": "b_mask.png"})
        config = {"manifest_csv": str(manifest), "data_root": str(tmp_path), "train_frac": 0.0, "val_frac": 0.0}
        images, label_paths, inline_labels = build_stream_index(config)
        assert len(images) == 2
        assert inline_labels == [3, None]
        assert label_paths[0] is None
        assert label_paths[1] is not None and label_paths[1].endswith("b_mask.png")


class TestLoadClassNames:
    def test_returns_mapping_when_file_present(self, tmp_path):
        manifest = tmp_path / "m.csv"
        manifest.write_text("input_path,label\n")
        mapping = tmp_path / "class_mapping.csv"
        with open(mapping, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", "class_name"])
            w.writeheader()
            w.writerow({"label": "0", "class_name": "abacus"})
            w.writerow({"label": "1", "class_name": "abaya"})
        names = load_class_names({"manifest_csv": str(manifest)})
        assert names == {0: "abacus", 1: "abaya"}

    def test_prefers_trimmed_class_index_over_class_mapping(self, tmp_path):
        """After scripts/trim_manifest.py's imagenet strategy remaps labels to
        a fresh 0..N-1 range, class_mapping.csv's original 0..999 labels no
        longer correspond to the manifest at all — class_index_trimmed.json
        must win when both files are present."""
        manifest = tmp_path / "m.csv"
        manifest.write_text("input_path,label\n")
        mapping = tmp_path / "class_mapping.csv"
        with open(mapping, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", "class_name"])
            w.writeheader()
            w.writerow({"label": "0", "class_name": "abacus"})
        trimmed = tmp_path / "class_index_trimmed.json"
        trimmed.write_text(json.dumps({"index_to_class": {"0": "zebra"}}))
        names = load_class_names({"manifest_csv": str(manifest)})
        assert names == {0: "zebra"}

    def test_returns_none_when_absent(self, tmp_path):
        manifest = tmp_path / "m.csv"
        manifest.write_text("input_path,label\n")
        assert load_class_names({"manifest_csv": str(manifest)}) is None


# ── offline_eval_classification ─────────────────────────────────────────────

class TestEvaluateClassification:
    def _make_fixture(self, tmp_path):
        """3 images: (pred=0,gt=0), (pred=1,gt=1), (pred=0,gt=1)."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        preds_dir = tmp_path / "run" / "predictions"
        preds_dir.mkdir(parents=True)

        rows = []
        for i, gt in enumerate([0, 1, 1]):
            (images_dir / f"img{i}.jpg").write_bytes(b"")
            rows.append({"step": i, "active_model": "m1", "planner": "naive"})
        for step, pred in enumerate([0, 1, 0]):
            (preds_dir / f"step_{step:06d}.txt").write_text(str(pred))

        pred_df = _predictions_df(rows)
        stream_images = [str(images_dir / f"img{i}.jpg") for i in range(3)]
        stream_labels = [None, None, None]
        stream_inline_labels = [0, 1, 1]
        return pred_df, stream_images, stream_labels, stream_inline_labels, tmp_path / "run"

    def test_confusion_matrix_derived_metrics(self, tmp_path):
        pred_df, imgs, labels, inline, run_path = self._make_fixture(tmp_path)
        result = evaluate_classification(
            {"num_classes": 2}, pred_df, imgs, labels, inline, interval_size=10, run_path=run_path,
        )
        overall = result["overall"]
        assert overall["n_samples"] == 3
        assert overall["accuracy"] == pytest.approx(2 / 3, abs=1e-6)
        assert overall["precision_macro"] == pytest.approx(0.75, abs=1e-6)
        assert overall["recall_macro"] == pytest.approx(0.75, abs=1e-6)
        assert overall["f1_macro"] == pytest.approx(2 / 3, abs=1e-6)
        assert overall["tp"] == 2
        assert overall["fp"] == 1
        assert overall["fn"] == 1
        assert overall["tn"] == 2

    def test_per_class_breakdown_present(self, tmp_path):
        pred_df, imgs, labels, inline, run_path = self._make_fixture(tmp_path)
        result = evaluate_classification(
            {"num_classes": 2}, pred_df, imgs, labels, inline, interval_size=10, run_path=run_path,
        )
        assert len(result["per_class"]) == 2
        assert {c["label"] for c in result["per_class"]} == {0, 1}

    def test_interval_accuracy(self, tmp_path):
        pred_df, imgs, labels, inline, run_path = self._make_fixture(tmp_path)
        result = evaluate_classification(
            {"num_classes": 2}, pred_df, imgs, labels, inline, interval_size=2, run_path=run_path,
        )
        # interval 0: steps 0,1 -> both correct -> acc=1.0; interval 1: step 2 -> wrong -> acc=0.0
        by_idx = {r["interval_idx"]: r for r in result["interval_results"]}
        assert by_idx[0]["accuracy"] == pytest.approx(1.0)
        assert by_idx[1]["accuracy"] == pytest.approx(0.0)

    def test_no_records_returns_none_overall(self, tmp_path):
        run_path = tmp_path / "empty_run"
        (run_path / "predictions").mkdir(parents=True)
        result = evaluate_classification(
            {"num_classes": 2}, _predictions_df([]), [], [], [], interval_size=10, run_path=run_path,
        )
        assert result["overall"] is None


# ── offline_eval_segmentation ───────────────────────────────────────────────

class TestSegmentationMath:
    def test_class_confusion_counts_and_miou(self):
        gt = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        pred = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        tp, fp, fn = _class_confusion_counts(pred, gt, num_classes=2, ignore_index=255)
        assert tp.tolist() == [1, 2]
        assert fp.tolist() == [0, 1]
        assert fn.tolist() == [1, 0]
        miou = _miou_from_counts(tp, fp, fn)
        expected = np.mean([1 / 2, 2 / 3])
        assert miou == pytest.approx(expected, abs=1e-6)

    def test_ignore_index_excluded(self):
        gt = np.array([[0, 255], [1, 1]], dtype=np.int64)
        pred = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        tp, fp, fn = _class_confusion_counts(pred, gt, num_classes=2, ignore_index=255)
        # the 255 pixel must not count toward class 1's fp
        assert tp.tolist() == [1, 2]
        assert fp.tolist() == [0, 0]
        assert fn.tolist() == [0, 0]


class TestEvaluateSegmentation:
    def _make_fixture(self, tmp_path):
        from PIL import Image

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        preds_dir = tmp_path / "run" / "predictions"
        preds_dir.mkdir(parents=True)

        gt = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        pred = np.array([[0, 1], [1, 1]], dtype=np.uint8)

        img_path = images_dir / "img0.jpg"
        Image.new("RGB", (2, 2)).save(img_path)
        label_path = images_dir / "gt0.png"
        Image.fromarray(gt, mode="L").save(label_path)
        Image.fromarray(pred, mode="L").save(preds_dir / "step_000000.png")

        pred_df = _predictions_df([{"step": 0, "active_model": "seg1", "planner": "naive"}])
        return pred_df, [str(img_path)], [str(label_path)], tmp_path / "run"

    def test_pooled_miou_matches_hand_computation(self, tmp_path):
        pred_df, imgs, labels, run_path = self._make_fixture(tmp_path)
        result = evaluate_segmentation(
            {"num_classes": 2, "ignore_index": 255}, pred_df, imgs, labels,
            interval_size=10, run_path=run_path,
        )
        expected = np.mean([1 / 2, 2 / 3])
        assert result["overall"]["miou"] == pytest.approx(expected, abs=1e-6)
        assert result["overall"]["pooled"] is True
        assert result["overall"]["n_images"] == 1

    def test_no_label_path_skips_step(self, tmp_path):
        pred_df, imgs, _labels, run_path = self._make_fixture(tmp_path)
        result = evaluate_segmentation(
            {"num_classes": 2}, pred_df, imgs, [None], interval_size=10, run_path=run_path,
        )
        assert result["overall"] is None


# ── offline_eval_detection ──────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_TORCHVISION, reason="torchvision not installed in this environment")
class TestEvaluateDetection:
    def _make_fixture(self, tmp_path):
        from PIL import Image

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        preds_dir = tmp_path / "run" / "predictions"
        preds_dir.mkdir(parents=True)

        # image 0: perfect match (pred box == GT box, IoU=1.0)
        img0 = images_dir / "img0.jpg"
        Image.new("RGB", (100, 100)).save(img0)
        lbl0 = images_dir / "lbl0.txt"
        lbl0.write_text("0 0.25 0.25 0.5 0.5\n")  # box [0,0,50,50] on a 100x100 image
        np.savez(
            preds_dir / "step_000000.npz",
            boxes=np.array([[0.0, 0.0, 50.0, 50.0]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
            classes=np.array([0], dtype=np.int32),
        )

        # image 1: prediction misses the GT entirely (IoU=0)
        img1 = images_dir / "img1.jpg"
        Image.new("RGB", (100, 100)).save(img1)
        lbl1 = images_dir / "lbl1.txt"
        lbl1.write_text("0 0.25 0.25 0.5 0.5\n")
        np.savez(
            preds_dir / "step_000001.npz",
            boxes=np.array([[60.0, 60.0, 90.0, 90.0]], dtype=np.float32),
            scores=np.array([0.8], dtype=np.float32),
            classes=np.array([0], dtype=np.int32),
        )

        pred_df = _predictions_df([
            {"step": 0, "active_model": "yolo_n", "planner": "naive"},
            {"step": 1, "active_model": "yolo_n", "planner": "naive"},
        ])
        return pred_df, [str(img0), str(img1)], [str(lbl0), str(lbl1)], tmp_path / "run"

    def test_map_matches_hand_computed_11point_ap(self, tmp_path):
        pred_df, imgs, labels, run_path = self._make_fixture(tmp_path)
        result = evaluate_detection(
            {"num_classes": 1}, pred_df, imgs, labels, interval_size=10, run_path=run_path,
        )
        # one perfect match (matches at any IoU threshold) + one total miss ->
        # recall caps at 0.5 -> 11-point AP = 6/11 at every threshold tested here.
        expected = 6 / 11
        overall = result["overall"]
        assert overall["map50"] == pytest.approx(expected, abs=1e-6)
        assert overall["map75"] == pytest.approx(expected, abs=1e-6)
        assert overall["map90"] == pytest.approx(expected, abs=1e-6)
        assert overall["n_images"] == 2

    def test_metric_names(self, tmp_path):
        pred_df, imgs, labels, run_path = self._make_fixture(tmp_path)
        result = evaluate_detection(
            {"num_classes": 1}, pred_df, imgs, labels, interval_size=10, run_path=run_path,
        )
        assert result["metric_names"] == ["map50", "map75", "map90"]

    def test_no_records_returns_none_overall(self, tmp_path):
        run_path = tmp_path / "empty_run"
        (run_path / "predictions").mkdir(parents=True)
        result = evaluate_detection(
            {"num_classes": 1}, _predictions_df([]), [], [], interval_size=10, run_path=run_path,
        )
        assert result["overall"] is None


# ── offline_eval.py dispatcher ──────────────────────────────────────────────

class TestDispatcherAndManifestMerge:
    def test_evaluate_run_merges_offline_task_metrics_into_manifest(self, tmp_path, monkeypatch):
        import experiments.offline_eval as oe

        run_path = tmp_path / "run"
        run_path.mkdir()
        (run_path / "predictions.csv").write_text("step,active_model,planner,proxy_acc,energy_uJ\n0,m1,naive,0.9,1.0\n")
        manifest_path = run_path / "run_manifest.json"
        manifest_path.write_text(json.dumps({"run_id": "run", "task_metrics": {"mean_proxy_acc": 0.9}}))

        config = {"task": "classification", "manifest_csv": str(tmp_path / "does_not_matter.csv")}
        monkeypatch.setattr(oe, "load_config", lambda dataset: config)
        monkeypatch.setattr(oe, "build_stream_index", lambda cfg: ([], [], []))
        monkeypatch.setattr(
            oe, "evaluate_classification",
            lambda *a, **k: {"overall": {"accuracy": 1.0, "precision_macro": 1.0, "recall_macro": 1.0,
                                          "f1_macro": 1.0, "tp": 1, "tn": 0, "fp": 0, "fn": 0, "n_samples": 1},
                              "per_class": [], "interval_results": [], "n_intervals": 0},
        )

        oe.evaluate_run(run_dir=str(run_path), dataset="fake", interval_size=10)

        merged = json.loads(manifest_path.read_text())
        assert "offline_task_metrics" in merged
        assert merged["offline_task_metrics"]["task"] == "classification"
        assert merged["offline_task_metrics"]["overall"]["accuracy"] == 1.0
        # live task_metrics (proxy-based) must survive alongside the offline ones
        assert merged["task_metrics"]["mean_proxy_acc"] == 0.9

    def test_evaluate_run_dispatches_by_task(self, tmp_path, monkeypatch):
        import experiments.offline_eval as oe

        run_path = tmp_path / "run"
        run_path.mkdir()
        (run_path / "predictions.csv").write_text("step,active_model,planner,proxy_acc,energy_uJ\n0,m1,naive,0.9,1.0\n")

        calls = []
        monkeypatch.setattr(oe, "load_config", lambda dataset: {"task": "detection"})
        monkeypatch.setattr(oe, "build_stream_index", lambda cfg: ([], [], []))
        monkeypatch.setattr(
            oe, "evaluate_detection",
            lambda *a, **k: calls.append("detection") or {"overall": None, "interval_results": [], "n_intervals": 0},
        )
        oe.evaluate_run(run_dir=str(run_path), dataset="fake", interval_size=10)
        assert calls == ["detection"]

    def test_unsupported_task_raises(self, tmp_path, monkeypatch):
        import experiments.offline_eval as oe

        run_path = tmp_path / "run"
        run_path.mkdir()
        (run_path / "predictions.csv").write_text("step,active_model,planner,proxy_acc,energy_uJ\n0,m1,naive,0.9,1.0\n")
        monkeypatch.setattr(oe, "load_config", lambda dataset: {"task": "weird_task"})
        monkeypatch.setattr(oe, "build_stream_index", lambda cfg: ([], [], []))
        with pytest.raises(ValueError, match="Unsupported task"):
            oe.evaluate_run(run_dir=str(run_path), dataset="fake", interval_size=10)
