"""tests/test_cv_initial_train.py — CV initial-train-if-missing gating (2026-08-30).

Mirrors _train_regression_models()'s "train if weights_path is absent" gate,
for CV. Unlike PRT fine-tune (pseudo-labeled, tested in test_cv_prt_vmr.py
with plain torch.nn substitutes), the three _initial_train_* task functions
build real torchvision/transformers/ultralytics pretrained models directly
(no injectable model object like the PRT functions take via model_store),
so they need real deps + network to execute and can't be substituted the
same way. These tests cover _train_cv_models_if_missing's gating/dispatch
logic only (which models are "missing", routing by cv_task, exception
containment) via monkeypatch — same dispatch-tested-not-behaviorally-verified
scope already accepted for segmentation/detection PRT (see DECISIONS_PENDING
DP21). Verify the three _initial_train_* bodies on the full-dependency
machine before relying on them for real experiments.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import experiments.run_experiment as run_experiment_module
from experiments.run_experiment import _train_cv_models_if_missing


def _dataset_config(models: dict) -> dict:
    return {"models": models, "num_classes": 7}


class TestTrainCvModelsIfMissingGating:
    def test_no_missing_weights_calls_nothing(self, tmp_path, monkeypatch):
        wp = tmp_path / "exists.pt"
        wp.write_bytes(b"x")
        calls = []
        monkeypatch.setattr(
            run_experiment_module, "_initial_train_torchvision_classifier",
            lambda *a, **k: calls.append(a) or True,
        )
        cfg = _dataset_config({"m1": {"weights_path": str(wp)}})
        _train_cv_models_if_missing(cfg, "classification", [], [], [], {}, tmp_path)
        assert calls == []

    def test_missing_classification_weights_dispatches_to_torchvision_trainer(self, tmp_path, monkeypatch):
        missing_wp = tmp_path / "does_not_exist.pt"
        calls = []
        monkeypatch.setattr(
            run_experiment_module, "_initial_train_torchvision_classifier",
            lambda name, wp, num_classes, paths, labels, thr: calls.append(
                (name, wp, num_classes, paths, labels)
            ) or True,
        )
        cfg = _dataset_config({"m1": {"weights_path": str(missing_wp)}})
        _train_cv_models_if_missing(
            cfg, "classification", ["a.jpg", "b.jpg"], [None, None], [1, 2], {}, tmp_path,
        )
        assert len(calls) == 1
        name, wp, num_classes, paths, labels = calls[0]
        assert name == "m1"
        assert wp == str(missing_wp)
        assert num_classes == 7
        assert paths == ["a.jpg", "b.jpg"]
        assert labels == [1, 2]

    def test_missing_segmentation_weights_dispatches_to_segformer_trainer(self, tmp_path, monkeypatch):
        missing_wp = tmp_path / "seg_missing.pt"
        calls = []
        monkeypatch.setattr(
            run_experiment_module, "_initial_train_segformer_segmentation",
            lambda *a, **k: calls.append(a) or True,
        )
        cfg = _dataset_config({"seg1": {"weights_path": str(missing_wp)}})
        _train_cv_models_if_missing(
            cfg, "segmentation", ["a.png"], ["a_mask.png"], [None], {}, tmp_path,
        )
        assert len(calls) == 1

    def test_missing_detection_weights_dispatches_to_yolo_trainer(self, tmp_path, monkeypatch):
        missing_wp = tmp_path / "det_missing.pt"
        calls = []
        monkeypatch.setattr(
            run_experiment_module, "_initial_train_yolo_detection",
            lambda *a, **k: calls.append(a) or True,
        )
        cfg = _dataset_config({"d1": {"weights_path": str(missing_wp)}})
        _train_cv_models_if_missing(
            cfg, "detection", ["a.jpg"], ["a.json"], [None], {}, tmp_path,
        )
        assert len(calls) == 1

    def test_unrecognised_cv_task_skips_without_crashing(self, tmp_path, monkeypatch):
        missing_wp = tmp_path / "missing.pt"
        cfg = _dataset_config({"m1": {"weights_path": str(missing_wp)}})
        # Should log a warning and return without raising or calling any trainer.
        _train_cv_models_if_missing(cfg, "unknown_task", [], [], [], {}, tmp_path)

    def test_trainer_exception_is_contained(self, tmp_path, monkeypatch):
        missing_wp = tmp_path / "missing.pt"

        def _boom(*a, **k):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(run_experiment_module, "_initial_train_torchvision_classifier", _boom)
        cfg = _dataset_config({"m1": {"weights_path": str(missing_wp)}})
        # Must not propagate — mirrors _load_cv_model_store's "None entry, not a crash" contract.
        _train_cv_models_if_missing(cfg, "classification", [], [], [], {}, tmp_path)

    def test_only_missing_models_are_trained(self, tmp_path, monkeypatch):
        present_wp = tmp_path / "present.pt"
        present_wp.write_bytes(b"x")
        missing_wp = tmp_path / "missing.pt"
        calls = []
        monkeypatch.setattr(
            run_experiment_module, "_initial_train_torchvision_classifier",
            lambda name, *a, **k: calls.append(name) or True,
        )
        cfg = _dataset_config({
            "present_model": {"weights_path": str(present_wp)},
            "missing_model": {"weights_path": str(missing_wp)},
        })
        _train_cv_models_if_missing(cfg, "classification", [], [], [], {}, tmp_path)
        assert calls == ["missing_model"]
