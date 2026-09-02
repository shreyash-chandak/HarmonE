"""
tests/test_cv_prt_vmr.py — Tests for CV periodic-retrain (PRT) + VMR fine-tuning.

Covers experiments/run_experiment.py's CV inline-finetune path across all
three CV task families (classification/segmentation/detection):
_select_finetune_params (shared, architecture-agnostic layer selection),
_do_cv_inline_finetune (dispatcher), _finetune_torchvision_classifier (full
coverage), _load_cv_model_store gating, and dispatch routing for the
segmentation/detection branches (_finetune_segformer_segmentation /
_finetune_yolo_detection are dispatch-tested via monkeypatch here, not given
full behavioural coverage — they need transformers/ultralytics installed,
which this dev environment does not have; verify those two on the actual
GPU/torchvision+transformers+ultralytics machine before relying on them).

Uses plain torch.nn models (no torchvision needed) so these tests run on any
machine with torch installed, matching the pattern already used by
tests/test_tent_and_augur.py.

Development note on the safety valve: pseudo-label self-training on a tiny
classifier was found, empirically, to reliably INCREASE max-softmax
confidence on held-out data (even out-of-distribution holdout) across a wide
range of seeds/LRs/epoch counts tried while building these tests — a
confirmation-bias effect consistent with known pseudo-labeling literature. A
max-confidence safety valve is well-suited to catching confidence COLLAPSE,
but should not be assumed to trigger from ordinary self-training instability
alone. The reject-path test below constructs an unsatisfiable tau_regression
directly rather than relying on training dynamics to produce one.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")
nn = torch.nn

import experiments.run_experiment as run_experiment_module
from experiments.run_experiment import (
    _select_finetune_params,
    _do_cv_inline_finetune,
    _load_cv_model_store,
)


def _tiny_classifier(n_classes=4, bias_class=0, image_size=8):
    """A small deterministic classifier: heavily biased toward `bias_class`
    regardless of input, so pseudo-label confidence is high and stable
    (avoids flaky tests tied to random-init softmax variance)."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(image_size * image_size * 3, 16),
        nn.ReLU(),
        nn.Linear(16, n_classes),
    )
    with torch.no_grad():
        final = model[-1]
        final.weight.zero_()
        final.bias.zero_()
        final.bias[bias_class] = 10.0
    model.eval()
    return model


def _make_test_images(tmp_path, n=10, size=8):
    from PIL import Image
    paths = []
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    return paths


def _simple_transform(img):
    arr = np.asarray(img.resize((8, 8)).convert("RGB"), dtype=np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1)


# ── _select_finetune_params ─────────────────────────────────────────────────

class TestSelectFinetuneParams:
    def test_selects_last_n_leaf_modules(self):
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2),
        )
        # 3 parameterised leaf modules total (the three Linear layers).
        params = _select_finetune_params(model, n_layers=1)
        last_linear = model[-1]
        expected_ids = {id(p) for p in last_linear.parameters(recurse=False)}
        assert {id(p) for p in params} == expected_ids

    def test_n_layers_larger_than_model_selects_all(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        params = _select_finetune_params(model, n_layers=100)
        all_params = [p for m in model.modules() if list(m.parameters(recurse=False))
                      for p in m.parameters(recurse=False)]
        assert len(params) == len(all_params)

    def test_zero_layers_selects_nothing(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        assert _select_finetune_params(model, n_layers=0) == []


# ── _do_cv_inline_finetune ──────────────────────────────────────────────────

class TestDoCvInlineFinetune:
    def test_wrong_model_type_returns_false(self):
        model_store = {"m1": {"type": "sklearn", "model": object()}}
        result = _do_cv_inline_finetune(
            "m1", model_store, {}, ["fake.jpg"], thresholds={}, drift_window=10,
        )
        assert result is False

    def test_missing_model_returns_false(self):
        result = _do_cv_inline_finetune(
            "missing", {}, {}, ["fake.jpg"], thresholds={}, drift_window=10,
        )
        assert result is False

    def test_empty_window_returns_false(self):
        model = _tiny_classifier()
        model_store = {"m1": {"type": "torchvision_classifier", "model": model,
                               "transform": _simple_transform, "arch": "fake", "num_classes": 4}}
        result = _do_cv_inline_finetune(
            "m1", model_store, {}, [], thresholds={}, drift_window=10,
        )
        assert result is False

    def test_too_few_confident_pseudo_labels_skips(self, tmp_path):
        model = _tiny_classifier()
        images = _make_test_images(tmp_path, n=10)
        model_store = {"m1": {"type": "torchvision_classifier", "model": model,
                               "transform": _simple_transform, "arch": "fake", "num_classes": 4}}
        models = {"m1": lambda p: None}
        result = _do_cv_inline_finetune(
            "m1", model_store, models, images,
            thresholds={"pseudo_label_threshold": 0.999999, "finetune_min_images": 5},
            drift_window=10,
        )
        assert result is False

    def test_successful_finetune_updates_predict_closure(self, tmp_path):
        model = _tiny_classifier(bias_class=0)
        images = _make_test_images(tmp_path, n=20)
        model_store = {"m1": {"type": "torchvision_classifier", "model": model,
                               "transform": _simple_transform, "arch": "fake", "num_classes": 4}}
        models = {"m1": lambda p: None}

        result = _do_cv_inline_finetune(
            "m1", model_store, models, images,
            thresholds={
                "pseudo_label_threshold": 0.5,
                "finetune_min_images": 5,
                "finetune_n_layers": 2,
                "finetune_epochs": 1,
                "finetune_lr": 1e-3,
                "tau_regression": 1.0,  # never trip the safety valve for this test
            },
            drift_window=20,
        )
        assert result is True
        # The predict closure must have been rebuilt (new function object).
        assert models["m1"] is not None
        out = models["m1"](images[0])
        assert set(out.keys()) == {"proxy", "pred"}
        assert 0.0 <= out["proxy"] <= 1.0

    def test_successful_finetune_actually_changes_weights(self, tmp_path):
        """Companion to test_successful_finetune_updates_predict_closure: proves
        the accepted fine-tune is a real weight update, not a no-op that would
        make the safety-valve test below meaningless."""
        model = _tiny_classifier(bias_class=0)
        images = _make_test_images(tmp_path, n=20)
        pre_state = {k: v.clone() for k, v in model.state_dict().items()}
        model_store = {"m1": {"type": "torchvision_classifier", "model": model,
                               "transform": _simple_transform, "arch": "fake", "num_classes": 4}}
        models = {"m1": lambda p: None}

        result = _do_cv_inline_finetune(
            "m1", model_store, models, images,
            thresholds={
                "pseudo_label_threshold": 0.5, "finetune_min_images": 5,
                "finetune_n_layers": 3, "finetune_epochs": 3,
                "finetune_lr": 1e-2, "tau_regression": 1.0,
            },
            drift_window=20,
        )
        assert result is True
        changed = any(not torch.equal(v, pre_state[k]) for k, v in model.state_dict().items())
        assert changed, "fine-tune accepted but no weights actually changed"

    def test_safety_valve_rejects_and_restores_weights(self, tmp_path):
        """tau_regression=-1.0 makes the reject condition
        (post_tune_conf < pre_tune_conf - tau_regression) equivalent to
        post_tune_conf < pre_tune_conf + 1.0 — unsatisfiable in the other
        direction, since confidence is bounded in [0, 1]: any pre_tune_conf
        <= 1.0 makes pre_tune_conf + 1.0 an upper bound post_tune_conf can
        never reach or exceed. This deterministically trips the valve
        regardless of which direction training moves confidence, without
        depending on training dynamics — isolating the restore mechanism
        (state_dict snapshot -> reload) from the confidence-comparison
        heuristic, which is unreliable to trigger via real training (see
        module docstring note above on the confirmation-bias effect)."""
        model = _tiny_classifier(bias_class=0)
        images = _make_test_images(tmp_path, n=20)
        pre_state = {k: v.clone() for k, v in model.state_dict().items()}
        model_store = {"m1": {"type": "torchvision_classifier", "model": model,
                               "transform": _simple_transform, "arch": "fake", "num_classes": 4}}
        models = {"m1": lambda p: None}

        result = _do_cv_inline_finetune(
            "m1", model_store, models, images,
            thresholds={
                "pseudo_label_threshold": 0.5, "finetune_min_images": 5,
                "finetune_n_layers": 3, "finetune_epochs": 3,
                "finetune_lr": 1e-2, "tau_regression": -1.0,
            },
            drift_window=20,
        )
        assert result is False
        for k, v in model.state_dict().items():
            assert torch.equal(v, pre_state[k]), f"weights not restored for {k}"


# ── _load_cv_model_store task gating ────────────────────────────────────────

class TestLoadCvModelStoreGating:
    """_load_cv_model_store now attempts to build all three CV task families
    (classification/segmentation/detection) — see the follow-up that extended
    PRT+VMR beyond classification-only. These tests cover the parts that
    don't require torchvision/transformers/ultralytics to be installed:
    missing-weights and unrecognised-task gating."""

    def test_missing_weights_file_gives_none_entry_regardless_of_task(self):
        for task in ("classification", "segmentation", "detection", "unknown_task"):
            store = _load_cv_model_store(
                {"models": {"m1": {"weights_path": "does_not_exist.pt"}}}, task,
            )
            assert store == {"m1": None}, f"task={task}"

    def test_unrecognised_task_with_real_weights_file_gives_none(self, tmp_path):
        wp = tmp_path / "weights.pt"
        wp.write_bytes(b"not a real checkpoint, just needs to exist")
        store = _load_cv_model_store(
            {"models": {"m1": {"weights_path": str(wp)}}}, "some_future_task",
        )
        assert store == {"m1": None}


# ── _do_cv_inline_finetune dispatch routing ─────────────────────────────────

class TestDispatchRouting:
    """Confirms _do_cv_inline_finetune routes to the correct per-task
    implementation based on model_store[name]["type"], without needing
    transformers/ultralytics installed to exercise the segmentation/detection
    bodies themselves (monkeypatched out)."""

    def test_routes_torchvision_classifier(self, monkeypatch):
        called = {}
        def fake(*args, **kwargs):
            called["fn"] = "classifier"
            return True
        monkeypatch.setattr(run_experiment_module, "_finetune_torchvision_classifier", fake)
        model_store = {"m1": {"type": "torchvision_classifier"}}
        result = _do_cv_inline_finetune("m1", model_store, {}, ["x.jpg"], thresholds={}, drift_window=10)
        assert result is True
        assert called["fn"] == "classifier"

    def test_routes_segformer_segmentation(self, monkeypatch):
        called = {}
        def fake(*args, **kwargs):
            called["fn"] = "segformer"
            return True
        monkeypatch.setattr(run_experiment_module, "_finetune_segformer_segmentation", fake)
        model_store = {"m1": {"type": "segformer_segmentation"}}
        result = _do_cv_inline_finetune("m1", model_store, {}, ["x.jpg"], thresholds={}, drift_window=10)
        assert result is True
        assert called["fn"] == "segformer"

    def test_routes_yolo_detection_with_run_path(self, monkeypatch, tmp_path):
        received = {}
        def fake(model_name, model_store, models, image_path_history, thresholds, drift_window, run_path):
            received["run_path"] = run_path
            return True
        monkeypatch.setattr(run_experiment_module, "_finetune_yolo_detection", fake)
        model_store = {"m1": {"type": "yolo_detection"}}
        result = _do_cv_inline_finetune(
            "m1", model_store, {}, ["x.jpg"], thresholds={}, drift_window=10, run_path=tmp_path,
        )
        assert result is True
        assert received["run_path"] == tmp_path

    def test_unknown_type_returns_false_without_calling_anything(self):
        model_store = {"m1": {"type": "something_else"}}
        result = _do_cv_inline_finetune("m1", model_store, {}, ["x.jpg"], thresholds={}, drift_window=10)
        assert result is False

    def test_missing_model_entry_returns_false(self):
        result = _do_cv_inline_finetune("missing", {}, {}, ["x.jpg"], thresholds={}, drift_window=10)
        assert result is False
