"""tests/test_cv_tent.py — TENT integration for CV classification (2026-08-30, DP25).

Covers experiments/run_experiment.py's _finetune_torchvision_classifier_tent
and its dispatch from _do_cv_inline_finetune via the "retrain_tactic": "tent"
config key. Uses a plain torch.nn model with BatchNorm2d (no torchvision
needed) — deliberately BatchNorm2d rather than BatchNorm1d-on-a-flattened-
vector (the pattern test_cv_prt_vmr.py uses for the supervised path): once
core.tta.tent.configure_model() sets track_running_stats=False, a BatchNorm
layer always computes fresh batch statistics regardless of train()/eval()
mode — for BatchNorm1d on a plain (N, C) tensor that breaks at N=1 (single-
image confidence checks, used both before and after adaptation), but
BatchNorm2d has H*W spatial samples per channel even at N=1, matching real
torchvision classifiers (which use BatchNorm2d on real image tensors) and
avoiding an artifact of the test harness rather than testing real behavior.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")
nn = torch.nn

import experiments.run_experiment as run_experiment_module
from experiments.run_experiment import (
    _do_cv_inline_finetune,
    _finetune_torchvision_classifier_tent,
    _restore_bn_tracking,
)
from core.tta.tent import softmax_entropy


def _tiny_bn_classifier(n_classes=4, seed=0, image_size=8):
    """Small conv+BN classifier — BatchNorm2d before the head, matching the
    conv/BN-block structure real torchvision classifiers use. Deliberately
    NOT the "zero the final layer's weight, fix a constant bias" trick
    test_cv_prt_vmr.py uses for stable predictions: TENT only ever trains
    BatchNorm params, never the final layer, and a zero final-layer weight
    makes the whole forward pass constant w.r.t. input — killing gradient
    flow to every upstream parameter, BN included, by construction. Default
    (random, seeded) initialization keeps output genuinely input-dependent
    so entropy-minimization gradients actually reach BN's affine params."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, n_classes),
    )
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


def _store(model):
    return {"m1": {"type": "torchvision_classifier", "model": model,
                    "transform": _simple_transform, "arch": "fake", "num_classes": 4}}


class TestFinetuneTorchvisionClassifierTent:
    def test_wrong_model_type_returns_false(self):
        result = _finetune_torchvision_classifier_tent(
            "m1", {"m1": {"type": "sklearn", "model": object()}}, {}, ["fake.jpg"],
            thresholds={}, drift_window=10,
        )
        assert result is False

    def test_missing_model_returns_false(self):
        result = _finetune_torchvision_classifier_tent(
            "missing", {}, {}, ["fake.jpg"], thresholds={}, drift_window=10,
        )
        assert result is False

    def test_empty_window_returns_false(self):
        model = _tiny_bn_classifier()
        result = _finetune_torchvision_classifier_tent(
            "m1", _store(model), {}, [], thresholds={}, drift_window=10,
        )
        assert result is False

    def test_too_few_images_skips(self, tmp_path):
        model = _tiny_bn_classifier()
        images = _make_test_images(tmp_path, n=5)
        result = _finetune_torchvision_classifier_tent(
            "m1", _store(model), {"m1": lambda p: None}, images,
            thresholds={"finetune_min_images": 20}, drift_window=10,
        )
        assert result is False

    def test_successful_adaptation_updates_predict_closure_and_rebinds(self, tmp_path):
        model = _tiny_bn_classifier()
        images = _make_test_images(tmp_path, n=20)
        models = {"m1": lambda p: None}
        result = _finetune_torchvision_classifier_tent(
            "m1", _store(model), models, images,
            thresholds={
                "finetune_min_images": 5, "tent_lr": 0.1, "tent_batch_size": 8,
                "tent_steps": 1, "tau_regression": 1.0,  # never trip the safety valve
            },
            drift_window=20,
        )
        assert result is True
        out = models["m1"](images[0])
        assert set(out.keys()) == {"proxy", "pred"}

    def test_only_bn_affine_params_change(self, tmp_path):
        """Everything except BatchNorm weight/bias must stay bit-identical —
        TENT's whole point is <1% of parameters move. BN running_mean/var/
        num_batches_tracked are excluded from the "must be identical" check
        for a different reason: the source statistics are discarded (paper:
        "statistics from the source data are discarded") and replaced by
        statistics estimated on the adaptation window (audit N7), so they
        change by design. They must still be present, so the state_dict loads
        strictly into a fresh model (the concurrent harness's reload path)."""
        model = _tiny_bn_classifier()
        images = _make_test_images(tmp_path, n=20)
        pre = {k: v.detach().clone() for k, v in model.state_dict().items()}
        store = _store(model)
        result = _finetune_torchvision_classifier_tent(
            "m1", store, {"m1": lambda p: None}, images,
            thresholds={
                "finetune_min_images": 5, "tent_lr": 0.5, "tent_batch_size": 8,
                "tent_steps": 3, "tau_regression": 1.0,
            },
            drift_window=20,
        )
        assert result is True
        post = store["m1"]["model"].state_dict()

        bn_names = [
            name for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)
        ]
        bn_affine_keys = {f"{n}.weight" for n in bn_names} | {f"{n}.bias" for n in bn_names}
        # Re-estimated on the adaptation window (audit N7).
        bn_discarded_keys = {
            f"{n}.{suffix}" for n in bn_names for suffix in ("running_mean", "running_var")
        }
        bn_untouched_keys = {f"{n}.num_batches_tracked" for n in bn_names}

        for key in pre:
            if key in bn_affine_keys or key in bn_discarded_keys or key in bn_untouched_keys:
                continue
            assert key in post and torch.equal(pre[key], post[key]), \
                f"non-BN param changed or vanished: {key}"

        # BN affine params must actually have moved (confirms adaptation ran).
        assert any(not torch.equal(pre[k], post[k]) for k in bn_affine_keys)
        # BN running stats are present again (re-estimated, not the source
        # ones), so a fresh model loads the adapted weights strictly.
        assert bn_discarded_keys <= set(post.keys())
        assert any(not torch.equal(pre[k], post[k]) for k in bn_discarded_keys)
        _tiny_bn_classifier().load_state_dict(post, strict=True)

    def test_entropy_decreases_after_adaptation(self, tmp_path):
        model = _tiny_bn_classifier()
        images = _make_test_images(tmp_path, n=20)

        def _batch_entropy():
            model.eval()
            from PIL import Image
            xs = torch.stack([_simple_transform(Image.open(p).convert("RGB")) for p in images[:8]])
            with torch.no_grad():
                return float(softmax_entropy(model(xs)).mean().item())

        pre_entropy = _batch_entropy()
        _finetune_torchvision_classifier_tent(
            "m1", _store(model), {"m1": lambda p: None}, images,
            thresholds={
                "finetune_min_images": 5, "tent_lr": 0.5, "tent_batch_size": 8,
                "tent_steps": 5, "tau_regression": 1.0,
            },
            drift_window=20,
        )
        post_entropy = _batch_entropy()
        assert post_entropy <= pre_entropy + 1e-4

    def test_safety_valve_rejects_and_restores_bn_tracking(self, tmp_path):
        """tau_regression=-1.0 makes the reject condition always true
        (post_tune_conf < pre_tune_conf + 1.0), same deterministic trick
        test_cv_prt_vmr.py uses for the supervised path's reject test."""
        model = _tiny_bn_classifier()
        images = _make_test_images(tmp_path, n=20)
        pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        models = {"m1": lambda p: None}

        result = _finetune_torchvision_classifier_tent(
            "m1", _store(model), models, images,
            thresholds={
                "finetune_min_images": 5, "tent_lr": 0.5, "tent_batch_size": 8,
                "tent_steps": 3, "tau_regression": -1.0,
            },
            drift_window=20,
        )
        assert result is False
        # predict closure must NOT have been rebound on rejection.
        assert models["m1"](None) is None

        # BN tracking must be fully restored — not left in "always use fresh
        # batch stats" mode — and weight values must match the pre-state.
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                assert m.track_running_stats is True
                assert m.running_mean is not None
                assert torch.equal(m.running_mean, pre_state[f"{name}.running_mean"])
                assert torch.equal(m.running_var, pre_state[f"{name}.running_var"])

        # A normal single-image inference must work post-rollback (would
        # raise if track_running_stats were still False and BN were still in
        # always-fresh-batch-stats mode with a since-deleted buffer).
        model.eval()
        from PIL import Image
        x = _simple_transform(Image.open(images[0]).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            model(x)  # must not raise


class TestRestoreBnTracking:
    def test_reregisters_buffers_and_reenables_tracking(self):
        model = _tiny_bn_classifier()
        pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        from core.tta.tent import configure_model
        configure_model(model)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert m.track_running_stats is False
                assert m.running_mean is None

        _restore_bn_tracking(model, pre_state)
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                assert m.track_running_stats is True
                assert torch.equal(m.running_mean, pre_state[f"{name}.running_mean"])
                assert torch.equal(m.running_var, pre_state[f"{name}.running_var"])


class TestDispatchByRetrainTactic:
    def test_default_retrain_tactic_uses_pseudo_label_path(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            run_experiment_module, "_finetune_torchvision_classifier",
            lambda *a, **k: calls.append("pseudo_label") or True,
        )
        monkeypatch.setattr(
            run_experiment_module, "_finetune_torchvision_classifier_tent",
            lambda *a, **k: calls.append("tent") or True,
        )
        model = _tiny_bn_classifier()
        _do_cv_inline_finetune(
            "m1", _store(model), {"m1": lambda p: None}, ["fake.jpg"],
            thresholds={}, drift_window=10,
        )
        assert calls == ["pseudo_label"]

    def test_retrain_tactic_tent_dispatches_to_tent_path(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            run_experiment_module, "_finetune_torchvision_classifier",
            lambda *a, **k: calls.append("pseudo_label") or True,
        )
        monkeypatch.setattr(
            run_experiment_module, "_finetune_torchvision_classifier_tent",
            lambda *a, **k: calls.append("tent") or True,
        )
        model = _tiny_bn_classifier()
        _do_cv_inline_finetune(
            "m1", _store(model), {"m1": lambda p: None}, ["fake.jpg"],
            thresholds={"retrain_tactic": "tent"}, drift_window=10,
        )
        assert calls == ["tent"]

    def test_retrain_tactic_only_affects_classification(self, monkeypatch):
        """segformer/yolo model types must ignore retrain_tactic entirely —
        TENT is classification-only (DP25). Dispatch must still route to the
        one real segmentation finetune function, never a TENT variant, no
        matter what retrain_tactic says."""
        calls = []
        monkeypatch.setattr(
            run_experiment_module, "_finetune_segformer_segmentation",
            lambda *a, **k: calls.append("segformer") or True,
        )
        model_store = {"m1": {"type": "segformer_segmentation", "model": object(),
                               "processor": object(), "arch": "fake", "num_classes": 19}}
        result = _do_cv_inline_finetune(
            "m1", model_store, {}, ["fake.jpg"],
            thresholds={"retrain_tactic": "tent"}, drift_window=10,
        )
        assert result is True
        assert calls == ["segformer"]
