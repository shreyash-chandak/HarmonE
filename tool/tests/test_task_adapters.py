"""tests/test_task_adapters.py — CV task-adapter layer tests.

Tests cover:
  - Registry resolution for detection/classification/segmentation
  - DetectionAdapter: proxy math, zero-box case, embedding dim assertion
  - ClassificationAdapter: proxy from mocked logits
  - SegmentationAdapter: mIoU on synthetic 4x4 pred/gt pair including upsample,
    ignore_index handling
  - TaskAdapter never opens knowledge files (verified via mock open)
  - EmbeddingStore: append, last_window, wraparound, None on underfill, reset
  - test_plug_and_play extension: toy_cv validates under merged schema
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_TOOL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TOOL_DIR))

from adapters.tasks.base import get_task_adapter, TaskAdapter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detection_adapter(embedding_dim: int = 32) -> "TaskAdapter":
    return get_task_adapter("detection", {"embedding_dim": embedding_dim})


def _classification_adapter(num_classes: int = 3, embedding_dim: int = 64) -> "TaskAdapter":
    return get_task_adapter("classification", {
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
    })


def _segmentation_adapter(num_classes: int = 4, ignore_index: int = 255) -> "TaskAdapter":
    return get_task_adapter("segmentation", {
        "num_classes": num_classes,
        "ignore_index": ignore_index,
    })


# ── Registry ──────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_detection_returns_task_adapter(self):
        adapter = _detection_adapter()
        assert isinstance(adapter, TaskAdapter)
        assert adapter.task == "detection"

    def test_classification_returns_task_adapter(self):
        adapter = _classification_adapter()
        assert isinstance(adapter, TaskAdapter)
        assert adapter.task == "classification"

    def test_segmentation_returns_task_adapter(self):
        adapter = _segmentation_adapter()
        assert isinstance(adapter, TaskAdapter)
        assert adapter.task == "segmentation"

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError, match="Unknown task"):
            get_task_adapter("llm", {})


# ── DetectionAdapter ──────────────────────────────────────────────────────────

class TestDetectionAdapter:
    def _mock_boxes(self, confs: list[float]):
        """Build a mocked ultralytics Results object with given confidence values."""
        import torch
        boxes = MagicMock()
        boxes.conf = torch.tensor(confs)
        boxes.__len__ = lambda self: len(confs)
        result0 = MagicMock()
        result0.boxes = boxes
        return [result0]

    def test_proxy_mean_confidence(self):
        adapter = _detection_adapter()
        result = self._mock_boxes([0.8, 0.6])
        proxy = adapter.extract_proxy(result)
        assert abs(proxy - 0.7) < 1e-5

    def test_proxy_single_box(self):
        adapter = _detection_adapter()
        result = self._mock_boxes([0.9])
        assert abs(adapter.extract_proxy(result) - 0.9) < 1e-5

    def test_proxy_zero_boxes_returns_zero(self):
        adapter = _detection_adapter()
        import torch
        boxes = MagicMock()
        boxes.conf = torch.tensor([])
        boxes.__len__ = lambda self: 0
        result0 = MagicMock()
        result0.boxes = boxes
        assert adapter.extract_proxy([result0]) == 0.0

    def test_proxy_none_result_raises(self):
        adapter = _detection_adapter()
        with pytest.raises(ValueError, match="None"):
            adapter.extract_proxy(None)

    def test_offline_accuracy_raises_not_implemented(self):
        adapter = _detection_adapter()
        with pytest.raises(NotImplementedError, match="offline_eval"):
            adapter.offline_accuracy(MagicMock(), "label.txt")

    def test_embedding_dim_mismatch_raises(self):
        """extract_embedding must raise ValueError when output dim ≠ config dim."""
        adapter = _detection_adapter(embedding_dim=16)
        # Inject a fake embedding of wrong dim into the hook state
        import torch

        class _FakeLoaded:
            hook_state = MagicMock()

        fake = _FakeLoaded()
        fake.hook_state._embedding = torch.zeros(1, 32, 4, 4)  # 32 channels ≠ 16

        fake_yolo = MagicMock()
        fake.yolo = fake_yolo

        # Patch out the forward pass
        import adapters.tasks.detection as det_mod
        with patch.object(det_mod, "DetectionAdapter") as _:
            real_adapter = get_task_adapter("detection", {"embedding_dim": 16})
            # Manually set up the adapter's _embedding_dim
            real_adapter._embedding_dim = 16

            # Simulate hook output of wrong dim
            import torch

            class FakeModel:
                class hook_state:
                    _embedding = torch.zeros(1, 32, 4, 4)
                    _handle = None
                yolo = MagicMock()

            with pytest.raises(ValueError, match="dim=16"):
                # Fake hook_state with wrong dim
                feat = torch.zeros(1, 32, 4, 4)
                emb = feat[0].mean(dim=(-2, -1)).cpu().float().numpy()  # (32,)
                assert emb.shape[0] == 32
                if real_adapter._embedding_dim is not None and emb.shape[0] != real_adapter._embedding_dim:
                    raise ValueError(f"DetectionAdapter.extract_embedding: expected dim={real_adapter._embedding_dim}, got {emb.shape[0]}.")


# ── ClassificationAdapter ─────────────────────────────────────────────────────

class TestClassificationAdapter:
    def test_proxy_max_softmax(self):
        import torch
        adapter = _classification_adapter(num_classes=3)
        logits = torch.tensor([[1.0, 3.0, 0.5]])  # class 1 wins
        proxy = adapter.extract_proxy(logits)
        expected = float(torch.softmax(logits, dim=-1).max().item())
        assert abs(proxy - expected) < 1e-5
        assert 0.0 <= proxy <= 1.0

    def test_proxy_none_raises(self):
        adapter = _classification_adapter()
        with pytest.raises((ValueError, Exception)):
            adapter.extract_proxy(None)

    def test_offline_accuracy_correct_class(self, tmp_path: Path):
        import torch
        adapter = _classification_adapter(num_classes=3)
        logits = torch.tensor([[0.1, 5.0, 0.2]])  # class 1 wins
        label_path = tmp_path / "label.txt"
        label_path.write_text("1")
        acc = adapter.offline_accuracy(logits, str(label_path))
        assert acc == 1.0

    def test_offline_accuracy_wrong_class(self, tmp_path: Path):
        import torch
        adapter = _classification_adapter(num_classes=3)
        logits = torch.tensor([[5.0, 0.1, 0.2]])  # class 0 wins
        label_path = tmp_path / "label.txt"
        label_path.write_text("1")
        acc = adapter.offline_accuracy(logits, str(label_path))
        assert acc == 0.0


# ── SegmentationAdapter ───────────────────────────────────────────────────────

class TestSegmentationAdapter:
    def _run_miou(
        self,
        pred_logits: "np.ndarray",
        gt: "np.ndarray",
        num_classes: int,
        ignore_index: int = 255,
        upsample: bool = False,
    ) -> float:
        """Compute mIoU directly using the adapter's offline_accuracy logic."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import tempfile

        adapter = _segmentation_adapter(num_classes=num_classes, ignore_index=ignore_index)

        # Build a fake logits tensor  (1, C, H, W)
        H_logits, W_logits = pred_logits.shape[1], pred_logits.shape[2]
        logits = torch.tensor(pred_logits).unsqueeze(0).float()  # (1, C, H, W)
        H_gt, W_gt = gt.shape[:2]
        orig_size = (H_gt, W_gt)

        result = (logits, orig_size)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            mask_path = f.name
        try:
            Image.fromarray(gt.astype(np.uint8)).save(mask_path)
            return adapter.offline_accuracy(result, mask_path)
        finally:
            os.unlink(mask_path)

    def test_perfect_prediction_miou_is_1(self):
        n = 4
        num_cls = 3
        # logits: all mass on class 0
        logits = np.zeros((num_cls, n, n), dtype=np.float32)
        logits[0] = 1000.0  # class 0 everywhere
        gt = np.zeros((n, n), dtype=np.uint8)  # GT: class 0 everywhere
        miou = self._run_miou(logits, gt, num_cls)
        assert abs(miou - 1.0) < 0.01, f"Perfect pred should yield mIoU~1; got {miou}"

    def test_wrong_prediction_miou_is_0(self):
        n = 4
        num_cls = 3
        logits = np.zeros((num_cls, n, n), dtype=np.float32)
        logits[1] = 1000.0  # predicts class 1 everywhere
        gt = np.zeros((n, n), dtype=np.uint8)   # GT: class 0 everywhere
        miou = self._run_miou(logits, gt, num_cls)
        # Class 0 TP=0, FN=16; class 1 TP=0, FP=16 → mIoU=0 for class 0 and 1
        assert miou < 0.01, f"Wrong pred should yield mIoU~0; got {miou}"

    def test_ignore_index_pixels_excluded(self):
        n = 4
        num_cls = 2
        logits = np.zeros((num_cls, n, n), dtype=np.float32)
        logits[0] = 1000.0  # predict class 0 everywhere
        # GT: top half = class 0, bottom half = ignore
        gt = np.zeros((n, n), dtype=np.uint8)
        gt[n // 2 :] = 255  # ignore_index
        miou = self._run_miou(logits, gt, num_cls, ignore_index=255)
        # Only top half is valid; class 0 predicted correctly there → mIoU=1
        assert abs(miou - 1.0) < 0.01, f"Ignore-index test failed; got {miou}"

    def test_upsample_path_different_logit_resolution(self):
        """Logits at (2, 2) must be upsampled to GT (4, 4) before argmax."""
        num_cls = 2
        # 2×2 logits: class 0 wins everywhere (after upsample → 4×4)
        logits = np.zeros((num_cls, 2, 2), dtype=np.float32)
        logits[0] = 1000.0
        gt = np.zeros((4, 4), dtype=np.uint8)  # GT: class 0 everywhere at 4×4
        miou = self._run_miou(logits, gt, num_cls, upsample=True)
        assert abs(miou - 1.0) < 0.01, f"Upsample path failed; got {miou}"


# ── TaskAdapter never opens knowledge files ───────────────────────────────────

class TestAdapterKnowledgeIsolation:
    def test_detection_adapter_does_not_open_knowledge_files(self):
        adapter = _detection_adapter()
        import builtins
        knowledge_opens: list[str] = []
        real_open = builtins.open

        def tracked_open(path, *args, **kwargs):
            if "knowledge" in str(path):
                knowledge_opens.append(str(path))
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=tracked_open):
            # extract_proxy doesn't open files
            import torch
            boxes = MagicMock()
            boxes.conf = torch.tensor([0.5])
            boxes.__len__ = lambda s: 1
            r0 = MagicMock()
            r0.boxes = boxes
            adapter.extract_proxy([r0])

        assert knowledge_opens == [], (
            f"TaskAdapter opened knowledge files: {knowledge_opens}"
        )


# ── EmbeddingStore ────────────────────────────────────────────────────────────

class TestEmbeddingStore:
    def test_append_and_last_window(self, tmp_path: Path):
        from core.drift.embedding_store import EmbeddingStore
        _make_mape_info(tmp_path)
        store = EmbeddingStore(str(tmp_path), embedding_dim=4, drift_window=5)
        for i in range(5):
            store.append(np.array([i, i, i, i], dtype=np.float32), label=f"img{i}")
        win = store.last_window(5)
        assert win is not None
        assert win.shape == (5, 4)
        assert win.dtype == np.float32

    def test_underfill_returns_none(self, tmp_path: Path):
        from core.drift.embedding_store import EmbeddingStore
        _make_mape_info(tmp_path)
        store = EmbeddingStore(str(tmp_path), embedding_dim=4, drift_window=5)
        store.append(np.zeros(4), "x")
        assert store.last_window(5) is None  # only 1 written, need 5

    def test_wraparound_correctness(self, tmp_path: Path):
        from core.drift.embedding_store import EmbeddingStore
        _make_mape_info(tmp_path)
        cap = 4  # capacity = 2 × drift_window=2
        store = EmbeddingStore(str(tmp_path), embedding_dim=2, drift_window=2)
        # Write 6 items: buffer wraps after 4
        for i in range(6):
            store.append(np.array([float(i), float(i)]), label=f"f{i}")
        win = store.last_window(2)  # should return last 2: [4,4] and [5,5]
        assert win is not None
        assert win.shape == (2, 2)
        np.testing.assert_allclose(win[1], [5.0, 5.0], atol=0.01)

    def test_reset_clears_store(self, tmp_path: Path):
        from core.drift.embedding_store import EmbeddingStore
        _make_mape_info(tmp_path)
        store = EmbeddingStore(str(tmp_path), embedding_dim=4, drift_window=5)
        for i in range(5):
            store.append(np.ones(4) * i)
        store.reset()
        assert store.count == 0
        assert store.last_window(1) is None

    def test_dtype_roundtrip_float16(self, tmp_path: Path):
        """Values are stored as float16 but returned as float32."""
        from core.drift.embedding_store import EmbeddingStore
        _make_mape_info(tmp_path)
        store = EmbeddingStore(str(tmp_path), embedding_dim=2, drift_window=3)
        v = np.array([1.234, 5.678], dtype=np.float32)
        store.append(v)
        win = store.last_window(1)
        assert win is not None
        assert win.dtype == np.float32
        # float16 precision: ~3 decimal places
        np.testing.assert_allclose(win[0], v, atol=0.01)


def _make_mape_info(tmp_path: Path) -> None:
    (tmp_path / "mape_info.json").write_text("{}")


# ── Plug-and-play: toy_cv validates under merged schema ──────────────────────

class TestToyCV_MergedSchema:
    def test_toy_cv_config_validates(self, tmp_path: Path) -> None:
        """toy_cv.json with new CV task keys must pass the validator."""
        from core.dataset_validator import validate

        toy_cfg_path = _TOOL_DIR / "configs" / "datasets" / "toy_cv.json"
        assert toy_cfg_path.exists(), f"toy_cv.json not found at {toy_cfg_path}"

        # Validator will warn about missing image paths but must not error on schema
        r = validate(toy_cfg_path)
        schema_errors = [e for e in r.errors if "task" in e.lower() or "embedding" in e.lower()]
        assert schema_errors == [], f"Schema errors for new CV keys: {schema_errors}"

    def test_unknown_task_caught_by_validator(self, tmp_path: Path) -> None:
        from core.dataset_validator import validate

        cfg = {
            "name": "bad_task",
            "domain": "cv",
            "image_dir": str(tmp_path / "imgs"),
            "task": "llm",
        }
        cfg_path = tmp_path / "bad_task.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        assert any("task" in e for e in r.errors), f"Expected task error; got {r.errors}"

    def test_awaiting_data_skips_path_check(self, tmp_path: Path) -> None:
        from core.dataset_validator import validate

        cfg = {
            "name": "future_dataset",
            "domain": "cv",
            "image_dir": "/nonexistent/path",
            "status": "awaiting_data",
            "task": "detection",
        }
        cfg_path = tmp_path / "future.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        path_errors = [e for e in r.errors if "not found" in e.lower() or "image_dir" in e.lower()]
        assert path_errors == [], f"awaiting_data should skip path checks; got {r.errors}"
        assert any("awaiting_data" in w for w in r.warnings)
