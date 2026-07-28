"""core/drift/embedding_store.py — Sidecar ring-buffer store for CV embeddings.

Implements R5-e: embeddings are stored as a float16 memory-mapped numpy array
(not appended to predictions.csv) to keep the monitor hot path lean.

Files on disk:
    knowledge/embeddings.f16.npy       — float16 ring buffer (capacity 2×drift_window)
    knowledge/embeddings_index.csv     — row index → image name mapping
    mape_info.json["embedding_cursor"] — write cursor (persistent across restarts)

The store wraps around when the buffer is full (oldest entry overwritten).
All reads return float32 (upcast from float16).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingStore:
    """Append-only ring buffer for inference embeddings.

    Args:
        knowledge_dir: Path to managed_system_*/knowledge/ directory.
        embedding_dim: Fixed dimension of each embedding vector.
        drift_window:  Number of most-recent embeddings needed for drift scoring.
                       Buffer capacity = 2 × drift_window.
    """

    _NPY_NAME = "embeddings.f16.npy"
    _IDX_NAME = "embeddings_index.csv"

    def __init__(
        self,
        knowledge_dir: str,
        embedding_dim: int,
        drift_window: int = 500,
    ) -> None:
        self._dir = Path(knowledge_dir)
        self._dim = embedding_dim
        self._capacity = 2 * drift_window
        self._cursor: int = 0
        self._count: int = 0  # total entries ever written (capped at capacity for reads)

        npy_path = self._dir / self._NPY_NAME
        if npy_path.exists():
            arr = np.load(str(npy_path))
            if arr.shape == (self._capacity, self._dim):
                self._buf = arr
            else:
                self._buf = self._new_buffer(npy_path)
        else:
            self._buf = self._new_buffer(npy_path)

        # Load persisted cursor
        mape_info_path = self._dir / "mape_info.json"
        if mape_info_path.exists():
            try:
                with open(mape_info_path) as f:
                    info = json.load(f)
                self._cursor = int(info.get("embedding_cursor", 0)) % self._capacity
                self._count = int(info.get("embedding_count", 0))
            except Exception:
                pass

    def _new_buffer(self, path: Path) -> np.ndarray:
        buf = np.zeros((self._capacity, self._dim), dtype=np.float16)
        np.save(str(path), buf)
        return buf

    # ── Public API ─────────────────────────────────────────────────────────────

    def append(self, vec: np.ndarray, label: str = "") -> None:
        """Append one embedding vector (any float dtype) and optional label."""
        v = np.asarray(vec, dtype=np.float16).reshape(-1)
        if v.shape[0] != self._dim:
            raise ValueError(
                f"EmbeddingStore.append: expected dim={self._dim}, got {v.shape[0]}"
            )
        self._buf[self._cursor] = v
        np.save(str(self._dir / self._NPY_NAME), self._buf)

        # Update index CSV
        idx_path = self._dir / self._IDX_NAME
        with open(idx_path, "a", newline="") as f:
            csv.writer(f).writerow([self._cursor, self._count, label])

        self._cursor = (self._cursor + 1) % self._capacity
        self._count += 1
        self._persist_cursor()

    def last_window(self, n: int) -> np.ndarray | None:
        """Return the n most recent embeddings as float32 array (n, dim).

        Returns None if fewer than n embeddings have been written.
        """
        if self._count < n:
            return None
        total_written = min(self._count, self._capacity)
        if total_written < n:
            return None

        # Gather the last n in chronological order
        indices = [
            (self._cursor - n + i) % self._capacity for i in range(n)
        ]
        rows = self._buf[indices].astype(np.float32)
        return rows

    def reset(self) -> None:
        """Zero the buffer and clear the index (called by run_reset.py)."""
        self._buf[:] = 0.0
        np.save(str(self._dir / self._NPY_NAME), self._buf)
        idx_path = self._dir / self._IDX_NAME
        if idx_path.exists():
            idx_path.unlink()
        self._cursor = 0
        self._count = 0
        self._persist_cursor()

    def _persist_cursor(self) -> None:
        """Write cursor + count back into mape_info.json (atomic)."""
        mape_info_path = self._dir / "mape_info.json"
        info: dict[str, Any] = {}
        if mape_info_path.exists():
            try:
                with open(mape_info_path) as f:
                    info = json.load(f)
            except Exception:
                pass
        info["embedding_cursor"] = self._cursor
        info["embedding_count"] = self._count
        tmp = mape_info_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(info, f, indent=2)
        os.replace(tmp, mape_info_path)

    @property
    def count(self) -> int:
        """Total embeddings ever appended (may exceed capacity)."""
        return self._count

    @property
    def dim(self) -> int:
        return self._dim
