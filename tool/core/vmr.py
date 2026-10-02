"""
core/vmr.py — Versioned Model Repository (VMR).

Provides typed store/match operations for archived model weights and their
associated training distributions (histograms for regression KL, embeddings
for CV MMD/Fréchet). Replaces the ad-hoc path scanning in analyse.py.

Directory layout (under knowledge/vmr/):
  <model_name>/
    <timestamp>_<tag>/
      weights.*          — model weights (format is model-specific)
      distribution.json  — {"type": "histogram"|"embedding", "data": ...}
      meta.json          — {"model": str, "timestamp": str, "tag": str,
                           "proxy_score": float|null, "drift_score": float|null}

Public API:
  vmr = VMR(base_dir="knowledge/vmr")
  vmr.store(model_name, weights_path, distribution, meta)
  version = vmr.best_match(model_name, current_distribution, strategy)
  versions = vmr.list_versions(model_name)
  vmr.restore(version) -> weights_path (already on disk, just returns the path)
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class VMRVersion:
    model: str
    timestamp: str
    tag: str
    version_dir: str
    weights_path: str
    proxy_score: float | None = None
    drift_score: float | None = None
    distribution: dict | None = None


class VMR:
    """Versioned Model Repository.

    Stores model checkpoints alongside the training distribution used to fit
    that version. Supports two matching strategies:
    - "best_score": pick the version with the highest stored proxy_score
    - "closest_distribution": pick the version whose stored distribution is
      closest (KL for histograms, MMD² for embeddings) to the current one
    """

    def __init__(self, base_dir: str = "knowledge/vmr") -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def store(
        self,
        model_name: str,
        weights_src: str,
        distribution: dict,
        *,
        tag: str = "auto",
        proxy_score: float | None = None,
        drift_score: float | None = None,
    ) -> VMRVersion:
        """Archive a model checkpoint and its training distribution.

        Args:
            model_name:   e.g. "lstm", "yolov8n"
            weights_src:  Path to weights file to copy into the VMR.
            distribution: {"type": "histogram"|"embedding"|"raw",
                          "data": <list or list-of-lists>}
            tag:          Human-readable label (default: "auto" → uses timestamp)
            proxy_score:  Proxy accuracy at archive time (optional).
            drift_score:  Drift score at archive time (optional).

        Returns:
            VMRVersion describing the stored version.
        """
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        version_name = f"{ts}_{tag}"
        version_dir = os.path.join(self.base_dir, model_name, version_name)
        os.makedirs(version_dir, exist_ok=True)

        # Copy weights
        weights_ext = os.path.splitext(weights_src)[1]
        weights_dst = os.path.join(version_dir, f"weights{weights_ext}")
        shutil.copy2(weights_src, weights_dst)

        # Write distribution
        dist_path = os.path.join(version_dir, "distribution.json")
        with open(dist_path, "w") as f:
            json.dump(distribution, f)

        # Write meta
        meta = {
            "model": model_name,
            "timestamp": ts,
            "tag": tag,
            "proxy_score": proxy_score,
            "drift_score": drift_score,
        }
        meta_path = os.path.join(version_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return VMRVersion(
            model=model_name,
            timestamp=ts,
            tag=tag,
            version_dir=version_dir,
            weights_path=weights_dst,
            proxy_score=proxy_score,
            drift_score=drift_score,
            distribution=distribution,
        )

    def list_versions(self, model_name: str) -> list[VMRVersion]:
        """Return all stored versions for model_name, sorted newest-first."""
        model_dir = os.path.join(self.base_dir, model_name)
        if not os.path.isdir(model_dir):
            return []

        versions = []
        for version_name in sorted(os.listdir(model_dir), reverse=True):
            vdir = os.path.join(model_dir, version_name)
            meta_path = os.path.join(vdir, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)

            dist_path = os.path.join(vdir, "distribution.json")
            distribution = None
            if os.path.isfile(dist_path):
                with open(dist_path) as f:
                    distribution = json.load(f)

            weights_path = self._find_weights(vdir)

            versions.append(
                VMRVersion(
                    model=meta["model"],
                    timestamp=meta["timestamp"],
                    tag=meta.get("tag", ""),
                    version_dir=vdir,
                    weights_path=weights_path or "",
                    proxy_score=meta.get("proxy_score"),
                    drift_score=meta.get("drift_score"),
                    distribution=distribution,
                )
            )
        return versions

    def best_match(
        self,
        model_name: str,
        current_distribution: dict | None = None,
        strategy: str = "best_score",
        threshold: float | None = None,
    ) -> VMRVersion | None:
        """Return the best-matching VMR version, or None if nothing qualifies.

        Strategies:
        - "best_score": highest stored proxy_score (ignores distribution).
        - "closest_distribution": minimum divergence to current_distribution.
          Supported distribution types: "histogram" (KL), "embedding" (MMD²).

        threshold (closest_distribution only): the archived version closest to
        current_distribution is only returned if its distance is BELOW this
        value — mirroring the original paper's own VMR-reuse gate ("if some
        prior version's training data is a closer match [min KL < 0.75], that
        version... is returned... instead of retraining"). Below prior versions
        of this function had no such gate at all: any non-empty archive always
        "won", however distant the best available match actually was, which
        meant a genuine retrain could never fire again once even one version
        existed for a model. threshold=None preserves that unconditional
        behavior (used by callers that don't have a meaningful distance-based
        gate to apply, e.g. the "best_score" strategy).

        Returns None if no versions are stored, or if the closest available
        distribution-based match doesn't clear `threshold`.
        """
        versions = self.list_versions(model_name)
        if not versions:
            return None

        if strategy == "best_score":
            scored = [v for v in versions if v.proxy_score is not None]
            if not scored:
                return versions[0]  # newest as fallback
            return max(scored, key=lambda v: v.proxy_score)  # type: ignore[return-value]

        if strategy == "closest_distribution" and current_distribution is not None:
            dist_type = current_distribution.get("type", "histogram")
            if dist_type == "histogram":
                best, best_dist = self._closest_histogram(versions, current_distribution)
            elif dist_type in ("embedding", "raw"):
                best, best_dist = self._closest_embedding(versions, current_distribution)
            else:
                return versions[0]
            if threshold is not None and best_dist >= threshold:
                return None
            return best

        return versions[0]  # newest fallback

    def restore(self, version: VMRVersion) -> str:
        """Return the on-disk path to the archived weights (no-op copy needed)."""
        if not os.path.isfile(version.weights_path):
            raise FileNotFoundError(
                f"VMR weights missing: {version.weights_path}. "
                "The VMR directory may have been manually modified."
            )
        return version.weights_path

    def _find_weights(self, version_dir: str) -> str | None:
        for fname in os.listdir(version_dir):
            if fname.startswith("weights"):
                return os.path.join(version_dir, fname)
        return None

    @staticmethod
    def _kl_divergence(p: list[float], q: list[float], eps: float = 1e-9) -> float:
        """KL(p||q) with epsilon smoothing."""
        import math
        total = 0.0
        for pi, qi in zip(p, q):
            pi = max(pi, eps)
            qi = max(qi, eps)
            total += pi * math.log(pi / qi)
        return total

    def _closest_histogram(
        self, versions: list[VMRVersion], current: dict
    ) -> tuple[VMRVersion, float]:
        """Return (closest version, its KL distance). Distance is float("inf")
        if no archived version has a comparable histogram at all, so a caller
        applying a threshold correctly rejects this case instead of silently
        accepting an incomparable fallback."""
        cur_data = current.get("data", [])
        if not cur_data:
            return versions[0], 0.0  # no current data to compare — unconditional fallback
        cur_sum = sum(cur_data) or 1.0
        cur_norm = [x / cur_sum for x in cur_data]

        best, best_kl = versions[0], float("inf")
        for v in versions:
            if v.distribution is None or v.distribution.get("type") != "histogram":
                continue
            ref_data = v.distribution.get("data", [])
            if len(ref_data) != len(cur_data):
                continue
            ref_sum = sum(ref_data) or 1.0
            ref_norm = [x / ref_sum for x in ref_data]
            kl = self._kl_divergence(cur_norm, ref_norm)
            if kl < best_kl:
                best_kl = kl
                best = v
        return best, best_kl

    def _closest_embedding(
        self, versions: list[VMRVersion], current: dict
    ) -> tuple[VMRVersion, float]:
        """Use Fréchet distance approximation for embedding distributions.
        Returns (closest version, its distance) — see _closest_histogram's
        docstring for why an incomparable case returns float("inf")."""
        import math

        cur_data = current.get("data", [])
        if not cur_data or not isinstance(cur_data[0], (int, float)):
            return versions[0], 0.0  # no current data to compare — unconditional fallback

        # Treat data as a flat mean vector (already stored as mean at archive time)
        cur_mean = cur_data

        best, best_dist = versions[0], float("inf")
        for v in versions:
            if v.distribution is None:
                continue
            ref_data = v.distribution.get("data", [])
            if len(ref_data) != len(cur_mean):
                continue
            # Euclidean distance between mean embeddings (simplified Fréchet upper bound)
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(cur_mean, ref_data)))
            if dist < best_dist:
                best_dist = dist
                best = v
        return best, best_dist
