"""
managed_system_cv/mape_logic/monitor.py — Monitor phase for CV managed system.

Phase 3 update:
- Accuracy signal computed via configurable AccuracyProxy (thresholds "proxy": "confidence" |
  "calibrated_confidence" | "agreement"; default: "confidence").
- Drift computed via configurable DriftDetector(s) (thresholds "drift_detector":
  "luminance_kl" | "mmd_embedding" | "frechet_embedding"; default: "luminance_kl").
- Separated EMA signals (ema_accuracy, ema_energy) updated alongside legacy ema_scores.
- Warmup returns {"kl_div": None} instead of triggering drift on partial data.
- All legacy return keys preserved for backward compatibility.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# ── resolve paths ──────────────────────────────────────────────────────────────
_MANAGED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TOOL_DIR = os.path.abspath(os.path.join(_MANAGED_DIR, ".."))
sys.path.insert(0, _TOOL_DIR)
sys.path.insert(0, _MANAGED_DIR)

from core.scoring import normalize_energy, update_ema, update_separated_emas
from utility.drift_utils import kl_divergence  # kept for legacy fallback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")


# ── helpers ────────────────────────────────────────────────────────────────────

def load_mape_info():
    with open(mape_info_file) as f:
        return json.load(f)


def save_mape_info(data):
    tmp = mape_info_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, mape_info_file)


def get_current_model():
    try:
        with open(model_file) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def _load_proxy(thresholds: dict):
    """Instantiate the configured AccuracyProxy."""
    proxy_name = thresholds.get("proxy", "confidence")
    calib_path = os.path.join(KNOWLEDGE_DIR, "calibration.json")
    if proxy_name == "calibrated_confidence":
        from core.proxies.calibrated_confidence import CalibratedConfidenceProxy
        return CalibratedConfidenceProxy(calibration_path=calib_path)
    elif proxy_name == "agreement":
        from core.proxies.agreement import AgreementProxy
        return AgreementProxy(
            iou_threshold=thresholds.get("agreement_iou_threshold", 0.5),
            subsample_frac=thresholds.get("agreement_subsample_frac", 0.1),
        )
    else:
        from core.proxies.confidence import ConfidenceProxy
        return ConfidenceProxy()


def _load_embedding_detector_with_reference(thresholds: dict, detector_name: str):
    """Instantiate an embedding detector and fit it from reference_embeddings.npz (§5.2/§5.3).

    Returns (detector, fitted: bool). fitted=False means the detector has no
    reference — score() will return None (I5).
    """
    tau_drift = thresholds.get("tau_drift", 0.07)
    window_size = int(thresholds.get("drift_window_size", 500))

    if detector_name == "mmd_embedding":
        from core.drift.mmd_embedding import MMDEmbeddingDetector
        det = MMDEmbeddingDetector(tau_drift=tau_drift, window_size=window_size)
    else:
        from core.drift.frechet_embedding import FrechetEmbeddingDetector
        det = FrechetEmbeddingDetector(tau_drift=tau_drift, window_size=window_size)

    # Persisted detector state (after first successful fit) takes priority
    det_state_path = os.path.join(KNOWLEDGE_DIR, f"detector_{detector_name}_fixed.npz")
    if os.path.exists(det_state_path):
        try:
            det.load(det_state_path)
            return det, True
        except Exception as exc:
            print(f"[MONITOR-CV] Could not load detector state {det_state_path}: {exc}; re-fitting.")

    # Bootstrap from reference_embeddings.npz produced by scripts/init_cv.py (§5.2)
    ref_npz_path = os.path.join(KNOWLEDGE_DIR, "reference_embeddings.npz")
    if not os.path.exists(ref_npz_path):
        print(
            f"[MONITOR-CV] {detector_name}: reference_embeddings.npz not found at "
            f"{ref_npz_path}. Run scripts/init_cv.py to generate it. "
            "Returning None score this cycle (I5)."
        )
        return det, False

    try:
        data = np.load(ref_npz_path, allow_pickle=True)
        ref_emb = data["embeddings"].astype(np.float32)  # (N, D)
        det.fit_reference(ref_emb)
        det.save(det_state_path)
        print(f"[MONITOR-CV] {detector_name}: fitted from reference_embeddings.npz ({len(ref_emb)} vectors, dim={ref_emb.shape[1]})")
        return det, True
    except Exception as exc:
        print(f"[MONITOR-CV] Could not fit {detector_name} from npz: {exc}. Score will be None.")
        return det, False


# ── main monitor functions ─────────────────────────────────────────────────────

def monitor_mape():
    info = load_mape_info()
    last_line = info["last_line"]
    current_model = get_current_model()
    if current_model is None:
        print("[MAPE] No current model found.")
        return None

    try:
        df = pd.read_csv(predictions_file, skiprows=range(1, last_line + 1))
        if df.empty:
            print("[MAPE] No new predictions to monitor.")
            event_counters = info.get("event_counters", {
                "model_switches": 0, "retrains": 0, "vmr_events": 0, "mape_k_energy_uJ": 0.0
            })
            simple_switch_counters = info.get("simple_switch_counters", {"simple_switches": 0})
            final_score = info["ema_scores"].get(current_model, 0.5)
            return {
                "confidence": 0.5, "energy": 0.0, "normalized_energy": 0.5,
                "score": final_score, "model_used": current_model,
                "model_switches": event_counters["model_switches"],
                "retrains": event_counters["retrains"],
                "vmr_events": event_counters["vmr_events"],
                "mape_k_energy_uJ": round(event_counters["mape_k_energy_uJ"], 2),
                "simple_switches": simple_switch_counters["simple_switches"],
            }
    except FileNotFoundError:
        print("[MAPE] Predictions file not found.")
        return None

    thresholds = json.load(open(thresholds_file))
    energy_min = thresholds.get("E_m", 0)
    energy_max = thresholds.get("E_M", 10_000_000)
    beta = thresholds.get("beta", 0.95)
    gamma = thresholds.get("gamma", 0.8)

    # ── proxy-based accuracy signal ────────────────────────────────────────────
    proxy = _load_proxy(thresholds)
    confs = df["confidence"].dropna().tolist() if "confidence" in df.columns else []
    inference_record = {"model": current_model, "confidences": confs}
    proxy_accuracy = proxy.score(inference_record)

    avg_energy = float(df["energy_uJ"].mean()) if "energy_uJ" in df.columns else 0.0
    energy_norm = normalize_energy(avg_energy, energy_min, energy_max)

    # Legacy score (backward compatible)
    score = beta * proxy_accuracy + (1.0 - beta) * (1.0 - energy_norm)
    prev_score = info["ema_scores"].get(current_model, 0.5)
    final_score = update_ema(prev_score, score, gamma)

    # Separated EMA signals (Phase 2+)
    update_separated_emas(info, current_model, proxy_accuracy, energy_norm, gamma)

    print(f"[MAPE] model={current_model}, proxy_acc={proxy_accuracy:.4f}, "
          f"energy_norm={energy_norm:.4f}, score={score:.4f}, ema={final_score:.4f}")

    info["ema_scores"][current_model] = final_score
    info["last_line"] += len(df)
    if "event_counters" not in info:
        info["event_counters"] = {
            "model_switches": 0, "retrains": 0, "vmr_events": 0, "mape_k_energy_uJ": 0.0
        }
    save_mape_info(info)

    event_counters = info["event_counters"]
    simple_switch_counters = info.get("simple_switch_counters", {"simple_switches": 0})
    print(f"📊 Switches: {event_counters['model_switches']}, Retrains: {event_counters['retrains']}, "
          f"VMR: {event_counters['vmr_events']}, MAPE-K Energy: {event_counters['mape_k_energy_uJ']:.2f} µJ")

    return {
        "confidence": proxy_accuracy,
        "energy": round(avg_energy, 2),
        "normalized_energy": energy_norm,
        "score": final_score,
        "model_used": current_model,
        "model_switches": event_counters["model_switches"],
        "retrains": event_counters["retrains"],
        "vmr_events": event_counters["vmr_events"],
        "mape_k_energy_uJ": round(event_counters["mape_k_energy_uJ"], 2),
        "simple_switches": simple_switch_counters["simple_switches"],
    }


def monitor_drift():
    """Compute drift signal using configured detector.

    Returns dict with "kl_div" key (may be None during warmup) for backward
    compatibility with analyse.py, plus "drift_detector" and "drift_score".
    """
    thresholds = json.load(open(thresholds_file))
    drift_reference = thresholds.get("drift_reference", "both")

    try:
        df = pd.read_csv(predictions_file)
    except FileNotFoundError:
        print("[DRIFT] Predictions file not found for drift monitoring.")
        return None

    detector_name = thresholds.get("drift_detector", "luminance_kl")

    # Luminance-KL: histogram path (fully backward compatible)
    if detector_name == "luminance_kl":
        return _monitor_drift_luminance(df, thresholds, drift_reference)

    # Embedding-based: read from EmbeddingStore ring buffer (§5.3)
    return _monitor_drift_embedding(thresholds, drift_reference, detector_name)


def _monitor_drift_luminance(df, thresholds, drift_reference) -> dict | None:
    """Legacy-compatible luminance histogram drift monitoring."""
    from core.drift.kl_fixed_ref import KLFixedRefDetector
    from core.drift.kl_rolling import KLRollingDetector

    tau_drift = thresholds.get("tau_drift", 0.07)

    if "histogram" not in df.columns:
        print("[DRIFT] 'histogram' column not found in predictions.csv.")
        return None

    cur_hists_str = df["histogram"].iloc[-1000:]
    cur_hists = np.array([np.fromstring(h, sep=" ") for h in cur_hists_str if isinstance(h, str) and h.strip()])
    if cur_hists.size == 0:
        return {"kl_div": None, "drift_detector": "luminance_kl", "drift_score": None}

    cur_dist = np.mean(cur_hists, axis=0)

    fixed_kl = None
    rolling_kl = None

    # Fixed-reference KL (primary)
    if drift_reference in ("fixed", "both"):
        detector_path = os.path.join(KNOWLEDGE_DIR, "detector_luminance_kl_fixed.json")
        ref_dist_path = os.path.join(KNOWLEDGE_DIR, "reference_distribution.json")

        det = LuminanceKLDetectorInline(tau_drift=tau_drift)
        if os.path.exists(detector_path):
            det.load_state(detector_path)
        elif os.path.exists(ref_dist_path):
            with open(ref_dist_path) as f:
                ref_data = json.load(f)
            ref_histograms = ref_data.get("histograms", [])
            if ref_histograms:
                ref_arr = np.array(ref_histograms)
                det.fit(np.mean(ref_arr, axis=0))
                det.save_state(detector_path)

        fixed_kl = det.score(cur_dist)

    # Rolling KL (secondary telemetry)
    if drift_reference in ("rolling", "both"):
        if len(df) >= 2000:
            ref_hists_str = df["histogram"].iloc[-2000:-1000]
            ref_hists = np.array([np.fromstring(h, sep=" ") for h in ref_hists_str if isinstance(h, str) and h.strip()])
            if ref_hists.size > 0:
                ref_dist = np.mean(ref_hists, axis=0)
                rolling_kl = kl_divergence(cur_dist, ref_dist)
                print(f"[DRIFT] Rolling KL (secondary): {rolling_kl:.4f}")

    primary_kl = fixed_kl if drift_reference in ("fixed", "both") else rolling_kl
    print(f"[DRIFT] Fixed-ref KL (primary): {primary_kl}")

    return {
        "kl_div": primary_kl,
        "drift_detector": "luminance_kl",
        "drift_score": primary_kl,
        "rolling_kl": rolling_kl,
    }


def _monitor_drift_embedding(thresholds: dict, drift_reference: str, detector_name: str) -> dict | None:
    """Embedding-based drift monitoring (MMD² or Fréchet) via EmbeddingStore (§5.3).

    Reads the current window from the EmbeddingStore ring buffer (written by
    inference.py). Returns None score if underfilled (I5 — no fabricated telemetry).
    Secondary mmd_local (rolling) emitted only when config emit_local_drift=true.
    """
    drift_window = int(thresholds.get("drift_window_size", 500))
    emit_local = bool(thresholds.get("emit_local_drift", False))

    # Load EmbeddingStore and retrieve current window
    try:
        from core.drift.embedding_store import EmbeddingStore
        emb_dim = int(thresholds.get("embedding_dim", 256))
        store = EmbeddingStore(KNOWLEDGE_DIR, embedding_dim=emb_dim, drift_window=drift_window)
        cur_window = store.last_window(drift_window)  # (N, D) float32 or None
    except Exception as exc:
        print(f"[DRIFT] Could not load EmbeddingStore: {exc}")
        return {"kl_div": None, "drift_score": None, "drift_detector": detector_name}

    if cur_window is None:
        print(f"[DRIFT] {detector_name}: embedding store underfilled (< {drift_window} vectors). "
              "Returning None score during warmup (I5).")
        return {"kl_div": None, "drift_score": None, "drift_detector": detector_name,
                "warmup": True}

    # Load/fit fixed-reference detector
    det, fitted = _load_embedding_detector_with_reference(thresholds, detector_name)
    if not fitted:
        return {"kl_div": None, "drift_score": None, "drift_detector": detector_name,
                "warmup": True}

    primary_score = det.score(cur_window)
    print(f"[DRIFT] {detector_name} (fixed-ref) score: {primary_score}")

    result: dict = {
        "kl_div": primary_score,   # backward compat key consumed by analyse.py
        "drift_score": primary_score,
        "drift_detector": detector_name,
    }

    # Optional rolling/local signal (NOT the trigger — R3)
    if emit_local and drift_reference in ("rolling", "both"):
        all_vecs = store.last_window(drift_window * 2)  # full 2x window
        if all_vecs is not None and len(all_vecs) >= drift_window * 2:
            ref_half = all_vecs[:drift_window]
            cur_half = all_vecs[drift_window:]
            if detector_name == "mmd_embedding":
                from core.drift.mmd_embedding import unbiased_mmd2, _median_bandwidth
                bw = _median_bandwidth(ref_half)
                local_score = round(unbiased_mmd2(ref_half, cur_half, bw), 8)
            else:
                from core.drift.frechet_embedding import frechet_distance
                mu1 = np.mean(ref_half.astype(np.float64), axis=0)
                mu2 = np.mean(cur_half.astype(np.float64), axis=0)
                s1 = np.cov(ref_half.T) if len(ref_half) > 1 else np.zeros((ref_half.shape[1],) * 2)
                s2 = np.cov(cur_half.T) if len(cur_half) > 1 else np.zeros_like(s1)
                local_score = round(frechet_distance(mu1, s1, mu2, s2), 6)
            result["mmd_local"] = local_score
            print(f"[DRIFT] {detector_name} (rolling/local) score: {local_score}")

    return result


# ── thin wrapper for inline luminance KL (avoids importing full LuminanceKLDetector) ──

class LuminanceKLDetectorInline:
    """Thin stateful KL detector used inside _monitor_drift_luminance()."""

    def __init__(self, tau_drift: float = 0.07):
        self.tau_drift = tau_drift
        self._ref_dist = None

    def fit(self, ref_dist: np.ndarray) -> None:
        self._ref_dist = ref_dist

    def score(self, cur_dist: np.ndarray) -> float | None:
        if self._ref_dist is None:
            return None
        return float(kl_divergence(cur_dist, self._ref_dist))

    def save_state(self, path: str) -> None:
        import json
        tmp = path + ".tmp"
        data = {"ref_dist": self._ref_dist.tolist() if self._ref_dist is not None else None}
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load_state(self, path: str) -> None:
        import json
        with open(path) as f:
            data = json.load(f)
        rd = data.get("ref_dist")
        self._ref_dist = np.array(rd) if rd is not None else None
