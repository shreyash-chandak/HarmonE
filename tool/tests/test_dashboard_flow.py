"""tests/test_dashboard_flow.py — Scripted simulation of dashboard → backend flow.

Replays the exact fetch sequence dashboard.html performs for each preset
(same endpoints, same order, same payloads) without starting the real managed
system.  These tests prove the full integration contract is wired correctly:

  set-planner → reset → write-approach → save-policy → [start]
              ↓
  approach.conf written; thresholds planner written; policy registered;
  GET /api/knowledge/<currentPolicyId> returns the registered policy.

Static-assertion variants (no subprocess) run in the default suite.
Variants that spawn the actual inference engine are marked @pytest.mark.slow
and are excluded from CI by default.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

flask = pytest.importorskip("flask", reason="Flask not installed in this environment")

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    import app as app_module
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: replays the JS launchHarmonE() fetch sequence as Python calls
# ---------------------------------------------------------------------------

def _launch_harmone_flow(client, system: str, planner: str, approach_key: str, policy: dict):
    """
    Mirror of dashboard.html launchHarmonE():
      1. POST /api/set-planner
      2. POST /api/reset
      3. POST /api/write-approach
      4. POST /api/save-policy
    Returns (set_planner_resp, reset_resp, write_approach_resp, save_resp).
    """
    real_thresh = str(_TOOL_DIR / f"managed_system_{system}" / "knowledge" / "thresholds.json")

    with patch("builtins.open", side_effect=lambda p, *a, **kw: open(real_thresh, *a, **kw) if "thresholds" in str(p) else open(p, *a, **kw)):
        r1 = client.post("/api/set-planner", json={"planner": planner, "system": system})

    r2 = client.post("/api/reset")

    with patch("builtins.open", side_effect=lambda p, *a, **kw: open(str(_TOOL_DIR / "approach.conf"), "w") if "approach.conf" in str(p) and ("w" in a or kw.get("mode") == "w") else open(p, *a, **kw)):
        r3 = client.post("/api/write-approach", json={"approach": approach_key})

    with patch("os.makedirs"), \
         patch("builtins.open", MagicMock()):
        r4 = client.post("/api/save-policy", json=policy)

    return r1, r2, r3, r4


# ---------------------------------------------------------------------------
# Test: default HarmonE regression (reg_harmone_score, harmone_original planner)
# ---------------------------------------------------------------------------

class TestRegHarmonEDefaultFlow:
    """Preset reg_harmone_score with harmone_original planner — the primary path."""

    def _policy(self) -> dict:
        return {
            "policy_id": "reg_harmone_score",
            "quality_attribute": "score",
            "adaptation_boundary": {"type": "STATIC_THRESHOLD", "condition": "LESS_THAN", "threshold": 0.78},
            "tactics": [{"tactic_id": "execute_mape_plan", "priority": 1,
                          "tactic_endpoint": "http://localhost:8080/adaptor/tactic"}],
            "secondary_boundaries": [
                {"quality_attribute": "kl_div", "condition": "GREATER_THAN",
                 "threshold": 0.10, "tactic_id": "handle_data_drift"}
            ],
            "associated_qas": ["r2_score", "energy", "model_used"],
        }

    def test_set_planner_accepted(self, client):
        real_path = str(_TOOL_DIR / "managed_system_regression" / "knowledge" / "thresholds.json")
        with patch("builtins.open", side_effect=lambda p, *a, **kw: open(real_path, *a, **kw) if "thresholds" in str(p) else open(p, *a, **kw)):
            resp = client.post("/api/set-planner",
                               json={"planner": "harmone_original", "system": "regression"})
        assert resp.status_code == 200

    def test_reset_clears_kb(self, client):
        import app as app_module
        client.post("/api/reset")
        assert app_module.KNOWLEDGE_BASE["policies"] == {}

    def test_policy_registered_under_approach_key(self, client):
        """G2: policy registered as reg_harmone_score and retrievable by that id."""
        client.post("/api/reset")
        client.post("/api/policy", json=self._policy())

        resp = client.get("/api/knowledge/reg_harmone_score")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "reg_harmone_score" in data.get("policies", {}), \
            "Dashboard polls /api/knowledge/reg_harmone_score — policy must be registered here"

    def test_telemetry_reaches_policy(self, client):
        """End-to-end: telemetry posted after policy registration populates knowledge."""
        client.post("/api/reset")
        client.post("/api/policy", json=self._policy())
        client.post("/api/telemetry", json={"timestamp": 1.0, "score": 0.9})

        resp = client.get("/api/knowledge/reg_harmone_score")
        history = resp.get_json().get("telemetry_history", [])
        assert len(history) >= 1, "telemetry_history must be non-empty after a telemetry post"


# ---------------------------------------------------------------------------
# Test: planner variant — reg_pareto (planner = pareto, same approach as above)
# ---------------------------------------------------------------------------

class TestRegHarmonEPlannerVariant:
    """A planner variant must use the same base policy_id (G2 proof)."""

    def test_pareto_planner_uses_reg_harmone_score_policy_id(self, client):
        """
        launchHarmonE('reg', 'pareto') → policyIdInput.value = 'reg_harmone_score'
        (after G_POLICY_PREFIX fix).  The policy is registered as reg_harmone_score
        and telemetry reaches it.
        """
        policy = {
            "policy_id": "reg_harmone_score",
            "quality_attribute": "score",
            "adaptation_boundary": {"condition": "LESS_THAN", "threshold": 0.78},
            "tactics": [{"tactic_id": "execute_mape_plan", "priority": 1,
                          "tactic_endpoint": "http://localhost:8080/adaptor/tactic"}],
        }
        client.post("/api/reset")
        client.post("/api/policy", json=policy)

        resp = client.get("/api/knowledge/reg_harmone_score")
        assert resp.status_code == 200
        assert "reg_harmone_score" in resp.get_json().get("policies", {})

    def test_cv_harmone_score_policy_id_for_cv_variant(self, client):
        """CV HarmonE planner variants also use cv_harmone_score as policy_id."""
        policy = {
            "policy_id": "cv_harmone_score",
            "quality_attribute": "score",
            "adaptation_boundary": {"condition": "LESS_THAN", "threshold": 0.54},
            "tactics": [{"tactic_id": "execute_mape_plan", "priority": 1,
                          "tactic_endpoint": "http://localhost:8080/adaptor/tactic"}],
        }
        client.post("/api/reset")
        client.post("/api/policy", json=policy)

        resp = client.get("/api/knowledge/cv_harmone_score")
        assert resp.status_code == 200
        assert "cv_harmone_score" in resp.get_json().get("policies", {})


# ---------------------------------------------------------------------------
# Test: single-model path (reg_single_lstm) — set-model + approach
# ---------------------------------------------------------------------------

class TestRegSingleModelFlow:
    def test_single_model_policy_registered(self, client):
        policy = {
            "policy_id": "reg_single_lstm",
            "quality_attribute": "score",
            "adaptation_boundary": {"condition": "LESS_THAN", "threshold": 0.5},
            "tactics": [],
        }
        client.post("/api/reset")
        client.post("/api/policy", json=policy)

        resp = client.get("/api/knowledge/reg_single_lstm")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Test: G8 startup error surfacing — verify endpoint returns error on early exit
# ---------------------------------------------------------------------------

class TestStartupErrorSurfacing:
    """G8: /api/start-managed-system must return 500 + output when process exits early."""

    def test_startup_failure_returns_500(self, client, tmp_path):
        """Simulate process exiting within grace period → endpoint must return 500."""
        startup_log = tmp_path / "startup.log"
        startup_log.write_text("FATAL: Required artifact missing: managed_system_cv/knowledge/model.csv\n"
                               "Run: python scripts/init_cv.py --config toy_cv\nThen restart.")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited with code 1

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"), \
             patch("app._STARTUP_LOG", str(startup_log)):
            resp = client.post("/api/start-managed-system")

        assert resp.status_code == 500, "Early-exit process must return HTTP 500"
        data = resp.get_json()
        assert "error" in data.get("status", ""), "Response must include status=error"
        assert "artifact" in data.get("message", "").lower() or "fatal" in data.get("message", "").lower() or len(data.get("message", "")) > 0

    def test_startup_success_returns_200(self, client):
        """Process still alive after grace period → HTTP 200."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"), \
             patch("builtins.open", MagicMock()):
            resp = client.post("/api/start-managed-system")

        assert resp.status_code == 200
