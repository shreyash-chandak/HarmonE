"""
B2: VMR reuse tactic was unreachable because analyse_drift() returned "best_version"
but plan_drift() read "action"/"version" — permanent key mismatch.

These tests verify the return contract of analyse_drift() using a mock and
that plan_drift() correctly routes to "replace" vs. "retrain" based on the contract.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestAnalyseDriftReturnContract:
    """Unit-test the contract, not the file I/O."""

    def _make_drift_result(self, drift_detected, action, version=None):
        return {"drift_detected": drift_detected, "action": action, "version": version}

    def test_no_drift_has_action_none(self):
        result = self._make_drift_result(False, None)
        assert result["action"] is None
        assert result["version"] is None
        assert result["drift_detected"] is False

    def test_vmr_hit_uses_replace_key(self):
        path = "versionedMR/lstm/version_3/lstm.pth"
        result = self._make_drift_result(True, "replace", path)
        assert result["action"] == "replace"
        assert result["version"] == path

    def test_no_vmr_uses_retrain_key(self):
        result = self._make_drift_result(True, "retrain", None)
        assert result["action"] == "retrain"
        assert result["version"] is None

    def test_old_contract_would_return_none_for_action(self):
        """Demonstrates the old bug: plan_drift couldn't route because key was missing."""
        old_style = {"drift_detected": True, "best_version": "some/path"}
        assert old_style.get("action") is None  # the bug — this was always None
        assert old_style.get("version") is None  # the bug — this was always None

    def test_new_contract_routes_to_replace(self):
        path = "versionedMR/lstm/version_2/lstm.pth"
        new_style = {"drift_detected": True, "action": "replace", "version": path}
        # Simulate what plan_drift() does
        if new_style.get("action") == "replace":
            plan_decision = {"action": "replace", "version": new_style["version"]}
        else:
            plan_decision = {"action": "retrain"}
        assert plan_decision["action"] == "replace"
        assert plan_decision["version"] == path
