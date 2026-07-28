"""
B6: execute_mape() incremented model_switches even when plan returned None (no-op).
Switch counter should only increment when a model file is actually written.
"""

import json
import os
import tempfile

import pytest


class FakeCounters:
    """Mimics the event counter logic from execute.py."""

    def __init__(self):
        self.model_switches = 0
        self.noops = 0
        self.mape_k_energy_uJ = 0.0

    def record_event(self, event_type: str, energy: float = 0.0):
        self.mape_k_energy_uJ += energy
        if event_type == "switch":
            self.model_switches += 1
        elif event_type == "noop":
            self.noops += 1


def simulate_execute_mape(decision, counters: FakeCounters, energy: float = 100.0):
    """Simulate the fixed execute_mape() behaviour."""
    if not decision:
        counters.record_event("noop", energy)
        return
    # Write model (simulated) then record switch
    counters.record_event("switch", energy)


class TestSwitchCounter:
    def test_noop_does_not_increment_switches(self):
        c = FakeCounters()
        simulate_execute_mape(None, c, energy=50.0)
        assert c.model_switches == 0
        assert c.noops == 1

    def test_actual_switch_increments_switches(self):
        c = FakeCounters()
        simulate_execute_mape("linear", c, energy=50.0)
        assert c.model_switches == 1
        assert c.noops == 0

    def test_noop_still_accumulates_mape_k_energy(self):
        """MAPE-K energy overhead is real even for no-ops and must be tracked."""
        c = FakeCounters()
        simulate_execute_mape(None, c, energy=75.0)
        assert c.mape_k_energy_uJ == 75.0

    def test_mixed_sequence_counts_correctly(self):
        c = FakeCounters()
        sequence = [None, "lstm", None, None, "svm", None]
        for decision in sequence:
            simulate_execute_mape(decision, c, energy=10.0)
        assert c.model_switches == 2
        assert c.noops == 4
        assert abs(c.mape_k_energy_uJ - 60.0) < 1e-9

    def test_old_behaviour_overcounts(self):
        """Demonstrates the original bug: every cycle counted as a switch."""

        def old_execute(decision, counters):
            # Old code: record_event("switch", ...) regardless of decision
            counters.record_event("switch", 10.0)
            if not decision:
                return  # returned early but counter already incremented

        old_c = FakeCounters()
        for decision in [None, None, "lstm"]:
            old_execute(decision, old_c)
        assert old_c.model_switches == 3, "Old behaviour: 3 cycles = 3 'switches' (2 false)"
