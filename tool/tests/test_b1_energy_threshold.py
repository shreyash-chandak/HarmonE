"""
B1: Dynamic energy threshold must be bidirectional.

The paper formula is tau_E(i+1) = clamp(tau_E(i) + delta*(E_ref - E_used), lo, hi).
When E_used < E_ref the threshold rises; when E_used > E_ref it falls.
The original code always increased the threshold, so the energy violation branch
of the planner was permanently disabled.
"""

import pytest
from core.scoring import update_energy_threshold


class TestUpdateEnergyThreshold:
    def test_threshold_rises_when_efficient(self):
        """E_used < E_ref: system is efficient, so threshold should relax upward."""
        result = update_energy_threshold(current=0.5, e_ref=0.7, e_used=0.3, delta=0.1)
        assert result > 0.5, "Threshold should increase when energy usage is below reference"
        assert abs(result - 0.54) < 1e-9  # 0.5 + 0.1*(0.7-0.3) = 0.54

    def test_threshold_falls_when_excessive(self):
        """E_used > E_ref: system is over budget, so threshold must tighten downward."""
        result = update_energy_threshold(current=0.5, e_ref=0.3, e_used=0.7, delta=0.1)
        assert result < 0.5, "Threshold should decrease when energy usage exceeds reference"
        assert abs(result - 0.46) < 1e-9  # 0.5 + 0.1*(0.3-0.7) = 0.46

    def test_threshold_stable_at_reference(self):
        """E_used == E_ref: threshold should not change."""
        result = update_energy_threshold(current=0.5, e_ref=0.5, e_used=0.5, delta=0.1)
        assert abs(result - 0.5) < 1e-9

    def test_clamps_to_hi(self):
        """Result must not exceed hi ceiling."""
        result = update_energy_threshold(current=0.95, e_ref=0.7, e_used=0.0, delta=0.1, hi=1.0)
        assert result <= 1.0

    def test_clamps_to_lo(self):
        """Result must not fall below lo floor."""
        result = update_energy_threshold(current=0.12, e_ref=0.0, e_used=1.0, delta=0.1, lo=0.1)
        assert result >= 0.1

    def test_original_broken_formula_always_increases(self):
        """Regression: verify the old formula was monotonically increasing."""
        original_max_energy = 1.0
        current = 0.5
        used_energy = 0.9  # high energy usage
        broken_result = current + 0.95 * (original_max_energy - used_energy)
        assert broken_result > current, "Confirms the old formula increased even under high energy usage"
        # The fixed formula should decrease in this case
        fixed_result = update_energy_threshold(
            current=current, e_ref=0.7, e_used=used_energy, delta=0.1
        )
        assert fixed_result < current, "Fixed formula should decrease when E_used (0.9) > E_ref (0.7)"
