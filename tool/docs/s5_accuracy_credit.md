# S5 redesign: run-level accuracy credit (2026-10-26)

Code: `core/planners/violation_aware.py` (S5), `core/planners/credit.py` (credit, cap,
required accuracy, known/unknown split), `core/planners/violation_aware_gated.py` (S5a,
the previous S5, kept unchanged as the "hard per-cycle constraints" ablation).

## Why

The previous S5 got stuck on the LSTM by design, not by a bug:

- its `min_accuracy` gate was a hard per-cycle constraint ridge and SVR almost never passed,
  and when the gate left no candidate it did nothing (121/131 PeMS energy violations were
  "stay on LSTM" no-ops);
- it required one model to satisfy both targets in the same cycle, although the targets are
  run-level averages that HarmonE meets by alternating between models;
- on drift it switched instead of adapting, judging the alternative by a pre-drift estimate;
- it trusted frozen accuracy estimates of models that last served during bad stretches.

## Mechanism

- **Credit** `B ← clip(B + (acc_batch − A_target), −B_max, +B_max)` every cycle in Monitor,
  from the batch accuracy (window R²) of whichever model served the batch — lagging batches
  and recovery-hold cycles included. `A_target` = the dataset's `min_accuracy` (the score
  target at E_ref = 0.5); no new calibration. Stored in `mape_info["s5_credit"]`, tracked for
  every planner, used only by S5.
- **Required accuracy** `a*_m = (min_score − (1−β)(1 − ê_m)) / β`: the accuracy at which
  model m, at its own energy estimate, would not itself trigger a score violation
  (`min_accuracy` is the same formula at ê = 0.5).
- **Affordability** `projected_B(m) = B − H × max(0, A_target − â_m) ≥ 0`.
- **Unknown models**: never observed, or last served more than `staleness_window` rows ago;
  the current model is always known. Their EMAs are never compared; they are ordered by cost
  class (ridge light, SVR medium, LSTM heavy).
- **Decision order**: drift (VMR replace / retrain, never a switch) → exploration probe
  (`explore_prob`) → no violation: noop → energy violation (cheaper candidates:
  already_cheapest / affordable_cheapest / unknown_probe / infeasible_fallback) → score
  violation (more accurate or costlier candidates: repay_debt / cheapest_sufficient /
  unknown_probe / most_accurate / current_is_best). Exact rules in the module docstring.

## Fixed values and their rationale (fixed before any run; not tuned on results)

| Value | Setting | Rationale |
|---|---|---|
| H (`s5_credit_horizon`) | 3 cycles | = `switch_hold_cycles`: an energy switch is held for 3 cycles (the original's `recovery_cycles`), so 3 cycles is the shortest stint a switch to a cheaper model commits the run to; affordability is judged over that stint. |
| Cap factor | 1.5 | `B_max = 1.5 × H × max(0, A_target − acc_naive_ridge)`: the run can bank enough credit for about one and a half held ridge stints, and no more, so credit cannot accumulate indefinitely during long LSTM stretches (which would later license long cheap stretches below target). Fallback 0.5 (with a warning) if ridge already meets the target. |
| `acc_naive_ridge` | naive ridge's mean per-batch window accuracy | the same quantity the calibration uses for `min_accuracy`; with the current rule (`min_accuracy` = midpoint of ridge and LSTM) `B_max = 2.25 × (acc_lstm − acc_ridge)`. |
| Staleness window | 2,600 rows | same key and value as pareto, so both planners judge staleness alike; it equals the drift cooldown (the original's 400 s after a drift action, at its pacing): an estimate older than one cooldown period may predate a drift the stream has gone through since. |
| Energy-violation margin | removed (S5 only) | HarmonE has none, and the 0.02 margin was part of what produced the no-ops; an energy violation now always switches unless the current model is already the cheapest. |
| Score-violation margin | `s5_switch_margin` 0.02 | kept only for the last-resort "most_accurate" branch, as before. |
| `explore_prob` | 0.05 | unchanged (same as pareto). |

## Logging

`mape_events.csv` columns for both S5 variants (blank otherwise): `s5_credit`, `s5_projected`,
`s5_required_acc` (of the chosen model, when it is known and switched to), `s5_reason`,
`s5_unknown_models`. S5a tracks B passively (same update and cap, never used in its decisions)
and labels its own branches (`drift_prefers_switch`, `energy_current_lowest_noop`, …).
`scripts/s5_report.py` summarises them; `scripts/check_runs.py` fails S5 runs whose credit
leaves ±cap, that have cycles without a reason, or (S5) an energy-violation noop other than
`already_cheapest` or a switch on drift.
