"""
scripts/s5_report.py — per-dataset diagnostics for the two S5 variants
(violation_aware, violation_aware_gated; 2026-10-26).

From each run's mape_events.csv, predictions.csv and run_manifest.json, per
dataset and variant (all seeds pooled unless noted):
  - counts of each s5_reason;
  - LSTM share of served predictions (mean over seeds, and per seed);
  - retrains + VMR restores per seed;
  - energy violations (after the recovery hold) that ended in a noop;
  - the credit B: median, p10, p90, % of cycles at +cap / −cap.
Also checks the smoke-test properties: every reason branch of the new S5 fired
at least once over all runs, B stayed within ±cap, no energy-violation noop
other than "already_cheapest".

Usage (from tool/):
    python3 scripts/s5_report.py --runs runs_seed1 runs_seed2 runs_seed3 runs_seed4 runs_seed5
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

DATASETS = [("pems_driftinduced", "PeMS"), ("spot_prices_driftinduced", "Spot"),
            ("uci_electricity_driftinduced", "UCI")]
VARIANTS = [("violation_aware", "S5"), ("violation_aware_gated", "S5a gated")]
NEW_S5_REASONS = ["drift_replace", "drift_retrain", "probe", "no_violation",
                  "already_cheapest", "affordable_cheapest", "unknown_probe", "infeasible_fallback",
                  "repay_debt", "cheapest_sufficient", "most_accurate", "current_is_best"]
GATED_REASONS = ["drift_prefers_switch", "drift_replace", "drift_retrain", "probe", "no_violation",
                 "energy_current_lowest_noop", "energy_unobserved_trial", "energy_margin_noop",
                 "energy_switch", "score_current_best_noop", "score_unobserved_trial",
                 "score_margin_noop", "score_switch"]


def load(run_dirs: list[Path]) -> dict:
    out: dict = collections.defaultdict(list)
    for rd in run_dirs:
        for mf in sorted(rd.glob("*_conc/run_manifest.json")):
            j = json.loads(mf.read_text())
            if j.get("planner") not in dict(VARIANTS):
                continue
            run = mf.parent
            thr = json.loads((run / "thresholds.json").read_text(encoding="utf-8"))
            with open(run / "mape_events.csv", newline="") as f:
                ev = list(csv.DictReader(f))
            with open(run / "predictions.csv", newline="") as f:
                served = [r["active_model"] for r in csv.DictReader(f)]
            ec = j.get("event_counters", {})
            out[(j["dataset"], j["planner"])].append({
                "seed": rd.name, "ev": ev, "cap": float(thr.get("s5_credit_cap", 0.5)),
                "lstm": served.count("lstm") / max(len(served), 1),
                "adapt": ec.get("retrains", 0) + ec.get("vmr_events", 0),
                "rt": ec.get("retrains", 0), "vmr": ec.get("vmr_events", 0),
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    data = load([Path(r) for r in args.runs])
    problems, fired = [], collections.Counter()

    for ds, short in DATASETS:
        for planner, label in VARIANTS:
            runs = data.get((ds, planner))
            if not runs:
                continue
            reasons = collections.Counter(e.get("s5_reason") or "(blank)" for r in runs for e in r["ev"])
            if planner == "violation_aware":
                fired.update(reasons)
            order = NEW_S5_REASONS if planner == "violation_aware" else GATED_REASONS
            print(f"\n### {short} — {label} ({len(runs)} run(s))\n")
            print("| s5_reason | cycles | % |")
            print("|---|---|---|")
            total = sum(reasons.values())
            for k in order + sorted(set(reasons) - set(order)):
                print(f"| {k} | {reasons.get(k, 0)} | {100 * reasons.get(k, 0) / max(total, 1):.1f} |")

            e_noop = [sum(1 for e in r["ev"] if e.get("violation_effective") == "energy"
                          and e["decision_action"] == "noop") for r in runs]
            e_all = [sum(1 for e in r["ev"] if e.get("violation_effective") == "energy") for r in runs]
            credit = np.array([float(e["s5_credit"]) for r in runs for e in r["ev"]
                               if e.get("s5_credit") not in (None, "")])
            caps = {r["cap"] for r in runs}
            cap = max(caps)
            at_hi = 100 * np.mean(np.isclose(credit, cap, atol=1e-6)) if credit.size else float("nan")
            at_lo = 100 * np.mean(np.isclose(credit, -cap, atol=1e-6)) if credit.size else float("nan")
            print()
            print("| Metric | Value |")
            print("|---|---|")
            print(f"| LSTM share, mean (per seed) | {100 * np.mean([r['lstm'] for r in runs]):.1f} % "
                  f"({', '.join(f'{100 * r['lstm']:.0f}' for r in runs)}) |")
            print(f"| retrains + VMR restores per seed | {', '.join(f'{r['rt']}+{r['vmr']}' for r in runs)} |")
            print(f"| energy violations ending in noop / all, per seed | "
                  f"{', '.join(f'{a}/{b}' for a, b in zip(e_noop, e_all))} |")
            if credit.size:
                print(f"| B median / p10 / p90 | {np.median(credit):+.4f} / {np.percentile(credit, 10):+.4f} / "
                      f"{np.percentile(credit, 90):+.4f} |")
                print(f"| % cycles at +cap / −cap (cap {', '.join(f'{c:g}' for c in sorted(caps))}) | "
                      f"{at_hi:.1f} / {at_lo:.1f} |")
                if np.any(np.abs(credit) > cap + 1e-6):
                    problems.append(f"{short} {label}: B outside ±{cap}")
            else:
                problems.append(f"{short} {label}: no s5_credit values")
            if planner == "violation_aware":
                bad = sum(1 for r in runs for e in r["ev"] if e.get("violation_effective") == "energy"
                          and e["decision_action"] == "noop" and e.get("s5_reason") != "already_cheapest")
                if bad:
                    problems.append(f"{short} S5: {bad} energy-violation noops other than already_cheapest")

    if fired:
        missing = [k for k in NEW_S5_REASONS if not fired.get(k)]
        print(f"\nNew S5 reason branches that never fired (all datasets): {', '.join(missing) or 'none'}")
    print("\n" + ("\n".join(problems) if problems else "credit within ±cap; no stay-on-model energy noops"))


if __name__ == "__main__":
    main()
