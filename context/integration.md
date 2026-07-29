# HarmonE Tool — Integration Hardening & Completeness Audit Guide

> **Audience:** Claude Sonnet 4.6, autonomous coding agent in `HarmonE-tool`,
> branch off the current main working branch as `fix/integration-hardening`.
> **Trigger:** The prototype sprint (execution.md) is reported complete
> (STATUS.md), and afterwards a dashboard change landed (DECISIONS.md D10:
> planner-selection modal, `/api/set-planner`, 12 presets, CV HarmonE
> re-enabled). The live documents now disagree with each other and, likely,
> with the code. The user observes UI ↔ backend discrepancies.
> **Mission:** (1) trace and verify every dashboard → backend → managed-system
> integration path end to end; (2) fix every inconsistency found; (3) audit
> the sprint against execution.md's Definition of Done and finish anything
> incomplete; (4) resynchronise all live documents to one truth.
> **Method — evidence over documents:** the uploaded docs contradict each
> other (specific cases below). For every check in this guide, the CODE and a
> RUNNING VERIFICATION are the truth; documents are then corrected to match.
> Never "fix" a discrepancy by editing only the doc unless you first proved
> the code is right.

---

## 0. Operating Rules

1. Read first, in order: `STATUS.md`, `DECISIONS.md` (esp. D10),
   `DECISIONS_PENDING.md`, `files.md`, `execution.md`, `RUNNING_ON_ARCH.md`,
   `CHANGES_FROM_PAPER.md`, then the actual `app.py`,
   `frontend/dashboard.html`, `run_managed_system.py`, both `manage.py` /
   `plan.py` files, and `experiments/run_reset.py`.
2. Every finding gets an ID `Gx-<n>` in a new working file
   `docs/checkpoints/CP7_integration_audit.md`: symptom → evidence (file:line
   or command output) → verdict (code bug / doc drift / both) → fix commit.
   That file is this task's deliverable alongside the fixes.
3. One commit per fix, referencing the finding ID. `pytest tool/tests/ -q`
   after every commit; suite stays green.
4. Stop-and-ask (report finding + two options + recommendation, then wait) if:
   a fix would change MAPE-K semantics, scoring math, or the VMR contract;
   an endpoint rename would break saved user workflows and both directions
   are defensible; the test count story (§3 G6) reveals silently deleted
   tests; git state is unexpected.
5. No new features. This is a hardening pass: wiring, contracts, validation,
   error surfacing, tests, docs. Anything feature-shaped goes to
   `DECISIONS_PENDING.md`.

---

## 1. The Contract Chain You Are Verifying

Every dashboard action must survive this chain intact. Keep it in front of
you; every finding below is a break somewhere along it.

```
dashboard.html button/preset (HARMONY_PRESETS key, policy_id, tactic ids)
  → fetch() calls to app.py REST endpoints (names, payloads, ordering)
    → files written: approach.conf, thresholds.json["planner"], model.csv,
      policies/<id>.json
      → run_managed_system.py startup: approach resolution, artifact health
        check, policy registration by filename prefix, subprocess spawn
        → manage.py: local approach.conf, command listener, tactic dispatch
          (execute_tactic_locally tactic-id cases)
          → plan.py dispatch: planner name read from thresholds.json →
            core planner registry → PlanDecision
            → execute.py: model.csv write, events, counters
              → monitor → telemetry POST → app.py policy matching by
                quality_attribute → KNOWLEDGE_BASE[policy_id]
                → dashboard poll GET /api/knowledge/<currentPolicyId>
                  → charts render
```

A mismatch at ANY arrow produces the classic symptom: everything "runs" and
the dashboard shows nothing, or shows stale/wrong data — which is what the
user is seeing.

---

## 2. Phase 1 — Build the Ground-Truth Inventories (before fixing anything)

### 2.1 Endpoint inventory (finding group G1)
Extract from `app.py` every `@app.route` (method, path, request schema,
response schema, side effects) into a table in the audit file. Then extract
every `fetch(`/`XMLHttpRequest` URL from `dashboard.html`. Diff the two.

Known documented contradictions you must resolve (docs currently claim ALL
of these; at most one set is real):
- `files.md` lists: `/api/telemetry`, `/api/policy`, `/api/knowledge`,
  `/api/set-model`, `/api/set-planner`, `/api/adaptor/upload`,
  `/api/set-approach`.
- `documentation.md` (Harmonica-era) + `RUNNING_ON_ARCH.md` Scenario G list:
  `/api/write-approach`, `/api/save-policy`, `/api/reset`,
  `/api/start-managed-system`, `/api/stop-managed-system`,
  `/api/upload-custom-mape`.
Determine the real set. Rules for the fix:
- If old endpoints were renamed, add thin alias routes for one release ONLY
  if the dashboard or tests still call the old names; otherwise document the
  rename in CHANGES_FROM_PAPER and update every doc.
- The reset / start / stop endpoints MUST exist and be reachable from the
  dashboard — the single-model and approach-switch flows depend on them per
  documentation.md §4.5. If D10's refactor dropped any, that is a code bug:
  restore.
- Every surviving endpoint gets a row in a new authoritative
  `docs/rest_endpoints.md` regeneration (it exists per documentation.md —
  refresh it; if the file is gone, recreate it).

### 2.2 Preset inventory (G2)
From `dashboard.html`, extract all `HARMONY_PRESETS` entries (files.md says
12) into a table: preset key → `policy_id` → quality_attribute → primary/
secondary boundaries → tactic ids → which planner the modal writes. From
`policies/`, list every policy JSON's `policy_id` and filename. From
`app.py`'s approach mapping (D10 says it was extended with
`reg_greedy_switch`, `reg_violation_aware`, `reg_pareto`,
`reg_random_switch`, `cv_greedy_switch`, `cv_violation_aware`), extract the
full dashboard-key → approach-token table. From `run_managed_system.py`,
confirm how `policy_prefix` is derived from the token and which policy files
each token therefore registers.

### 2.3 Planner-source inventory (G3)
Grep every read and write of the `planner` key: `/api/set-planner` writes
where; `plan.py` `dispatch_plan()` reads from where and WHEN (import time vs
per-call); does `approach.conf` carry a planner field at all? `files.md`
line for `approach.conf` says it "Sets system, run_mode, and planner name";
DECISIONS D10 says approach.conf stays unchanged and only
`thresholds.json["planner"]` changes. Exactly one is true. Establish it,
then fix the losing document AND remove any dead code path that reads a
planner from approach.conf if it exists (two config sources for the same
knob is a bug even if currently harmless).

---

## 3. Phase 2 — Findings to Verify and Fix (the known/suspected breaks)

Work through in order. For each: verify → classify → fix → test → doc.

### G1 — Endpoint drift (§2.1). Fix per the rules above.

### G2 — Dashboard polls a policy_id that never receives telemetry
**Risk:** With 12 presets, if the planner-variant presets (e.g. a
`reg_pareto` preset) define their own `policy_id`, the dashboard sets
`currentPolicyId` to it and polls `/api/knowledge/reg_pareto` — but
`run_managed_system.py` registers policies from JSON files by prefix
(`reg_harmone*` for token `reg_harmone`), and telemetry is matched to those
registered policy_ids. Result: charts permanently empty while the system
runs fine.
**Verify:** for each of the 12 presets, statically trace: preset policy_id ==
the policy_id inside the JSON file(s) the mapped approach token registers?
**Fix if broken (choose the minimal-surface option):** planner-variant
presets must reuse the base policy_id (`reg_harmone_score` /
`cv_harmone_score`) for registration and polling; the planner name is
carried ONLY via `/api/set-planner`. If instead each variant registers its
own policy JSON via `POST /api/policy` (dashboard-pushed, not file-scanned),
verify `push_telemetry` matching still routes telemetry to it AND that
`register_policies_with_acp` doesn't then double-register a second policy
watching the same quality_attribute (two policies on `score` = duplicate
tactic firing on every violation — check `receive_telemetry` loops over ALL
policies). Duplicate-firing is a real bug class here; add a server-side
guard: registering a policy whose (quality_attribute, boundary) duplicates
an existing active policy logs a warning and replaces rather than appends.

### G3 — Two sources of truth for the active planner (§2.3). Single source =
`thresholds.json["planner"]` (per D10). Additionally verify freshness:
`dispatch_plan` must read thresholds per planning cycle (or manage.py must
reload on each tactic), because `/api/set-planner` fires before
`start-managed-system` in the modal flow but users can also change planners
between sessions without restarting `app.py`. Add a test: write planner A,
plan once, write planner B, plan again → registry received B.

### G4 — Modal call-ordering vs state resets
**Verify the exact JS sequence** of the modal flow (read the code; expected
shape per documentation.md §4.5: reset → write-approach → set-planner →
[set-model] → POST policy → save-policy → start). Then check two clobber
hazards:
1. Does the approach-switch endpoint (or anything it calls) rewrite/copy
   `thresholds.json` after `set-planner` ran? If yes, planner choice is
   silently lost — reorder the JS or make the endpoint preserve the key.
2. Does `experiments/run_reset.py` or any startup reset touch
   `thresholds.json["planner"]`? It must NOT (run_reset resets run STATE,
   thresholds are config). Add an assertion test.
Also verify the modal flow still performs the reset+stop steps the pre-D10
flow did — if D10's new path skips `POST /api/reset` or the stop of a
running managed system before switching, stale-state bugs L2/L6 partially
return through the UI door. The invariant: EVERY dashboard-initiated
approach launch passes through stop → reset → configure → start, in that
order, regardless of which button started it.

### G5 — `random_switch` exists twice with different meanings
Modal planner `random_switch` (runs under the HarmonE score+drift policy)
vs the standalone Switch approach (`reg_switch`/`cv_switch`, own policy on
r2/confidence, historically the paper's "Switch" baseline). Both are
legitimate; the UI must disambiguate. **Fix:** label the modal option
"Random switch (HarmonE policy)" and the approach button "Switch baseline
(paper)"; add one sentence to README's configuration section explaining the
difference; verify the two produce distinct `run_manifest`/telemetry
identifiers so results can't be conflated later.

### G6 — Test-count story is inconsistent: 210 vs 290
STATUS.md (dated 2026-07-29): "210/210". RUNNING_ON_ARCH: "All 210".
files.md CP6 entry: "290/290". **Verify:** run `pytest tool/tests/ -q
--collect-only | tail -1` and a full run. If the real number is ≥290, update
STATUS + RUNNING_ON_ARCH. If it is <290, tests were deleted or a collection
error is silently hiding files (check `pytest -q` warnings for collection
errors — a broken import in one test file silently drops its tests). A
collection error is a P0 bug: fix the import, never delete the file.
Whatever the number: STATUS.md and RUNNING_ON_ARCH.md must state the same
one, sourced from the run you executed.

### G7 — CV drift tactic may still be a no-op (`pass`)
documentation.md records CV `execute_tactic_locally`: `"handle_data_drift"
→ (currently commented out / pass; placeholder)`. D10 re-enabled CV HarmonE
in the UI. If that `pass` survived, CV HarmonE silently ignores every drift
violation — the UI advertises a capability the backend drops on the floor.
**Verify** the current CV `manage.py` dispatch. **Fix:** wire
`handle_data_drift` → `execute_drift(trigger="acp")` symmetric to
regression; add a test (command file containing the tactic → execute_drift
called, via monkeypatch). If it was intentionally disabled for a reason you
can find (comment/commit), stop-and-ask instead.

### G8 — Dashboard must surface managed-system startup failure
CV HarmonE from the UI now hits the L1-b artifact health check (missing
init artifacts, missing YOLO weights, missing BDD images) and the L3 arch
guard. **Verify** what `/api/start-managed-system` (or its renamed
successor) returns when the wrapper exits within ~5 s, and what the
dashboard does with it. **Fix:** the endpoint waits a short grace period
(3–5 s), polls the child, and on early exit returns HTTP 500 with the last
~20 lines of the wrapper's output; the dashboard shows that message in an
error banner instead of starting the polling loop. Test with a deliberately
missing artifact. (This converts the worst UI symptom — "I clicked start
and nothing happens" — into an actionable message.)

### G9 — `/api/set-planner` server-side validation + bandit exclusion
files.md says the modal offers 5 planners (no `naive`, no `bandit`).
**Verify** the endpoint validates the planner name against the actual
registry (`core.planners.base.get_planner`) and rejects unknowns AND
rejects `bandit` (its `plan()` raises NotImplementedError — selecting it
from any client would crash the MAPE cycle at runtime). Also decide `naive`:
it is a valid registry planner; either include it in the modal (it's the S1
baseline — cheap win, recommended) or document why single-model presets
cover it. Add endpoint tests: valid name per system → correct file written;
`bandit` → 400 with message; unknown → 400; missing system field → 400.

### G10 — Energy backend config has two homes
RUNNING_ON_ARCH says `energy_meter` defaults to `"null"` "in your dataset
config or grid YAML"; files.md shows `energy_meter` inside both
`thresholds.json` files. **Verify** which file the LIVE system reads and
which the HEADLESS harness reads. Likely intended: live → thresholds.json;
headless → dataset config/grid override. That split is acceptable ONLY if
documented as a rule; better: harness override WRITES the effective value
into the per-run thresholds copy so `run_manifest` + artifacts are
self-describing (check it already does — CHANGES Phase 5 suggests yes).
Document the precedence in README config reference + RUNNING_ON_ARCH §3.
While here: confirm `probe_energy.py` cache file location matches what
`core/energy.py` reads (earlier plans placed `.energy_backends.json` in
knowledge/ — a path mismatch between writer and reader means the cache
never hits; verify one canonical path).

### G11 — "thresholds.json is never written by runtime code" is now false
`/api/set-planner` writes it. Reconcile the files.md description:
thresholds.json is written by CONFIG-TIME actors (user edits, set-planner
endpoint, harness per-run copies) and never by the MAPE runtime
(monitor/analyse/plan/execute) — verify that stronger claim by grepping
writes, then state it precisely. (`current_energy_threshold` — check where
the dynamic threshold is persisted: if analyse writes it into
thresholds.json, the claim is false twice and the dynamic value belongs in
mape_info.json instead; documentation.md's mape_info schema already has
`current_energy_threshold`, so verify analyse writes THERE and not to
thresholds.)

### G12 — Preset ↔ policy-file ↔ tactic-id closure (the full matrix)
Using §2.2's inventories, verify for EVERY preset: (a) mapped approach token
resolves to an existing managed_system dir; (b) prefix-scan finds ≥1 policy
file; (c) every tactic_id in that policy (primary + secondary) has a case in
the target domain's `execute_tactic_locally`; (d) the preset's
quality_attribute is actually emitted by that domain's `monitor_mape()`
payload (e.g. `r2_score` exists in regression telemetry; `confidence` in
CV); (e) dashboard `validModels` whitelist matches the domain's model set.
Record the 12-row matrix in the audit file with PASS/FAIL per column. Fix
every FAIL. This matrix IS the deliverable the user asked for when they said
"make sure the integration between everything is seamless".

### G13 — CORS and dashboard origin
The dashboard is served from :8000 and calls :5000. Confirm `flask-cors` is
still applied to the app after the D10 edits (a refactor that recreated the
Flask app object can silently drop `CORS(app)`), and that the new endpoints
(`set-planner`, upload) are covered. Symptom if broken: every UI action
fails only in the browser while curl works — a classic "UI vs backend
discrepancy". Add a header check to the new endpoint tests
(`Access-Control-Allow-Origin` present on an OPTIONS/GET).

### G14 — Missing checkpoints CP4 and CP5
`docs/checkpoints/` contains CP0–CP3 and CP6 — CP4 (WSL smoke) and CP5
(docs verification) were skipped even though STATUS claims a WSL smoke on
2026-07-28. The dashboard changed AFTER that smoke (D10), so the old smoke
no longer covers the live UI path. **Fix:** produce CP4 and CP5 now, as
defined in execution.md §10, with one upgrade to CP4: the live-system part
must exercise the NEW dashboard flow. Since you cannot click a browser,
implement §4.2's scripted dashboard-flow simulation and run it as the CP4
evidence, plus the wrapper-level WSL smoke for both domains.

### G15 — README existence and accuracy
`files.md`'s tree omits the repo root files; execution.md §8.1 required a
full README rewrite, and CP5 (its verification) is missing. **Verify** a
root `README.md` exists and matches the §8.1 section list; if absent or
stale, write it now per that spec. Then update it with this pass's outcomes:
the planner modal (with the G5 disambiguation), the endpoint reference
pointer, the energy-config precedence rule (G10). Every command in it must
be re-executed by you before committing (execution.md's rule stands).
Cross-check README against RUNNING_ON_ARCH so the two never duplicate
divergent instructions — README covers WSL quickstart + pointers;
RUNNING_ON_ARCH remains the Arch deep-dive; each links the other.

### G16 — STATUS.md refresh
After all fixes: correct the test count (G6), add a Phase 8 row
("Integration hardening — CP7"), fix the smoke-test block if the
`approach.conf` format changed (G3 outcome), and align the "Key File
Locations" table with files.md. STATUS, files.md, RUNNING_ON_ARCH,
DECISIONS, CHANGES must tell one story; grep each for the losing claims
identified in G1/G3/G6/G10/G11 and purge them.

---

## 4. Phase 3 — New Tests That Lock the Integration Down

### 4.1 `tests/test_api_endpoints.py`
Flask test client against `app.py`:
- every endpoint from the §2.1 inventory: happy path + malformed payload;
- `set-planner`: valid/invalid/bandit/missing-system (G9);
- policy registration duplicate-guard behavior (G2);
- reset actually clears the three KNOWLEDGE_BASE dicts;
- None-valued `kl_div` telemetry ingested without firing secondary
  boundaries (regression guard for L4 via the API surface);
- CORS header present (G13).

### 4.2 `tests/test_dashboard_flow.py` — the scripted UI simulation
This is the centerpiece. A test (and a runnable script
`scripts/simulate_dashboard.py --preset <key>`) that replays the EXACT
fetch sequence dashboard.html performs for a given preset — same endpoints,
same order, same payloads (extract them from the JS; if the JS and this
script ever diverge, that divergence is itself the bug, so add a comment
block in dashboard.html marking the canonical sequence and pointing here).
For at least these four presets: `reg_harmone_score` (default planner),
one regression planner variant (e.g. pareto), `reg_single_*` (set-model
path), and one CV preset (expected to fail-fast cleanly on missing
artifacts in CI → asserts the G8 error surface, and to succeed when toy_cv
artifacts exist).
Assertions after the sequence, WITHOUT starting real inference where
possible (mock/spawn-with---max-steps): approach.conf token correct;
thresholds planner correct; policy registered under the polled policy_id;
`GET /api/knowledge/<currentPolicyId>` returns the registered policy (the
G2 proof); for the live variant, after N seconds telemetry_history is
non-empty and its rows carry the preset's quality_attribute.
Mark the spawning variants with a `slow`/`integration` pytest marker; the
static-assertion variants run in the default suite.

### 4.3 Extend `test_live_run_fixes.py`
- run_reset preserves `thresholds.json` entirely, including `planner` (G4);
- CV `execute_tactic_locally("handle_data_drift")` dispatches to
  `execute_drift` (G7);
- planner re-read freshness (G3).

---

## 5. Phase 4 — Completeness Audit vs execution.md Definition of Done

Walk execution.md §11 item by item; record PASS/GAP in the audit file:
1. CP0–CP6 reports — GAP known (CP4, CP5; fix via G14).
2. Full suite green incl. new tests — after §4.
3. WSL boot both domains — re-verify NOW on the current code (post-D10),
   using `harmone_start_wsl.sh` + `harmone_stop.sh`; confirm the pidfile
   stop leaves zero orphans (`pgrep -f inference.py` empty).
4. Task-agnostic CV behind TaskAdapter + fixed-reference embedding drift +
   pinned embedding model — spot-verify in code (not docs): CV
   `inference.py` has no `YOLO(` outside the detection task adapter/loader;
   `mmd_embedding.fit_reference` loads the npz; `embedding_model` honored
   when it differs from the active model (unit test exists? if not, add).
5. R1–R6 applied — grep proofs: no `import pyRAPL`; no root
   `scripts/init_scaler.py`; `evaluate_run_against_labels.py` spelled
   correctly; cv_guide §8.4 carries the SUPERSEDED note; CHANGES Phase 4
   describes pyJoules (the R2 rewrite actually landed — read it).
6. Six dataset docs + awaiting_data skeletons + validator SKIPPED-vs-FAIL —
   run `scripts/validate_dataset.py` against all 10 configs; expect
   PASS×4 (pems_node1, bdd100k?, toys — bdd100k passes only if images
   present; if absent it must be awaiting-tolerant or the config gets
   `status`), SKIPPED×5, zero FAIL. Fix validator or configs until true.
7. README — G15.
8. Push hygiene — re-run the CP6 greps (absolute paths, pyRAPL, tracked
   generated files) since new commits landed after CP6; then the final push
   step in §6.

Also close the loop on two stale DECISIONS_PENDING items: DP1 still says
"datasets not yet chosen" while DP-A above it says decided — delete/merge
DP1 into DP-A; DP5 is marked partially resolved — finish its resolution
line (the config keys exist; state that and close it or name what remains).

---

## 6. Phase 5 — Final Verification Protocol and Push

Run in order; each step's output goes into `docs/checkpoints/CP7_final.md`:
1. `pytest tool/tests/ -q` full — record the real count (G6 truth).
2. `scripts/simulate_dashboard.py` for the four presets (§4.2) — attach
   trimmed output.
3. WSL live smoke, regression: `./harmone_start_wsl.sh`, ≥3 telemetry
   cycles, dashboard endpoint returns growing telemetry_history for the
   DEFAULT preset's policy_id, `./harmone_stop.sh`, zero orphans.
4. WSL live smoke, CV on toy_cv (yolo_n, CPU, luminance_kl) — same checks;
   plus one forced drift-tactic command through the command file to prove
   G7's fix end-to-end.
5. Headless: one `run_experiment.py` run + `metrics.py aggregate` on it.
6. Invariants I1–I8 from execution.md §3 re-asserted with one addition —
   **I9: UI truthfulness** — every control visible in the dashboard maps to
   a verified backend behavior (the G12 matrix all-PASS is the evidence).
7. Doc-consistency grep: the losing claims from G1/G3/G6/G10/G11 appear in
   zero files.
8. Update `files.md`, `STATUS.md`, `CHANGES_FROM_PAPER.md` (new section:
   "Integration hardening — CP7", listing G-findings fixed),
   `DECISIONS.md` (append D11: endpoint set + planner single-source +
   UI error surfacing, with rationale), `DECISIONS_PENDING.md` (§5 cleanup).
9. Commit, merge to the main working branch, tag `v2.1-integrated`, push.
   Push problems → stop-and-ask; never force-push.

---

## 7. Definition of Done

1. `CP7_integration_audit.md` exists with every G1–G16 finding resolved
   (verdict + fix commit) and the 12-preset G12 matrix all-PASS.
2. CP4, CP5 (backfilled on current code) and CP7_final reports exist.
3. New test files (§4) green in the default suite; slow markers documented;
   one true test count stated identically in STATUS, RUNNING_ON_ARCH, and
   CP7_final.
4. Dashboard flows verified by script: default HarmonE, planner variant,
   single-model, and CV — including the failure path showing a remediation
   banner instead of silence.
5. Exactly one source of truth each for: active planner, energy backend
   (with documented precedence), endpoint names, approach tokens.
6. CV drift tactic executes (G7) or a stop-and-ask record explains why not.
7. All live documents mutually consistent; DECISIONS_PENDING cleaned.
8. Tagged `v2.1-integrated` and pushed.