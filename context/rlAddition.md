# Claude Sonnet Agent — LinUCB Bandit Planner Implementation

> **Mission**: Implement the LinUCB contextual bandit planner (S7) in the HarmonE-tool
> codebase. This resolves DP3 and DP8 from `DECISIONS_PENDING.md`. The planner currently
> exists as a documented stub in `core/planners/bandit.py` raising `NotImplementedError`.
> Your job is to replace that stub with a complete, tested, production-quality implementation
> and wire it into every layer of the system that touches planner selection.
>
> **Before writing a single line of code**, read the following files in this order:
> 1. `tool/docs/files.md` — the live file inventory; treat it as your map of what exists
> 2. `tool/DECISIONS.md` — every architectural decision made so far; do not contradict these
> 3. `tool/DECISIONS_PENDING.md` — what is deferred and why; understand DP3/DP8/DP10/DP11
> 4. `tool/STATUS.md` — current implementation state; understand what is green and what is not
> 5. `tool/CHANGES_FROM_PAPER.md` — what diverges from the published papers and why
> 6. `core/planners/bandit.py` — the existing stub; read the full specification comment
> 7. `core/planners/base.py` — the `Planner` ABC, `PlanningContext` dataclass, `Decision`
>    dataclass, and `get_planner()` registry
> 8. `core/planners/pareto.py` — the most recent complete planner; follow its patterns exactly
> 9. `managed_system_regression/mape_logic/plan.py` — how `dispatch_plan()` works
> 10. `managed_system_regression/mape_logic/execute.py` — how decisions are enacted and
>     where energy is measured
> 11. `managed_system_regression/mape_logic/manage.py` — the MAPE loop lifecycle
> 12. `core/scoring.py` — the canonical `S_i` formula and EMA update
> 13. `core/energy.py` — `EnergyMeter` API; do not use pyRAPL anywhere
> 14. `tests/test_planners.py` — existing planner test patterns; follow these exactly
>
> After reading those, run `pytest tool/tests/ -q` and confirm the suite is green before
> touching anything. Record the output. If anything is red, report it and stop.

---

## Operating Rules

**Do not proceed past any of these without stopping and reporting:**

1. If any file you need to read is missing from the tree — report what you expected
   and what was found instead.
2. If two live documents contradict each other on something not already noted in
   `DECISIONS.md` or `CHANGES_FROM_PAPER.md` — report both contradicting passages.
3. If any change would alter the scoring formula `S_i = β·A_i + (1−β)(1−Ē_i)`,
   the EMA update, Eq. 3 threshold dynamics, or the VMR store/match semantics —
   report it; you should not need to change any of these.
4. If a test that was green before you started turns red and you cannot fix it
   within 15 minutes — report the full traceback rather than marking it xfail
   or deleting it.
5. If you encounter anything that requires `sudo` beyond what is already documented
   in `scripts/setup_energy_permissions.sh` — report it.

**Commit discipline:**
- Branch: `feat/bandit-planner` off the current working branch.
- Conventional commits. One logical change per commit.
- Include the DP reference in commit messages: e.g.
  `feat: implement LinUCB bandit planner (DP3/DP8)`.
- **Every commit that changes code structure must also update `files.md`.**
  A commit that adds a file but does not update `files.md` is incomplete.
- Run `pytest tool/tests/ -x -q` after every section. The suite must stay green.

**Live document discipline — this is mandatory, not optional:**
- After every significant milestone (each numbered section below), update the
  relevant live documents in the same commit as the code:
  - `DECISIONS_PENDING.md`: mark DP3 and DP8 as DECIDED with the decision made
  - `DECISIONS.md`: add a new D-entry for the bandit planner design choices
  - `STATUS.md`: update the bandit row in the planner table from "stub" to the
    current state
  - `CHANGES_FROM_PAPER.md`: add an entry if the bandit planner represents a
    departure from or extension of the published approaches
  - `files.md`: update the `core/planners/bandit.py` row description

---

## Section 1 — Reality Check (Do This Before Any Code)

Read the actual content of `core/planners/bandit.py`. The stub should contain a
specification comment describing the intended algorithm. Read it carefully — it is
your primary algorithm specification. If the comment is incomplete or absent, check
`DECISIONS_PENDING.md` DP3 and DP8 for the specification.

Then verify the following facts by reading the listed files. If any fact is wrong,
report it before proceeding.

**Fact 1**: `core/planners/base.py` exports:
- `PlanningContext` — a dataclass containing at minimum: `violation` (str: "score",
  "energy", or "drift"), `current_model` (str), `ema_scores` (dict mapping model
  name to float), `ema_accuracy` (dict), `ema_energy` (dict), `thresholds` (dict),
  `knowledge_dir` (str or Path).
- `Decision` — a dataclass with at minimum: `action` (str: "switch", "retrain",
  "vmr_replace", "noop"), `target_model` (str or None).
- `Planner` ABC with abstract method `plan(ctx: PlanningContext) -> Decision`.
- `get_planner(name: str) -> Planner` registry function.

Verify these are present. If the actual signatures differ, record the actual
signatures — your implementation must match what actually exists, not what is
described here.

**Fact 2**: `core/planners/pareto.py` registers itself in the planner registry.
Confirm how registration works (decorator, explicit call in `__init_subclass__`,
or explicit registration call in `base.py`). Your bandit planner must register
itself using exactly the same mechanism.

**Fact 3**: `managed_system_regression/mape_logic/plan.py` contains `dispatch_plan()`
which reads `thresholds.json["planner"]` to select which planner to call. Confirm
the key name and the exact code path. Confirm what value the thresholds.json must
contain to route to the bandit planner (e.g. `"bandit"`, `"linucb"`, or
`"s7_bandit"` — read the actual code to find out, or the stub's comment).

**Fact 4**: `STATUS.md` records that DP11 means CV domain does NOT call
`dispatch_plan()` — it calls `plan_mape()` directly. This means the bandit
planner will only be wired for regression in this implementation. CV bandit
support is explicitly deferred as DP11. Confirm this is still the case and do
not attempt to wire the bandit into CV plan.py in this sprint.

**Fact 5**: `tests/test_planners.py` already tests S1–S6 using `PlanningContext`
mocks. Read the test patterns. Your tests for S7 must follow exactly the same
pattern.

After verifying all five facts, record any discrepancies from what is described
above, and write them as a brief reality-check note at the top of the first
checkpoint doc (`docs/checkpoints/CP_bandit.md`).

---

## Section 2 — Algorithm Specification (Read, Confirm, Then Implement)

This section specifies the algorithm. Do not implement anything until you have
read this in full.

### 2.1 What LinUCB Does

LinUCB (Linear Upper Confidence Bound) is a contextual bandit algorithm. At each
decision point it receives a context vector describing the current system state.
For each available arm (model) it computes a score that balances expected reward
(exploitation) and uncertainty (exploration). It selects the arm with the highest
score.

The algorithm assumes expected reward is a linear function of the context. This is
appropriate here because the context features (EMA slopes, drift signal magnitude,
normalised energy) have roughly linear relationships with which model will perform
best under those conditions.

### 2.2 Context Vector — What the Bandit Sees

The context vector is assembled from `PlanningContext` at the moment a planning
decision is triggered. All values must be normalised to approximately [0, 1] or
[-1, 1] for numerical stability.

The context vector contains these features in this exact order (dimension = 16
for a 3-model regression spectrum; dimension scales with model count):

```python
def _build_context(ctx: PlanningContext) -> np.ndarray:
    """
    Build the context vector from a PlanningContext.

    All values normalised. The vector is 10 + 2*len(models) dimensional.
    For a 3-model spectrum: 10 + 6 = 16 dimensions.
    """
    models = list(ctx.ema_scores.keys())  # consistent ordering

    # Violation type — one-hot (3 dims)
    violation_score  = float(ctx.violation == "score")
    violation_energy = float(ctx.violation == "energy")
    violation_drift  = float(ctx.violation == "drift")

    # Current system state (5 dims)
    current_ema    = float(np.clip(ctx.ema_scores.get(ctx.current_model, 0.5), 0, 1))
    current_energy = float(np.clip(
        ctx.thresholds.get("current_normalised_energy", 0.5), 0, 1
    ))
    drift_signal   = float(np.clip(ctx.thresholds.get("current_kl_div") or 0.0, 0, 2) / 2.0)
    ema_slope      = float(np.clip(ctx.thresholds.get("ema_slope", 0.0), -1, 1))
    steps_since    = float(np.clip(
        ctx.thresholds.get("steps_since_last_switch", 0) / 500.0, 0, 1
    ))

    # Budget signals (2 dims)
    retrains_norm  = float(np.clip(
        ctx.thresholds.get("retrains_this_run", 0) / 10.0, 0, 1
    ))
    vmr_size_norm  = float(np.clip(
        ctx.thresholds.get("vmr_size", 0) / 20.0, 0, 1
    ))

    # Per-model EMA accuracy and energy (2 × len(models) dims)
    model_ema_acc  = [float(np.clip(ctx.ema_accuracy.get(m, 0.5), 0, 1)) for m in models]
    model_ema_eng  = [float(np.clip(ctx.ema_energy.get(m, 0.5), 0, 1))   for m in models]

    features = (
        [violation_score, violation_energy, violation_drift,
         current_ema, current_energy, drift_signal, ema_slope,
         steps_since, retrains_norm, vmr_size_norm]
        + model_ema_acc
        + model_ema_eng
    )
    return np.array(features, dtype=np.float32)
```

If `PlanningContext` does not have `ema_accuracy` or `ema_energy` attributes
(because they were added in Phase 2 and may not have been backfilled to
all context construction sites), do not crash — fall back to using `ema_scores`
for both and add a comment flagging the limitation for future resolution.

Check that `mape_info.json` contains `ema_accuracy` and `ema_energy` dicts
by reading `managed_system_regression/mape_logic/execute.py`. If they are
present, they are available. If not, file this as a known limitation in the
checkpoint doc.

### 2.3 Action Space

The arms are all models except the currently active model:
```python
candidates = [m for m in ctx.ema_scores.keys() if m != ctx.current_model]
```

If candidates is empty (only one model exists), return a noop Decision immediately
without invoking the bandit logic.

### 2.4 LinUCB Per-Arm Matrices

For each arm `a` in the full model set (not just candidates), maintain:
- `A[a]`: a `d × d` matrix initialised to `np.eye(d)` where d = context dimension
- `b[a]`: a `d`-dimensional vector initialised to `np.zeros(d)`
- `theta[a]`: derived as `np.linalg.inv(A[a]) @ b[a]` (computed at selection time)

At selection time for each candidate arm `a`:
```python
A_inv = np.linalg.inv(A[a])
theta = A_inv @ b[a]
expected_reward = theta @ context           # exploitation term
uncertainty = alpha * np.sqrt(context @ A_inv @ context)  # exploration bonus
ucb_score[a] = expected_reward + uncertainty
```

Select the arm with the highest UCB score. Log all UCB scores for diagnostics.

### 2.5 Reward Function

The reward is computed ONE monitoring interval after the decision was made,
when the new model has accumulated real EMA data. The reward captures sustainability
improvement per unit of switching energy cost:

```python
def _compute_reward(
    ema_before: float,
    ema_after: float,
    energy_before: float,   # normalised Ē_i before switch
    energy_after: float,    # normalised Ē_i after switch
    switch_cost_J: float,   # MAPE-K energy for this switch in joules
    w_acc: float = 0.7,
    w_energy: float = 0.3,
) -> float:
    """
    Reward = sustainability gain per joule of switching cost.

    Positive when the new model improves accuracy and/or reduces energy.
    Negative when the switch makes things worse.
    Clipped to [-10, 10] to prevent gradient explosion.
    """
    delta_acc    = ema_after - ema_before                              # positive = better
    delta_energy = energy_before - energy_after                        # positive = less energy

    # Normalise to comparable scales
    delta_acc_norm    = float(np.clip(delta_acc / 0.5, -1.0, 1.0))
    delta_energy_norm = float(np.clip(
        delta_energy / max(energy_before, 1e-9), -1.0, 1.0
    ))

    gain = w_acc * delta_acc_norm + w_energy * delta_energy_norm
    cost = max(switch_cost_J, 1e-9)    # avoid division by zero
    return float(np.clip(gain / cost, -10.0, 10.0))
```

`w_acc` and `w_energy` are config-driven. They should default to match the paper's
`beta` parameter where possible (beta = accuracy weight in S_i formula). Read
`thresholds.json` for the beta value; use `w_acc = beta` and `w_energy = 1 - beta`
as defaults if not explicitly configured.

### 2.6 Delayed Reward — The Pending Decision Pattern

The bandit cannot compute the reward immediately after selecting an arm, because
the new model needs at least one monitoring interval to accumulate real EMA data.
The reward observation is therefore delayed.

At decision time:
1. Select the arm and build a `PendingDecision` record containing:
   `context`, `arm`, `ema_before`, `energy_before`, `switch_cost_J`, `timestamp`
2. Persist the `PendingDecision` to `knowledge/bandit_pending.json` (atomic write,
   same `.tmp` → `os.replace()` pattern used everywhere else in the codebase)
3. Return the `Decision` to the caller as normal

At the NEXT monitoring interval (in `manage.py`'s MAPE loop, BEFORE calling
monitor/analyse/plan):
1. Check if `knowledge/bandit_pending.json` exists
2. If it exists and `time.time() - pending["timestamp"] >= thresholds["monitoring_interval_s"]`
   (i.e., at least one full monitoring interval has passed):
   a. Read current `ema_scores[arm]` and `current_normalised_energy` from `mape_info.json`
   b. Compute the reward
   c. Call `bandit.observe_outcome(context, arm, reward)` which updates A[arm] and b[arm]
   d. Save bandit state
   e. Delete `bandit_pending.json`
3. If it exists but not enough time has passed: skip for this cycle

The pending check belongs in `manage.py`'s loop, not in `plan.py`. Do not call
`observe_outcome` from inside `dispatch_plan()` — the MAPE loop orchestrates this.

**Important**: if the planner changes (a different planner fires before the pending
decision is resolved — e.g. a drift tactic fires), the pending decision should be
resolved using whatever EMA data is available at the next opportunity, even if the
model switch is no longer the active one. The reward will be inaccurate but this
is an edge case; do not over-engineer. Just resolve it and move on.

### 2.7 State Persistence

The bandit's `A` and `b` matrices must persist across runs so learning accumulates.
State is stored in `knowledge/bandit_state.json`. This file must:
- Be written atomically (`.tmp` → `os.replace()`)
- Be EXCLUDED from `experiments/run_reset.py` — the bandit should keep learning
  across runs; resetting it defeats the purpose
- Be KEYED by dataset_id — `{"pems_node2": {"A": {...}, "b": {...}, ...}, ...}` —
  so that switching datasets does not contaminate the learned policy
- Store `total_decisions` and `total_updates` counters alongside A and b

Read `experiments/run_reset.py` and confirm it does not currently touch
`bandit_state.json`. If it does, remove that deletion (and document why in a
comment). If it does not, add a comment in `run_reset.py` explaining that
`bandit_state.json` is intentionally preserved:
```python
# bandit_state.json is intentionally NOT reset between runs.
# The LinUCB bandit (S7) accumulates learning across runs.
# To reset the bandit, delete knowledge/bandit_state.json manually.
```

### 2.8 Alpha (Exploration Parameter)

`alpha` is the exploration-exploitation trade-off parameter. Higher alpha = more
exploration. It must be configurable via `thresholds.json["bandit_alpha"]` with
a default of `1.0`. No decay schedule in this implementation — fixed alpha,
documented in the paper as a hyperparameter. Add a note in `DECISIONS.md` (new
D-entry) recording this choice and the rationale (small sample count, fixed alpha
sufficient for LinUCB's regret guarantees, tuned via sensitivity analysis at
alpha ∈ {0.1, 0.5, 1.0}).

---

## Section 3 — Implementation

Now implement. Work in this exact order. Run the relevant test after each step.

### Step 3.1 — Core algorithm in `core/planners/bandit.py`

Replace the stub entirely. The new file must:

**Class: `LinUCBBandit`**

Not a `Planner` subclass — this is the raw algorithm, separate from the planning
logic. The `BanditPlanner` class (step 3.2) wraps this.

```python
class LinUCBBandit:
    """
    LinUCB contextual bandit.

    Maintains per-arm A matrices and b vectors.
    State persists to knowledge/bandit_state.json keyed by dataset_id.
    Learning accumulates across runs (bandit_state.json is never reset
    by run_reset.py — see experiments/run_reset.py for the explicit comment).

    Thread safety: not thread-safe. The MAPE loop calls plan() and
    observe_outcome() from the same thread in manage.py — this is correct.
    Do not add locks.
    """
```

Required methods:
- `__init__(self, models: list[str], context_dim: int, alpha: float, w_acc: float,
  w_energy: float, state_path: str | Path)`
- `_init_arms(self)` — sets A[m] = np.eye(d), b[m] = np.zeros(d) for all models
- `_load_state(self, dataset_id: str)` — loads A, b, counters from JSON for this
  dataset_id; calls _init_arms() if no entry exists for this dataset_id
- `_save_state(self, dataset_id: str)` — atomic JSON write
- `select_action(self, context: np.ndarray, current_model: str,
  candidates: list[str]) -> str` — LinUCB arm selection; raises ValueError if
  candidates is empty; logs all UCB scores at DEBUG level
- `record_pending(self, context: np.ndarray, action: str, ema_before: float,
  energy_before: float, switch_cost_J: float) -> dict` — returns the pending
  record dict (caller is responsible for persisting it)
- `observe_outcome(self, context: np.ndarray, action: str, reward: float)` —
  updates A[action] and b[action]; increments total_updates; saves state
- `compute_reward(self, ema_before: float, ema_after: float, energy_before: float,
  energy_after: float, switch_cost_J: float) -> float` — see §2.5 formula
- `get_stats(self) -> dict` — returns total_decisions, total_updates, has_pending
  (checks bandit_pending.json existence), and per-arm theta L2 norm

**Function: `build_context(ctx: PlanningContext, models: list[str]) -> np.ndarray`**

Standalone function (not a method) implementing §2.2 exactly. Models list must
be passed in to ensure consistent ordering across calls.

**Function: `load_or_create_bandit(thresholds: dict, models: list[str],
knowledge_dir: str | Path) -> LinUCBBandit`**

Factory function. Reads `thresholds["bandit_alpha"]` (default 1.0),
`thresholds["beta"]` (for w_acc default), context_dim = 10 + 2*len(models),
state_path = knowledge_dir / "bandit_state.json". Constructs and returns a
`LinUCBBandit`. Called once at managed system startup in `manage.py`.

**No module-level bandit instance.** The bandit is instantiated by `manage.py`
and passed to the planner. Do not use a global or module-level singleton.

### Step 3.2 — `BanditPlanner` in `core/planners/bandit.py`

Below `LinUCBBandit`, add the `BanditPlanner` class that implements the `Planner` ABC:

```python
class BanditPlanner(Planner):
    """
    S7 — LinUCB contextual bandit planner.

    Registered as "bandit" in the planner registry.

    Requires a LinUCBBandit instance to be injected at construction time.
    The bandit instance is owned by manage.py and shared across MAPE cycles.
    This class is a thin adapter between the Planner interface and LinUCBBandit.

    Pending decision lifecycle:
    - plan() calls bandit.select_action() and bandit.record_pending()
    - The pending dict is returned in Decision.metadata["bandit_pending"]
    - manage.py persists it to knowledge/bandit_pending.json
    - On the next MAPE cycle, manage.py checks for bandit_pending.json,
      resolves the reward, and calls bandit.observe_outcome()

    Decision fallback:
    If bandit.select_action() raises for any reason (empty candidates,
    numerical error in matrix inversion), fall back to the greedy switch
    (best EMA alternative) and log a WARNING. Do NOT raise.
    """
    name = "bandit"

    def __init__(self, bandit: LinUCBBandit, dataset_id: str):
        self.bandit = bandit
        self.dataset_id = dataset_id

    def plan(self, ctx: PlanningContext) -> Decision:
        ...
```

The `plan()` method must:
1. Call `_build_context(ctx)` to get the context vector
2. Build the candidates list (all models except current)
3. If candidates is empty, return `Decision(action="noop", target_model=None)`
4. Try `self.bandit.select_action(context, ctx.current_model, candidates)`;
   on any exception, log WARNING and fall back to the greedy alternative
   (highest EMA score among candidates)
5. Record the pending decision
6. Return `Decision(action="switch", target_model=selected_model,
   metadata={"bandit_pending": pending_record, "dataset_id": self.dataset_id})`
7. Call `self.bandit._total_decisions += 1` (or however the counter is tracked)

Register the planner: at the bottom of `bandit.py`, call
`register_planner("bandit", BanditPlanner)` or use whatever registration
mechanism the other planners use (read `base.py` to confirm).

**Important**: `BanditPlanner.__init__` takes a `bandit` argument. This means
`get_planner("bandit")` cannot be called with just a name — it needs a bandit
instance. Look at how the planner registry instantiates planners in `base.py`.
If the registry calls `PlnrClass()` with no arguments, you need to handle this.
Options:
- Modify the registry to support constructor kwargs (preferred if other planners
  may also need this)
- Use a factory function instead of a class in the registry
- Store the bandit instance at the module level in `bandit.py` and have
  `BanditPlanner.__init__` read it from there

Read `base.py` and `dispatch_plan()` to understand the instantiation pattern,
then choose the approach that requires the fewest changes to existing code.
Document your choice in `DECISIONS.md`.

### Step 3.3 — Wire manage.py (regression)

Read `managed_system_regression/mape_logic/manage.py` fully before touching it.

Add the bandit lifecycle to the MAPE loop. The manage.py changes are:

**At startup** (before the loop begins):
```python
# After loading thresholds and before the MAPE loop:
from core.planners.bandit import load_or_create_bandit
_bandit = None
if thresholds.get("planner") == "bandit":
    dataset_id = thresholds.get("dataset_id", "unknown")
    _bandit = load_or_create_bandit(thresholds, list_of_models, KNOWLEDGE_DIR)
    _bandit._load_state(dataset_id)
    logger.info(f"LinUCB bandit loaded for dataset '{dataset_id}': "
                f"{_bandit.total_decisions} prior decisions")
```

`list_of_models` must come from `thresholds.json["models"]` or the same source
that plan.py uses to enumerate available models. Read manage.py and plan.py to
find this.

**At the top of each MAPE cycle** (BEFORE calling monitor/analyse/plan):
```python
# Resolve any pending bandit reward
if _bandit is not None:
    pending_path = KNOWLEDGE_DIR / "bandit_pending.json"
    if pending_path.exists():
        _resolve_bandit_pending(_bandit, pending_path, mape_info, thresholds)
```

Add the helper function in manage.py (not in bandit.py — this is orchestration):
```python
def _resolve_bandit_pending(
    bandit: "LinUCBBandit",
    pending_path: Path,
    mape_info: dict,
    thresholds: dict,
) -> None:
    """
    Resolve a pending bandit reward if enough time has passed since the
    decision was made. Reads current EMA state, computes reward, updates
    bandit matrices, deletes the pending file.

    Called at the top of each MAPE cycle. Safe to call with a non-existent
    pending file (caller guards with exists() check).
    """
    try:
        with open(pending_path) as f:
            pending = json.load(f)

        elapsed = time.time() - pending["timestamp"]
        min_interval = thresholds.get("monitoring_interval_s", 5.0)

        if elapsed < min_interval:
            return  # too soon — wait another cycle

        # Read current state from mape_info
        arm = pending["action"]
        ema_after   = mape_info.get("ema_scores", {}).get(arm, 0.5)
        energy_after = mape_info.get("current_normalised_energy", 0.5)

        # Reconstruct context from the stored array
        context = np.array(pending["context"], dtype=np.float32)

        reward = bandit.compute_reward(
            ema_before=pending["ema_before"],
            ema_after=ema_after,
            energy_before=pending["energy_before"],
            energy_after=energy_after,
            switch_cost_J=pending["switch_cost_J"],
            w_acc=thresholds.get("w_acc", thresholds.get("beta", 0.95)),
            w_energy=thresholds.get("w_energy", 1.0 - thresholds.get("beta", 0.95)),
        )

        bandit.observe_outcome(context, arm, reward)

        pending_path.unlink()
        logger.info(
            f"[Bandit] Resolved pending: arm={arm}, reward={reward:.4f}, "
            f"ema_delta={ema_after - pending['ema_before']:.4f}"
        )
    except Exception as e:
        logger.warning(f"[Bandit] Failed to resolve pending reward: {e}. "
                       f"Deleting pending file to avoid stale state.")
        pending_path.unlink(missing_ok=True)
```

**Passing the bandit to the planner**: `dispatch_plan()` in `plan.py` currently
calls `get_planner(name)` and then calls `planner.plan(ctx)`. For the bandit,
the planner needs the `LinUCBBandit` instance. Handle this by modifying how the
planner is instantiated for the bandit case. The cleanest approach:

In `manage.py`, after creating `_bandit`, store it somewhere that `dispatch_plan()`
can access it. Options:
- Pass `_bandit` as an argument to `dispatch_plan()` (requires changing plan.py's
  function signature, but is clean and testable)
- Register the bandit instance in a module-level slot in `bandit.py` that
  `BanditPlanner.__init__` reads when called with no arguments

Read `dispatch_plan()` and choose the approach that requires the fewest changes.
Document your choice in `DECISIONS.md`.

**After a switch decision is executed** (in execute.py or back in manage.py):
If the decision came from the bandit (i.e. `decision.metadata.get("bandit_pending")`
is not None), persist the pending record to `knowledge/bandit_pending.json`:
```python
if decision.metadata and "bandit_pending" in decision.metadata:
    pending_path = KNOWLEDGE_DIR / "bandit_pending.json"
    tmp = str(pending_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(decision.metadata["bandit_pending"], f)
    os.replace(tmp, str(pending_path))
```

This should happen in `execute.py` after `model.csv` has been written (i.e.
after the switch is committed), not before. Check whether `execute.py` has
access to `decision.metadata` or whether you need to pass additional information.
Read execute.py before deciding.

### Step 3.4 — Dashboard and API changes

**`frontend/dashboard.html`**: The planner modal currently lists 5 planners
(harmone_original, greedy_switch, violation_aware, pareto, random_switch).
Add `bandit` as a sixth option with the label "LinUCB Bandit (S7)".

Read the dashboard HTML to find the `PLANNERS` array. Add:
```javascript
{ id: "bandit", label: "LinUCB Bandit (S7) — contextual bandit, learns across runs" }
```

**`app.py`**: The `/api/set-planner` endpoint validates the planner name against
a hardcoded allowlist. Read `DECISIONS.md` D11 which says the valid set is
`{harmone_original, greedy_switch, violation_aware, pareto, random_switch}` and
that `bandit` is excluded because it raises `NotImplementedError`. Now that
bandit is implemented, add `"bandit"` to the valid set.

Read `app.py` to find the validation list and update it. Also read
`tests/test_api_endpoints.py` — it tests that `bandit` returns HTTP 400. That
test must be updated to expect HTTP 200 now that the implementation exists.
Update the test to confirm the round-trip: set-planner to "bandit" returns 200,
subsequent GET of thresholds.json shows `"planner": "bandit"`.

**`thresholds.json` for both domains**: Add documentation comments (if the
format supports them — JSON does not; use a `_comment_bandit` key) and add:
- `"bandit_alpha": 1.0` — exploration parameter
- `"bandit_context_dim": 16` — computed at runtime but stored for validation
- `"dataset_id": ""` — must be filled in per-dataset; bandit state is keyed by this

Read the existing `thresholds.json` files to understand the conventions and add
these keys in a logical place (near the planner key).

### Step 3.5 — `run_reset.py` protection and `experiments/run_grid.py` awareness

Read `experiments/run_reset.py`. Confirm it does not delete `bandit_state.json`.
If it does, remove that line. Add the explicit comment (§2.7).

Read `experiments/run_grid.py`. The grid runs multiple (dataset × planner × seed)
combinations. For the bandit planner, cross-seed learning is intentional
(the bandit learns across seeds — that is the point). Confirm that `run_grid.py`
does not reset knowledge state between grid runs in a way that would delete
`bandit_state.json`. If it calls `run_reset()`, verify that `run_reset()` does
not touch the bandit file.

Also confirm that `run_grid.py` sets `dataset_id` in thresholds.json before each
run (so the bandit state is keyed correctly). If it does not, add that step:
```python
# Before running each (dataset, planner, seed) combination:
if planner == "bandit":
    thresholds["dataset_id"] = dataset_id
    # write thresholds.json
```

Read the actual grid code to see where configuration is set before each run.

---

## Section 4 — Tests

Write tests in `tests/test_bandit_planner.py`. Follow the patterns in
`tests/test_planners.py` exactly. Use the same mock patterns, fixture styles,
and assertion conventions already established.

The test file must cover:

**Group 1 — LinUCBBandit unit tests** (mock filesystem with `tmp_path`):

1. `test_bandit_init_creates_identity_matrices`: newly constructed bandit has
   A = I for each arm, b = 0 for each arm, total_decisions = 0.

2. `test_bandit_state_persists_across_instances`: call `observe_outcome` on
   a bandit, then construct a new bandit from the same state file. The new
   bandit's A and b matrices match the first. Tests cross-run learning.

3. `test_bandit_state_keyed_by_dataset_id`: bandit with dataset_id "pems" and
   dataset_id "uci" get independent A/b matrices in the same state file.

4. `test_select_action_excludes_current`: `select_action` never returns the
   current model regardless of EMA scores.

5. `test_select_action_pure_exploitation_when_alpha_zero`: with alpha=0.0,
   the planner always selects the arm with the highest expected reward (theta @ x),
   given a known A and b that produce a known theta.

6. `test_select_action_raises_on_empty_candidates`: `ValueError` when candidates
   is an empty list.

7. `test_compute_reward_positive_on_improvement`: when ema_after > ema_before
   and energy_after < energy_before, reward is positive.

8. `test_compute_reward_negative_on_regression`: when ema_after < ema_before
   and energy_after > energy_before, reward is negative.

9. `test_compute_reward_clipped`: extreme inputs produce reward in [-10, 10].

10. `test_observe_outcome_updates_matrices`: after calling `observe_outcome`
    with a known context and reward, verify A[arm] ≠ I (has been updated) and
    b[arm] ≠ 0. Verify the update satisfies the LinUCB update equations:
    `A_new = A_old + outer(x, x)` and `b_new = b_old + r * x`.

11. `test_observe_outcome_increments_counter`: total_updates increments by 1.

12. `test_get_stats_returns_required_keys`: result dict has keys
    `total_decisions`, `total_updates`, `has_pending`.

**Group 2 — BanditPlanner integration tests** (mock PlanningContext):

13. `test_bandit_planner_returns_switch_decision`: with 3 models, current="lstm",
    plan() returns a Decision with action="switch" and target_model in ["lr", "svr"].

14. `test_bandit_planner_returns_noop_when_no_candidates`: with 1 model, plan()
    returns Decision(action="noop").

15. `test_bandit_planner_decision_contains_pending_metadata`: the returned
    Decision has metadata["bandit_pending"] containing context, action, timestamp,
    ema_before, energy_before.

16. `test_bandit_planner_fallback_on_numerical_error`: if select_action raises
    (monkeypatch it to raise ValueError), plan() does NOT raise — it falls back
    to greedy selection and logs a WARNING.

17. `test_bandit_planner_registered_in_registry`: `get_planner("bandit")` does not
    raise; the returned object is a BanditPlanner or a callable that produces one.
    (Adapt to whatever registration pattern exists — the test should confirm the
    registry can route to the bandit.)

**Group 3 — pending resolution tests** (mock manage.py helpers):

18. `test_resolve_pending_too_soon_skips`: `_resolve_bandit_pending` called with
    a pending file timestamped 0.1s ago and monitoring_interval_s=5 does nothing
    (bandit.observe_outcome not called, pending file still exists).

19. `test_resolve_pending_resolves_after_interval`: pending file timestamped 10s
    ago with monitoring_interval_s=5 → observe_outcome called, pending file deleted.

20. `test_resolve_pending_handles_corrupt_json`: if bandit_pending.json contains
    invalid JSON, the file is deleted and no exception propagates to the caller.

**Group 4 — API/dashboard tests** (add to existing test files, not a new file):

In `tests/test_api_endpoints.py`, update the test that currently asserts that
"bandit" returns HTTP 400. Change it to assert HTTP 200. Add an assertion that
the thresholds.json is updated with `"planner": "bandit"`.

**Run the full suite after writing all tests:**
```bash
pytest tool/tests/ -q
```
All 295+ tests must pass (293 existing + new bandit tests). Report the count.

---

## Section 5 — Live Document Updates

After the implementation and tests pass, update every live document. Do this
in a single commit with message:
`docs: update live documents for S7 LinUCB bandit (DP3/DP8 closed)`.

### `DECISIONS_PENDING.md`

Mark DP3 and DP8 as DECIDED:

```markdown
## ✅ DECIDED — DP3 / DP8: LinUCB Bandit Planner (S7)

**Decided (August 2026):** LinUCB contextual bandit implemented in
`core/planners/bandit.py`. Key decisions:

- **Algorithm**: LinUCB with fixed alpha (not decaying). Fixed alpha chosen
  because sample count per run (~12 decisions) is too low for meaningful
  decay; fixed alpha provides LinUCB's theoretical regret guarantees.
- **Context**: 10 + 2×|models| dimensional vector. Features: violation type
  (one-hot), current EMA, normalised energy, drift signal, EMA slope,
  steps-since-switch, retrain count, VMR size, per-model EMA accuracy and
  energy.
- **Reward**: sustainability gain (weighted accuracy improvement + energy
  reduction) per joule of switching cost. Delayed by one monitoring interval.
- **State persistence**: `knowledge/bandit_state.json` keyed by dataset_id.
  NOT reset between runs. `run_reset.py` explicitly preserves this file.
- **Sensitivity analysis**: alpha ∈ {0.1, 0.5, 1.0} to be run in experiments.
- **CV domain**: bandit NOT wired in CV domain (DP11 still pending).
- **Dashboard**: "bandit" added to planner modal as option 6.
- **API**: `/api/set-planner` now accepts "bandit" (previously HTTP 400).
```

Also add entries for any new sub-decisions that came up during implementation.

### `DECISIONS.md`

Add a new decision entry at the bottom:

```markdown
## D12 — LinUCB Bandit Planner Design (S7)

**Decision:** [summarise your key design choices made during implementation —
particularly how you handled BanditPlanner constructor injection, how manage.py
receives the bandit instance, and how the pending-reward lifecycle is orchestrated]

**Rationale:** [explain why you chose the approach you chose over the alternatives]

**Alternatives considered:** [list the alternatives you considered for constructor
injection, pending resolution placement, etc.]
```

### `STATUS.md`

Update the planners section. Change the bandit row from:
```
| `bandit.py` | **S7 — LinUCB stub.** Raises `NotImplementedError`. |
```
to:
```
| `bandit.py` | **S7 — LinUCB contextual bandit.** Full implementation. State persists to `knowledge/bandit_state.json` keyed by `dataset_id`. Wired into regression `dispatch_plan()` and `manage.py` pending-reward lifecycle. NOT wired into CV domain (DP11). Dashboard modal enabled; `/api/set-planner` accepts "bandit". |
```

Also update the Known Gaps table:
- Remove the bandit from any "deferred" rows if present
- Add: `DP11 — CV bandit wiring | ⬜ PENDING | CV bandit comparison rows`
  (if not already there)

### `CHANGES_FROM_PAPER.md`

Add a new section after the most recent phase entry:

```markdown
## S7 — LinUCB Bandit Planner (Extension beyond both papers)

The HarmonE and Harmonica papers do not include a bandit-based planner.
The LinUCB implementation adds a contextual bandit (S7) that learns which
model to select given the current system state, replacing the ε-greedy
heuristic of the original planner with a statistically principled online
learning approach.

**Key differences from the published approach:**
- The bandit maintains per-arm A/b matrices that accumulate learning across
  all runs on a dataset (not reset between runs)
- Reward is delayed by one monitoring interval to allow the new model to
  accumulate real EMA data before updating the bandit
- State is keyed by dataset_id so different datasets have independent policies
- Alpha is a configurable hyperparameter (default 1.0); sensitivity analysis
  at {0.1, 0.5, 1.0} is planned

**Paper representation:** The bandit will be presented as a novel planner
contribution (Strategy 7) in the journal extension, compared against S1–S6
in the experiment grid on all six datasets.
```

### `files.md`

Update the `core/planners/bandit.py` entry to reflect the full implementation.
Also add `knowledge/bandit_state.json` and `knowledge/bandit_pending.json` to the
`managed_system_regression/knowledge/` section with descriptions.

If any new files were created during implementation (e.g. `tests/test_bandit_planner.py`),
add them to the tests section.

---

## Section 6 — Final Verification

Run the complete verification sequence. Report every result.

```bash
# 1. Full test suite
cd tool && pytest tests/ -q
# Expected: all prior tests green + new bandit tests green
# Record the total count

# 2. Confirm bandit is not reachable in CV (DP11 check)
grep -rn "bandit\|LinUCB" managed_system_cv/
# Expected: no results (bandit not wired in CV)

# 3. Confirm no pyRAPL
grep -rn "import pyRAPL\|from pyRAPL" .
# Expected: no results

# 4. Confirm no absolute paths
grep -rn '"/home/\|"D:/' core/ managed_system_regression/ managed_system_cv/
# Expected: no results

# 5. Confirm bandit_state.json excluded from run_reset
grep -n "bandit" experiments/run_reset.py
# Expected: only the preservation comment, no deletion code

# 6. Confirm bandit added to dashboard
grep -n "bandit\|LinUCB" frontend/dashboard.html
# Expected: the new PLANNERS entry

# 7. Confirm API allowlist updated
grep -n "bandit\|valid_planners\|VALID_PLANNERS" app.py
# Expected: "bandit" in the allowlist

# 8. Invariant checklist — verify these manually by code inspection:
# I1 (MAPE separation): bandit is called from plan.py only, not from
#    monitor.py, analyse.py, or execute.py. Pending resolution is in manage.py.
# I2 (scoring formula): bandit.compute_reward() does NOT redefine S_i.
#    It uses ema_before/after which are the outputs of core/scoring.py.
# I5 (no fabricated telemetry): bandit returns 0.0 reward on corrupt
#    pending file (test 20 covers this), never fabricates EMA values.
# I7 (config over code): bandit_alpha, w_acc, w_energy, dataset_id all
#    come from thresholds.json. The planner name "bandit" in thresholds.json
#    is the single source of truth for whether the bandit runs.
```

Write the results of this verification into `docs/checkpoints/CP_bandit.md`.
The checkpoint doc must include:
1. Reality check notes from Section 1
2. Any design choices that differed from what this prompt specified (and why)
3. Test count before and after
4. All grep outputs from the verification sequence
5. Invariant checklist I1–I8 PASS/FAIL (the bandit should not violate any)
6. A one-paragraph summary of what was built

---

## Section 7 — Push

```bash
# Ensure suite is green
pytest tool/tests/ -q

# Final status check
git status    # must be clean (all changes committed)
git log --oneline -10   # review commits

# Merge to working branch and tag
git checkout <working-branch>
git merge feat/bandit-planner --no-ff -m "feat: add S7 LinUCB bandit planner (DP3/DP8)"
git tag v2.1-bandit

# Push
git push origin <working-branch>
git push origin v2.1-bandit
```

If push is rejected or the remote is not configured: stop and report. Never
force-push.

---

## What Success Looks Like

At the end of this task:

1. `core/planners/bandit.py` contains a complete, working `LinUCBBandit` and
   `BanditPlanner` — no `NotImplementedError` anywhere in the file.

2. `pytest tool/tests/ -q` reports all tests green including all new bandit tests.

3. Setting `thresholds.json["planner"] = "bandit"` and starting the regression
   managed system causes the bandit planner to fire on violations, log UCB scores
   at DEBUG level, persist a pending file after each switch, and resolve it on
   the next cycle.

4. `knowledge/bandit_state.json` grows across runs (does not reset).

5. `experiments/run_reset.py` explicitly preserves `bandit_state.json` with a
   comment explaining why.

6. Dashboard modal shows "LinUCB Bandit (S7)" as a sixth planner option.

7. `/api/set-planner` with `{"planner": "bandit", "system": "reg"}` returns HTTP 200.

8. All four live documents (`DECISIONS_PENDING.md`, `DECISIONS.md`, `STATUS.md`,
   `CHANGES_FROM_PAPER.md`) and `files.md` are updated and internally consistent.

9. `docs/checkpoints/CP_bandit.md` exists with the full verification record.

10. The branch is merged and tagged on the remote.