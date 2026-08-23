Absolutely. I would treat this as the **implementation/design specification for S6 and S7**, not just a list of suggestions. The goal is that an agent can implement it without having to make architectural decisions on its own.

---

# HarmonE S6/S7 Planner Improvement Specification

## 1. Purpose

This document specifies the redesign of:

* **S6 — Pareto Planner**
* **S7 — Bandit Planner**

The objective is to make both planners properly reflect HarmonE's two sustainability objectives:

[
\boxed{\text{maximize prediction accuracy}}
]

and

[
\boxed{\text{minimize energy consumption}}
]

while preserving HarmonE's existing MAPE-K architecture, model-switching mechanism, drift/VMR handling, and existing S1–S5 baselines.

**S8 / MDP/RL is explicitly out of scope.**

The redesign should not modify the semantics of S1–S5.

---

# 2. Existing Architecture

All planners currently receive a `PlanningContext` containing:

* violation type;
* legacy combined EMA score;
* separate accuracy EMA;
* separate energy EMA;
* current model;
* available models;
* thresholds;
* drift information;
* history;
* current stream step. 

The planner returns one of:

```text
switch
replace
retrain
noop
```

and does not directly perform file I/O. 

This interface should remain unchanged unless a new field is genuinely necessary.

The original HarmonE monitor defines:

[
S_i=\beta A_i+(1-\beta)(1-\bar E_i)
]

where (A_i) is accuracy and (\bar E_i) is normalized energy, with an EMA maintained over the score. 

The new planners should **not discard this score**, because it remains useful for comparison with the original HarmonE planner. However, S6 and S7 must use the separate accuracy and energy signals for their actual optimization.

---

# 3. Design Principles

Both planners must obey these principles.

### P1 — Accuracy and energy are separate objectives

Do not collapse them into:

[
\beta A+(1-\beta)(1-E)
]

inside S6 or S7.

The combined HarmonE score remains available for baseline comparison, but the new planners explicitly model:

[
(A,-E)
]

as two separate quantities.

---

### P2 — `min_score` must not silently mean `min_accuracy`

The current S6 effectively uses:

```text
min_score
```

as the lower bound for:

```text
ema_accuracy
```

This is semantically ambiguous and must be removed.

Use explicit configuration names:

```text
min_accuracy
max_energy
energy_reference
```

instead.

S5 currently has the same ambiguity: it compares `ema_accuracy` against `min_score`. 

That ambiguity should **not** be carried into S6/S7.

---

### P3 — No planner may select a model based on stale/unavailable information without accounting for it

Every model needs:

* accuracy estimate;
* energy estimate;
* estimate freshness;
* optionally, uncertainty.

Missing observations must not silently become:

```text
accuracy = 0
energy = 1
```

and thereby distort the Pareto front or bandit.

---

### P4 — Switching has a cost

A switch is not free.

The planner must account for:

[
C_{switch}
]

where available.

However, switching cost must not corrupt the accuracy/energy objective measurements themselves.

---

### P5 — Do not switch unnecessarily

If the current model already satisfies the relevant constraints and there is no meaningful expected improvement, return:

```text
noop
```

This prevents oscillation.

---

# 4. S6 — Improved Pareto Planner

## 4.1 Objective

S6 should answer:

> **Given the currently observed operating conditions, which available model lies on the accuracy/energy Pareto frontier and best satisfies HarmonE's current sustainability requirements?**

It should **not** simply select the model closest to an arbitrary `(score, energy)` point.

The current implementation constructs a Pareto front correctly in principle, but then uses:

[
\max(
w_A\max(0,A_{min}-A_m),
w_E\max(0,E_m-E_{ref})
)
]

and ties are broken by lower energy. 

The one-sided clipping creates large regions where models become indistinguishable once they satisfy both targets.

That behavior must be removed.

---

# 5. S6 Data Model

For every model (m), obtain:

```text
A[m] = current accuracy estimate
E[m] = current normalized energy estimate
```

where:

[
A[m]\in[0,1]
]

and:

[
E[m]\ge0
]

with energy normalized consistently across all models.

The planner must use:

```text
ctx.ema_accuracy
ctx.ema_energy
```

as the primary existing estimates.

Do **not** use:

```text
ctx.ema_scores
```

for Pareto geometry.

---

# 6. S6 Step 1 — Validate Estimates

Before constructing the front:

### Required

A model is eligible only if it has valid:

```text
accuracy
energy
```

values.

Do not silently substitute:

```text
accuracy = 0
energy = 1
```

for missing observations.

If a model has never been observed, mark it:

```text
unknown
```

and handle it through the exploration policy described below.

For the initial implementation, the simplest policy is:

> An unobserved model is eligible for controlled exploration but must not be allowed to dominate observed models using fabricated estimates.

---

# 7. S6 Step 2 — Construct the Pareto Front

For models (a) and (b):

Model (a) dominates model (b) iff:

[
A_a\ge A_b
]

and

[
E_a\le E_b
]

with at least one strict inequality.

Therefore:

[
Pareto =
{m:\nexists n\text{ that dominates }m}
]

The current implementation's dominance test is already conceptually correct. 

Keep this portion.

---

# 8. S6 Step 3 — Apply Hard Constraints

The Pareto front should then be filtered according to the **current violation**.

This is important because Pareto optimality alone does not tell HarmonE what it should do.

## Energy violation

When:

```text
ctx.violation == "energy"
```

the planner is trying to reduce energy without sacrificing acceptable accuracy.

Define:

[
A_{min}
]

as the minimum acceptable accuracy.

Eligible models:

[
F_E=
{m\in Pareto:A_m\ge A_{min}}
]

If:

[
F_E\neq\emptyset
]

only these models are considered.

If no Pareto model satisfies the accuracy constraint, use the complete Pareto front and select the model that minimizes the accuracy violation while still reducing energy.

---

## Score/performance violation

When:

```text
ctx.violation == "score"
```

the planner must restore adequate performance while remaining energy-aware.

Define:

[
E_{max}
]

as the maximum acceptable normalized energy.

Eligible models:

[
F_A=
{m\in Pareto:E_m\le E_{max}}
]

If this set is non-empty, select from it.

If no Pareto model satisfies the energy constraint, use the full Pareto front and select the model with the smallest energy violation.

---

## Drift violation

Do **not** change existing drift handling.

S6 should continue to:

1. use VMR replacement if an appropriate version exists;
2. otherwise retrain.

The current implementation delegates drift to VMR/retraining. 

---

# 9. S6 Step 4 — Choose Within the Feasible Pareto Set

Do **not** use the current asymmetric clipped Chebyshev distance.

Instead use an **achievement-style reference-point scalarization** over the remaining Pareto candidates.

Reference-point scalarization is well established in multi-objective optimization, including Chebyshev/achievement scalarizing functions. These methods allow a preference/reference point to select a particular region of a Pareto front rather than simply collapsing the objectives into a weighted sum. ([Springer][1])

Define the desired reference:

[
R=(A_{ref},E_{ref})
]

where:

* higher accuracy is better;
* lower energy is better.

For each candidate:

[
d_A(m)=\frac{|A_m-A_{ref}|}{s_A}
]

[
d_E(m)=\frac{|E_m-E_{ref}|}{s_E}
]

where (s_A,s_E) are fixed normalization scales.

Then:

[
D(m)=
\max(
w_A d_A(m),
w_E d_E(m)
)
+
\epsilon
(w_A d_A(m)+w_E d_E(m))
]

with:

```text
epsilon > 0
```

small, e.g.:

```text
epsilon = 0.01
```

The chosen model is:

[
m^*=\arg\min_{m\in F}D(m)
]

The small augmentation term prevents excessive ties/weakly efficient choices, consistent with augmented Chebyshev scalarization. ([Springer][1])

---

# 10. S6 Reference Point

Do **not** define the reference point using:

```text
min_score
```

Use:

```text
pareto_accuracy_reference
pareto_energy_reference
```

Recommended defaults:

```text
pareto_accuracy_reference = min_accuracy
pareto_energy_reference = energy_reference
```

This means the planner tries to find a Pareto model close to the architect's desired operating point.

The reference point may be:

* achievable;
* partially achievable;
* completely unachievable.

That is acceptable; reference-point methods are specifically designed to express aspiration levels/preferences even when the desired point is not attainable. ([Springer][2])

---

# 11. S6 Step 5 — Current Model Hysteresis

The planner must not switch merely because another model is microscopically better.

Define:

[
D_{current}
]

and:

[
D_{candidate}
]

A switch occurs only if:

[
D_{candidate}+\delta < D_{current}
]

where:

```text
pareto_switch_margin = δ
```

For example:

```text
δ = 0.02
```

This creates hysteresis and prevents:

```text
A → B → A → B → A
```

when the estimates fluctuate slightly.

---

# 12. S6 Step 6 — Switching Cost

If switching energy is measurable, incorporate it as a **decision cost**, not as a permanent change to the model's inference-energy estimate.

For example:

[
D'(m)=D(m)+\lambda_C C_{switch}(m)
]

where (C_{switch}) is normalized.

Do not mix:

```text
model inference energy
```

with:

```text
model switching energy
```

because these represent different quantities.

---

# 13. S6 Required Decision Logic

The final S6 algorithm should therefore be:

```text
if no violation:
    NOOP

if drift violation:
    existing VMR/retrain logic

collect valid A[m], E[m]

construct Pareto front

if energy violation:
    filter Pareto front by A[m] >= min_accuracy
    if empty:
        choose Pareto model minimizing accuracy violation

if score violation:
    filter Pareto front by E[m] <= max_energy
    if empty:
        choose Pareto model minimizing energy violation

within resulting candidate set:
    compute augmented Chebyshev/reference-point distance

apply switching-cost penalty

if current model is sufficiently close to best candidate:
    NOOP
else:
    SWITCH
```

---

# 14. What S6 Should NOT Do

S6 must not:

* use `min_score` as accuracy;
* use `ema_scores` as its primary objective;
* treat accuracy and energy as one scalar;
* select the current model merely because it is Pareto-optimal;
* fabricate missing model measurements;
* ignore switching cost;
* switch for insignificant improvements;
* perform model learning/training itself;
* modify VMR behavior.

---

# 15. S7 — Constrained Contextual Bandit

## 15.1 Objective

S7 should answer a different question from S6:

> **Can HarmonE learn, from experience, which model performs best under the current operating context while respecting an energy constraint?**

This is a **contextual constrained bandit**, not an ordinary MAB.

The distinction matters because the outcome of a model depends on the current state of the data/system.

Contextual bandits explicitly model the reward conditional on the current context and chosen action. LinUCB is an established contextual-bandit method. ([arXiv][3])

More importantly, constrained contextual bandits explicitly address settings where actions produce both reward and resource consumption under a constraint. ([Proceedings of Machine Learning Research][4])

This is the correct theoretical direction for S7.

---

# 16. S7 Fundamental Change

The current S7 does:

```text
context
   ↓
LinUCB
   ↓
scalar reward
   ↓
model
```

Its reward is:

[
\frac{
w_A\Delta A+w_E\Delta E
}{
C_{switch}
}
]

and the current implementation even records:

```text
switch_cost_J = 0.0
```

at decision time.  

That formulation must be removed.

---

# 17. S7 Two Separate Prediction Models

For each model (m), maintain two predictors:

### Accuracy model

[
\hat A(x,m)
]

predicting expected accuracy.

### Energy model

[
\hat E(x,m)
]

predicting expected normalized energy.

The models must be learned independently.

Do **not** learn:

[
\hat R(x,m)
]

where reward has already combined accuracy and energy.

This preserves the two-dimensional structure.

---

# 18. S7 Context

The existing context already contains useful features:

* violation type;
* current accuracy;
* current energy;
* drift;
* EMA slope;
* steps since switch;
* retrain count;
* VMR size;
* per-model accuracy;
* per-model energy. 

Keep these, but restructure the representation so the model/action relationship is explicit.

For every candidate model (m), construct:

[
\phi(x,m)
]

containing:

```text
global context
+
current model information
+
candidate model's historical performance
+
candidate/current differences
```

At minimum:

```text
violation_score
violation_energy
current_accuracy
current_energy
current_model_one_hot
candidate_accuracy
candidate_energy
candidate_accuracy_minus_current
candidate_energy_minus_current
ema_slope
drift_signal
steps_since_switch
retrain_count
```

The important addition is:

> **candidate-specific context.**

The current implementation uses one context vector and feeds it independently to each arm. 

The improved version should explicitly represent:

[
(x,m)
]

rather than just (x).

---

# 19. S7 Accuracy Estimator

Use a linear contextual model initially:

[
\hat A(x,m)=\theta_A^T\phi(x,m)
]

with confidence:

[
U_A(x,m)=
\theta_A^T\phi(x,m)
+
\alpha_A
\sqrt{
\phi^T A_A^{-1}\phi
}
]

This preserves the computational simplicity of LinUCB while making it a genuine contextual estimator.

There is no need to introduce a neural model at this stage.

---

# 20. S7 Energy Estimator

Similarly:

[
\hat E(x,m)=\theta_E^T\phi(x,m)
]

but energy is a **cost**, so the planner should be conservative.

Use:

[
U_E(x,m)=
\theta_E^T\phi(x,m)
+
\alpha_E
\sqrt{
\phi^T A_E^{-1}\phi
}
]

Interpretation:

> `U_E` is an optimistic/high estimate of energy consumption.

This is intentional.

If the planner is uncertain whether a model consumes a lot of energy, it should not pretend the model is cheap.

---

# 21. S7 Energy Constraint

Define:

[
E_{budget}
]

as the maximum acceptable normalized energy.

For each candidate:

[
\text{energy_violation}(m)
==========================

\max(0,U_E(x,m)-E_{budget})
]

The planner should prefer models that satisfy:

[
U_E(x,m)\le E_{budget}.
]

---

# 22. S7 Energy Debt / Lyapunov Queue

The most important new component is an energy-debt variable.

Define:

[
Q_{t+1}
=======

\max
\left(
0,
Q_t+E_t-E_{budget}
\right)
]

where:

* (Q_t) = accumulated energy debt;
* (E_t) = observed energy;
* (E_{budget}) = target energy.

Interpretation:

```text
Q = 0
    energy budget is healthy

Q ↑
    system has been consuming too much energy

Q ↓
    system has accumulated energy credit
```

This is inspired by constrained contextual-bandit approaches using Lyapunov optimization, where constraint violations influence future decisions rather than treating each decision independently. LOE2D is an established example of this approach. ([Proceedings of Machine Learning Research][5])

This is particularly appropriate for HarmonE because the original system already has the conceptual notion of accumulated energy credit/debt.

---

# 23. S7 Selection Objective

For each candidate model define:

[
Score(m)
========

## U_A(x,m)

\lambda_t U_E(x,m)
]

where:

[
\lambda_t
=========

\lambda_0+\eta Q_t
]

with:

* (\lambda_0) = baseline energy penalty;
* (\eta) = energy-debt sensitivity.

Therefore:

### Low energy debt

[
Q_t\approx0
]

means:

```text
accuracy is strongly considered
```

### High energy debt

[
Q_t\gg0
]

means:

```text
energy becomes increasingly important
```

This gives the bandit a dynamic preference instead of a fixed:

```text
beta = 0.95
```

for every situation.

---

# 24. Hard Constraint vs Soft Objective

Use the following hierarchy.

### First:

Reject models whose **upper-confidence energy estimate** is catastrophically above the budget, unless no feasible model exists.

### Second:

Among feasible models:

[
\arg\max U_A(x,m)-\lambda_tU_E(x,m)
]

### Third:

If no model is predicted feasible:

choose the model minimizing:

[
U_E(x,m)-E_{budget}
]

subject to preserving the best available accuracy.

This prevents S7 from selecting an extremely expensive model merely because its expected accuracy is slightly better.

---

# 25. S7 Exploration

Exploration remains necessary because the bandit has to learn model/context relationships.

However, exploration must be **constraint-aware**.

Do not randomly explore every model.

Instead:

```text
if candidate is well-known:
    normal exploitation/exploration

if candidate has high uncertainty:
    exploration bonus increases

if candidate's upper energy confidence exceeds safety limit:
    exploration is suppressed
```

Thus exploration occurs primarily among models that are plausible candidates.

This is much more appropriate than unconstrained exploration in a sustainability-sensitive system.

---

# 26. S7 Reward / Update

The existing reward:

[
\frac{w_A\Delta A+w_E\Delta E}{C_{switch}}
]

must be removed.

Instead, after the observation window completes, record:

```text
context x
candidate model m
observed accuracy A
observed energy E
switch cost C_switch
```

Then update **both models independently**:

[
A_A \leftarrow A_A+\phi\phi^T
]

[
b_A \leftarrow b_A+A\phi
]

and:

[
A_E \leftarrow A_E+\phi\phi^T
]

[
b_E \leftarrow b_E+E\phi.
]

This gives the planner two learned functions:

```text
context + model → accuracy

context + model → energy
```

rather than one opaque reward function.

---

# 27. S7 Reward Attribution Must Be Fixed

The current implementation waits for one monitoring interval and then compares the selected model's EMA before and after the action. 

This is too ambiguous.

The pending decision must record:

```text
decision_step
decision_timestamp
context
selected_model
accuracy_before
energy_before
current_model
```

After the evaluation window:

```text
accuracy_after
energy_after
actual_switch_energy
observation_window
```

The update must use the metrics attributable to the interval in which the selected model was actually active.

Do not compare arbitrary global EMAs that may include observations from before the switch.

---

# 28. S7 Minimum Evaluation Window

A switch must be allowed to run for a minimum number of observations before its outcome is evaluated.

Define:

```text
bandit_min_observation_steps
```

For example:

```text
bandit_min_observation_steps = monitoring_interval_steps
```

The exact value should come from configuration.

Do not hard-code `40 seconds` as the learning horizon merely because the current implementation defaults to a 40-second monitoring interval. The current code does exactly this when resolving pending rewards. 

---

# 29. S7 Switching Cost

Switching cost should be logged separately:

[
C_{switch}
]

but **must not divide the reward by near-zero values**.

The current:

[
\max(C_{switch},10^{-9})
]

division is especially problematic when `switch_cost_J=0`. 

Instead:

* inference energy is modeled by (E(x,m));
* switching energy is logged separately;
* switching cost can be incorporated as a small decision penalty.

For example:

[
Score'(m)
=========

Score(m)-\lambda_C C_{switch}(m)
]

with a properly normalized (C_{switch}).

Never divide by (C_{switch}).

---

# 30. S7 Persistence

The current bandit persists:

```text
knowledge/bandit_state.json
```

by dataset ID. 

Retain persistence.

However, the state must now contain separate estimator parameters.

Conceptually:

```json
{
  "dataset_id": {
    "accuracy_model": {
      "A": "...",
      "b": "..."
    },
    "energy_model": {
      "A": "...",
      "b": "..."
    },
    "energy_debt": 0.0,
    "decisions": 0,
    "updates": 0
  }
}
```

Do not retain the old scalar-reward `A/b` state format without migration/versioning.

Add a state version:

```text
state_version: 2
```

so stale S7 state cannot silently contaminate the new planner.

---

# 31. S7 Dataset Isolation

State must remain isolated by dataset.

The existing implementation already does this by keying state on `dataset_id`. 

Retain this behavior.

A bandit trained on one dataset must not automatically learn model/context relationships from another dataset.

---

# 32. S7 Decision Procedure

The complete algorithm should be:

```text
1. Check violation.

2. If no violation:
       NOOP.

3. If drift:
       delegate to existing VMR/retrain logic.

4. Construct current context x.

5. Construct candidate feature vector φ(x,m)
   for every non-current model.

6. Predict:
       accuracy UCB
       energy upper confidence bound

7. Update energy debt Q.

8. Determine feasible candidates.

9. If feasible candidates exist:
       choose argmax(
           accuracy_UCB
           - λ(Q) * energy_UCB
           - switching_cost_penalty
       )

10. If no feasible candidate:
       choose candidate minimizing energy violation,
       subject to preserving acceptable accuracy.

11. Apply minimum-switch hysteresis.

12. If improvement is insufficient:
       NOOP.

13. Otherwise:
       SWITCH.

14. Record pending observation.

15. After the evaluation window:
       observe actual accuracy and energy.

16. Update accuracy estimator.

17. Update energy estimator.

18. Update energy debt.

19. Persist state.
```

---

# 33. S7 Must NOT Do

S7 must not:

* use a single scalar reward combining accuracy and energy;
* divide reward by switching energy;
* use zero switching cost as a denominator;
* update from arbitrary global EMA differences;
* randomly explore models that are clearly unsafe energetically;
* treat every context as equivalent;
* ignore the energy budget;
* reset learning every decision;
* use the old scalar LinUCB state format;
* become an MDP/RL implementation.

---

# 34. S6 vs S7 — They Must Remain Meaningfully Different

This distinction is important for the experiment.

## S6

**Explicit multi-objective optimization**

```text
Current observations
        ↓
Accuracy × Energy
        ↓
Pareto front
        ↓
Constraints/reference preference
        ↓
Model
```

S6 does not learn a predictive policy.

It directly reasons over the current estimated model trade-offs.

---

## S7

**Learned constrained contextual decision-making**

```text
Current context
        ↓
Learn Accuracy(x,m)
Learn Energy(x,m)
        ↓
Energy constraint/debt
        ↓
Contextual exploration
        ↓
Model
        ↓
Observed outcome
        ↓
Update models
```

S7 learns from previous decisions.

This is the key experimental distinction.

---

# 35. Expected Research Progression

The planner hierarchy then becomes:

| Stage  | Planner                           | Main idea                                          |
| ------ | --------------------------------- | -------------------------------------------------- |
| S1     | Naive                             | Fixed/no adaptation baseline                       |
| S2     | Random Switch                     | Random model selection                             |
| S3     | Greedy Switch                     | Historical best model                              |
| S4     | Original HarmonE                  | ε-greedy scalar HarmonE                            |
| S5     | Violation-Aware                   | Hand-designed constraint-aware selection           |
| **S6** | **Pareto**                        | Explicit accuracy-energy Pareto optimization       |
| **S7** | **Constrained Contextual Bandit** | Learn context-dependent accuracy/energy trade-offs |
| S8     | MDP/RL                            | **Deferred**                                       |

The existing S4 is the bug-fixed published HarmonE strategy: ε-greedy exploration over non-current models, with VMR/retraining for drift. 

This gives you a clean progression from:

[
\text{heuristic}
\rightarrow
\text{multi-objective optimization}
\rightarrow
\text{online learning}
]

rather than adding increasingly complicated algorithms without a clear conceptual progression.

---

# 36. Evaluation Requirements

Do **not** evaluate these planners only using the final combined HarmonE score.

Report at least:

### Accuracy

[
A_{mean}
]

and the task-specific accuracy metric.

### Energy

[
E_{total}
]

and:

[
E_{per\ inference}
]

where appropriate.

### Sustainability/quality trade-off

Report:

[
(A,E)
]

pairs and the resulting empirical Pareto frontier.

### Constraint performance

Report:

```text
accuracy violations
energy violations
total violation duration
maximum energy overshoot
```

### Adaptation

Report:

```text
number of switches
number of retrains
switch frequency
time to recover from violation
```

### Cost of adaptation

Report:

```text
switching energy
retraining energy
total adaptation energy
```

This is important because a planner that obtains a good steady-state point by switching models constantly is not necessarily better.

---

# 37. S6-Specific Evaluation

For S6 additionally report:

```text
Pareto-front size
fraction of decisions selecting Pareto models
dominated selections
constraint-feasible selections
NOOP-on-violation count
```

The last metric is particularly important because the current implementation can select the current model and return `noop` during a violation. 

After the redesign, a `NOOP` during a violation should have an explicit reason such as:

```text
current model remains within hysteresis margin
```

rather than:

```text
current model happens to be Pareto optimal
```

---

# 38. S7-Specific Evaluation

Report:

```text
bandit decisions
bandit updates
exploration decisions
exploitation decisions
accuracy prediction error
energy prediction error
constraint violations
energy debt over time
switches caused by high energy debt
switches caused by accuracy degradation
```

Also report the **learning curve**.

The key question is not simply:

> "Did S7 obtain a good final result?"

It is:

> **"Does S7 improve its decisions as it observes more contexts and outcomes?"**

---

# 39. Critical Ablations

The final experiment should include at least these ablations.

### S6

```text
S6-full
S6-without-hysteresis
S6-without-switch-cost
S6-without-hard-constraints
```

This establishes what is actually responsible for its performance.

### S7

```text
S7-full
S7-without-context
S7-without-energy-debt
S7-without-uncertainty
S7-scalar-reward baseline
```

The last one is essentially the current LinUCB formulation.

That lets you demonstrate that the improvement comes from:

1. contextual modeling;
2. separate objectives;
3. constraint handling;

rather than merely changing the implementation.

---

# 40. Recommended Configuration

Use explicit names.

```json
{
  "min_accuracy": 0.80,
  "max_energy": 0.70,

  "pareto_accuracy_reference": 0.80,
  "pareto_energy_reference": 0.70,

  "pareto_accuracy_weight": 0.5,
  "pareto_energy_weight": 0.5,

  "pareto_augmentation": 0.01,
  "pareto_switch_margin": 0.02,

  "bandit_accuracy_alpha": 1.0,
  "bandit_energy_alpha": 1.0,

  "bandit_lambda_0": 0.5,
  "bandit_energy_debt_eta": 1.0,

  "bandit_switch_penalty": 0.01,

  "bandit_min_observation_steps": 1
}
```

These are **starting values**, not experimentally justified final values. They should be tuned using the same protocol for every dataset/planner.

Most importantly, there should be **no hidden defaults that change the mathematical meaning of a parameter**.

---

# 41. Implementation Order

Do not implement S6 and S7 simultaneously.

## Phase 1 — Fix S6

1. Rename/remove ambiguous `min_score` usage.
2. Validate separate accuracy/energy estimates.
3. Keep existing dominance calculation.
4. Implement constraint filtering.
5. replace asymmetric clipped distance with augmented reference-point scalarization.
6. add hysteresis.
7. add switch-cost penalty.
8. add detailed decision logging.
9. test edge cases.

---

## Phase 2 — Validate S6

Test at least:

```text
all models dominated except one
all models Pareto optimal
current model Pareto optimal
current model violates energy
current model violates accuracy
no model satisfies accuracy constraint
no model satisfies energy constraint
missing model estimate
two identical models
one available model
zero energy
very small energy
equal Pareto distances
```

Only after these pass should S6 enter experiments.

---

# 42. Phase 3 — Replace S7

1. Remove scalar reward formulation.
2. Remove reward/switch-cost division.
3. Implement candidate-specific context.
4. Implement separate accuracy estimator.
5. Implement separate energy estimator.
6. implement confidence bounds.
7. implement energy debt.
8. implement constrained action selection.
9. implement proper observation windows.
10. implement separate model updates.
11. version persistent state.
12. implement hysteresis.
13. add diagnostics.

---

# 43. Phase 4 — Validate S7

Test:

```text
one model
two models
three models
unseen model
unseen context
high energy debt
zero energy debt
accuracy violation
energy violation
both objectives violated
no feasible model
switch cost > 0
switch cost = 0
missing observations
pending observation
restart with persisted state
new dataset
```

The planner must never crash or silently corrupt its learning state in these cases.

---

# 44. The Core Design Decision

The most important thing to preserve throughout the implementation is this:

### S6 asks:

> **"What is the best point on the currently observed accuracy-energy Pareto frontier?"**

### S7 asks:

> **"Given the current context, what model do I expect to provide the best accuracy while respecting my energy constraint, and how should my belief change after observing the result?"**

Those are genuinely different algorithms.

S6 is therefore the **explicit multi-objective optimizer**.

S7 is the **adaptive online learner**.

That gives the S1–S7 progression a coherent research story rather than seven arbitrary planners.

---

## 45. Research Basis

The proposed S6 design uses established Pareto/reference-point scalarization ideas; weighted Chebyshev and achievement scalarizing functions are standard ways of selecting preferred points from a Pareto set without reducing the entire problem to a simple weighted sum. ([Springer][1])

The proposed S7 is grounded in three established lines of work:

1. **Contextual bandits / LinUCB** — learning action quality conditional on context. ([arXiv][3])
2. **Multi-objective contextual bandits** — treating outcomes as a reward vector and reasoning about a context-dependent Pareto front rather than immediately collapsing objectives into one scalar. ([arXiv][6])
3. **Constrained contextual bandits / bandits with resource constraints** — learning decisions while respecting resource budgets. ([Proceedings of Machine Learning Research][4])
4. **Lyapunov-based constrained contextual bandits** — using accumulated constraint violation to influence future decisions, which motivates S7's energy-debt mechanism. ([Proceedings of Machine Learning Research][5])

So this isn't "let's throw a fancy algorithm at HarmonE." The progression is:

[
\boxed{
\text{Pareto optimization}
\rightarrow
\text{contextual learning}
\rightarrow
\text{constraint-aware contextual learning}
}
]

while **deliberately stopping before MDP/RL**.

That is the version of S6/S7 I would implement and evaluate.

[1]: https://link.springer.com/article/10.1007/s11047-018-9685-y?utm_source=chatgpt.com "A tutorial on multiobjective optimization: fundamentals and evolutionary methods | Natural Computing | Springer Nature Link"
[2]: https://link.springer.com/article/10.1007/s00291-018-0540-4?utm_source=chatgpt.com "Decision making in multiobjective optimization problems under uncertainty: balancing between robustness and quality | OR Spectrum | Springer Nature Link"
[3]: https://arxiv.org/abs/1003.0146?utm_source=chatgpt.com "A Contextual-Bandit Approach to Personalized News Article Recommendation"
[4]: https://proceedings.mlr.press/v49/agrawal16.html?utm_source=chatgpt.com "An efficient algorithm for contextual bandits with knapsacks, and an extension to concave objectives"
[5]: https://proceedings.mlr.press/v247/guo24a.html?utm_source=chatgpt.com "Stochastic Constrained Contextual Bandits via Lyapunov Optimization Based Estimation to Decision Framework"
[6]: https://arxiv.org/abs/1708.05655?utm_source=chatgpt.com "Multi-objective Contextual Multi-armed Bandit with a Dominant Objective"
