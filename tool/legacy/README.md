# tool/legacy/

Legacy files preserved for reference. These are superseded by newer implementations
but retained to keep git history interpretable.

| File | Superseded by |
|---|---|
| `simulator.py` | `adapters/regression_csv.py` + `experiments/run_experiment.py` |
| `train_models.py` | `managed_system_regression/train.py` (live system) and `experiments/run_experiment.py::_train_regression_models()` (harness, inline-trains on first run). Original-repo copy retained as a hyperparameter reference — see `context/DECISIONS_PENDING.md` DP14. |
