"""
core/config_keys.py — top-level dataset-config keys that some component reads
(2026-10-26).

`unread_keys(config)` lists the keys of a dataset config that nothing reads, so
dead keys (bandit_min_observation_steps, bandit_switch_margin, drift_reference
until 2026-10-26) are reported instead of silently looking like settings.
experiments/run_experiment._load_dataset_config warns on them. Documentation
keys are exempt: anything starting with "_" and provenance keys ending in
"_source".

When a component starts reading a new key, add it here;
tests/test_s5_credit.py checks that every key listed is referenced somewhere in
the code.
"""

from __future__ import annotations

READ_KEYS = frozenset({
    # data / adapter
    "adapter", "data_path", "data_root", "device", "domain", "image_dir", "manifest_csv",
    "name", "status", "stream_path", "task", "train_frac", "train_path", "train_window_rows",
    "val_frac", "value_column", "seq_length", "num_classes", "ignore_index", "models",
    # energy
    "E_M", "E_m", "E_ref", "energy_budget_rho", "energy_meter", "energy_metering",
    "energy_reference", "max_energy", "delta",
    # accuracy / scoring
    "accuracy_budget_rho", "accuracy_signal", "alpha", "beta", "gamma", "min_accuracy",
    "min_score", "proxy", "w_acc", "w_e",
    # cadence / harness
    "background_retrain", "max_batches_ahead", "monitor_interval", "prt_interval",
    "reload_model_per_step", "stream_delay_s", "switch_cooldown_s", "switch_hold_cycles",
    "switch_hold_mode",
    # drift / VMR
    "drift_accuracy_drop", "drift_confirmation", "drift_cooldown_rows", "drift_detector",
    "drift_window_size", "tau_drift", "embedding_dim", "embedding_model",
    "embedding_sample_every",
    # retraining / CV adaptation
    "retrain_regularisation", "retrain_tactic", "ema_head_start", "finetune_batch_size",
    "finetune_epochs", "finetune_lr", "finetune_min_images", "finetune_n_layers",
    "finetune_window_images", "pseudo_label_threshold", "tau_regression", "tent_batch_size",
    "tent_lr", "tent_steps",
    # planners
    "planner", "explore_prob", "staleness_window",
    "s5_switch_margin", "s5_drift_prefers_switch", "s5_credit_horizon", "s5_credit_cap",
    "pareto_augmentation", "pareto_switch_cost_lambda", "pareto_switch_margin",
    "bandit_alpha", "bandit_mu", "bandit_switch_cost",
})


def unread_keys(config: dict) -> list[str]:
    """Top-level keys of a dataset config that no component reads."""
    return sorted(k for k in config
                  if k not in READ_KEYS and not k.startswith("_") and not k.endswith("_source"))
