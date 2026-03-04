"""
Train orchestrator: reads config.yaml major_categories and config_control.json.
Runs mvp_train.py once per active run × category, applying each run's overrides.
All runs write to the same dir (outputs/spike-anchored/{category}/models/); each run
appends a record with run_name to training_history.json. metadata_spike-anchored.json
is the result of the last run.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from config import load_config


def load_control(control_path: str):
    """
    Load config_control.json. Supports:
    - { "runs": [ {...}, ... ] }
    - [ {...}, ... ]  (top-level array of runs)
    Returns list of run dicts.
    """
    path = Path(control_path)
    if not path.exists():
        raise FileNotFoundError(f"Control config not found: {control_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("runs", [])


def apply_control_to_config(cfg, run_spec: dict, category: str) -> None:
    """
    Apply one run's overrides to a loaded Config (base + config_{category}.yaml).
    run_spec: one element from config_control.json "runs" (model_params, train_params, etc.).
    """
    # model_params -> model.*
    mp = run_spec.get("model_params") or {}
    if mp.get("hidden_size") is not None:
        cfg.set("model.hidden_size", int(mp["hidden_size"]))
    if mp.get("n_layers") is not None:
        cfg.set("model.n_layers", int(mp["n_layers"]))
    if mp.get("dropout") is not None:
        cfg.set("model.dropout_prob", float(mp["dropout"]))
    if mp.get("emb_dim") is not None:
        cfg.set("model.cat_emb_dim", int(mp["emb_dim"]))

    # train_params -> training.*
    tp = run_spec.get("train_params") or {}
    if tp.get("lr") is not None:
        cfg.set("training.learning_rate", float(tp["lr"]))
    if tp.get("epochs") is not None:
        cfg.set("training.epochs", int(tp["epochs"]))
    if tp.get("batch_size") is not None:
        cfg.set("training.batch_size", int(tp["batch_size"]))
    if tp.get("weight_decay") is not None:
        cfg.set("training.weight_decay", float(tp["weight_decay"]))

    # penalty_settings -> category_specific_params[category].*
    ps = run_spec.get("penalty_settings") or {}
    if ps:
        if ps.get("over_pred") is not None:
            cfg.set(f"category_specific_params.{category}.over_pred_penalty", float(ps["over_pred"]))
        if ps.get("under_pred") is not None:
            cfg.set(f"category_specific_params.{category}.under_pred_penalty", float(ps["under_pred"]))
        if ps.get("monday_weight") is not None:
            cfg.set(f"category_specific_params.{category}.monday_loss_weight", float(ps["monday_weight"]))
        if ps.get("mean_error_weight") is not None:
            cfg.set(f"category_specific_params.{category}.mean_error_weight", float(ps["mean_error_weight"]))

    # data_scope -> window.* and data.feature_cols (exclude list)
    ds = run_spec.get("data_scope") or {}
    if ds.get("input_days") is not None:
        cfg.set("window.input_size", int(ds["input_days"]))
    if ds.get("output_days") is not None:
        cfg.set("window.horizon", int(ds["output_days"]))
    excluded = ds.get("excluded_features")
    if excluded is not None:
        feature_cols = cfg.data.get("feature_cols", [])
        if isinstance(feature_cols, list):
            feature_cols = [f for f in feature_cols if f not in excluded]
            cfg.set("data.feature_cols", feature_cols)

    # Ensure only this category is trained when using this merged config
    cfg.set("data.major_categories", [category])


def main():
    script_dir = Path(__file__).resolve().parent
    config_dir = script_dir / "config"
    base_config_path = config_dir / "config.yaml"
    control_path = config_dir / "config_control.json"

    if not base_config_path.exists():
        print(f"ERROR: Base config not found: {base_config_path}")
        sys.exit(1)

    # Load base config to read major_categories (config.yaml data.major_categories)
    base = load_config(config_path=str(base_config_path))
    major_categories = base.data.get("major_categories", [])
    if not major_categories:
        print("ERROR: data.major_categories is empty in config/config.yaml. Nothing to train.")
        sys.exit(1)

    runs = load_control(str(control_path))
    active_runs = [r for r in runs if r.get("active", True)]
    if not active_runs:
        print("ERROR: No active runs in config_control.json (set \"active\": true for at least one run).")
        sys.exit(1)

    print(f"Loaded config_control.json from {control_path}")
    print(f"Runs: {len(runs)} total, {len(active_runs)} active")
    for r in active_runs:
        print(f"  - {r.get('run_name', 'unnamed')}")
    print(f"Major categories: {major_categories}")

    mvp_train_script = script_dir / "mvp_train.py"
    if not mvp_train_script.exists():
        print(f"ERROR: mvp_train.py not found: {mvp_train_script}")
        sys.exit(1)

    run_idx = 0
    for run_spec in active_runs:
        run_name = run_spec.get("run_name") or f"run_{run_idx}"
        run_idx += 1

        for category in major_categories:
            print("\n" + "=" * 60)
            print(f"Orchestrator: run '{run_name}' | category '{category}'")
            print("=" * 60)

            # Load base + config_{CATEGORY}.yaml
            cfg = load_config(config_path=str(base_config_path), category=category)
            apply_control_to_config(cfg, run_spec, category)

            # Single output dir per category (no run_name subdir); run_name is recorded in training_history.json
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, dir=str(script_dir)
            ) as f:
                temp_path = f.name
            try:
                cfg.save(temp_path)
                cmd = [sys.executable, str(mvp_train_script), "--config", temp_path, "--run-name", run_name]
                ret = subprocess.run(cmd, cwd=str(script_dir))
                if ret.returncode != 0:
                    print(f"mvp_train.py exited with code {ret.returncode} for run={run_name} category={category}")
                    sys.exit(ret.returncode)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    print("\n" + "=" * 60)
    print("Orchestrator: all active runs finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
