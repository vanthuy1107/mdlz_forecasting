from src.forecast import (
    forecast_commit_month,
)
from src.utils import month_range
from src.training import train_model_for_cutoff
from src.data import FeatureEngineer
from src.training import Trainer          # for get_latest_walkforward_checkpoint
import pandas as pd
import numpy as np
from pathlib import Path

def run_monthly_walkforward(
    full_actuals: pd.DataFrame,
    config,
    device,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    checkpoint_dir: str = "checkpoints/walkforward",
    resume_from_existing: bool = True,
    verbose: bool = True,
):
    """
    Strict production-style monthly walk-forward simulation
    using FeatureEngineer as single source of truth.

    Warm-start / incremental training
    ----------------------------------
    Instead of re-training from random weights every month, each month
    *fine-tunes* the weights produced by the previous month:

    1. At the end of month N's training, the trainer saves
       ``<checkpoint_dir>/ckpt_<YYYY-MM>.pth``.
    2. At the start of month N+1, that file is loaded before any gradient
       steps are taken, so the model only needs to adapt to the new month's
       data rather than relearn everything from scratch.

    ``resume_from_existing=True`` (default) means that if you re-run the
    pipeline after an interruption the loop automatically picks up from the
    latest saved checkpoint instead of starting cold.

    Args:
        full_actuals (pd.DataFrame):    Complete ground-truth time-series.
        config:                         Config object.
        device (torch.device):          Compute device.
        test_start (pd.Timestamp):      First month to forecast.
        test_end (pd.Timestamp):        Exclusive upper bound (loop stops before this).
        checkpoint_dir (str):           Where to store monthly ``.pth`` files.
        resume_from_existing (bool):    If True, auto-detect the latest checkpoint
                                        already in *checkpoint_dir* and use it as
                                        the warm-start for the first iteration.
        verbose (bool):                 Print progress.
    """

    data_cfg = config.data
    time_col = data_cfg["time_col"]
    brand_col = data_cfg["brand_col"]
    brand_id_col = data_cfg["brand_id_col"]
    target_col = data_cfg["target_col"]
    feature_cols = data_cfg["feature_cols"]

    # --------------------------------------------------
    # Ground truth (never modified)
    # --------------------------------------------------
    full_actuals = full_actuals.sort_values(time_col).copy()

    # --------------------------------------------------
    # What production "knows"
    # --------------------------------------------------
    simulation_data = full_actuals[
        full_actuals[time_col] < test_start
    ].copy()

    # --------------------------------------------------
    # Checkpoint directory + warm-start seed
    # --------------------------------------------------
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Try to pick up where we left off if the caller allows it
    last_checkpoint_path = None
    if resume_from_existing:
        last_checkpoint_path = Trainer.get_latest_walkforward_checkpoint(checkpoint_dir)
        if last_checkpoint_path and verbose:
            print(f"🔁 Warm-starting from existing checkpoint: {last_checkpoint_path.name}")

    all_commit_results = []
    all_eval_results = []

    current_month = test_start.replace(day=1)
    last_train_fe = None

    while current_month < test_end:

        month_start, month_end = month_range(current_month)
        
        fe = FeatureEngineer(config)

        if verbose:
            print("\n" + "="*80)
            print(f"🚀 MONTH {month_start:%Y-%m}")
            print("="*80)

        # ==================================================
        # STEP 1 — Train
        # ==================================================
        train_raw = simulation_data.copy()

        if train_raw.empty:
            print("⚠ No training data")
            current_month += pd.DateOffset(months=1)
            continue

        if verbose:
            print(f"[TRAIN]")
            print(f"  Train period: {train_raw[time_col].min().date()} → {train_raw[time_col].max().date()}")
            print(f"  Train rows:   {len(train_raw):,}")
            print(f"  Brands:       {train_raw[brand_col].nunique()}")

        train_fe = fe.fit_transform(train_raw)
        last_train_fe = train_fe.copy()
          
        model = train_model_for_cutoff(
            data=train_fe,
            config=config,
            train_cutoff=month_start,
            checkpoint_path=last_checkpoint_path,
            checkpoint_save_dir=checkpoint_dir,
            month=month_start,
        )

        # Track the checkpoint just saved so the next month can load it
        last_checkpoint_path = Trainer.get_latest_walkforward_checkpoint(checkpoint_dir)

        model.to(device)
        model.eval()

        # ==================================================
        # STEP 2 — Forecast
        # ==================================================
        if verbose:
            print(f"\n[FORECAST]")
            print(f"  Forecast range: {month_start} → {month_end}")

        commit_df = forecast_commit_month(
            model=model,
            feature_engineer=fe,
            history=train_fe,
            forecast_start=month_start,
            forecast_end=month_end,
            time_col=time_col,
            brand_col=brand_col,
            brand_id_col=brand_id_col,
            feature_cols=feature_cols,
            input_size=config.window["input_size"],
            device=device,
            horizon=config.window["horizon"]
        )

        commit_df["commit_month"] = month_start
        all_commit_results.append(commit_df)

        if verbose:
            print(f"  Predictions generated: {len(commit_df):,}")

        # ==================================================
        # STEP 3 — Receive actuals
        # ==================================================
        actual_month = full_actuals[
            (full_actuals[time_col] >= month_start) &
            (full_actuals[time_col] <= month_end)
        ].copy()

        if actual_month.empty:
            print("⚠ No actuals for month")
            current_month += pd.DateOffset(months=1)
            continue

        if verbose:
            print(f"\n[ACTUALS RECEIVED]")
            print(f"  Actual rows: {len(actual_month):,}")

        # ==================================================
        # STEP 4 — Evaluate
        # ==================================================
        eval_df = commit_df.merge(
            actual_month,
            on=[time_col, brand_col],
            how="left"
        ).rename(columns={target_col: "actual"})
        all_eval_results.append(eval_df)

        # ==================================================
        # STEP 5 — Update simulation_data with actuals
        # ==================================================
        simulation_data = (
            pd.concat([simulation_data, actual_month], ignore_index=True)
            .sort_values([brand_col, time_col])
            .reset_index(drop=True)
        )
        
        if verbose:
            print(f"\n[SIMULATION DATA UPDATED]")
            print(f"  New history max date: {simulation_data[time_col].max()}")
            print(f"  Total rows in history: {len(simulation_data):,}")

        current_month += pd.DateOffset(months=1)

    # ==================================================
    # SAVE FINAL TRAIN_FE
    # ==================================================
    if last_train_fe is not None:
        output_df = (
            last_train_fe
            .sort_values([brand_col, time_col])
            .reset_index(drop=True)
        )[[brand_col] + [time_col] + [target_col] + feature_cols]

        output_df.to_csv("final_train_fe.csv", index=False)

        if verbose:
            print("\n💾 Final train_fe saved to final_train_fe.csv")

    return (
        pd.concat(all_commit_results, ignore_index=True),
        pd.concat(all_eval_results, ignore_index=True),
    )

