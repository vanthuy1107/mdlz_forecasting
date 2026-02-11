from src.forecast import (
    forecast_commit_month,
)
from src.utils import (
    month_range,
    generate_accuracy_report
)
from src.training import train_model_for_cutoff
from src.data import FeatureEngineer
import pandas as pd
import numpy as np

def run_monthly_walkforward(
    full_actuals: pd.DataFrame,
    config,
    device,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    verbose: bool = True,
):
    """
    Strict production-style monthly walk-forward simulation
    using FeatureEngineer as single source of truth.
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

    all_commit_results = []
    all_eval_results = []

    current_month = test_start.replace(day=1)

    while current_month < test_end:

        month_start, month_end = month_range(current_month)

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
            print(f"  Train period: {train_raw[time_col].min()} → {train_raw[time_col].max()}")
            print(f"  Train rows:   {len(train_raw):,}")
            print(f"  Brands:       {train_raw[brand_col].nunique()}")

        fe = FeatureEngineer(config)
        train_fe = fe.fit_transform(train_raw, month_start)

        if verbose:
            print(f"  Feature rows: {len(train_fe):,}")

        model = train_model_for_cutoff(
            data=train_fe,
            config=config,
            train_cutoff=month_start,
        )

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
            target_col="residual",
            input_size=config.window["input_size"],
            device=device,
            verbose=False,   # change to True if debugging deep
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
        # STEP 5 — Update simulation_data
        # ==================================================
        simulation_data = (
            pd.concat([simulation_data, actual_month], ignore_index=True)
            .sort_values(time_col)
        )

        if verbose:
            print(f"\n[SIMULATION DATA UPDATED]")
            print(f"  New history max date: {simulation_data[time_col].max()}")
            print(f"  Total rows in history: {len(simulation_data):,}")

        current_month += pd.DateOffset(months=1)


    return (
        pd.concat(all_commit_results, ignore_index=True),
        pd.concat(all_eval_results, ignore_index=True),
    )

