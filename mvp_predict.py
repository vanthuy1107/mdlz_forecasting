"""
Monthly Walk-Forward Deployment Simulation (Production Style)
"""

from pathlib import Path
import time

import pandas as pd
import torch

from config import load_config
from src.data import DataReader
from src.inference.deploy import run_monthly_walkforward
from src.utils import (
    generate_accuracy_report, 
    seed_everything, 
    plot_all_monthly_forecasts,
    plot_all_monthly_history,
    SEED
)

seed_everything(SEED)


def main():
    print("=" * 80)
    print("MONTHLY WALK-FORWARD DEPLOYMENT SIMULATION")
    print("=" * 80)

    global_start_time = time.perf_counter()
    # --------------------------------------------------
    # 1️⃣ Load config
    # --------------------------------------------------
    config = load_config()
    data_cfg = config.data
    infer_cfg = config.inference

    time_col = data_cfg["time_col"]
    brand_col = data_cfg["brand_col"]
    brand_id_col = data_cfg["brand_id_col"]
    target_col = data_cfg["target_col"]

    test_start = pd.to_datetime(infer_cfg["test_start"])
    test_end = pd.to_datetime(infer_cfg["test_end"]) + pd.Timedelta(days=1)

    device = torch.device(config.training["device"])

    # --------------------------------------------------
    # 2️⃣ Load RAW data only
    # --------------------------------------------------
    reader = DataReader(
        data_dir=data_cfg["data_dir"],
        file_pattern=data_cfg["file_pattern"],
    )

    raw_data = reader.load(years=data_cfg["years"])
    raw_data.columns = raw_data.columns.str.lower()
    raw_data = raw_data[~raw_data[brand_col].isin(["KINH DO CAKE", "LU"])]
    raw_data = raw_data[
        [brand_col, time_col, target_col]
    ]

    raw_data[time_col] = pd.to_datetime(raw_data[time_col])
    # Create stable brand_id mapping
    unique_brands = sorted(raw_data[brand_col].dropna().astype(str).unique())
    brand2id = {b: i for i, b in enumerate(unique_brands)}

    raw_data[brand_id_col] = (
        raw_data[brand_col]
        .astype(str)
        .map(brand2id)
    )

    # --------------------------------------------------
    # Run them once to get history plots and re-comment
    # -------------------------------------------------- 

    # history_df = raw_data[raw_data[time_col] < test_start]
    # history_df = history_df.rename(
    #     columns={
    #         brand_col: brand_col.lower(),
    #         time_col: time_col.lower(),
    #         target_col: "actual"
    # })

    # plot_all_monthly_history(
    #     df=history_df,   # must contain brand, date, actual
    #     test_start=test_start,
    #     show=False
    # )

    print(f"[READY] Raw data loaded: {raw_data.shape}")
    print(f"Date range: {raw_data[time_col].min().date()} → {raw_data[time_col].max().date()}")

    # --------------------------------------------------
    # 3️⃣ Run monthly walk-forward simulation
    # --------------------------------------------------
    sim_start = time.perf_counter()

    commit_results, eval_results = run_monthly_walkforward(
        full_actuals=raw_data,
        config=config,
        device=device,
        test_start=test_start,
        test_end=test_end,
    )
    eval_results = eval_results.rename(columns={"CUBE_OUT": "actual"})

    sim_end = time.perf_counter()
    print("\n" + "="*80)
    print(f"SIMULATION FINISHED")
    print(f"Total simulation time: {(sim_end - sim_start):.2f} seconds")
    print("="*80)

    # --------------------------------------------------
    # 4️⃣ Save results
    # --------------------------------------------------
    output_dir = Path("outputs/mvp_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    commit_path = output_dir / "commit_forecast_monthly.csv"
    commit_results.to_csv(commit_path, index=False)
    print(f"[SAVED] Commit forecast → {commit_path}")

    report_df, report_str = generate_accuracy_report(
        eval_results,
        output_path="outputs/final_accuracy_report.txt",
    )
    # --------------------------------------------------
    # 5. Plot charts
    # --------------------------------------------------
    plot_all_monthly_forecasts(
        df=eval_results,
        plot_baseline=True,
        report_df=report_df
    )

    total_time = time.perf_counter() - global_start_time
    print(f"\nTOTAL RUNTIME: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
