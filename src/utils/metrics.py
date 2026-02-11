"""
Clean, vectorized metrics for demand / volume forecasting.

Design principles:
- Baseline-aware SFA
- Zero-demand safe
- Volume-weighted monthly aggregation
- Brand-balanced overall score
- Fully vectorized (no slow row-wise apply)
"""

import numpy as np
import pandas as pd
import os

# ============================================================
# 1️⃣ Vectorized Daily SFA
# ============================================================

def compute_daily_sfa(
    y_true,
    y_pred,
    baseline,
    p: float = 0.1,
    eps: float = 1e-6,
):
    """
    Vectorized SFA accuracy.

    near-zero threshold = p% of baseline
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    baseline = np.maximum(np.asarray(baseline, dtype=float), eps)

    near_zero_thresh = p * baseline

    # Zero-demand mask
    zero_mask = np.abs(y_true) <= near_zero_thresh

    # Case 1: near-zero demand
    zero_acc = (np.abs(y_pred) <= near_zero_thresh).astype(float)

    # Case 2: normal demand
    normal_acc = 1.0 - np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    normal_acc = np.clip(normal_acc, 0.0, 1.0)

    return np.where(zero_mask, zero_acc, normal_acc)


# ============================================================
# 2️⃣ Monthly SFA (Volume Weighted)
# ============================================================

def calculate_monthly_sfa(
    df: pd.DataFrame,
    brand_col: str,
    date_col: str,
    actual_col: str,
    forecast_col: str,
    baseline_col: str,
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Daily SFA
    df["daily_sfa"] = compute_daily_sfa(
        df[actual_col],
        df[forecast_col],
        df[baseline_col],
    )

    # Year-Month period
    df["year_month"] = df[date_col].dt.to_period("M")

    # Volume-weighted monthly SFA
    monthly = (
        df.groupby([brand_col, "year_month"])
        .apply(
            lambda g: np.average(
                g["daily_sfa"],
                weights=np.abs(g[actual_col]) + 1e-6,
            )
        )
        .reset_index(name="monthly_sfa")
    )

    return monthly


# ============================================================
# 3️⃣ Overall Business Accuracy
# ============================================================

def calculate_overall_sfa(monthly_sfa_df, brand_col):
    """
    Brand-balanced overall SFA.
    Each brand contributes equally.
    """

    brand_level = (
        monthly_sfa_df
        .groupby(brand_col)["monthly_sfa"]
        .mean()
    )

    return brand_level.mean()


# ============================================================
# 4️⃣ Full Text Report
# ============================================================

def generate_accuracy_report(
    df: pd.DataFrame,
    actual_col="actual",
    forecast_col="predicted",
    baseline_col="baseline",
    brand_col="brand",
    date_col="date",
    output_path=None,
):
    """
    Returns:
        report_str (cumulative grouped report)
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[actual_col])

    # -----------------------------------
    # Compute monthly SFA (UNCHANGED)
    # -----------------------------------
    monthly_sfa = calculate_monthly_sfa(
        df,
        brand_col=brand_col,
        date_col=date_col,
        actual_col=actual_col,
        forecast_col=forecast_col,
        baseline_col=baseline_col,
    )

    overall_acc = calculate_overall_sfa(
        monthly_sfa,
        brand_col=brand_col,
    )

    # -----------------------------------
    # OPTIMIZATION: Precompute monthly sums ONCE
    # -----------------------------------
    df["year_month"] = df[date_col].dt.to_period("M")

    monthly_sum = (
        df.groupby([brand_col, "year_month"], as_index=False)
          .agg(
              actual_sum=(actual_col, "sum"),
              forecast_sum=(forecast_col, "sum"),
          )
    )

    # -----------------------------------
    # Merge SFA + sums
    # -----------------------------------
    report_df = (
        monthly_sfa.merge(
            monthly_sum,
            on=[brand_col, "year_month"],
            how="left"
        )
        .rename(columns={"monthly_sfa": "accuracy"})   
        .assign(month=lambda x: x["year_month"].astype(str))
        .sort_values([brand_col, "month"])
        .reset_index(drop=True)
    )


    # -----------------------------------
    # Build cumulative text report
    # -----------------------------------
    lines = []
    lines.append("=== FORECAST ACCURACY REPORT ===")
    lines.append(f"Overall SFA Accuracy: {overall_acc * 100:.2f}%")
    lines.append("")

    for brand, g_brand in report_df.groupby(brand_col, sort=True):

        lines.append(f"--- Brand: {brand} ---")

        for _, r in g_brand.iterrows():
            lines.append(
                f"  {r['month']}: "
                f"Accuracy={r['accuracy'] * 100:.1f}% | "
                f"Actual={r['actual_sum']:.0f} | "
                f"Forecast={r['forecast_sum']:.0f}"
            )

        lines.append("")

    report_str = "\n".join(lines)

    # -----------------------------------
    # Save (overwrite)
    # -----------------------------------
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_str)

    return report_df, report_str




__all__ = [
    "generate_accuracy_report",
    "calculate_monthly_sfa",
    "calculate_overall_sfa",
]
