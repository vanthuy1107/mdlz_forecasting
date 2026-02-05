"""
Clean, consistent metrics for demand / volume forecasting.

Design principles:
- Aggregate-first metrics (volume-weighted)
- No per-row averaging for business KPIs
- Safe handling of zeros
- Clear separation between pointwise errors and business accuracy

All functions accept numpy arrays or pandas Series.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    return np.asarray(x)

def _first_step(y):
    """
    If horizon > 1, return y[..., 0].
    Works for scalars, 1D, and 2D arrays.
    """
    y = _to_numpy(y)

    if y.ndim >= 2:
        return y[..., 0]

    return y


# def sfa_daily_accuracy(
#     y_true,
#     y_pred,
#     baseline,
#     p=0.1,
#     min_eps=5.0
# ):
#     """
#     Baseline-aware SFA daily accuracy.
    
#     Near-zero is defined relative to baseline:
#         |y_true| <= max(p * |baseline|, min_eps)
#     """

#     # y_true = _first_step(y_true).astype(float)
#     # y_pred = _first_step(y_pred).astype(float)
#     # baseline = _first_step(baseline).astype(float)

#     acc = np.zeros_like(y_true, dtype=float)

#     # dynamic near-zero threshold
#     dyn_eps = np.maximum(np.abs(baseline) * p, min_eps)

#     near_zero = np.abs(y_true) <= dyn_eps
#     non_zero = ~near_zero

#     # standard SFA
#     acc[non_zero] = 1.0 - np.abs(y_pred[non_zero] - y_true[non_zero]) / y_true[non_zero]

#     # zero-demand logic (relative)
#     acc[near_zero] = (np.abs(y_pred[near_zero]) <= dyn_eps[near_zero]).astype(float)

#     return np.clip(acc, 0.0, 1.0)

def sfa_daily_accuracy(
    y_true: float,
    y_pred: float,
    baseline: float,
    p: float = 0.1,
    eps: float = 1e-6,
):
    """
    Scalar SFA accuracy for 1-day-ahead forecast with baseline-aware zero handling.

    near-zero threshold = p% of baseline
    """

    y_true = float(y_true)
    y_pred = float(y_pred)
    baseline = max(float(baseline), eps)

    # dynamic near-zero threshold
    near_zero_thresh = p * baseline

    if abs(y_true) <= near_zero_thresh:
        # zero-demand logic
        return float(abs(y_pred) <= near_zero_thresh)
    else:
        acc = 1.0 - abs(y_pred - y_true) / max(abs(y_true), eps)
        return float(np.clip(acc, 0.0, 1.0))



def calculate_sfa_by_brand_and_date(
    df,
    actual_col="actual",
    forecast_col="predicted",
    brand_col="brand",
    date_col="date",
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["sfa_accuracy"] = sfa_daily_accuracy(
        df[actual_col].values,
        df[forecast_col].values
    )

    return (
        df
        .groupby([brand_col, date_col])["sfa_accuracy"]
        .mean()
        .reset_index()
    )

def calculate_sfa_by_brand_and_month(
    df: pd.DataFrame,
    brand_col: str,
    date_col: str,
    actual_col: str,
    forecast_col: str,
    baseline_col: str,
):
    df = df.copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df["year_month"] = df[date_col].dt.to_period("M")

    # ✅ DAILY SFA (row-wise, scalar-safe)
    df["daily_sfa"] = df.apply(
        lambda r: sfa_daily_accuracy(
            r[actual_col],
            r[forecast_col],
            r[baseline_col],
        ),
        axis=1,
    )

    # ✅ MONTHLY aggregation
    monthly = (
        df.groupby([brand_col, "year_month"])
        .apply(
            lambda g: np.average(
                g["daily_sfa"],
                weights=g[actual_col].abs() + 1e-6,
            )
        )
        .reset_index(name="monthly_sfa")
    )

    return monthly


# ---------------------------------------------------------------------
# High-level report (text-friendly)
# ---------------------------------------------------------------------

def generate_accuracy_report(
    df,
    actual_col="actual",
    forecast_col="predicted",
    baseline_col="baseline",
    brand_col="brand",
    date_col="date",
    brand_name_map=None,
    output_path=None,
):
    """
    Generate a clean text report using DAILY-AVERAGED SFA accuracy.
    """
    if brand_name_map is None:
        brand_name_map = {}

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # ------------------------------------------------------------------
    # Daily SFA (row-level)
    # ------------------------------------------------------------------
    df["daily_sfa"] = df.apply(
        lambda r: sfa_daily_accuracy(r[actual_col], r[forecast_col], r[baseline_col]),
        axis=1,
    )

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    monthly_sfa = calculate_sfa_by_brand_and_month(
        df,
        actual_col=actual_col,
        forecast_col=forecast_col,
        baseline_col=baseline_col,
        brand_col=brand_col,
        date_col=date_col,
    )

    # ------------------------------------------------------------------
    # Overall accuracy = avg over brands (each brand = avg over months)
    # ------------------------------------------------------------------
    brand_level_acc = (
        monthly_sfa
        .groupby(brand_col)["monthly_sfa"]
        .mean()
    )

    overall_acc = brand_level_acc.mean()

    # daily_sfa = calculate_sfa_by_brand_and_date(
    #     df,
    #     actual_col=actual_col,
    #     forecast_col=forecast_col,
    #     brand_col=brand_col,
    #     date_col=date_col,
    # )

    # ------------------------------------------------------------------
    # Report rendering
    # ------------------------------------------------------------------
    lines = []
    lines.append("=== FORECAST ACCURACY REPORT ===")
    lines.append(f"Overall SFA Accuracy (Monthly/Brand Avg): {overall_acc * 100:.2f}%")
    lines.append("")

    for brand, g_brand in monthly_sfa.groupby(brand_col):
        brand_name = brand_name_map.get(brand, str(brand))
        lines.append(f"--- Brand: {brand_name} ---")

        for _, row in g_brand.sort_values("year_month").iterrows():
            month = str(row["year_month"])   # Period → string
            acc = row["monthly_sfa"]

            g_month_raw = df[
                (df[brand_col] == brand) &
                (df[date_col].dt.to_period("M") == row["year_month"])
            ]

            lines.append(
                f"  {month}: "
                f"Accuracy={acc * 100:.1f}% | "
                f"Actual={g_month_raw[actual_col].sum():.0f} | "
                f"Forecast={g_month_raw[forecast_col].sum():.0f}"
            )

        lines.append("")

    report_str = "\n".join(lines)
    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"✓ Accuracy report saved to: {output_path}")

    return report_str




__all__ = [ 
    'generate_accuracy_report', 
]