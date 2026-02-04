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


# ---------------------------------------------------------------------
# Pointwise error metrics (diagnostic)
# ---------------------------------------------------------------------

def mae(y_true, y_pred):
    """Mean Absolute Error (unweighted)."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Squared Error (unweighted)."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------
# Volume-weighted business metrics (recommended)
# ---------------------------------------------------------------------

def wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error.
    WAPE = sum(|y - yhat|) / sum(y)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    denom = np.sum(y_true)
    if denom == 0:
        return 0.0 if np.sum(y_pred) == 0 else 1.0

    return np.sum(np.abs(y_true - y_pred)) / denom


def sfa_daily_accuracy(y_true, y_pred, eps=1e-6):
    y_true = _to_numpy(y_true).astype(float)
    y_pred = _to_numpy(y_pred).astype(float)

    acc = np.zeros_like(y_true, dtype=float)

    near_zero = np.abs(y_true) <= eps
    non_zero = ~near_zero

    # normal SFA for non-zero actuals
    acc[non_zero] = 1.0 - np.abs(y_pred[non_zero] - y_true[non_zero]) / y_true[non_zero]

    # zero-demand handling
    acc[near_zero] = (np.abs(y_pred[near_zero]) <= eps).astype(float)

    acc = np.clip(acc, 0.0, 1.0)
    return acc


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
    df,
    actual_col="actual",
    forecast_col="predicted",
    brand_col="brand",
    date_col="date",
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M").astype(str)

    df["daily_sfa"] = sfa_daily_accuracy(
        df[actual_col].values,
        df[forecast_col].values
    )

    return (
        df
        .groupby([brand_col, "month"])["daily_sfa"]
        .mean()
        .reset_index(name="sfa_accuracy")
    )



# ---------------------------------------------------------------------
# High-level report (text-friendly)
# ---------------------------------------------------------------------

def generate_accuracy_report(
    df,
    actual_col="actual",
    forecast_col="predicted",
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
        lambda r: sfa_daily_accuracy(r[actual_col], r[forecast_col]),
        axis=1,
    )

    # Overall accuracy = average of all daily accuracies
    overall_acc = df["daily_sfa"].mean()
    overall_wape = wape(df[actual_col], df[forecast_col])

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    monthly_sfa = calculate_sfa_by_brand_and_month(
        df,
        actual_col=actual_col,
        forecast_col=forecast_col,
        brand_col=brand_col,
        date_col=date_col,
    )

    daily_sfa = calculate_sfa_by_brand_and_date(
        df,
        actual_col=actual_col,
        forecast_col=forecast_col,
        brand_col=brand_col,
        date_col=date_col,
    )

    # ------------------------------------------------------------------
    # Report rendering
    # ------------------------------------------------------------------
    lines = []
    lines.append("=== FORECAST ACCURACY REPORT ===")
    lines.append(f"Overall SFA Accuracy (Daily Avg): {overall_acc * 100:.2f}%")
    lines.append(f"Overall WAPE: {overall_wape * 100:.2f}%")
    lines.append("")

    for brand, g_brand in monthly_sfa.groupby(brand_col):
        brand_name = brand_name_map.get(brand, str(brand))
        lines.append(f"--- Brand: {brand_name} ---")

        for _, row in g_brand.sort_values("month").iterrows():
            month = row["month"]
            acc = row["sfa_accuracy"]

            g_month_raw = df[
                (df[brand_col] == brand) &
                (df[date_col].dt.to_period("M").astype(str) == month)
            ]

            w = wape(g_month_raw[actual_col], g_month_raw[forecast_col])

            lines.append(
                f"  {month}: "
                f"Accuracy={acc * 100:.1f}% | "
                f"WAPE={w * 100:.1f}% | "
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