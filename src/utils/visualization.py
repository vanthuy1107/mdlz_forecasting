"""Visualization utilities."""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path



def plot_monthly_forecast(
    df,
    brand,
    month_str,
    output_dir: str = "outputs/plots",
    show: bool = False,
    plot_baseline: bool = False,
    report_df: pd.DataFrame | None = None,
):

    # Filter data for this brand and month
    mask = (df['brand'] == brand) & (
        df['date'].dt.to_period('M').astype(str) == month_str
    )
    df_filtered = df[mask].sort_values('date')

    if len(df_filtered) == 0:
        print(f"    [WARNING] No data for brand={brand}, month={month_str}")
        return

    # Create output directory using brand name
    brand_output_dir = Path(output_dir) / brand
    brand_output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # TAKE FIRST HORIZON STEP ONLY
    # --------------------------------------------------
    y_true = df_filtered['actual'].values
    y_pred = df_filtered['predicted'].values
    dates = df_filtered['date'].values
    if plot_baseline:
        y_base = df_filtered['baseline'].values

    # Plot
    plt.figure(figsize=(14, 6))

    date_labels = [pd.Timestamp(d).strftime('%m-%d') for d in dates]
    x_pos = range(len(dates))

    plt.plot(
        x_pos, y_true,
        label="Actual",
        marker="o",
        linewidth=2,
        markersize=6,
        color="blue"
    )

    if plot_baseline:
        plt.plot(
            x_pos, y_base,
            label="Baseline",
            marker="s",
            linewidth=2,
            linestyle="--",
            markersize=5,
            color="gray"
        )

    plt.plot(
        x_pos, y_pred,
        label="Predicted",
        marker="x",
        linewidth=2,
        markersize=6,
        color="red"
    )

    plt.xticks(
        x_pos[::max(1, len(x_pos)//10)],
        date_labels[::max(1, len(x_pos)//10)],
        rotation=45
    )

    plt.title(f"Monthly Forecast - {brand} - {month_str}")
    plt.xlabel("Date")
    plt.ylabel("Volume (CBM)")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # --------------------------------------------------
    # STATS (t+1 ONLY)
    # --------------------------------------------------
    accuracy_text = "Accuracy: N/A"

    if report_df is not None:
        match = report_df[
            (report_df["brand"] == brand) &
            (report_df["month"] == month_str)
        ]

        if len(match) > 0:
            acc = 100 * float(match["accuracy"].iloc[0])
            accuracy_text = f"Accuracy: {acc:.1f}%"

    plt.text(
        0.5, 0.95,
        accuracy_text,
        transform=plt.gca().transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9
    )

    plt.tight_layout()

    filename = f"{brand}_{month_str}.png"
    filepath = brand_output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"    - Saved: {filepath}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_monthly_forecasts(
    df,
    output_dir: str = "outputs/plots",
    show: bool = False,
    plot_baseline: bool = False,
    report_df: pd.DataFrame | None = None
):
    """
    Generate monthly forecast plots:
        outputs/
            plots/
                BrandA/
                    BrandA_2025-01.png
                    BrandA_2025-02.png
                BrandB/
                    BrandB_2025-01.png
                    ...
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Get unique brands
    brands = sorted(df["brand"].unique())

    for brand in brands:

        # Get available months for this brand
        brand_df = df[df["brand"] == brand]
        months = (
            brand_df["date"]
            .dt.to_period("M")
            .astype(str)
            .unique()
        )

        print(f"\n[Brand] {brand} → {len(months)} months")

        for month_str in sorted(months):

            plot_monthly_forecast(
                df=df,
                brand=brand,
                month_str=month_str,
                output_dir=output_dir,
                show=show,
                plot_baseline=plot_baseline,
                report_df=report_df,
            )



def plot_monthly_history(
    df,
    brand,
    month_str,
    test_start,
    output_dir: str = "outputs/history_plots",
    show: bool = False,
):
    """
    Plot historical actuals only (before test_start)
    """

    test_start = pd.to_datetime(test_start)

    # Filter brand + month + before test_start
    mask = (
        (df["brand"] == brand) &
        (df["date"] < test_start) &
        (df["date"].dt.to_period("M").astype(str) == month_str)
    )

    df_filtered = df[mask].sort_values("date")

    if len(df_filtered) == 0:
        return

    # Create brand folder
    brand_output_dir = Path(output_dir) / brand
    brand_output_dir.mkdir(parents=True, exist_ok=True)

    y_true = df_filtered["actual"].values
    dates = df_filtered["date"].values

    plt.figure(figsize=(14, 6))

    date_labels = [pd.Timestamp(d).strftime('%m-%d') for d in dates]
    x_pos = range(len(dates))

    plt.plot(
        x_pos,
        y_true,
        label="Actual",
        marker="o",
        linewidth=2,
        markersize=5
    )

    plt.xticks(
        x_pos[::max(1, len(x_pos)//10)],
        date_labels[::max(1, len(x_pos)//10)],
        rotation=45
    )

    plt.title(f"Historical Actual - {brand} - {month_str}")
    plt.xlabel("Date")
    plt.ylabel("Volume (CBM)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f"{brand}_{month_str}.png"
    filepath = brand_output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    print(f"    - Saved history: {filepath}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_monthly_history(
    df,
    test_start,
    output_dir: str = "outputs/history_plots",
    show: bool = False,
):
    """
    Generate monthly historical plots (actual only) before test_start.

    outputs/
        history_plots/
            BrandA/
                BrandA_2023-01.png
                BrandA_2023-02.png
            BrandB/
                ...
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    test_start = pd.to_datetime(test_start)

    # Keep only history
    df = df[df["date"] < test_start]

    brands = sorted(df["brand"].unique())

    for brand in brands:

        brand_df = df[df["brand"] == brand]

        months = (
            brand_df["date"]
            .dt.to_period("M")
            .astype(str)
            .unique()
        )

        print(f"\n[History Brand] {brand} → {len(months)} months")

        for month_str in sorted(months):

            plot_monthly_history(
                df=df,
                brand=brand,
                month_str=month_str,
                test_start=test_start,
                output_dir=output_dir,
                show=show,
            )


