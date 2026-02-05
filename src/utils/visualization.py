"""Visualization utilities."""
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path


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

def plot_difference(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    save_path: str = "../outputs/prediction_vs_truth.png",
    show: bool = True
):
    """
    Plot comparison of true vs predicted values.
    
    Args:
        y_true: True values (real scale).
        y_pred: Predicted values (real scale).
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    plt.figure(figsize=(20, 6))
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # DEBUG: Check values before plotting
    if len(y_true) > 0:
        print(f"  [plot_difference DEBUG] y_true: min={y_true.min():.4f}, max={y_true.max():.4f}, mean={y_true.mean():.4f}, zeros={np.sum(y_true == 0)}/{len(y_true)}")
        print(f"  [plot_difference DEBUG] y_true first 5: {y_true[:5] if len(y_true) >= 5 else y_true}")
    
    times = range(len(y_true))
    
    plt.plot(times, y_true, label="True Outbound", marker="o")
    plt.plot(times, y_pred, label="Predicted Outbound", marker="x")
    
    plt.title("Outbound Prediction Daily")
    plt.xlabel("Day")
    plt.ylabel("Outbound Quantity")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create directory if needed
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_monthly_forecast(
    df,
    brand,
    brand_name,
    month_str,
    output_dir: str = "outputs/plots",
    show: bool = False,
    plot_baseline: bool = False,  
):

    # Filter data for this brand and month
    mask = (df['brand'] == brand) & (
        df['date'].dt.to_period('M').astype(str) == month_str
    )
    df_filtered = df[mask].sort_values('date')

    if len(df_filtered) == 0:
        print(f"    [WARNING] No data for brand={brand} ({brand_name}), month={month_str}")
        return

    # Create output directory using brand name
    brand_output_dir = Path(output_dir) / brand_name
    brand_output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # TAKE FIRST HORIZON STEP ONLY
    # --------------------------------------------------
    y_true = _first_step(df_filtered['actual'].values)
    y_pred = _first_step(df_filtered['predicted'].values)
    dates = df_filtered['date'].values
    if plot_baseline:
        y_base = _first_step(df_filtered['baseline'].values)

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

    plt.title(f"Monthly Forecast - {brand_name} - {month_str}")
    plt.xlabel("Date")
    plt.ylabel("Volume (CBM)")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # --------------------------------------------------
    # STATS (t+1 ONLY)
    # --------------------------------------------------
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    accuracy = (
        100 * (1 - np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())
        if np.abs(y_true).sum() > 0 else 0
    )

    stats_text = f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | Accuracy: {accuracy:.1f}%"
    plt.text(
        0.5, 0.95,
        stats_text,
        transform=plt.gca().transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9
    )

    plt.tight_layout()

    filename = f"{brand_name}_{month_str}.png"
    filepath = brand_output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"    - Saved: {filepath}")

    if show:
        plt.show()
    else:
        plt.close()


        
def plot_learning_curve(
    train_losses: Union[list, np.ndarray],
    val_losses: Union[list, np.ndarray],
    save_path: str = "../outputs/learning_curve.png",
    show: bool = True
):
    """
    Plot learning curve (train loss vs validation loss).
    
    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    train_losses = np.asarray(train_losses)
    val_losses = np.asarray(val_losses)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='x', label='Validation Loss')
    
    plt.title("Learning Curve (Train vs Validation Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create directory if needed
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()

