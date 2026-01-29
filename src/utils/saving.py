"""Utilities for saving predictions and results."""
import os
import numpy as np
import pandas as pd
from typing import Union, Optional
from pathlib import Path


def _to_scalar(x) -> float:
    """Convert tensor or array to scalar float."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def save_pred_actual_txt(
    filepath: str,
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    index: Optional[Union[np.ndarray, list, range]] = None,
    header: str = "PREDICTION RESULT"
):
    """
    Save predictions and actual values to a text file.
    
    Args:
        filepath: Path to output file.
        y_true: True values.
        y_pred: Predicted values.
        index: Optional index values for each prediction.
        header: Header text for the file.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    
    assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{header}\n")
        f.write("=" * 80 + "\n")
        
        if index is not None:
            f.write(f"{'IDX':>6} | {'ACTUAL':>15} | {'PRED':>15} | {'ERROR':>15}\n")
            f.write("-" * 80 + "\n")
            for i, a, p in zip(index, y_true, y_pred):
                a = _to_scalar(a)
                p = _to_scalar(p)
                f.write(
                    f"{i:>6} | "
                    f"{a:>15,.2f} | "
                    f"{p:>15,.2f} | "
                    f"{(p - a):>15,.2f}\n"
                )
        else:
            f.write(f"{'ACTUAL':>15} | {'PRED':>15} | {'ERROR':>15}\n")
            f.write("-" * 60 + "\n")
            for a, p in zip(y_true, y_pred):
                a = _to_scalar(a)
                p = _to_scalar(p)
                f.write(
                    f"{a:>15,.2f} | "
                    f"{p:>15,.2f} | "
                    f"{(p - a):>15,.2f}\n"
                )
        
        f.write("=" * 80 + "\n")


def save_window_samples(
    filepath: str,
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 5,
    start_idx: int = 0,
    header: str = "SLIDING WINDOW SAMPLES"
):
    """
    Save sliding window samples to a text file for inspection.
    
    Args:
        filepath: Path to output file.
        X: Input windows of shape (N_samples, T, d_features).
        y: Target values of shape (N_samples,) or (N_samples, label_size).
        n_samples: Number of samples to save.
        start_idx: Starting index for samples.
        header: Header text for the file.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    N, T, d_emb = X.shape
    
    # Reshape y if needed
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    
    end_idx = min(start_idx + n_samples, N)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{header}\n")
        f.write("=" * 90 + "\n")
        f.write(f"X shape: {X.shape} | y shape: {y.shape}\n")
        f.write("=" * 90 + "\n")
        
        for i in range(start_idx, end_idx):
            f.write(f"\nSample #{i}\n")
            f.write("-" * 90 + "\n")
            
            f.write("INPUT WINDOW (time → features):\n")
            
            for t in range(T):
                step_idx = T - t - 1
                values = X[i, t]
                
                values_str = " | ".join(
                    f"f{j}: {values[j]:>10.6f}" for j in range(d_emb)
                )
                
                f.write(f"  t-{step_idx:>2} : {values_str}\n")
            
            f.write("\nLABEL:\n")
            f.write(f"  y = {y[i]:,.6f}\n")
        
        f.write("\n" + "=" * 90 + "\n")


def save_monthly_forecast(
    test_results_monthly: pd.DataFrame,
    output_path: str = "outputs/forecast_on_month_start.csv"
):
    """
    Save monthly forecast results (30-day windows starting from day 1 of month).
    
    Args:
        test_results_monthly: DataFrame with columns ['predicted', 'actual', 'category', 'date'].
        output_path: Path to save the CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure date column is datetime for consistent formatting
    if 'date' in test_results_monthly.columns:
        test_results_monthly = test_results_monthly.copy()
        test_results_monthly['date'] = pd.to_datetime(test_results_monthly['date'])
    
    # Save to CSV
    test_results_monthly.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Saved monthly forecast to: {output_path}")
    print(f"  - Total rows: {len(test_results_monthly)}")
    print(f"  - Columns: {list(test_results_monthly.columns)}")


def export_chunk_windows(
    chunk_id: int,
    preds: np.ndarray,        # (N, horizon)
    y_true: np.ndarray,       # (N, horizon)
    cats: np.ndarray,         # (N,)
    window_dates: np.ndarray, # (N,)
    input_size: int,
    horizon: int,
    output_dir: str = "debug_windows"
):
    """
    Export all windows of a chunk to a CSV file (1 row = 1 window).
    """

    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for i in range(len(window_dates)):
        t0 = pd.Timestamp(window_dates[i])

        row = {
            "chunk_id": chunk_id,
            "category": int(cats[i]),
            "window_start_date": t0.date(),

            "input_start_date": (t0 - pd.Timedelta(days=input_size)).date(),
            "input_end_date": (t0 - pd.Timedelta(days=1)).date(),

            "horizon_start_date": t0.date(),
            "horizon_end_date": (t0 + pd.Timedelta(days=horizon - 1)).date(),

            "y_true": y_true[i].tolist(),
            "y_pred": preds[i].tolist(),
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)

    filepath = os.path.join(
        output_dir,
        f"chunk_{chunk_id:02d}_windows.csv"
    )
    df_out.to_csv(filepath, index=False)

    print(
        f"       📄 Exported {len(df_out)} windows → {filepath}"
    )



