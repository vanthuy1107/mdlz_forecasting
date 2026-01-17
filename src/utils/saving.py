"""Utilities for saving predictions and results."""
import numpy as np
from typing import Union, Optional


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

