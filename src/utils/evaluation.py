"""Evaluation metrics for forecasting models."""
import numpy as np
from typing import Tuple, Optional


def quantile_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.9,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, int, int, dict]:
    """
    Calculate quantile coverage metric for P90 predictions.
    
    Quantile coverage measures what percentage of actual values fall below
    the predicted quantile (P90). For a well-calibrated P90 model, we expect
    approximately 90% of actual values to be below the prediction.
    
    This is the primary success metric for quantile regression models,
    replacing traditional MSE/RMSE which target the mean.
    
    Args:
        y_true: Actual values array (can be 1D or 2D, will be flattened).
        y_pred: Predicted quantile values array (same shape as y_true, will be flattened).
        quantile: Target quantile (default: 0.9 for P90).
        mask: Optional boolean mask to exclude certain samples from calculation
              (e.g., off-season zeros). Shape should match flattened y_true/y_pred.
    
    Returns:
        Tuple of:
        - coverage_rate: Percentage of actual values below prediction (0.0 to 1.0)
        - covered_count: Number of samples where y_true < y_pred
        - total_count: Total number of samples (after masking if provided)
        - details: Dictionary with additional metrics:
            - coverage_rate: Same as first return value
            - target_coverage: Expected coverage (quantile value, e.g., 0.9)
            - coverage_error: Difference between actual and target coverage
            - under_coverage: Number of samples where y_true > y_pred (violations)
            - over_coverage: Number of samples where y_true < y_pred (covered)
            - mean_excess: Mean of (y_true - y_pred) for violations (when y_true > y_pred)
    
    Example:
        >>> y_true = np.array([100, 200, 150, 180])
        >>> y_pred = np.array([120, 190, 160, 170])  # P90 predictions
        >>> coverage, covered, total, details = quantile_coverage(y_true, y_pred, quantile=0.9)
        >>> print(f"Coverage: {coverage:.2%}")  # Should be close to 90%
    """
    # Flatten arrays for calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    if len(y_true_flat) != len(y_pred_flat):
        raise ValueError(
            f"Shape mismatch after flattening: y_true {y_true_flat.shape} vs y_pred {y_pred_flat.shape}"
        )
    
    # Apply mask if provided (e.g., exclude off-season zeros)
    if mask is not None:
        mask_flat = mask.flatten()
        if len(mask_flat) != len(y_true_flat):
            raise ValueError(
                f"Mask shape mismatch: mask {mask_flat.shape} vs y_true {y_true_flat.shape}"
            )
        y_true_flat = y_true_flat[mask_flat]
        y_pred_flat = y_pred_flat[mask_flat]
    
    if len(y_true_flat) == 0:
        return 0.0, 0, 0, {
            'coverage_rate': 0.0,
            'target_coverage': quantile,
            'coverage_error': -quantile,
            'under_coverage': 0,
            'over_coverage': 0,
            'mean_excess': 0.0
        }
    
    # Calculate coverage: percentage of samples where actual < prediction
    # For P90, we want ~90% of actual values to be below the prediction
    covered_mask = y_true_flat < y_pred_flat
    covered_count = np.sum(covered_mask)
    total_count = len(y_true_flat)
    coverage_rate = covered_count / total_count if total_count > 0 else 0.0
    
    # Calculate violations (under-coverage): samples where actual > prediction
    violation_mask = y_true_flat > y_pred_flat
    violation_count = np.sum(violation_mask)
    
    # Calculate mean excess for violations (how much actual exceeds prediction)
    if violation_count > 0:
        excess_values = y_true_flat[violation_mask] - y_pred_flat[violation_mask]
        mean_excess = np.mean(excess_values)
    else:
        mean_excess = 0.0
    
    # Coverage error: difference from target coverage
    coverage_error = coverage_rate - quantile
    
    details = {
        'coverage_rate': coverage_rate,
        'target_coverage': quantile,
        'coverage_error': coverage_error,
        'under_coverage': violation_count,
        'over_coverage': covered_count,
        'mean_excess': mean_excess
    }
    
    return coverage_rate, covered_count, total_count, details


def calculate_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: Optional[float] = None,
    mask: Optional[np.ndarray] = None
) -> dict:
    """
    Calculate comprehensive forecast evaluation metrics.
    
    For quantile regression (P90), quantile coverage is the primary metric.
    Traditional metrics (MAE, RMSE) are kept for reference.
    
    Args:
        y_true: Actual values array.
        y_pred: Predicted values array.
        quantile: Optional quantile value (e.g., 0.9 for P90). If provided, calculates quantile coverage.
        mask: Optional boolean mask to exclude samples from calculation.
    
    Returns:
        Dictionary with metrics:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - mse: Mean Squared Error
        - quantile_coverage: Quantile coverage rate (if quantile provided)
        - quantile_coverage_details: Detailed quantile coverage metrics (if quantile provided)
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
        y_true_flat = y_true_flat[mask_flat]
        y_pred_flat = y_pred_flat[mask_flat]
    
    # Calculate traditional metrics
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mse': mse
    }
    
    # Calculate quantile coverage if quantile is provided
    if quantile is not None:
        coverage_rate, covered_count, total_count, details = quantile_coverage(
            y_true, y_pred, quantile=quantile, mask=mask
        )
        metrics['quantile_coverage'] = coverage_rate
        metrics['quantile_coverage_details'] = details
    
    return metrics
