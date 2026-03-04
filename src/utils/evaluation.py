"""Evaluation metrics for forecasting models."""
import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict, Any


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


def calculate_segmented_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    day_of_week: Optional[np.ndarray] = None,
    spike_quantile: float = 0.8
) -> Dict[str, Any]:
    """
    Calculate segmented error metrics for spike-aware loss analysis.
    
    Segments:
    - Spike vs non-spike: Top spike_quantile (default 80th percentile) of y_true as spikes
    - Weekday (Mon/Wed/Fri): If day_of_week provided (0=Mon, 6=Sun)
    - Prediction bias: Over-prediction vs under-prediction frequency
    
    Args:
        y_true: Actual values (flattened or 2D).
        y_pred: Predicted values (same shape as y_true).
        day_of_week: Optional day of week (0-6) per sample. Monday=0, Wednesday=2, Friday=4.
        spike_quantile: Quantile threshold for spike (default 0.8 = top 20%).
    
    Returns:
        Dictionary with segmented MAE, RMSE, and bias metrics.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    n = len(y_true_flat)
    if n == 0:
        return {}
    
    error = y_pred_flat - y_true_flat
    
    # Spike vs non-spike (top 20% of actuals = spikes, matching spike_aware_mse)
    threshold = np.quantile(y_true_flat, spike_quantile)
    is_spike = y_true_flat > threshold
    
    spike_mae = float(np.mean(np.abs(error[is_spike]))) if np.sum(is_spike) > 0 else None
    spike_rmse = float(np.sqrt(np.mean(error[is_spike] ** 2))) if np.sum(is_spike) > 0 else None
    non_spike_mae = float(np.mean(np.abs(error[~is_spike]))) if np.sum(~is_spike) > 0 else None
    non_spike_rmse = float(np.sqrt(np.mean(error[~is_spike] ** 2))) if np.sum(~is_spike) > 0 else None
    
    result = {
        'spike_vs_non_spike': {
            'threshold_y_true': float(threshold),
            'spike_count': int(np.sum(is_spike)),
            'non_spike_count': int(np.sum(~is_spike)),
            'spike_mae': spike_mae,
            'spike_rmse': spike_rmse,
            'non_spike_mae': non_spike_mae,
            'non_spike_rmse': non_spike_rmse,
        },
        'prediction_bias': {
            'over_pred_count': int(np.sum(error > 0)),
            'under_pred_count': int(np.sum(error <= 0)),
            'over_pred_pct': float(np.mean(error > 0)) * 100,
            'under_pred_pct': float(np.mean(error <= 0)) * 100,
        }
    }
    
    # Weekday performance (Mon=0, Wed=2, Fri=4)
    if day_of_week is not None:
        dow_flat = np.asarray(day_of_week).flatten()
        if len(dow_flat) == n:
            mon_mask = dow_flat == 0
            wed_mask = dow_flat == 2
            fri_mask = dow_flat == 4
            hvd_mask = mon_mask | wed_mask | fri_mask  # high-volume days
            
            def _metrics(mask):
                if np.sum(mask) == 0:
                    return None, None
                err = error[mask]
                return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))
            
            mon_mae, mon_rmse = _metrics(mon_mask)
            wed_mae, wed_rmse = _metrics(wed_mask)
            fri_mae, fri_rmse = _metrics(fri_mask)
            hvd_mae, hvd_rmse = _metrics(hvd_mask)
            
            result['weekday_performance'] = {
                'monday_mae': mon_mae,
                'monday_rmse': mon_rmse,
                'monday_count': int(np.sum(mon_mask)),
                'wednesday_mae': wed_mae,
                'wednesday_rmse': wed_rmse,
                'wednesday_count': int(np.sum(wed_mask)),
                'friday_mae': fri_mae,
                'friday_rmse': fri_rmse,
                'friday_count': int(np.sum(fri_mask)),
                'high_volume_days_mae': hvd_mae,
                'high_volume_days_rmse': hvd_rmse,
            }
    
    return result


def residual_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute skewness and kurtosis of residuals (y_true - y_pred).
    Useful for detecting systematic bias in highly skewed data (e.g., DRY spikes).

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        mask: Optional boolean mask to exclude samples.

    Returns:
        Dict with residual_skewness, residual_kurtosis, residual_mean, residual_std.
    """
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    if len(y_true_flat) != len(y_pred_flat):
        raise ValueError("Shape mismatch between y_true and y_pred")
    if mask is not None:
        mask_flat = np.asarray(mask).flatten()
        y_true_flat = y_true_flat[mask_flat]
        y_pred_flat = y_pred_flat[mask_flat]
    residuals = y_true_flat - y_pred_flat
    if len(residuals) < 2:
        return {
            'residual_skewness': float('nan'),
            'residual_kurtosis': float('nan'),
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
        }
    return {
        'residual_skewness': float(stats.skew(residuals)),
        'residual_kurtosis': float(stats.kurtosis(residuals)),
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(np.std(residuals)),
    }


def percentile_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentiles: Optional[list] = None,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute percentile-based absolute errors (P90, P95) to understand
    worst-case failures during spike days.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        percentiles: List of percentiles (default [90, 95]).
        mask: Optional boolean mask to exclude samples.

    Returns:
        Dict with p90_abs_error, p95_abs_error, etc.
    """
    if percentiles is None:
        percentiles = [90, 95]
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    if len(y_true_flat) != len(y_pred_flat):
        raise ValueError("Shape mismatch between y_true and y_pred")
    if mask is not None:
        mask_flat = np.asarray(mask).flatten()
        y_true_flat = y_true_flat[mask_flat]
        y_pred_flat = y_pred_flat[mask_flat]
    abs_errors = np.abs(y_true_flat - y_pred_flat)
    if len(abs_errors) == 0:
        return {f'p{p}_abs_error': float('nan') for p in percentiles}
    result = {}
    for p in percentiles:
        result[f'p{p}_abs_error'] = float(np.percentile(abs_errors, p))
    return result


def spike_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: Optional[float] = None,
    spike_quantile: float = 0.8,
    mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Treat spikes as a classification problem: predict whether a day will exceed threshold.
    Returns Precision, Recall, and F1-Score for spike prediction.

    Args:
        y_true: Actual values.
        y_pred: Predicted values (used as spike prediction if > threshold).
        threshold: Explicit spike threshold. If None, use quantile of y_true.
        spike_quantile: Quantile of y_true to define threshold (default 0.8 = top 20%).
        mask: Optional boolean mask to exclude samples.

    Returns:
        Dict with precision, recall, f1_score, threshold, tp, fp, fn, tn, support.
    """
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    if len(y_true_flat) != len(y_pred_flat):
        raise ValueError("Shape mismatch between y_true and y_pred")
    if mask is not None:
        mask_flat = np.asarray(mask).flatten()
        y_true_flat = y_true_flat[mask_flat]
        y_pred_flat = y_pred_flat[mask_flat]
    if threshold is None:
        threshold = float(np.quantile(y_true_flat, spike_quantile))
    y_true_spike = (y_true_flat > threshold).astype(int)
    # Predicted spike: model predicts above threshold
    y_pred_spike = (y_pred_flat > threshold).astype(int)
    tp = int(np.sum((y_true_spike == 1) & (y_pred_spike == 1)))
    fp = int(np.sum((y_true_spike == 0) & (y_pred_spike == 1)))
    fn = int(np.sum((y_true_spike == 1) & (y_pred_spike == 0)))
    tn = int(np.sum((y_true_spike == 0) & (y_pred_spike == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'spike_precision': float(precision),
        'spike_recall': float(recall),
        'spike_f1_score': float(f1),
        'spike_threshold': float(threshold),
        'spike_tp': tp,
        'spike_fp': fp,
        'spike_fn': fn,
        'spike_tn': tn,
        'spike_support': int(np.sum(y_true_spike)),
    }
