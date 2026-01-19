"""Prediction script for full year 2025 using MVP test model.

This script loads the trained model from mvp_test.py and makes predictions
for full year 2025 data, filtering to DRY category only.

Two modes are supported:
1. Teacher Forcing (Test Evaluation): Uses actual ground truth QTY values from 2025 as features
2. Recursive (Production Forecast): Uses model's own predictions as inputs for future dates
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List

from config import load_config
from src.data import (
    DataReader,
    ForecastDataset,
    slicing_window_category,
    encode_categories,
    add_holiday_features,
    add_temporal_features,
    aggregate_daily,
    apply_scaling,
    inverse_transform_scaling
)
from src.data.preprocessing import get_us_holidays, get_vietnam_holidays
from src.models import RNNWithCategory
from src.training import Trainer
from src.utils import plot_difference


def solar_to_lunar_date(solar_date: date) -> tuple:
    """
    Convert solar (Gregorian) date to lunar (Vietnamese) date approximation.
    
    This is a simplified approximation. For production, use a proper lunar calendar library.
    Tet (Lunar New Year) typically falls between Jan 20 - Feb 20 in solar calendar.
    
    Args:
        solar_date: Gregorian date.
    
    Returns:
        Tuple of (lunar_month, lunar_day) where lunar_month is 1-12.
    """
    # Simplified conversion: use solar month/day as approximation
    # This will be refined with actual lunar calendar data
    # For now, Tet is typically in late Jan / early Feb
    if solar_date.month == 1 and solar_date.day >= 20:
        # Late January = start of lunar year (month 1)
        lunar_month = 1
        lunar_day = solar_date.day - 19  # Approximate offset
    elif solar_date.month == 2:
        # February continues lunar month 1 or moves to month 2
        if solar_date.day <= 10:
            lunar_month = 1
            lunar_day = solar_date.day + 12  # Continue from Jan
        else:
            lunar_month = 2
            lunar_day = solar_date.day - 10
    else:
        # Approximate: lunar months are roughly aligned with solar months
        lunar_month = solar_date.month
        lunar_day = solar_date.day
    
    # Clamp to valid ranges
    lunar_month = max(1, min(12, lunar_month))
    lunar_day = max(1, min(30, lunar_day))  # Lunar months have 29-30 days
    
    return lunar_month, lunar_day


def add_weekend_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    is_weekend_col: str = "is_weekend",
    day_of_week_col: str = "day_of_week"
) -> pd.DataFrame:
    """Add weekend and day-of-week features."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    df[day_of_week_col] = df[time_col].dt.dayofweek
    df[is_weekend_col] = (df[day_of_week_col] >= 5).astype(int)
    return df


def add_lunar_calendar_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    lunar_month_col: str = "lunar_month",
    lunar_day_col: str = "lunar_day"
) -> pd.DataFrame:
    """Add lunar calendar features for Vietnamese holiday prediction."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    lunar_dates = df[time_col].dt.date.apply(solar_to_lunar_date)
    df[lunar_month_col] = [ld[0] for ld in lunar_dates]
    df[lunar_day_col] = [ld[1] for ld in lunar_dates]
    return df


def add_holiday_features_vietnam(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    holiday_indicator_col: str = "holiday_indicator",
    days_until_holiday_col: str = "days_until_next_holiday",
    days_since_holiday_col: str = "days_since_holiday"
) -> pd.DataFrame:
    """Add Vietnamese holiday-related features to DataFrame."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    extended_max = max_date + timedelta(days=365)
    holidays = get_vietnam_holidays(min_date, extended_max)
    holiday_set = set(holidays)
    df[holiday_indicator_col] = 0
    df[days_until_holiday_col] = np.nan
    df[days_since_holiday_col] = np.nan
    
    for idx, row in df.iterrows():
        current_date = row[time_col].date()
        if current_date in holiday_set:
            df.at[idx, holiday_indicator_col] = 1
        
        next_holiday = None
        for holiday in holidays:
            if holiday > current_date:
                next_holiday = holiday
                break
        df.at[idx, days_until_holiday_col] = (next_holiday - current_date).days if next_holiday else 365
        
        last_holiday = None
        for holiday in sorted(holidays, reverse=True):
            if holiday <= current_date:
                last_holiday = holiday
                break
        df.at[idx, days_since_holiday_col] = (current_date - last_holiday).days if last_holiday else 365
    
    df[days_until_holiday_col] = df[days_until_holiday_col].fillna(365)
    df[days_since_holiday_col] = df[days_since_holiday_col].fillna(365)
    return df


def add_rolling_and_momentum_features(
    df: pd.DataFrame,
    target_col: str = "QTY",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    rolling_7_col: str = "rolling_mean_7d",
    rolling_30_col: str = "rolling_mean_30d",
    momentum_col: str = "momentum_3d_vs_14d"
) -> pd.DataFrame:
    """Add rolling mean and momentum features to reduce model inertia."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([cat_col, time_col]).reset_index(drop=True)
    df[rolling_7_col] = np.nan
    df[rolling_30_col] = np.nan
    df[momentum_col] = np.nan
    
    for cat, group in df.groupby(cat_col, sort=False):
        cat_mask = df[cat_col] == cat
        cat_indices = df[cat_mask].index
        rolling_7 = group[target_col].rolling(window=7, min_periods=1).mean()
        rolling_30 = group[target_col].rolling(window=30, min_periods=1).mean()
        rolling_3 = group[target_col].rolling(window=3, min_periods=1).mean()
        rolling_14 = group[target_col].rolling(window=14, min_periods=1).mean()
        df.loc[cat_indices, rolling_7_col] = rolling_7.values
        df.loc[cat_indices, rolling_30_col] = rolling_30.values
        momentum = rolling_3 - rolling_14
        df.loc[cat_indices, momentum_col] = momentum.values
    
    for col in [rolling_7_col, rolling_30_col, momentum_col]:
        df[col] = df[col].ffill().bfill().fillna(0)
    return df


def calculate_accuracy(y_true, y_pred):
    """
    Calculate %accuracy based on the formula:
    
    Total_Error = Σ |y_pred - y_true|
    Total_Actual = Σ y_true
    
    If |Total_Error| > Total_Actual:
        ⟹ Accuracy = 0%
    
    If |Total_Error| ≤ Total_Actual:
        ⟹ %Bias = (|Total_Error| / Total_Actual) × 100%
        ⟹ Accuracy = 100% - %Bias
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
    
    Returns:
        Accuracy as a percentage (0-100)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid_mask.sum() == 0:
        return np.nan
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Calculate Total_Error (sum of absolute errors)
    total_error = np.sum(np.abs(y_pred_valid - y_true_valid))
    
    # Calculate Total_Actual (sum of actual values)
    total_actual = np.sum(np.abs(y_true_valid))
    
    # Handle edge cases
    if total_actual == 0:
        # If there are no actual values, return NaN
        return np.nan
    
    # Check if |Total_Error| > Total_Actual
    if total_error > total_actual:
        return 0.0
    
    # Calculate %Bias and Accuracy
    percent_bias = (total_error / total_actual) * 100.0
    accuracy = 100.0 - percent_bias
    
    return accuracy


def scan_previous_runs(predictions_base_dir: Path):
    """
    Scan previous prediction runs to extract metrics for comparison.
    
    Args:
        predictions_base_dir: Base directory containing run directories (e.g., outputs/mvp_test/predictions/)
    
    Returns:
        List of dictionaries with run metadata and metrics, sorted by timestamp (oldest first)
    """
    previous_runs = []
    
    if not predictions_base_dir.exists():
        return previous_runs
    
    # Scan for directories matching run_YYYYMMDD_HHMMSS pattern
    for item in predictions_base_dir.iterdir():
        if item.is_dir() and item.name.startswith('run_'):
            run_id = item.name
            summary_path = item / "summary.txt"
            tf_csv_path = item / "predictions_teacher_forcing.csv"
            rec_csv_path = item / "predictions_recursive.csv"
            
            run_data = {
                'run_id': run_id,
                'timestamp': None,
                'loss_function': 'Unknown',
                'tf_mae': None,
                'tf_rmse': None,
                'tf_mse': None,
                'rec_mae': None,
                'rec_rmse': None,
                'rec_mse': None
            }
            
            # Try to parse timestamp from run_id
            try:
                # run_YYYYMMDD_HHMMSS
                timestamp_str = run_id.replace('run_', '')
                run_data['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            except ValueError:
                pass
            
            # Try to extract metrics from CSV files
            try:
                if tf_csv_path.exists():
                    tf_df = pd.read_csv(tf_csv_path)
                    if 'abs_error' in tf_df.columns and 'error' in tf_df.columns:
                        run_data['tf_mae'] = tf_df['abs_error'].mean()
                        run_data['tf_mse'] = (tf_df['error'] ** 2).mean()
                        run_data['tf_rmse'] = np.sqrt(run_data['tf_mse'])
            except Exception as e:
                pass
            
            try:
                if rec_csv_path.exists():
                    rec_df = pd.read_csv(rec_csv_path)
                    if 'abs_error' in rec_df.columns and 'error' in rec_df.columns:
                        # Filter out rows without actuals
                        rec_df_with_actuals = rec_df[rec_df['actual'].notna()]
                        if len(rec_df_with_actuals) > 0:
                            run_data['rec_mae'] = rec_df_with_actuals['abs_error'].mean()
                            run_data['rec_mse'] = ((rec_df_with_actuals['actual'] - rec_df_with_actuals['predicted']) ** 2).mean()
                            run_data['rec_rmse'] = np.sqrt(run_data['rec_mse'])
            except Exception as e:
                pass
            
            # Try to extract loss function from metadata.json
            metadata_path = item / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        if 'training_config' in metadata and 'loss_function' in metadata['training_config']:
                            run_data['loss_function'] = metadata['training_config']['loss_function']
                except Exception:
                    pass
            
            previous_runs.append(run_data)
    
    # Sort by timestamp (oldest first)
    previous_runs.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)
    
    return previous_runs


def generate_comparison_table(current_metrics: dict, previous_runs: list) -> str:
    """
    Generate a formatted comparison table for historical runs.
    
    Args:
        current_metrics: Dictionary with current run metrics
        previous_runs: List of previous run data dictionaries
    
    Returns:
        Formatted string table
    """
    if not previous_runs:
        return "No previous runs found for comparison.\n"
    
    table_lines = []
    table_lines.append("\nHistorical Comparison Table")
    table_lines.append("=" * 120)
    table_lines.append(
        f"{'Run ID':<18} {'Loss Function':<20} {'TF MAE':<12} {'TF RMSE':<12} {'TF MSE':<12} "
        f"{'REC MAE':<12} {'REC RMSE':<12} {'REC MSE':<12}"
    )
    table_lines.append("-" * 120)
    
    # Add previous runs
    for run in previous_runs:
        run_id_short = run['run_id'].replace('run_', '')[:15] if run['run_id'] else 'Unknown'
        loss_fn = run['loss_function'][:18] if run['loss_function'] else 'Unknown'
        tf_mae = f"{run['tf_mae']:.4f}" if run['tf_mae'] is not None else "N/A"
        tf_rmse = f"{run['tf_rmse']:.4f}" if run['tf_rmse'] is not None else "N/A"
        tf_mse = f"{run['tf_mse']:.4f}" if run['tf_mse'] is not None else "N/A"
        rec_mae = f"{run['rec_mae']:.4f}" if run['rec_mae'] is not None else "N/A"
        rec_rmse = f"{run['rec_rmse']:.4f}" if run['rec_rmse'] is not None else "N/A"
        rec_mse = f"{run['rec_mse']:.4f}" if run['rec_mse'] is not None else "N/A"
        
        table_lines.append(
            f"{run_id_short:<18} {loss_fn:<20} {tf_mae:<12} {tf_rmse:<12} {tf_mse:<12} "
            f"{rec_mae:<12} {rec_rmse:<12} {rec_mse:<12}"
        )
    
    # Add current run with special marker
    run_id_current = current_metrics.get('run_id', 'CURRENT')[:15]
    loss_fn_current = current_metrics.get('loss_function', 'Unknown')[:18]
    tf_mae_current = f"{current_metrics['tf_mae']:.4f}" if current_metrics.get('tf_mae') is not None else "N/A"
    tf_rmse_current = f"{current_metrics['tf_rmse']:.4f}" if current_metrics.get('tf_rmse') is not None else "N/A"
    tf_mse_current = f"{current_metrics['tf_mse']:.4f}" if current_metrics.get('tf_mse') is not None else "N/A"
    rec_mae_current = f"{current_metrics['rec_mae']:.4f}" if current_metrics.get('rec_mae') is not None else "N/A"
    rec_rmse_current = f"{current_metrics['rec_rmse']:.4f}" if current_metrics.get('rec_rmse') is not None else "N/A"
    rec_mse_current = f"{current_metrics['rec_mse']:.4f}" if current_metrics.get('rec_mse') is not None else "N/A"
    
    table_lines.append("-" * 120)
    table_lines.append(
        f"{run_id_current + ' (*)':<18} {loss_fn_current:<20} {tf_mae_current:<12} {tf_rmse_current:<12} {tf_mse_current:<12} "
        f"{rec_mae_current:<12} {rec_rmse_current:<12} {rec_mse_current:<12}"
    )
    table_lines.append("-" * 120)
    table_lines.append("(*) Current run\n")
    
    return "\n".join(table_lines)


def analyze_improvement(current_metrics: dict, previous_runs: list) -> str:
    """
    Analyze if current run is an improvement or regression compared to best previous run.
    
    Args:
        current_metrics: Dictionary with current run metrics
        previous_runs: List of previous run data dictionaries
    
    Returns:
        Formatted string with improvement analysis
    """
    if not previous_runs:
        return "No previous runs available for comparison.\n"
    
    # Find best previous run (lowest TF MAE)
    best_run = None
    best_tf_mae = float('inf')
    
    for run in previous_runs:
        if run['tf_mae'] is not None and run['tf_mae'] < best_tf_mae:
            best_tf_mae = run['tf_mae']
            best_run = run
    
    if best_run is None or current_metrics.get('tf_mae') is None:
        return "Unable to determine improvement (missing metrics).\n"
    
    current_tf_mae = current_metrics['tf_mae']
    improvement_pct = ((best_tf_mae - current_tf_mae) / best_tf_mae) * 100
    
    analysis_lines = []
    analysis_lines.append("Improvement Analysis")
    analysis_lines.append("-" * 70)
    analysis_lines.append(f"Best previous run: {best_run['run_id']}")
    analysis_lines.append(f"  Best TF MAE: {best_tf_mae:.4f}")
    analysis_lines.append(f"  Loss function: {best_run['loss_function']}")
    analysis_lines.append(f"\nCurrent run TF MAE: {current_tf_mae:.4f}")
    
    if improvement_pct > 0:
        analysis_lines.append(f"\n✓ IMPROVEMENT: {improvement_pct:.2f}% better (lower MAE)")
    elif improvement_pct < 0:
        analysis_lines.append(f"\n✗ REGRESSION: {abs(improvement_pct):.2f}% worse (higher MAE)")
    else:
        analysis_lines.append(f"\n= NO CHANGE: Same MAE as best previous run")
    
    # Compare RMSE and MSE too
    if best_run['tf_rmse'] is not None and current_metrics.get('tf_rmse') is not None:
        rmse_improvement = ((best_run['tf_rmse'] - current_metrics['tf_rmse']) / best_run['tf_rmse']) * 100
        if rmse_improvement > 0:
            analysis_lines.append(f"  RMSE: {rmse_improvement:.2f}% improvement")
        else:
            analysis_lines.append(f"  RMSE: {abs(rmse_improvement):.2f}% regression")
    
    analysis_lines.append("")
    
    return "\n".join(analysis_lines)


def load_model_for_prediction(model_path: str, config):
    """Load trained model from checkpoint and scaler."""
    # Load metadata first to get the num_categories used during training
    model_dir = Path(model_path).parent
    metadata_path = model_dir / "metadata.json"
    
    # Get num_categories from metadata if available, otherwise from config
    num_categories = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Get num_categories from model_config in metadata
        if 'model_config' in metadata and 'num_categories' in metadata['model_config']:
            num_categories = metadata['model_config']['num_categories']
    
    # Fallback to config if metadata doesn't have it
    if num_categories is None:
        model_config = config.model
        num_categories = model_config.get('num_categories')
    
    if num_categories is None:
        raise ValueError("num_categories must be found in model metadata or config")
    
    # Get category_filter from training metadata to know which category(ies) model was trained on
    trained_category_filter = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if 'data_config' in metadata and 'category_filter' in metadata['data_config']:
            trained_category_filter = metadata['data_config']['category_filter']
    
    print(f"  - Loading model with num_categories={num_categories} (from trained model)")
    if trained_category_filter:
        print(f"  - Model was trained on category: {trained_category_filter}")
    else:
        print(f"  - Model was trained on: all categories (num_categories={num_categories})")
    
    # Build model with same architecture (use num_categories from trained model)
    model_config = config.model
    model = RNNWithCategory(
        num_categories=num_categories,
        cat_emb_dim=model_config['cat_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        output_dim=model_config['output_dim']
    )
    
    # Load checkpoint
    device = torch.device(config.training['device'])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  - Model loaded from: {model_path}")
    print(f"  - Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    # Load scaler from same directory as model (model_dir already defined above)
    scaler_path = model_dir / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  - Scaler loaded from: {scaler_path}")
        print(f"    Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")
    else:
        print(f"  [WARNING] Scaler not found at {scaler_path}, predictions will be in scaled space")
    
    return model, device, scaler, trained_category_filter


def prepare_prediction_data(data, config, cat2id, scaler=None):
    """
    Prepare data for prediction using the same preprocessing as training.
    
    Args:
        data: DataFrame with raw data
        config: Configuration object
        cat2id: Category to ID mapping from training
        scaler: Optional StandardScaler for QTY values (if None, no scaling applied)
    
    Returns:
        Prepared DataFrame ready for window creation
    """
    data_config = config.data
    time_col = data_config['time_col']
    cat_col = data_config['cat_col']
    cat_id_col = data_config['cat_id_col']
    
    # Ensure time column is datetime and sort
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)
    
    # Add temporal features (before aggregation)
    print("  - Adding temporal features...")
    data = add_temporal_features(
        data,
        time_col=time_col,
        month_sin_col="month_sin",
        month_cos_col="month_cos",
        dayofmonth_sin_col="dayofmonth_sin",
        dayofmonth_cos_col="dayofmonth_cos"
    )
    
    # Add weekend features (before aggregation)
    print("  - Adding weekend features (is_weekend, day_of_week)...")
    data = add_weekend_features(
        data,
        time_col=time_col,
        is_weekend_col="is_weekend",
        day_of_week_col="day_of_week"
    )
    
    # Add lunar calendar features (before aggregation)
    print("  - Adding lunar calendar features (lunar_month, lunar_day)...")
    data = add_lunar_calendar_features(
        data,
        time_col=time_col,
        lunar_month_col="lunar_month",
        lunar_day_col="lunar_day"
    )
    
    # Add Vietnamese holiday features (before aggregation)
    print("  - Adding Vietnamese holiday features...")
    data = add_holiday_features_vietnam(
        data,
        time_col=time_col,
        holiday_indicator_col="holiday_indicator",
        days_until_holiday_col="days_until_next_holiday",
        days_since_holiday_col="days_since_holiday"
    )
    
    # Daily aggregation: Group by date and category, sum QTY
    # This matches the training pipeline
    print("  - Aggregating to daily totals by category...")
    samples_before = len(data)
    data = aggregate_daily(
        data,
        time_col=time_col,
        cat_col=cat_col,
        target_col=data_config['target_col']
    )
    samples_after = len(data)
    print(f"    Samples: {samples_before} -> {samples_after} (one row per date per category)")
    
    # Add rolling mean and momentum features (after aggregation)
    print("  - Adding rolling mean and momentum features (7d, 30d, momentum)...")
    data = add_rolling_and_momentum_features(
        data,
        target_col=data_config['target_col'],
        time_col=time_col,
        cat_col=cat_col,
        rolling_7_col="rolling_mean_7d",
        rolling_30_col="rolling_mean_30d",
        momentum_col="momentum_3d_vs_14d"
    )
    
    # Encode categories using the same mapping from training
    print("  - Encoding categories...")
    data = data.copy()
    data[cat_id_col] = data[cat_col].map(cat2id)
    
    # Check for unknown categories
    unknown_cats = data[data[cat_id_col].isna()][cat_col].unique()
    if len(unknown_cats) > 0:
        print(f"  [WARNING] Unknown categories found: {unknown_cats}")
        print(f"  [WARNING] These will be filtered out")
        data = data[data[cat_id_col].notna()].copy()
    
    # Apply scaling if scaler is provided (matches training pipeline)
    if scaler is not None:
        print(f"  - Applying scaling to {data_config['target_col']} values...")
        data = apply_scaling(data, scaler, target_col=data_config['target_col'])
    
    return data


def create_prediction_windows(data, config):
    """
    Create sliding windows for prediction and extract corresponding dates.
    
    Returns:
        Tuple of (X_pred, y_pred, cat_pred, dates) where dates are the
        prediction dates corresponding to each window.
    """
    window_config = config.window
    data_config = config.data
    
    feature_cols = data_config['feature_cols']
    target_col = [data_config['target_col']]
    cat_id_col = data_config['cat_id_col']
    time_col_name = data_config['time_col']
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    
    print(f"  - Creating prediction windows (input_size={input_size}, horizon={horizon})...")
    
    # Use slicing_window_category but also extract dates
    X_pred, y_pred, cat_pred = slicing_window_category(
        data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=[cat_id_col],
        time_col=[time_col_name]
    )
    
    # Extract dates corresponding to each prediction window
    # The date for prediction at index i corresponds to data point at input_size + i
    # Normalize to date only (remove time component)
    dates = []
    cat_col_name = cat_id_col
    
    for cat, g in data.groupby(cat_col_name, sort=False):
        g = g.sort_values(time_col_name)
        # Extract dates for each window (date at prediction point)
        # Normalize to date only (remove time)
        for i in range(len(g) - input_size - horizon + 1):
            pred_datetime = g.iloc[i + input_size][time_col_name]
            # Convert to date only (normalize time to 00:00:00)
            if pd.api.types.is_datetime64_any_dtype(pd.Series([pred_datetime])):
                pred_date = pd.to_datetime(pred_datetime).normalize()  # Keeps as datetime but time = 00:00:00
            else:
                pred_date = pd.to_datetime(pred_datetime).normalize()
            dates.append(pred_date)
    
    dates = pd.to_datetime(dates).normalize()  # Ensure all dates are normalized
    
    print(f"  - Created {len(X_pred)} prediction windows")
    return X_pred, y_pred, cat_pred, dates


def predict_recursive(
    model,
    device,
    initial_window_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    config,
    cat_id: int
):
    """
    Recursive prediction mode: uses model's own predictions as inputs.
    
    This function simulates a true production forecast where future QTY values
    are unknown. It uses the model's previous predictions to build the input
    window for subsequent predictions.
    
    Args:
        model: Trained PyTorch model (already on device and in eval mode)
        device: PyTorch device
        initial_window_data: DataFrame with last 30 days of historical data (Dec 2024)
                            Must have all feature columns and be sorted by time
        start_date: First date to predict (e.g., date(2025, 1, 1))
        end_date: Last date to predict (e.g., date(2025, 1, 31))
        config: Configuration object
        cat_id: Category ID (integer)
    
    Returns:
        DataFrame with columns: date, predicted, actual (if available)
    """
    window_config = config.window
    data_config = config.data
    
    input_size = window_config['input_size']
    feature_cols = data_config['feature_cols']
    target_col = data_config['target_col']
    time_col = data_config['time_col']
    
    # Validate initial window has enough data
    if len(initial_window_data) < input_size:
        raise ValueError(
            f"Initial window must have at least {input_size} samples, "
            f"got {len(initial_window_data)}"
        )
    
    # Get last input_size rows as the initial window
    window = initial_window_data.tail(input_size).copy()
    window = window.sort_values(time_col).reset_index(drop=True)
    
    predictions = []
    current_date = start_date
    
    print(f"  - Starting recursive prediction from {start_date} to {end_date}")
    print(f"  - Initial window: {window[time_col].min()} to {window[time_col].max()}")
    
    while current_date <= end_date:
        # Create feature vector for current_date using current window
        # Calendar features will be computed for current_date
        # QTY values come from the window (which includes previous predictions)
        
        # Get the feature values from the window (last input_size timesteps)
        # The window contains: [t-29, t-28, ..., t-1] where t is current_date
        window_features = window[feature_cols].values  # Shape: (input_size, n_features)
        
        # We need to update the last row's calendar features for current_date
        # But keep QTY from the window (which may be predicted)
        # For all features except QTY, compute for current_date
        
        # Create a row for current_date with correct calendar features
        current_datetime = pd.Timestamp(current_date)
        
        # Compute temporal features for current_date
        month = current_datetime.month
        dayofmonth = current_datetime.day
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        dayofmonth_sin = np.sin(2 * np.pi * (dayofmonth - 1) / 31)
        dayofmonth_cos = np.cos(2 * np.pi * (dayofmonth - 1) / 31)
        
        # Compute weekend features for current_date
        day_of_week = current_datetime.dayofweek  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Compute lunar calendar features for current_date
        lunar_month, lunar_day = solar_to_lunar_date(current_date)
        
        # Compute Vietnamese holiday features for current_date
        extended_end = end_date + timedelta(days=365)
        holidays = get_vietnam_holidays(current_date, extended_end)
        holiday_set = set(holidays)
        
        holiday_indicator = 1 if current_date in holiday_set else 0
        
        # Calculate days until next holiday
        next_holiday = None
        for holiday in holidays:
            if holiday > current_date:
                next_holiday = holiday
                break
        
        if next_holiday:
            days_until_next_holiday = (next_holiday - current_date).days
        else:
            days_until_next_holiday = 365
        
        # Calculate days since last holiday
        last_holiday = None
        for holiday in sorted(holidays, reverse=True):
            if holiday <= current_date:
                last_holiday = holiday
                break
        
        days_since_holiday = (current_date - last_holiday).days if last_holiday else 365
        
        # Compute rolling means and momentum from window QTY values
        qty_values = window[target_col].values
        rolling_mean_7d = np.mean(qty_values[-7:]) if len(qty_values) >= 7 else np.mean(qty_values)
        rolling_mean_30d = np.mean(qty_values) if len(qty_values) >= 30 else np.mean(qty_values)
        rolling_mean_3d = np.mean(qty_values[-3:]) if len(qty_values) >= 3 else np.mean(qty_values)
        rolling_mean_14d = np.mean(qty_values[-14:]) if len(qty_values) >= 14 else np.mean(qty_values)
        momentum_3d_vs_14d = rolling_mean_3d - rolling_mean_14d
        
        # Create input tensor from current window
        # The window contains features from dates [t-30, t-29, ..., t-1]
        # Calendar features in the window are correct for their respective dates
        # QTY values in the window include previous predictions for dates >= start_date
        X_window = torch.tensor(
            window_features,
            dtype=torch.float32
        ).unsqueeze(0).to(device)  # Shape: (1, input_size, n_features)
        
        cat_tensor = torch.tensor([cat_id], dtype=torch.long).to(device)
        
        # Make prediction (in scaled space if scaler was used)
        with torch.no_grad():
            pred_scaled = model(X_window, cat_tensor).cpu().item()
        
        # Store prediction (will be inverse transformed later if scaler is available)
        predictions.append({
            'date': current_date,
            'predicted': pred_scaled
        })
        
        # Update window: remove oldest row, add new row with prediction
        # Create new row with predicted QTY and calendar features for current_date
        new_row = window.iloc[-1:].copy()  # Copy last row as template
        new_row[time_col] = current_datetime
        new_row[target_col] = pred_scaled  # Use predicted QTY (in scaled space if scaler used)
        new_row['month_sin'] = month_sin
        new_row['month_cos'] = month_cos
        new_row['dayofmonth_sin'] = dayofmonth_sin
        new_row['dayofmonth_cos'] = dayofmonth_cos
        new_row['is_weekend'] = is_weekend
        new_row['day_of_week'] = day_of_week
        new_row['lunar_month'] = lunar_month
        new_row['lunar_day'] = lunar_day
        new_row['holiday_indicator'] = holiday_indicator
        new_row['days_until_next_holiday'] = days_until_next_holiday
        new_row['days_since_holiday'] = days_since_holiday
        new_row['rolling_mean_7d'] = rolling_mean_7d
        new_row['rolling_mean_30d'] = rolling_mean_30d
        new_row['momentum_3d_vs_14d'] = momentum_3d_vs_14d
        
        # Remove oldest row and append new row
        window = pd.concat([window.iloc[1:], new_row], ignore_index=True)
        window = window.sort_values(time_col).reset_index(drop=True)
        
        # Move to next date
        current_date += timedelta(days=1)
    
    predictions_df = pd.DataFrame(predictions)
    return predictions_df


def get_historical_window_data(
    historical_data: pd.DataFrame,
    end_date: date,
    config,
    num_days: int = 30
):
    """
    Extract the last N days of historical data to initialize recursive prediction window.
    
    Args:
        historical_data: DataFrame with historical data (e.g., 2024 data)
        end_date: Last date to include (e.g., date(2024, 12, 31))
        config: Configuration object
        num_days: Number of days to extract (default: 30, matching input_size)
    
    Returns:
        DataFrame with last num_days of data, sorted by time
    """
    time_col = config.data['time_col']
    
    # Filter to dates up to end_date
    historical_data = historical_data[
        pd.to_datetime(historical_data[time_col]).dt.date <= end_date
    ].copy()
    
    # Sort by time
    historical_data = historical_data.sort_values(time_col).reset_index(drop=True)
    
    # Get last num_days (group by date first, then take last num_days of unique dates)
    # Since there may be multiple rows per date (different categories), we need to handle this
    historical_data['date_only'] = pd.to_datetime(historical_data[time_col]).dt.date
    unique_dates = historical_data['date_only'].unique()
    unique_dates = sorted(unique_dates)
    
    if len(unique_dates) < num_days:
        print(f"  [WARNING] Only {len(unique_dates)} unique dates available, requested {num_days}")
        selected_dates = unique_dates
    else:
        selected_dates = unique_dates[-num_days:]
    
    window_data = historical_data[
        historical_data['date_only'].isin(selected_dates)
    ].copy()
    
    # Remove temporary column
    window_data = window_data.drop(columns=['date_only'])
    
    return window_data.sort_values(time_col).reset_index(drop=True)


def main():
    """Main prediction function with both Teacher Forcing and Recursive modes."""
    print("=" * 80)
    print("FULL YEAR 2025 PREDICTION - TEACHER FORCING vs RECURSIVE MODES")
    print("=" * 80)
    
    # Load configuration
    print("\n[1/7] Loading configuration...")
    config = load_config()
    
    # Override to match MVP test settings
    config.set('data.years', [2024])  # For loading category mapping reference
    data_config = config.data
    
    # Load 2024 data to get category mapping and historical window
    print("\n[2/7] Loading historical 2024 data...")
    data_reader = DataReader(
        data_dir=data_config['data_dir'],
        file_pattern=data_config['file_pattern']
    )
    
    try:
        ref_data = data_reader.load(years=[2024])
    except FileNotFoundError:
        print("[WARNING] Trying pattern-based loading...")
        ref_data = data_reader.load_by_file_pattern(
            years=[2024],
            file_prefix=data_config.get('file_prefix', 'Outboundreports')
        )
    
    # Get category mode from config
    inference_config = config.inference
    category_mode = inference_config.get('category_mode', 'single')  # Default to 'single' for backward compatibility
    category_filter = inference_config.get('category_filter', 'DRY')  # Default to 'DRY' for backward compatibility
    
    print(f"  - Category mode: {category_mode}")
    if category_mode == 'single':
        print(f"  - Category filter: {category_filter}")
    
    # Encode all categories from reference data (don't filter yet - need full mapping)
    # Note: We encode here to get cat2id mapping, but num_categories for model will come from trained model metadata
    _, cat2id, num_categories_in_data = encode_categories(ref_data, data_config['cat_col'])
    # Don't overwrite config.model.num_categories here - it will be loaded from trained model metadata
    
    print(f"  - Category mapping: {cat2id}")
    print(f"  - Number of categories in data: {num_categories_in_data}")
    
    # Determine which categories to predict
    # Filter out NaN values and convert to string to handle mixed types
    unique_ref_cats = ref_data[data_config['cat_col']].dropna().astype(str).unique().tolist()
    available_categories = sorted([cat for cat in unique_ref_cats if cat.lower() != 'nan'])
    print(f"  - Available categories in reference data: {available_categories}")
    
    if category_mode == 'single':
        if category_filter not in available_categories:
            raise ValueError(f"Category '{category_filter}' not found in reference data. Available: {available_categories}")
        categories_to_predict = [category_filter]
    elif category_mode == 'all':
        categories_to_predict = available_categories
    else:
        raise ValueError(f"Invalid category_mode: {category_mode}. Must be 'all' or 'single'")
    
    print(f"  - Categories to predict: {categories_to_predict}")
    
    # Load 2025 data
    print("\n[4/7] Loading 2025 data...")
    data_2025_path = Path("dataset/test/data_2025.csv")
    
    if not data_2025_path.exists():
        raise FileNotFoundError(
            f"2025 data file not found at: {data_2025_path.absolute()}"
        )
    
    print(f"  - Loading from: {data_2025_path}")
    try:
        data_2025 = pd.read_csv(data_2025_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        try:
            data_2025 = pd.read_csv(data_2025_path, encoding='latin-1', low_memory=False)
        except Exception as e:
            data_2025 = pd.read_csv(data_2025_path, encoding='cp1252', low_memory=False)
    print(f"  - Loaded {len(data_2025)} samples")
    
    # Filter to full year 2025 (keep all categories for now - will filter per category later)
    print("\n[5/7] Filtering data...")
    cat_col = data_config['cat_col']
    time_col = data_config['time_col']
    
    # Filter to full year 2025 first
    if not pd.api.types.is_datetime64_any_dtype(data_2025[time_col]):
        data_2025[time_col] = pd.to_datetime(data_2025[time_col])
    
    data_2025 = data_2025[
        (data_2025[time_col] >= '2025-01-01') & 
        (data_2025[time_col] < '2026-01-01')
    ].copy()
    print(f"  - After full year 2025 filter: {len(data_2025)} samples")
    
    # Check which categories are available in 2025 data
    # Filter out NaN values and convert to string to handle mixed types
    unique_cats = data_2025[cat_col].dropna().astype(str).unique().tolist()
    available_2025_categories = sorted([cat for cat in unique_cats if cat.lower() != 'nan'])
    print(f"  - Available categories in 2025 data: {available_2025_categories}")
    
    # Filter categories_to_predict to only those available in both reference and 2025 data
    categories_to_predict = [cat for cat in categories_to_predict if cat in available_2025_categories]
    
    if len(categories_to_predict) == 0:
        raise ValueError(f"No matching categories found between reference data and 2025 data. "
                        f"Reference: {available_categories}, 2025: {available_2025_categories}")
    
    print(f"  - Final categories to predict: {categories_to_predict}")
    
    # Load trained model (to get scaler before data preparation)
    # Determine model directory based on inference category_mode
    # Match the directory structure from mvp_test.py (outputs/mvp_test{suffix}/models/)
    print("\n[6/7] Loading trained model...")
    base_output_dir = "outputs/mvp_test"
    
    # Determine suffix based on inference category_mode
    # Note: This should match what mvp_test.py uses when training
    # For "all": uses suffix "_all" -> outputs/mvp_test_all/models/
    # For "single": uses suffix "_{category}" -> outputs/mvp_test_{category}/models/
    # Default (no suffix): outputs/mvp_test/models/
    
    if category_mode == 'all':
        model_suffix = "_all"
    elif category_mode == 'single' and category_filter:
        model_suffix = f"_{category_filter}"
    else:
        model_suffix = ""  # Default: outputs/mvp_test/models/
    
    model_dir_path = Path(f"{base_output_dir}{model_suffix}/models")
    model_path = model_dir_path / "best_model.pth"
    
    # If model not found at expected path, try default location as fallback
    if not model_path.exists():
        default_model_path = Path(f"{base_output_dir}/models/best_model.pth")
        if default_model_path.exists():
            print(f"  [WARNING] Model not found at expected path: {model_path}")
            print(f"  [INFO] Using default model path: {default_model_path}")
            model_path = default_model_path
            model_dir_path = default_model_path.parent  # Update model_dir_path to match
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path} or {default_model_path}. "
                f"Please run mvp_test.py first to train the model with "
                f"data.category_mode='{category_mode}'."
            )
    else:
        print(f"  - Using model from: {model_path}")
    
    model, device, scaler, trained_category_filter = load_model_for_prediction(str(model_path), config)
    
    # IMPORTANT: Filter predictions to only categories the model was trained on
    # If model was trained on a single category, we can only predict that category
    if trained_category_filter is not None:
        print(f"\n[INFO] Model was trained on category '{trained_category_filter}' only.")
        print(f"       Filtering predictions to match training data (ignoring inference.category_mode).")
        categories_to_predict = [trained_category_filter] if trained_category_filter in categories_to_predict else []
        if len(categories_to_predict) == 0:
            raise ValueError(
                f"Model was trained on category '{trained_category_filter}', "
                f"but this category is not available in the prediction data. "
                f"Available categories: {available_2025_categories}"
            )
        print(f"  - Updated categories to predict: {categories_to_predict}")
    
    # Prepare data with scaler if available (matching training pipeline)
    print("\n[6.5/7] Preparing data with scaling...")
    historical_data_prepared = prepare_prediction_data(ref_data.copy(), config, cat2id, scaler)
    data_2025_prepared = prepare_prediction_data(data_2025, config, cat2id, scaler)
    
    # Get last 30 days of December 2024 for initial window (after aggregation/scaling)
    historical_window = get_historical_window_data(
        historical_data_prepared,
        end_date=date(2024, 12, 31),
        config=config,
        num_days=config.window['input_size']
    )
    print(f"  - Historical window: {len(historical_window)} samples")
    print(f"  - Date range: {historical_window[data_config['time_col']].min()} to {historical_window[data_config['time_col']].max()}")
    
    # Recreate windows after scaling (all categories or filtered by categories_to_predict)
    X_pred, y_actual, cat_pred, pred_dates = create_prediction_windows(data_2025_prepared, config)
    
    # Filter predictions to selected categories if needed
    # Filter if: single category mode OR model was trained on single category
    if category_mode == 'single' or (trained_category_filter is not None and len(categories_to_predict) == 1):
        # Filter windows to selected category only
        cat_id = cat2id.get(categories_to_predict[0])
        if cat_id is None:
            raise ValueError(f"Category '{categories_to_predict[0]}' not found in category mapping")
        mask = cat_pred == cat_id
        X_pred = X_pred[mask]
        y_actual = y_actual[mask]
        cat_pred = cat_pred[mask]
        pred_dates = pred_dates[mask] if hasattr(pred_dates, '__getitem__') else [pred_dates[i] for i in range(len(pred_dates)) if mask[i]]
        print(f"  - Filtered prediction windows to category '{categories_to_predict[0]}' (category ID: {cat_id})")
        print(f"  - Prediction windows after filtering: {len(X_pred)}")
    
    # ========================================================================
    # MODE 1: TEACHER FORCING (Test Evaluation) - Uses actual 2025 values
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODE 1: TEACHER FORCING (Test Evaluation)")
    print("=" * 80)
    print(f"Using actual ground truth {data_config['target_col']} values from 2025 as features.")
    print("This mode is suitable for model evaluation but not production forecasting.")
    
    # Create dataset and dataloader
    pred_dataset = ForecastDataset(X_pred, y_actual, cat_pred)
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=config.training['test_batch_size'],
        shuffle=False
    )
    
    # Create trainer for prediction
    trainer = Trainer(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        device=device
    )
    
    y_true_tf, y_pred_tf = trainer.predict(pred_loader)
    
    print(f"  - Predictions made: {len(y_pred_tf)} samples")
    print(f"  - Prediction date range: {pred_dates.min()} to {pred_dates.max()}")
    
    # Inverse transform predictions and actuals if scaler is available
    if scaler is not None:
        print("  - Inverse transforming scaled predictions to original scale...")
        y_true_tf_unscaled = inverse_transform_scaling(y_true_tf.flatten(), scaler)
        y_pred_tf_unscaled = inverse_transform_scaling(y_pred_tf.flatten(), scaler)
        # Clip negative predictions to 0 (QTY cannot be negative)
        negative_count = np.sum(y_pred_tf_unscaled < 0)
        if negative_count > 0:
            print(f"  [WARNING] Clipping {negative_count} negative predictions to 0 (QTY must be >= 0)")
            y_pred_tf_unscaled = np.maximum(y_pred_tf_unscaled, 0.0)
    else:
        y_true_tf_unscaled = y_true_tf.flatten()
        y_pred_tf_unscaled = y_pred_tf.flatten()
        # Clip negative predictions to 0 even if no scaler
        negative_count = np.sum(y_pred_tf_unscaled < 0)
        if negative_count > 0:
            print(f"  [WARNING] Clipping {negative_count} negative predictions to 0 (QTY must be >= 0)")
            y_pred_tf_unscaled = np.maximum(y_pred_tf_unscaled, 0.0)
    
    # Calculate metrics on unscaled values
    mse_tf = np.mean((y_true_tf_unscaled - y_pred_tf_unscaled) ** 2)
    mae_tf = np.mean(np.abs(y_true_tf_unscaled - y_pred_tf_unscaled))
    rmse_tf = np.sqrt(mse_tf)
    accuracy_tf = calculate_accuracy(y_true_tf_unscaled, y_pred_tf_unscaled)
    
    print(f"\n  - MSE:  {mse_tf:.4f}")
    print(f"  - MAE:  {mae_tf:.4f}")
    print(f"  - RMSE: {rmse_tf:.4f}")
    if not np.isnan(accuracy_tf):
        print(f"  - Accuracy: {accuracy_tf:.2f}%")
    
    # ========================================================================
    # MODE 2: RECURSIVE (Production Forecast) - Uses model's own predictions
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODE 2: RECURSIVE (Production Forecast)")
    print("=" * 80)
    print("Using model's own predictions as inputs for future dates.")
    print("This mode simulates true production forecasting.")
    
    # For recursive mode, process each category separately
    # Get category ID for the category being processed
    if category_mode == 'single':
        cat_id = cat2id.get(categories_to_predict[0])
        if cat_id is None:
            raise ValueError(f"Category '{categories_to_predict[0]}' not found in category mapping")
        process_categories = [categories_to_predict[0]]
    else:
        # For "all" mode, process each category separately in recursive mode
        process_categories = categories_to_predict
    
    # Filter historical window to same category(s) - process first category for now
    # (For "all" mode with multiple categories, we'd need to loop here, but let's keep it simple for now)
    current_category = process_categories[0] if len(process_categories) > 0 else categories_to_predict[0]
    cat_id = cat2id.get(current_category)
    if cat_id is None:
        raise ValueError(f"Category '{current_category}' not found in category mapping")
    
    historical_window_filtered = historical_window[
        historical_window[data_config['cat_id_col']] == cat_id
    ].copy()
    
    if len(historical_window_filtered) < config.window['input_size']:
        # If not enough data for this category, use all data and duplicate
        print(f"  [WARNING] Only {len(historical_window_filtered)} samples for category, need {config.window['input_size']}")
        historical_window_filtered = historical_window[
            historical_window[data_config['cat_id_col']] == cat_id
        ].copy()
        # Take last available samples and repeat if needed
        if len(historical_window_filtered) > 0:
            last_row = historical_window_filtered.iloc[-1:].copy()
            while len(historical_window_filtered) < config.window['input_size']:
                historical_window_filtered = pd.concat([historical_window_filtered, last_row], ignore_index=True)
        else:
            raise ValueError(f"No historical data found for category ID {cat_id}")
    
    # Get last input_size samples
    historical_window_filtered = historical_window_filtered.tail(config.window['input_size']).copy()
    historical_window_filtered = historical_window_filtered.sort_values(time_col).reset_index(drop=True)
    
    # Run recursive prediction
    recursive_preds = predict_recursive(
        model=model,
        device=device,
        initial_window_data=historical_window_filtered,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        config=config,
        cat_id=cat_id
    )
    
    # Get actual values from original data (before scaling, already aggregated)
    # Re-prepare without scaler to get original scale actuals
    data_2025_unscaled = prepare_prediction_data(data_2025.copy(), config, cat2id, scaler=None)
    actuals_by_date = data_2025_unscaled.groupby(
        pd.to_datetime(data_2025_unscaled[time_col]).dt.date
    )[data_config['target_col']].sum().reset_index()
    actuals_by_date.columns = ['date', 'actual']
    actuals_by_date['date'] = pd.to_datetime(actuals_by_date['date']).dt.date
    
    recursive_results = recursive_preds.merge(
        actuals_by_date,
        on='date',
        how='left'
    )
    
    # Inverse transform predictions if scaler is available
    if scaler is not None:
        print("  - Inverse transforming recursive predictions to original scale...")
        recursive_results['predicted_unscaled'] = inverse_transform_scaling(
            recursive_results['predicted'].values, scaler
        )
        # Clip negative predictions to 0 (QTY cannot be negative)
        negative_count = np.sum(recursive_results['predicted_unscaled'] < 0)
        if negative_count > 0:
            print(f"  [WARNING] Clipping {negative_count} negative recursive predictions to 0 (QTY must be >= 0)")
            recursive_results['predicted_unscaled'] = np.maximum(recursive_results['predicted_unscaled'], 0.0)
    else:
        recursive_results['predicted_unscaled'] = recursive_results['predicted']
        # Clip negative predictions to 0 even if no scaler
        negative_count = np.sum(recursive_results['predicted_unscaled'] < 0)
        if negative_count > 0:
            print(f"  [WARNING] Clipping {negative_count} negative recursive predictions to 0 (QTY must be >= 0)")
            recursive_results['predicted_unscaled'] = np.maximum(recursive_results['predicted_unscaled'], 0.0)
    
    # Calculate metrics (only for dates with actuals, using unscaled values)
    recursive_results_with_actuals = recursive_results[recursive_results['actual'].notna()].copy()
    
    if len(recursive_results_with_actuals) > 0:
        mse_rec = np.mean((recursive_results_with_actuals['actual'] - recursive_results_with_actuals['predicted_unscaled']) ** 2)
        mae_rec = np.mean(np.abs(recursive_results_with_actuals['actual'] - recursive_results_with_actuals['predicted_unscaled']))
        rmse_rec = np.sqrt(mse_rec)
        accuracy_rec = calculate_accuracy(
            recursive_results_with_actuals['actual'].values,
            recursive_results_with_actuals['predicted_unscaled'].values
        )
        
        print(f"\n  - Predictions made: {len(recursive_preds)} samples")
        print(f"  - Samples with actuals: {len(recursive_results_with_actuals)}")
        print(f"  - MSE:  {mse_rec:.4f}")
        print(f"  - MAE:  {mae_rec:.4f}")
        print(f"  - RMSE: {rmse_rec:.4f}")
        if not np.isnan(accuracy_rec):
            print(f"  - Accuracy: {accuracy_rec:.2f}%")
    else:
        print(f"\n  - Predictions made: {len(recursive_preds)} samples")
        print(f"  - No actual values available for comparison")
        mse_rec = mae_rec = rmse_rec = accuracy_rec = np.nan
    
    # ========================================================================
    # COMPARISON AND SAVING
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: TEACHER FORCING vs RECURSIVE")
    print("=" * 80)
    
    # Prepare teacher forcing results for comparison
    if isinstance(pred_dates, pd.DatetimeIndex):
        tf_dates = pred_dates.date
    else:
        tf_dates = pd.to_datetime(pred_dates).dt.date
    
    tf_results = pd.DataFrame({
        'date': tf_dates,
        'actual': y_true_tf_unscaled,
        'predicted': y_pred_tf_unscaled
    })
    tf_results = tf_results.groupby('date').agg({
        'actual': 'sum',
        'predicted': 'sum'
    }).reset_index()
    
    # Compare metrics
    print(f"\nMetrics Comparison:")
    print(f"{'Metric':<15} {'Teacher Forcing':<20} {'Recursive':<20} {'Difference':<20}")
    print("-" * 75)
    if not np.isnan(mae_rec):
        print(f"{'MAE':<15} {mae_tf:<20.4f} {mae_rec:<20.4f} {mae_rec - mae_tf:<20.4f}")
        print(f"{'RMSE':<15} {rmse_tf:<20.4f} {rmse_rec:<20.4f} {rmse_rec - rmse_tf:<20.4f}")
        print(f"{'MSE':<15} {mse_tf:<20.4f} {mse_rec:<20.4f} {mse_rec - mse_tf:<20.4f}")
        if not np.isnan(accuracy_tf) and not np.isnan(accuracy_rec):
            accuracy_diff = accuracy_rec - accuracy_tf
            print(f"{'Accuracy':<15} {accuracy_tf:<20.2f}% {accuracy_rec:<19.2f}% {accuracy_diff:<19.2f}%")
        print(f"\nError increase: {(mae_rec / mae_tf - 1) * 100:.2f}% (MAE)")
        print(f"Error increase: {(rmse_rec / rmse_tf - 1) * 100:.2f}% (RMSE)")
    else:
        print(f"{'MAE':<15} {mae_tf:<20.4f} {'N/A':<20}")
        print(f"{'RMSE':<15} {rmse_tf:<20.4f} {'N/A':<20}")
        if not np.isnan(accuracy_tf):
            print(f"{'Accuracy':<15} {accuracy_tf:<20.2f}% {'N/A':<20}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Create timestamped run directory
    # Use the same suffix logic as model loading to match directory structure
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = "outputs/mvp_test"
    
    # Determine suffix based on inference category_mode (same logic as model loading)
    if category_mode == 'all':
        model_suffix = "_all"
    elif category_mode == 'single' and category_filter:
        model_suffix = f"_{category_filter}"
    else:
        model_suffix = ""  # Default: outputs/mvp_test/predictions/
    
    predictions_base_dir = Path(f"{base_output_dir}{model_suffix}/predictions")
    predictions_base_dir.mkdir(parents=True, exist_ok=True)
    output_dir = predictions_base_dir / f"run_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - Using timestamped directory: {output_dir}")
    
    # Load metadata from model directory for comparison (before saving anything)
    # Use the same model_dir_path that was determined during model loading
    model_dir = model_dir_path  # Use the model_dir_path from earlier (line 1038)
    metadata_source = model_dir / "metadata.json"
    if metadata_source.exists():
        with open(metadata_source, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        loss_function = metadata.get('training_config', {}).get('loss_function', 'Unknown')
    else:
        print(f"  - Warning: metadata.json not found in model directory")
        loss_function = 'Unknown'
        metadata = None
    
    # Scan previous runs BEFORE saving current run (to exclude current run from comparison)
    print("  - Scanning previous runs for comparison...")
    all_runs = scan_previous_runs(predictions_base_dir)
    # Filter out current run if it somehow got included
    current_run_id = f"run_{run_timestamp}"
    previous_runs = [run for run in all_runs if run['run_id'] != current_run_id]
    print(f"    Found {len(previous_runs)} previous run(s)")
    
    # Save Teacher Forcing results
    tf_output_path = output_dir / "predictions_teacher_forcing.csv"
    tf_results['date'] = pd.to_datetime(tf_results['date']).dt.strftime('%m/%d/%Y')
    tf_results['error'] = tf_results['actual'] - tf_results['predicted']
    tf_results['abs_error'] = np.abs(tf_results['error'])
    tf_results = tf_results.sort_values('date')
    tf_results.to_csv(tf_output_path, index=False)
    print(f"  - Teacher Forcing results: {tf_output_path}")
    
    # Save Recursive results (use unscaled predictions for output)
    rec_output_path = output_dir / "predictions_recursive.csv"
    recursive_results['date'] = pd.to_datetime(recursive_results['date']).dt.strftime('%m/%d/%Y')
    # Use unscaled predictions for error calculation if available
    pred_col_for_error = 'predicted_unscaled' if 'predicted_unscaled' in recursive_results.columns else 'predicted'
    recursive_results['predicted'] = recursive_results[pred_col_for_error]
    recursive_results['error'] = recursive_results['actual'] - recursive_results['predicted']
    recursive_results['abs_error'] = np.abs(recursive_results['error'])
    recursive_results = recursive_results.sort_values('date')
    # Drop intermediate columns for cleaner output
    output_cols = ['date', 'predicted', 'actual', 'error', 'abs_error']
    recursive_results = recursive_results[[c for c in output_cols if c in recursive_results.columns]]
    recursive_results.to_csv(rec_output_path, index=False)
    print(f"  - Recursive results: {rec_output_path}")
    
    # Copy metadata.json from model directory if it exists
    if metadata_source.exists():
        metadata_dest = output_dir / "metadata.json"
        shutil.copy2(metadata_source, metadata_dest)
        print(f"  - Copied metadata.json from model directory to: {metadata_dest}")
    
    # Prepare current run metrics
    current_metrics = {
        'run_id': f"run_{run_timestamp}",
        'loss_function': loss_function,
        'tf_mae': mae_tf,
        'tf_rmse': rmse_tf,
        'tf_mse': mse_tf,
        'rec_mae': mae_rec if not np.isnan(mae_rec) else None,
        'rec_rmse': rmse_rec if not np.isnan(rmse_rec) else None,
        'rec_mse': mse_rec if not np.isnan(mse_rec) else None
    }
    
    # Generate comparison table and improvement analysis
    comparison_table = generate_comparison_table(current_metrics, previous_runs)
    improvement_analysis = analyze_improvement(current_metrics, previous_runs)
    
    # Save comparison summary (renamed to summary.txt)
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Full Year 2025 Prediction Comparison: Teacher Forcing vs Recursive\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Run ID: run_{run_timestamp}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        category_info = f"Full year 2025, {'ALL categories' if category_mode == 'all' else f'{categories_to_predict[0]} category only'}\n"
        f.write(f"Data: {category_info}")
        f.write(f"Data source: dataset/test/data_2025.csv\n\n")
        
        f.write("Teacher Forcing Mode (Test Evaluation):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Description: Uses actual ground truth {data_config['target_col']} values from 2025 as features\n")
        f.write(f"  Suitable for: Model evaluation on test set\n")
        f.write(f"  Number of predictions: {len(y_pred_tf)}\n")
        f.write(f"  Date range: {pred_dates.min()} to {pred_dates.max()}\n")
        f.write(f"  MSE:  {mse_tf:.4f}\n")
        f.write(f"  MAE:  {mae_tf:.4f}\n")
        f.write(f"  RMSE: {rmse_tf:.4f}\n")
        if not np.isnan(accuracy_tf):
            f.write(f"  Accuracy: {accuracy_tf:.2f}%\n")
        f.write("\n")
        
        f.write("Recursive Mode (Production Forecast):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Description: Uses model's own predictions as inputs\n")
        f.write(f"  Suitable for: Production forecasting of unknown future dates\n")
        f.write(f"  Number of predictions: {len(recursive_preds)}\n")
        f.write(f"  Date range: 2025-01-01 to 2025-12-31\n")
        if not np.isnan(mae_rec):
            f.write(f"  MSE:  {mse_rec:.4f}\n")
            f.write(f"  MAE:  {mae_rec:.4f}\n")
            f.write(f"  RMSE: {rmse_rec:.4f}\n")
            if not np.isnan(accuracy_rec):
                f.write(f"  Accuracy: {accuracy_rec:.2f}%\n")
            f.write("\n")
        else:
            f.write(f"  MSE:  N/A (no actual values available)\n")
            f.write(f"  MAE:  N/A (no actual values available)\n")
            f.write(f"  RMSE: N/A (no actual values available)\n")
            f.write(f"  Accuracy: N/A (no actual values available)\n\n")
        
        if not np.isnan(mae_rec):
            f.write("Comparison:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  MAE increase:  {(mae_rec / mae_tf - 1) * 100:.2f}%\n")
            f.write(f"  RMSE increase: {(rmse_rec / rmse_tf - 1) * 100:.2f}%\n")
            f.write(f"  MSE increase:  {(mse_rec / mse_tf - 1) * 100:.2f}%\n")
            f.write(f"\n  Note: Recursive mode shows higher error due to error accumulation.\n")
            f.write(f"  This is expected and represents true production forecast performance.\n")
        
        # Add historical comparison
        f.write("\n" + comparison_table)
        f.write("\n" + improvement_analysis)
    
    print(f"  - Summary saved to: {summary_path}")
    
    # Generate plots
    if len(y_true_tf) > 0:
        plot_path_tf = output_dir / "predictions_teacher_forcing.png"
        n_samples = min(100, len(y_true_tf))
        plot_difference(
            y_true_tf[:n_samples],
            y_pred_tf[:n_samples],
            save_path=str(plot_path_tf),
            show=False
        )
        print(f"  - Teacher Forcing plot: {plot_path_tf}")
    
    if len(recursive_results_with_actuals) > 0:
        plot_path_rec = output_dir / "predictions_recursive.png"
        pred_col_for_plot = 'predicted_unscaled' if 'predicted_unscaled' in recursive_results_with_actuals.columns else 'predicted'
        plot_difference(
            recursive_results_with_actuals['actual'].values,
            recursive_results_with_actuals[pred_col_for_plot].values,
            save_path=str(plot_path_rec),
            show=False
        )
        print(f"  - Recursive plot: {plot_path_rec}")
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

