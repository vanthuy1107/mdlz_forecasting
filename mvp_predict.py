"""Prediction script for January 2025 using MVP test model.

This script loads the trained model from mvp_test.py and makes predictions
for January 2025 data, filtering to DRY category only.

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
import re
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List

from config import load_config, load_holidays
from src.data import (
    DataReader,
    ForecastDataset,
    RollingGroupScaler,
    slicing_window_category,
    encode_categories,
    add_holiday_features,
    add_temporal_features,
    aggregate_daily,
    add_cbm_density_features,
    apply_scaling,
    inverse_transform_scaling
)
from src.data.preprocessing import (
    get_us_holidays,
    get_vietnam_holidays,
    add_day_of_week_cyclical_features,
    add_eom_features,
    add_weekday_volume_tier_features,
    apply_sunday_to_monday_carryover,
    add_operational_status_flags,
)
from src.models import RNNWithCategory
from src.training import Trainer
from src.utils import plot_difference, upload_to_google_sheets, GSPREAD_AVAILABLE
from src.utils.google_sheets import upload_history_prediction


###############################################################################
# Holiday and Lunar Calendar Utilities
###############################################################################

# NOTE:
# We keep a single source of truth for Vietnamese holidays (including Tet)
# so that both the discrete holiday indicators and the continuous
# "days-to-lunar-event" features stay perfectly aligned.
# Holidays are now loaded from config/holidays.yaml for easier maintenance.
VIETNAM_HOLIDAYS_BY_YEAR = load_holidays(holiday_type="model")


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
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
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
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    lunar_dates = df[time_col].dt.date.apply(solar_to_lunar_date)
    df[lunar_month_col] = [ld[0] for ld in lunar_dates]
    df[lunar_day_col] = [ld[1] for ld in lunar_dates]
    return df


def add_lunar_cyclical_features(
    df: pd.DataFrame,
    lunar_month_col: str = "lunar_month",
    lunar_day_col: str = "lunar_day",
    lunar_month_sin_col: str = "lunar_month_sin",
    lunar_month_cos_col: str = "lunar_month_cos",
    lunar_day_sin_col: str = "lunar_day_sin",
    lunar_day_cos_col: str = "lunar_day_cos",
) -> pd.DataFrame:
    """
    Add sine/cosine cyclical encodings for the lunar calendar.

    Mirrors the training-time implementation so that the feature
    representation seen during prediction matches what the model
    was trained on.
    """
    df = df.copy()

    # Ensure base lunar features exist
    if lunar_month_col not in df.columns or lunar_day_col not in df.columns:
        raise ValueError(
            "Lunar calendar columns not found. "
            "Call add_lunar_calendar_features before add_lunar_cyclical_features."
        )

    # Month: 1-12 -> [0, 2π)
    df[lunar_month_sin_col] = np.sin(2 * np.pi * (df[lunar_month_col] - 1) / 12.0)
    df[lunar_month_cos_col] = np.cos(2 * np.pi * (df[lunar_month_col] - 1) / 12.0)

    # Day: 1-30 -> [0, 2π). We use 30 as an upper bound for simplicity.
    df[lunar_day_sin_col] = np.sin(2 * np.pi * (df[lunar_day_col] - 1) / 30.0)
    df[lunar_day_cos_col] = np.cos(2 * np.pi * (df[lunar_day_col] - 1) / 30.0)

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
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    
    # Filter out rows with NaT dates before processing
    valid_mask = df[time_col].notna()
    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()
    
    if len(df_valid) == 0:
        # If no valid dates, set all features to default values and return
        df[holiday_indicator_col] = 0
        df[days_until_holiday_col] = 365
        df[days_since_holiday_col] = 365
        return df
    
    min_date = df_valid[time_col].min().date()
    max_date = df_valid[time_col].max().date()
    extended_max = max_date + timedelta(days=365)
    holidays = get_vietnam_holidays(min_date, extended_max)
    holiday_set = set(holidays)
    df[holiday_indicator_col] = 0
    df[days_until_holiday_col] = np.nan
    df[days_since_holiday_col] = np.nan
    
    for idx, row in df_valid.iterrows():
        try:
            current_date = row[time_col].date()
        except (AttributeError, ValueError):
            # Skip if date conversion fails
            continue
        
        if current_date in holiday_set:
            df.at[idx, holiday_indicator_col] = 1
        
        next_holiday = None
        for holiday in holidays:
            try:
                if holiday > current_date:
                    next_holiday = holiday
                    break
            except (TypeError, ValueError):
                # Skip if comparison fails (e.g., NaT)
                continue
        df.at[idx, days_until_holiday_col] = (next_holiday - current_date).days if next_holiday else 365
        
        last_holiday = None
        for holiday in sorted(holidays, reverse=True):
            try:
                if holiday <= current_date:
                    last_holiday = holiday
                    break
            except (TypeError, ValueError):
                # Skip if comparison fails (e.g., NaT)
                continue
        df.at[idx, days_since_holiday_col] = (current_date - last_holiday).days if last_holiday else 365
    
    df[days_until_holiday_col] = df[days_until_holiday_col].fillna(365)
    df[days_since_holiday_col] = df[days_since_holiday_col].fillna(365)
    return df


def get_tet_start_dates(start_year: int, end_year: int) -> List[date]:
    """
    Get Tet (Lunar New Year) *start dates* for a year range.

    These are the anchor points for the "days_to_tet" continuous feature,
    representing the surge window that the model struggles with.
    """
    tet_dates: List[date] = []
    for year in range(start_year, end_year + 1):
        if year in VIETNAM_HOLIDAYS_BY_YEAR:
            tet_window = VIETNAM_HOLIDAYS_BY_YEAR[year]["tet"]
            if tet_window:
                # Use the first day of Tet as the event start
                tet_dates.append(tet_window[0])

    # Remove duplicates and sort
    tet_dates = sorted(list(set(tet_dates)))
    return tet_dates


def add_days_to_tet_feature(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    days_to_tet_col: str = "days_to_tet",
) -> pd.DataFrame:
    """
    Add a continuous "days_to_tet" feature based on the lunar Tet window.

    For each date, this feature is the number of days until the *start* of
    the next Tet holiday period (Lunar New Year). When the date falls inside
    the Tet window itself, the value is 0.

    This smooth countdown signal helps the model anticipate Tet-driven
    demand surges well before they appear in the immediate look-back window.
    """
    df = df.copy()

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')

    # Filter out rows with NaT dates before processing
    valid_mask = df[time_col].notna()
    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()
    
    # Initialize the column with default value
    df[days_to_tet_col] = 365
    
    if len(df_valid) == 0:
        # If no valid dates, set all to default and return
        return df

    min_date = df_valid[time_col].min().date()
    max_date = df_valid[time_col].max().date()

    # Extend a bit so all dates have a "next Tet"
    extended_max = max_date + timedelta(days=365)
    tet_start_dates = get_tet_start_dates(min_date.year, extended_max.year)

    if not tet_start_dates:
        # Fallback: no Tet dates configured, set large constant
        return df

    tet_start_dates = sorted(tet_start_dates)

    for idx, row in df_valid.iterrows():
        try:
            current_date = row[time_col].date()
        except (AttributeError, ValueError):
            # Skip if date conversion fails
            continue

        # Find the next Tet start on or after current_date
        next_tet = None
        for tet_date in tet_start_dates:
            try:
                if tet_date >= current_date:
                    next_tet = tet_date
                    break
            except (TypeError, ValueError):
                # Skip if comparison fails (e.g., NaT)
                continue

        if next_tet is None:
            # If we're beyond the last configured Tet, use a large value
            df.at[idx, days_to_tet_col] = 365
        else:
            try:
                df.at[idx, days_to_tet_col] = (next_tet - current_date).days
            except (TypeError, ValueError):
                # Fallback to default if calculation fails
                df.at[idx, days_to_tet_col] = 365

    df[days_to_tet_col] = df[days_to_tet_col].fillna(365)
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
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
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
    Calculate %accuracy based on the formula using
    SUM OF ABSOLUTE ERRORS over the whole period:
    
        Total_Error = Σ |y_pred - y_true|
        Total_Actual = Σ |y_true|
    
    If Total_Error > Total_Actual:
        ⟹ Accuracy = 0%
    
    If Total_Error ≤ Total_Actual:
        ⟹ %Bias = (Total_Error / Total_Actual) × 100%
        ⟹ Accuracy = 100% - %Bias
    
    This corresponds to "accuracy by each day then abs", i.e. we take the
    absolute error for every daily data point before summing.
    
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
    
    # Check if Total_Error > Total_Actual
    if total_error > total_actual:
        return 0.0
    
    # Calculate %Bias and Accuracy
    percent_bias = (total_error / total_actual) * 100.0
    accuracy = 100.0 - percent_bias
    
    return accuracy


def calculate_accuracy_sum_before_abs(y_true, y_pred):
    """
    Calculate %accuracy using SUM BEFORE ABS, i.e.:
    
        Total_Error_signed = | Σ (y_pred - y_true) |
        Total_Actual       = Σ |y_true|
    
    If Total_Error_signed > Total_Actual:
        ⟹ Accuracy = 0%
    
    If Total_Error_signed ≤ Total_Actual:
        ⟹ %Bias = (Total_Error_signed / Total_Actual) × 100%
        ⟹ Accuracy = 100% - %Bias
    
    This lets us see the net bias where over- and under-forecasting
    can cancel each other before taking the absolute value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid_mask.sum() == 0:
        return np.nan

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Sum BEFORE absolute: net signed error, then take abs once
    total_error_signed = np.abs(np.sum(y_pred_valid - y_true_valid))

    # Total_Actual is still the sum of absolute actual values
    total_actual = np.sum(np.abs(y_true_valid))

    if total_actual == 0:
        return np.nan

    if total_error_signed > total_actual:
        return 0.0

    percent_bias = (total_error_signed / total_actual) * 100.0
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
    # Load metadata first to get the training-time model/data config
    model_dir = Path(model_path).parent
    metadata_path = model_dir / "metadata.json"
    
    # ------------------------------------------------------------------
    # 1) Recover num_categories and full model architecture from metadata
    # ------------------------------------------------------------------
    num_categories = None
    trained_model_config = None
    trained_feature_cols = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Get full model_config (includes num_categories, input_dim, etc.)
        trained_model_config = metadata.get('model_config', {})
        if 'num_categories' in trained_model_config:
            num_categories = trained_model_config['num_categories']

        # Also recover the exact feature column list used during training
        trained_data_config = metadata.get("data_config", {})
        trained_feature_cols = trained_data_config.get("feature_cols")
    
    # Fallback to config if metadata doesn't have it
    if num_categories is None:
        model_config = config.model
        num_categories = model_config.get('num_categories')
    
    if num_categories is None:
        raise ValueError("num_categories must be found in model metadata or config")
    
    # Get category_filter from training metadata to know which category(ies) model was trained on
    trained_category_filter = None
    trained_cat2id = None  # Training-time category mapping
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if 'data_config' in metadata and 'category_filter' in metadata['data_config']:
            trained_category_filter = metadata['data_config']['category_filter']
        
        # Try to extract category mapping from log_summary
        log_summary = metadata.get('log_summary', '')
        # Look for "Category mapping: {...}" in log_summary
        match = re.search(r"Category mapping: ({[^}]+})", log_summary)
        if match:
            try:
                # Parse the dictionary string from log_summary
                trained_cat2id_str = match.group(1)
                # Convert single quotes to double quotes for JSON parsing
                trained_cat2id_str = trained_cat2id_str.replace("'", '"')
                trained_cat2id = json.loads(trained_cat2id_str)
            except:
                pass

    # If we have the training-time feature list, push it into the live config
    # so that window creation uses the exact same ordering and dimensionality.
    if trained_feature_cols is not None:
        config.set("data.feature_cols", list(trained_feature_cols))
    
    print(f"  - Loading model with num_categories={num_categories} (from trained model)")
    if trained_category_filter:
        print(f"  - Model was trained on category: {trained_category_filter}")
    else:
        print(f"  - Model was trained on: all categories (num_categories={num_categories})")
    if trained_cat2id:
        print(f"  - Training-time category mapping: {trained_cat2id}")
    
    # ------------------------------------------------------------------
    # 2) Build model with the *exact* architecture used during training
    #    (input_dim, hidden_size, n_layers, etc. come from metadata)
    # ------------------------------------------------------------------
    if trained_model_config is not None:
        # Override config.model with training-time values for safety
        model_config = config.model
        for k, v in trained_model_config.items():
            model_config[k] = v
    else:
        model_config = config.model

    model = RNNWithCategory(
        num_categories=num_categories,
        cat_emb_dim=model_config['cat_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        output_dim=model_config['output_dim'],
        use_layer_norm=model_config.get('use_layer_norm', True),
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
        # print(f"    Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")
    else:
        print(f"  [WARNING] Scaler not found at {scaler_path}, predictions will be in scaled space")
    
    return model, device, scaler, trained_category_filter, trained_cat2id


def prepare_prediction_data(data, config, cat2id, scaler : RollingGroupScaler, trained_cat2id=None):
    """
    Prepare data for prediction using the same preprocessing as training.
    
    Args:
        data: DataFrame with raw data
        config: Configuration object
        cat2id: Category to ID mapping from prediction data (may include all categories)
        scaler: Optional StandardScaler for QTY values (if None, no scaling applied)
        trained_cat2id: Training-time category mapping (used for remapping to match model)
    
    Returns:
        Prepared DataFrame ready for window creation
    """
    data_config = config.data
    time_col = data_config['time_col']
    cat_col = data_config['cat_col']
    cat_id_col = data_config['cat_id_col']
    
    # Ensure time column is datetime and sort
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col], dayfirst=True, errors='coerce')
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
    
    # Add cyclical day-of-week encoding (sin/cos)
    print("  - Adding cyclical day-of-week features (day_of_week_sin, day_of_week_cos)...")
    data = add_day_of_week_cyclical_features(
        data,
        time_col=time_col,
        day_of_week_sin_col="day_of_week_sin",
        day_of_week_cos_col="day_of_week_cos"
    )
    
    # Add weekday volume tier features
    print("  - Adding weekday volume tier features (weekday_volume_tier, is_high_volume_weekday)...")
    data = add_weekday_volume_tier_features(
        data,
        time_col=time_col,
        weekday_volume_tier_col="weekday_volume_tier",
        is_high_volume_weekday_col="is_high_volume_weekday"
    )
    
    # Add End-of-Month (EOM) surge features
    print("  - Adding EOM features (is_EOM, days_until_month_end)...")
    data = add_eom_features(
        data,
        time_col=time_col,
        is_eom_col="is_EOM",
        days_until_month_end_col="days_until_month_end",
        eom_window_days=3
    )
    
    # Add lunar calendar features (before aggregation)
    print("  - Adding lunar calendar features (lunar_month, lunar_day)...")
    data = add_lunar_calendar_features(
        data,
        time_col=time_col,
        lunar_month_col="lunar_month",
        lunar_day_col="lunar_day"
    )

    # Lunar cyclical encodings (sine/cosine) to mirror training-time features
    print("  - Adding lunar cyclical features (sine/cosine)...")
    data = add_lunar_cyclical_features(
        data,
        lunar_month_col="lunar_month",
        lunar_day_col="lunar_day",
        lunar_month_sin_col="lunar_month_sin",
        lunar_month_cos_col="lunar_month_cos",
        lunar_day_sin_col="lunar_day_sin",
        lunar_day_cos_col="lunar_day_cos",
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
    
    # Feature engineering: continuous countdown to Tet (lunar event)
    print("  - Adding Tet countdown feature (days_to_tet)...")
    data = add_days_to_tet_feature(
        data,
        time_col=time_col,
        days_to_tet_col="days_to_tet",
    )
    
    # Daily aggregation: Group by date and category, sum target
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

    # Context-aware operational status flags on the daily series.
    # As in training, if full calendar reindexing is desired, perform the
    # reindex step before this call and then rely on these flags +
    # anomaly-aware baselines to preserve interpretability.
    print("  - Tagging Holiday_OFF, Weekend_Downtime, and Operational_Anomalies...")
    data = add_operational_status_flags(
        data,
        time_col=time_col,
        target_col=data_config['target_col'],
        status_col="operational_status",
        expected_zero_flag_col="is_expected_zero",
        anomaly_flag_col="is_operational_anomaly",
    )

    # Apply Sunday-to-Monday demand carryover (same as training)
    print("  - Applying Sunday-to-Monday demand carryover...")
    data = apply_sunday_to_monday_carryover(
        data,
        time_col=time_col,
        cat_col=cat_col,
        target_col=data_config['target_col'],
        actual_col=data_config['target_col']
    )
    print("    - Monday's target now includes Sunday's demand (captures backlog accumulation)")

    # CBM/QTY density features (including last-year prior), same as training
    print("  - Adding CBM density features (cbm_per_qty, cbm_per_qty_last_year)...")
    data = add_cbm_density_features(
        data,
        cbm_col=data_config['target_col'],  # e.g., "Total CBM"
        qty_col="Total QTY",
        time_col=time_col,
        cat_col=cat_col,
        density_col="cbm_per_qty",
        density_last_year_col="cbm_per_qty_last_year",
    )
    
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
    
    # If we have training-time category mapping, remap to match model's expected IDs
    if trained_cat2id is not None:
        print(f"  - Remapping categories to match training-time mapping: {trained_cat2id}")
        # Only keep categories that the model was trained on
        trained_categories = set(trained_cat2id.keys())
        data_before = len(data)
        data = data[data[cat_col].isin(trained_categories)].copy()
        data_after = len(data)
        if data_before > data_after:
            print(f"  - Filtered out {data_before - data_after} samples with categories not in training data")
        
        # Remap to training-time IDs
        data[cat_id_col] = data[cat_col].map(trained_cat2id)
    else:
        # Use prediction-time mapping (fallback)
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
        data = scaler.transform(data)
        #data = apply_scaling(data, scaler, target_col=data_config['target_col'])
    
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


def apply_sunday_to_monday_carryover_predictions(
    predictions_df: pd.DataFrame,
    date_col: str = 'date',
    pred_col: str = 'predicted'
) -> pd.DataFrame:
    """
    Apply Sunday-to-Monday carryover rule to predictions.
    
    This post-processing step ensures predictions follow the same rule as training data:
    - Sunday predictions are forced to 0 (non-operational day)
    - Sunday's prediction value is added to the following Monday's prediction
    
    This maintains consistency between training target transformation and inference outputs.
    
    Args:
        predictions_df: DataFrame with 'date' and prediction column (e.g., 'predicted' or 'predicted_unscaled')
        date_col: Name of date column
        pred_col: Name of prediction column to adjust
    
    Returns:
        DataFrame with adjusted predictions where Sunday=0 and Monday includes Sunday's value
        (date column type is preserved)
    """
    df = predictions_df.copy()
    
    # Helper function to extract date object from various types
    def get_date_obj(d):
        if isinstance(d, date):
            return d
        elif isinstance(d, pd.Timestamp):
            return d.date()
        elif pd.api.types.is_datetime64_any_dtype(pd.Series([d])):
            return pd.to_datetime(d).date()
        else:
            return pd.to_datetime(d).date()
    
    # Extract date objects for processing (preserve original column)
    df['_date_obj'] = df[date_col].apply(get_date_obj)
    
    # Sort by date to ensure proper ordering
    df = df.sort_values('_date_obj').reset_index(drop=True)
    
    # Process each row to apply carryover
    for i in range(len(df)):
        current_date = df.loc[i, '_date_obj']
        
        # Check if current date is Sunday (weekday == 6)
        if current_date.weekday() == 6:  # Sunday
            sunday_value = df.loc[i, pred_col]
            
            # Find the next Monday (must be exactly 1 day later)
            if i + 1 < len(df):
                next_date = df.loc[i + 1, '_date_obj']
                days_diff = (next_date - current_date).days
                
                # Only apply if next day is Monday and exactly 1 day later
                if next_date.weekday() == 0 and days_diff == 1:  # Next day is Monday
                    # Add Sunday's value to Monday
                    df.loc[i + 1, pred_col] = df.loc[i + 1, pred_col] + sunday_value
            
            # Set Sunday to 0
            df.loc[i, pred_col] = 0.0
    
    # Clean up temporary column (original date_col is preserved)
    df = df.drop(columns=['_date_obj'])
    
    return df


def predict_direct_multistep(
    model,
    device,
    initial_window_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    config,
    cat_id: int
):
    """
    Direct multi-step prediction mode: predicts entire forecast horizon at once.
    
    This function addresses "Exposure Bias" by outputting the entire forecast
    window (e.g., 30 days) in a single forward pass, preventing error accumulation
    from recursive forecasting where errors compound step-by-step.
    
    Args:
        model: Trained PyTorch model (already on device and in eval mode)
        device: PyTorch device
        initial_window_data: DataFrame with last input_size days of historical data
                            Must have all feature columns and be sorted by time
        start_date: First date to predict (e.g., date(2025, 1, 1))
        end_date: Last date to predict (e.g., date(2025, 1, 30))
        config: Configuration object
        cat_id: Category ID (integer)
    
    Returns:
        DataFrame with columns: date, predicted, actual (if available)
    """
    window_config = config.window
    data_config = config.data
    
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    feature_cols = data_config['feature_cols']
    target_col = data_config['target_col']
    time_col = data_config['time_col']
    
    # Validate initial window has enough data
    if len(initial_window_data) < input_size:
        raise ValueError(
            f"Initial window must have at least {input_size} samples, "
            f"got {len(initial_window_data)}"
        )
    
    # Validate prediction range matches horizon
    num_days_to_predict = (end_date - start_date).days + 1
    if num_days_to_predict != horizon:
        print(f"  [WARNING] Prediction range ({num_days_to_predict} days) doesn't match horizon ({horizon}). "
              f"Will predict {horizon} days starting from {start_date}")
    
    # Get last input_size rows as the initial window
    window = initial_window_data.tail(input_size).copy()
    window = window.sort_values(time_col).reset_index(drop=True)
    
    print(f"  - Starting direct multi-step prediction from {start_date} to {end_date}")
    print(f"  - Initial window: {window[time_col].min()} to {window[time_col].max()}")
    print(f"  - Model will output {horizon} predictions at once (direct multi-step)")
    
    # Create input tensor from current window
    window_features = window[feature_cols].values  # Shape: (input_size, n_features)
    X_window = torch.tensor(
        window_features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)  # Shape: (1, input_size, n_features)
    
    cat_tensor = torch.tensor([cat_id], dtype=torch.long).to(device)
    
    # Make prediction: model outputs entire horizon at once
    with torch.no_grad():
        pred_scaled = model(X_window, cat_tensor).cpu().numpy()  # Shape: (1, output_dim)
    
    # Check if model supports direct multi-step (output_dim == horizon)
    # or if we need to handle single-step model (output_dim == 1)
    pred_scaled = pred_scaled.squeeze(0) if pred_scaled.ndim > 1 else pred_scaled
    model_output_dim = pred_scaled.shape[0] if pred_scaled.ndim > 0 else 1
    
    if model_output_dim == 1 and horizon > 1:
        # Model was trained for single-step prediction, but we need multi-step
        # Repeat the single prediction for all days in horizon
        print(f"  [INFO] Model outputs single value (output_dim=1), but horizon={horizon}.")
        print(f"         Repeating prediction for all {horizon} days.")
        single_pred = float(pred_scaled[0] if pred_scaled.ndim > 0 else pred_scaled)
        pred_scaled = np.repeat(single_pred, horizon)
    elif model_output_dim < horizon:
        # Model outputs fewer values than needed
        print(f"  [WARNING] Model output_dim ({model_output_dim}) < horizon ({horizon}).")
        print(f"           Repeating last prediction for remaining days.")
        # Repeat the last prediction for remaining days
        if pred_scaled.ndim == 0:
            pred_scaled = np.array([pred_scaled])
        pred_scaled = np.concatenate([
            pred_scaled,
            np.repeat(pred_scaled[-1], horizon - model_output_dim)
        ])
    elif model_output_dim > horizon:
        # Take only the first horizon predictions
        pred_scaled = pred_scaled[:horizon]
    
    # Ensure pred_scaled is 1D array with horizon elements
    if pred_scaled.ndim == 0:
        pred_scaled = np.repeat(pred_scaled, horizon)
    elif len(pred_scaled) != horizon:
        # Safety check: if still not matching, repeat or truncate
        if len(pred_scaled) < horizon:
            pred_scaled = np.concatenate([pred_scaled, np.repeat(pred_scaled[-1], horizon - len(pred_scaled))])
        else:
            pred_scaled = pred_scaled[:horizon]
    
    # Generate dates for the forecast horizon
    prediction_dates = [start_date + timedelta(days=i) for i in range(horizon)]
    
    # Apply holiday masking: force predictions to 0 on Vietnamese holidays
    extended_end = end_date + timedelta(days=365)
    holidays = get_vietnam_holidays(start_date, extended_end)
    holiday_set = set(holidays)
    
    predictions = []
    for i, pred_date in enumerate(prediction_dates):
        is_holiday = pred_date in holiday_set
        is_sunday = pred_date.weekday() == 6  # 0=Monday, 6=Sunday
        
        # Force to 0 on holidays or Sundays (Sunday will be handled by carryover post-processing)
        pred_value = 0.0 if (is_holiday or is_sunday) else float(pred_scaled[i])
        
        predictions.append({
            'date': pred_date,
            'predicted': pred_value
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    # Apply Sunday-to-Monday carryover: Sunday's value moves to Monday, Sunday becomes 0
    predictions_df = apply_sunday_to_monday_carryover_predictions(
        predictions_df,
        date_col='date',
        pred_col='predicted'
    )
    
    return predictions_df


def predict_direct_multistep_rolling(
    model,
    device,
    initial_window_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    config,
    cat_id: int
):
    """
    Predict for a long date range by looping through chunks of horizon days.
    
    This function extends predict_direct_multistep() to handle date ranges longer
    than the model's horizon (e.g., predicting a full year when horizon=30 days).
    It predicts in rolling chunks, updating the window with predictions from each
    chunk to use as input for the next chunk.
    
    Args:
        model: Trained PyTorch model (already on device and in eval mode)
        device: PyTorch device
        initial_window_data: DataFrame with last input_size days of historical data
                            Must have all feature columns and be sorted by time
        start_date: First date to predict (e.g., date(2025, 1, 1))
        end_date: Last date to predict (e.g., date(2025, 12, 31))
        config: Configuration object
        cat_id: Category ID (integer)
    
    Returns:
        DataFrame with columns: date, predicted
    """
    window_config = config.window
    data_config = config.data
    
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    feature_cols = data_config['feature_cols']
    target_col = data_config['target_col']
    time_col = data_config['time_col']
    
    # Validate initial window has enough data
    if len(initial_window_data) < input_size:
        raise ValueError(
            f"Initial window must have at least {input_size} samples, "
            f"got {len(initial_window_data)}"
        )
    
    # Initialize window with historical data
    window = initial_window_data.tail(input_size).copy()
    window = window.sort_values(time_col).reset_index(drop=True)
    
    all_predictions = []
    current_start = start_date
    chunk_num = 0
    
    total_days = (end_date - start_date).days + 1
    print(f"  - Starting rolling prediction for {total_days} days (horizon={horizon} days per chunk)")
    print(f"  - Date range: {start_date} to {end_date}")
    print(f"  - Initial window: {window[time_col].min()} to {window[time_col].max()}")
    
    while current_start <= end_date:
        chunk_num += 1
        # Calculate end date for this chunk (min of horizon days or remaining days)
        chunk_end = min(
            current_start + timedelta(days=horizon - 1),
            end_date
        )
        chunk_days = (chunk_end - current_start).days + 1
        
        print(f"\n  [Chunk {chunk_num}] Predicting {current_start} to {chunk_end} ({chunk_days} days)...")
        
        # Predict this chunk using direct multi-step
        chunk_predictions = predict_direct_multistep(
            model=model,
            device=device,
            initial_window_data=window,
            start_date=current_start,
            end_date=chunk_end,
            config=config,
            cat_id=cat_id
        )
        
        all_predictions.append(chunk_predictions)
        
        # Update window with predictions from this chunk
        # We need to convert predictions to feature rows and append to window
        print(f"  - Updating window with {len(chunk_predictions)} predictions...")
        
        # Get the last row from window as a template
        last_row_template = window.iloc[-1:].copy()
        
        # For each predicted date, create a feature row
        new_rows = []
        for _, pred_row in chunk_predictions.iterrows():
            pred_date = pred_row['date']
            pred_value = pred_row['predicted']  # Already in scaled space
            
            # Create new row based on template
            new_row = last_row_template.copy()
            new_row[time_col] = pd.Timestamp(pred_date)
            new_row[target_col] = pred_value  # Use predicted value
            
            # Compute temporal features for this date
            pred_datetime = pd.Timestamp(pred_date)
            month = pred_datetime.month
            dayofmonth = pred_datetime.day
            new_row['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
            new_row['dayofmonth_sin'] = np.sin(2 * np.pi * (dayofmonth - 1) / 31)
            new_row['dayofmonth_cos'] = np.cos(2 * np.pi * (dayofmonth - 1) / 31)
            
            # Compute weekend features
            day_of_week = pred_datetime.dayofweek
            new_row['is_weekend'] = 1 if day_of_week >= 5 else 0
            new_row['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            new_row['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Compute weekday volume tier
            if day_of_week == 2:  # Wednesday
                weekday_volume_tier = 2
            elif day_of_week == 4:  # Friday
                weekday_volume_tier = 1
            elif day_of_week in [1, 3]:  # Tuesday, Thursday
                weekday_volume_tier = 0
            else:
                weekday_volume_tier = -1
            new_row['weekday_volume_tier'] = weekday_volume_tier
            new_row['is_high_volume_weekday'] = 1 if day_of_week in [2, 4] else 0
            
            # Compute lunar calendar features
            lunar_month, lunar_day = solar_to_lunar_date(pred_date)
            new_row['lunar_month'] = lunar_month
            new_row['lunar_day'] = lunar_day
            
            # Compute holiday features
            extended_end = end_date + timedelta(days=365)
            holidays = get_vietnam_holidays(pred_date, extended_end)
            holiday_set = set(holidays)
            new_row['holiday_indicator'] = 1 if pred_date in holiday_set else 0
            
            # Days until next holiday
            next_holiday = None
            for holiday in holidays:
                if holiday > pred_date:
                    next_holiday = holiday
                    break
            new_row['days_until_next_holiday'] = (next_holiday - pred_date).days if next_holiday else 365
            
            # Days since last holiday
            last_holiday = None
            for holiday in sorted(holidays, reverse=True):
                if holiday <= pred_date:
                    last_holiday = holiday
                    break
            new_row['days_since_holiday'] = (pred_date - last_holiday).days if last_holiday else 365
            
            # EOM features
            is_eom = pred_datetime.day >= 28  # Approximate end of month
            new_row['is_EOM'] = 1 if is_eom else 0
            days_in_month = (pred_datetime.replace(day=28) + timedelta(days=4)).day
            days_until_month_end = days_in_month - pred_datetime.day
            new_row['days_until_month_end'] = max(0, days_until_month_end)
            
            new_rows.append(new_row)
        
        # Append new rows to window
        if new_rows:
            new_rows_df = pd.concat(new_rows, ignore_index=True)
            window = pd.concat([window, new_rows_df], ignore_index=True)
            window = window.sort_values(time_col).reset_index(drop=True)
            
            # Recompute rolling features for the updated window
            # This ensures rolling_mean_7d, rolling_mean_30d, and momentum are correct
            qty_values = window[target_col].values
            
            for i in range(len(window)):
                # Rolling mean 7d: last 7 values
                start_idx_7d = max(0, i - 6)
                rolling_mean_7d = np.mean(qty_values[start_idx_7d:i+1])
                
                # Rolling mean 30d: last 30 values (or all available if less)
                start_idx_30d = max(0, i - 29)
                rolling_mean_30d = np.mean(qty_values[start_idx_30d:i+1])
                
                # Momentum: 3d vs 14d
                start_idx_3d = max(0, i - 2)
                start_idx_14d = max(0, i - 13)
                rolling_mean_3d = np.mean(qty_values[start_idx_3d:i+1]) if i >= 2 else np.mean(qty_values[:i+1])
                rolling_mean_14d = np.mean(qty_values[start_idx_14d:i+1]) if i >= 13 else np.mean(qty_values[:i+1])
                momentum_3d_vs_14d = rolling_mean_3d - rolling_mean_14d
                
                window.loc[i, 'rolling_mean_7d'] = rolling_mean_7d
                window.loc[i, 'rolling_mean_30d'] = rolling_mean_30d
                window.loc[i, 'momentum_3d_vs_14d'] = momentum_3d_vs_14d
            
            # Keep only the last input_size rows for next iteration
            window = window.tail(input_size).copy()
            window = window.sort_values(time_col).reset_index(drop=True)
        
        # Move to next chunk
        current_start = chunk_end + timedelta(days=1)
    
    # Combine all predictions
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        # Remove duplicates (in case of overlap) by keeping first occurrence
        final_predictions = final_predictions.drop_duplicates(subset=['date'], keep='first')
        final_predictions = final_predictions.sort_values('date').reset_index(drop=True)
        print(f"\n  - Completed rolling prediction: {len(final_predictions)} total predictions")
        return final_predictions
    else:
        return pd.DataFrame(columns=['date', 'predicted'])


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
    DEPRECATED: Recursive prediction mode (replaced by direct multi-step).
    
    This function is kept for backward compatibility but should not be used.
    Use predict_direct_multistep() instead to avoid exposure bias.
    
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
    print("  [WARNING] Using deprecated recursive prediction. Consider using direct multi-step instead.")
    
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
        
        # Compute cyclical day-of-week encoding (sin/cos)
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Compute weekday volume tier features
        # 2 = Wednesday (highest), 1 = Friday (high), 0 = Tuesday/Thursday (low), -1 = Monday/Saturday/Sunday (neutral)
        if day_of_week == 2:  # Wednesday
            weekday_volume_tier = 2
        elif day_of_week == 4:  # Friday
            weekday_volume_tier = 1
        elif day_of_week in [1, 3]:  # Tuesday, Thursday
            weekday_volume_tier = 0
        else:  # Monday, Saturday, Sunday
            weekday_volume_tier = -1
        
        is_high_volume_weekday = 1 if day_of_week in [2, 4] else 0  # Wednesday or Friday
        
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

        # ------------------------------------------------------------------
        # Holiday Volume Zero‑Constraint (mask output, keep rolling window)
        # ------------------------------------------------------------------
        # For warehouse day‑off dates (Vietnam holidays), we **force** the
        # published prediction to 0, but we keep the *unmasked* prediction
        # inside the recursive window so that rolling_mean_7d/30d and
        # momentum features don't collapse around temporary zeros.
        is_holiday = bool(holiday_indicator == 1)

        # Value used for model‑internal rolling statistics
        pred_for_rolling = pred_scaled
        # Value exposed to users / saved to CSV (zero on holidays)
        pred_for_output = 0.0 if is_holiday else pred_scaled

        # Store prediction for current_date (scaled space; will be inverse‑transformed later)
        predictions.append({
            'date': current_date,
            'predicted': pred_for_output
        })
        
        # Update window: remove oldest row, add new row with prediction
        # Create new row with predicted QTY and calendar features for current_date
        new_row = window.iloc[-1:].copy()  # Copy last row as template
        new_row[time_col] = current_datetime
        # Use unmasked prediction for the recursive state so rolling features
        # see the "true" model belief even when output is later masked to 0.
        new_row[target_col] = pred_for_rolling
        new_row['month_sin'] = month_sin
        new_row['month_cos'] = month_cos
        new_row['dayofmonth_sin'] = dayofmonth_sin
        new_row['dayofmonth_cos'] = dayofmonth_cos
        new_row['is_weekend'] = is_weekend
        new_row['day_of_week'] = day_of_week
        new_row['day_of_week_sin'] = day_of_week_sin
        new_row['day_of_week_cos'] = day_of_week_cos
        new_row['weekday_volume_tier'] = weekday_volume_tier
        new_row['is_high_volume_weekday'] = is_high_volume_weekday
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
    # Load configuration
    print("\n[1/7] Loading configuration...")
    config = load_config()
    
    # Note: We'll determine historical_year after parsing prediction dates
    # This config setting is not critical as we load historical data directly
    data_config = config.data

    # -----------------------------------------------------------------------
    # 0. Resolve prediction horizon from config.inference so the time range
    #    can be adjusted easily without touching code.
    # -----------------------------------------------------------------------
    inference_config = config.inference or {}
    prediction_data_path_cfg = inference_config.get(
        "prediction_data_path",
        "dataset/test/data_2025.csv",
    )
    prediction_start_str = inference_config.get("prediction_start", "2025-01-01")
    prediction_end_str = inference_config.get("prediction_end", "2025-01-31")

    # Convert to date objects
    prediction_start_date = pd.to_datetime(prediction_start_str).date()
    prediction_end_date = pd.to_datetime(prediction_end_str).date()
    if prediction_end_date < prediction_start_date:
        raise ValueError(
            f"inference.prediction_end ({prediction_end_str}) "
            f"must be on or after inference.prediction_start ({prediction_start_str})"
        )

    # Determine historical year: use the year before prediction start year
    # (e.g., if predicting 2026, use 2025 as historical reference)
    historical_year = prediction_start_date.year - 1

    # For pandas filtering we use an exclusive upper bound (end + 1 day)
    prediction_filter_start = prediction_start_date.isoformat()
    prediction_filter_end = (prediction_end_date + timedelta(days=1)).isoformat()

    print(
        f"PREDICTION WINDOW: {prediction_start_date} to {prediction_end_date} "
        f"(loaded from config.inference)"
    )
    horizon_days = (config.window or {}).get("horizon", 30)
    requested_days = (prediction_end_date - prediction_start_date).days + 1
    if requested_days > horizon_days:
        print(
            f"[INFO] Requested {requested_days} days exceeds horizon ({horizon_days} days). "
            f"Will use rolling prediction to cover the full date range from "
            f"{prediction_start_date} to {prediction_end_date}."
        )
    print("=" * 80)
    print(f"\n[2/7] Loading historical {historical_year} data...")
    data_reader = DataReader(
        data_dir=data_config['data_dir'],
        file_pattern=data_config['file_pattern']
    )
    
    try:
        ref_data = data_reader.load(years=[historical_year])
    except FileNotFoundError:
        print("[WARNING] Trying pattern-based loading...")
        ref_data = data_reader.load_by_file_pattern(
            years=[historical_year],
            file_prefix=data_config.get('file_prefix', 'Outboundreports')
        )
    
    # Get category mode from TRAINING config (use the same setting for train & prediction)
    # This ensures we don't have two separate places to configure category_mode/category_filter.
    # mvp_test.py also reads these from config.data.
    data_config = config.data
    category_mode = data_config.get('category_mode', 'single')  # Default to 'single' for backward compatibility
    category_filter = data_config.get('category_filter', 'DRY')  # Default to 'DRY' for backward compatibility
    
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
    elif category_mode == 'each':
        # For 'each' mode, determine which categories have trained models
        # Use major_categories if specified, otherwise use all available categories
        major_categories = data_config.get("major_categories", [])
        if major_categories:
            categories_to_predict = [cat for cat in major_categories if cat in available_categories]
            print(f"  - 'each' mode: Will process major_categories: {categories_to_predict}")
        else:
            categories_to_predict = available_categories
            print(f"  - 'each' mode: Will process all available categories: {categories_to_predict}")
    else:
        raise ValueError(f"Invalid category_mode: {category_mode}. Must be 'all', 'single', or 'each'")
    
    print(f"  - Categories to predict: {categories_to_predict}")
    
    # Load prediction data (e.g., 2025 file)
    print("\n[4/7] Loading prediction data...")
    data_2025_path = Path(prediction_data_path_cfg)
    
    if not data_2025_path.exists():
        raise FileNotFoundError(
            f"Prediction data file not found at: {data_2025_path.absolute()}"
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
    
    # Filter to desired prediction window (keep all categories for now - will filter per category later)
    print("\n[5/7] Filtering data...")
    cat_col = data_config['cat_col']
    time_col = data_config['time_col']
    
    # Filter to configured prediction window first
    if not pd.api.types.is_datetime64_any_dtype(data_2025[time_col]):
        # The prediction CSV can contain dates like "13/01/2025" (dd/mm/YYYY).
        # Use a robust parser that supports mixed formats and day-first dates.
        data_2025[time_col] = pd.to_datetime(
            data_2025[time_col],
            format="mixed",
            dayfirst=True,
        )
    
    # Store original data to check date range if filtering returns empty
    data_2025_original = data_2025.copy()
    
    data_2025 = data_2025[
        (data_2025[time_col] >= prediction_filter_start)
        & (data_2025[time_col] < prediction_filter_end)
    ].copy()
    print(
        f"  - After date filter [{prediction_filter_start} .. {prediction_filter_end}): "
        f"{len(data_2025)} samples"
    )
    
    # If no data after filtering, provide helpful error message with available date range
    if len(data_2025) == 0:
        if len(data_2025_original) > 0:
            min_date = data_2025_original[time_col].min()
            max_date = data_2025_original[time_col].max()
            raise ValueError(
                f"No data found in prediction file for the specified date range "
                f"[{prediction_filter_start} .. {prediction_filter_end}).\n"
                f"Available date range in file: {min_date.date()} to {max_date.date()}\n"
                f"Please update config.inference.prediction_start and prediction_end to "
                f"match the available date range, or use a different prediction data file."
            )
        else:
            raise ValueError(
                f"Prediction data file is empty or contains no valid data."
            )
    
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
    
    # For "each" mode, we need to check that models exist for all categories
    if category_mode == 'each':
        print("\n[6/7] Checking for trained models for each category...")
        base_output_dir = "outputs/mvp_test"
        missing_models = []
        for cat in categories_to_predict:
            model_path = Path(f"{base_output_dir}_{cat}/models/best_model.pth")
            if not model_path.exists():
                missing_models.append(cat)
        if missing_models:
            raise FileNotFoundError(
                f"Models not found for categories: {missing_models}. "
                f"Please run mvp_test.py with category_mode='each' first to train all category models."
            )
        print(f"  - All required models found for {len(categories_to_predict)} category(ies)")
    
    # Load trained model (to get scaler before data preparation)
    # Determine model directory based on inference category_mode
    # Match the directory structure from mvp_test.py (outputs/mvp_test{suffix}/models/)
    print("\n[6/7] Loading trained model...")
    base_output_dir = "outputs/mvp_test"
    
    # Determine suffix based on inference category_mode
    # Note: This should match what mvp_test.py uses when training
    # For "all": uses suffix "_all" -> outputs/mvp_test_all/models/
    # For "single": uses suffix "_{category}" -> outputs/mvp_test_{category}/models/
    # For "each": will be handled per-category in the loop below
    # Default (no suffix): outputs/mvp_test/models/
    
    if category_mode == 'all':
        model_suffix = "_all"
    elif category_mode == 'single' and category_filter:
        model_suffix = f"_{category_filter}"
    else:
        model_suffix = ""  # Default: outputs/mvp_test/models/
    
    # For "each" mode, we'll process each category separately with its own model
    # Store all results to combine later
    if category_mode == 'each':
        all_category_results = []
        
        for cat_idx, current_cat in enumerate(categories_to_predict, 1):
            print(f"\n{'=' * 80}")
            print(f"PROCESSING CATEGORY {cat_idx}/{len(categories_to_predict)}: {current_cat}")
            print(f"{'=' * 80}")
            
            # Load this category's model
            model_suffix = f"_{current_cat}"
            model_dir_path = Path(f"{base_output_dir}{model_suffix}/models")
            model_path = model_dir_path / "best_model.pth"
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found for category '{current_cat}' at {model_path}. "
                    f"Please run mvp_test.py with category_mode='each' first."
                )
            
            print(f"  - Using model from: {model_path}")
            model, device, scaler, trained_category_filter, trained_cat2id = load_model_for_prediction(str(model_path), config)
            
            # Filter data to this category only
            cat_to_process = [current_cat]
            
            # Process this category (will be continued below with rest of prediction logic)
            # For now, we'll handle it in the main flow by temporarily setting variables
            all_category_results.append({
                'category': current_cat,
                'model': model,
                'device': device,
                'scaler': scaler,
                'trained_category_filter': trained_category_filter,
                'trained_cat2id': trained_cat2id
            })
        
        # For "each" mode, we need to load a model for Teacher Forcing evaluation
        # We'll use the first category's model as a placeholder, but process all categories
        # The recursive section will handle loading each category's model separately
        print(f"\n[NOTE] 'each' mode: Will process all {len(categories_to_predict)} categories.")
        print(f"       Teacher Forcing will use first category's model as placeholder.")
        print(f"       Recursive mode will load each category's model separately.")
        current_cat = categories_to_predict[0]
        model_suffix = f"_{current_cat}"
        model_dir_path = Path(f"{base_output_dir}{model_suffix}/models")
        model_path = model_dir_path / "best_model.pth"
        print(f"  - Using model from: {model_path} (for Teacher Forcing placeholder)")
        model, device, scaler, trained_category_filter, trained_cat2id = load_model_for_prediction(str(model_path), config)
        # Keep all categories_to_predict - don't filter to just first category
        # categories_to_predict remains as is with all categories
    else:
        # Normal flow for "all" or "single" mode
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
        
        model, device, scaler, trained_category_filter, trained_cat2id = load_model_for_prediction(str(model_path), config)
    
    # IMPORTANT: Filter predictions to only categories the model was trained on
    # If model was trained on a single category, we can only predict that category
    # EXCEPTION: For "each" mode, we want to process all categories separately, so don't filter here
    if category_mode != 'each' and trained_category_filter is not None:
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
    elif category_mode == 'each':
        print(f"\n[INFO] 'each' mode: Will process all {len(categories_to_predict)} categories with their respective models.")
    
    # Prepare data with scaler if available (matching training pipeline)
    # For "each" mode, we'll prepare data separately per category, so skip global preparation
    if category_mode != 'each':
        print("\n[6.5/7] Preparing data with scaling...")
        historical_data_prepared = prepare_prediction_data(ref_data.copy(), config, cat2id, scaler, trained_cat2id)
        data_2025_prepared = prepare_prediction_data(data_2025, config, cat2id, scaler, trained_cat2id)
        
        # Get last N days of historical year for initial window (after aggregation/scaling)
        historical_window = get_historical_window_data(
            historical_data_prepared,
            end_date=date(historical_year, 12, 31),
            config=config,
            num_days=config.window['input_size']
        )
        print(f"  - Historical window: {len(historical_window)} samples")
        print(f"  - Date range: {historical_window[data_config['time_col']].min()} to {historical_window[data_config['time_col']].max()}")
        
        # Recreate windows after scaling (all categories or filtered by categories_to_predict)
        X_pred, y_actual, cat_pred, pred_dates = create_prediction_windows(data_2025_prepared, config)
    else:
        # For "each" mode, data preparation will happen per-category in the recursive loop
        print("\n[6.5/7] Data preparation will be done per-category in recursive mode.")
        X_pred, y_actual, cat_pred, pred_dates = np.array([]), np.array([]), np.array([]), []
        historical_window = pd.DataFrame()
    
    # Filter predictions to selected categories if needed
    # Filter if: single category mode OR model was trained on single category (but not "each" mode)
    if category_mode == 'single' or (category_mode != 'each' and trained_category_filter is not None and len(categories_to_predict) == 1):
        # Filter windows to selected category only
        # Use trained_cat2id if available, otherwise fall back to cat2id
        mapping_to_use = trained_cat2id if trained_cat2id is not None else cat2id
        cat_id = mapping_to_use.get(categories_to_predict[0])
        if cat_id is None:
            raise ValueError(f"Category '{categories_to_predict[0]}' not found in category mapping")
        mask = cat_pred == cat_id
        X_pred = X_pred[mask]
        y_actual = y_actual[mask]
        cat_pred = cat_pred[mask]
        pred_dates = pred_dates[mask] if hasattr(pred_dates, '__getitem__') else [pred_dates[i] for i in range(len(pred_dates)) if mask[i]]
        print(f"  - Filtered prediction windows to category '{categories_to_predict[0]}' (category ID: {cat_id})")
        print(f"  - Prediction windows after filtering: {len(X_pred)}")
    elif category_mode == 'each':
        # For "each" mode, Teacher Forcing will be skipped (only Recursive mode processes all categories)
        print("\n[NOTE] Teacher Forcing mode skipped for 'each' mode.")
        print("       All categories will be processed in Recursive mode with their respective models.")
    
    # ========================================================================
    # MODE 1: TEACHER FORCING (Test Evaluation) - Uses actual 2025 values
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODE 1: TEACHER FORCING (Test Evaluation)")
    print("=" * 80)
    if category_mode == 'each':
        print("[NOTE] Teacher Forcing is skipped in 'each' mode. Only Recursive mode processes categories.")
        print("       This is expected - Recursive mode will handle all categories independently.")
    else:
        print(f"Using actual ground truth {data_config['target_col']} values from 2025 as features.")
        print("This mode is suitable for model evaluation but not production forecasting.")
    
    # Skip Teacher Forcing for "each" mode
    if category_mode == 'each':
        print("  - Skipping Teacher Forcing evaluation for 'each' mode")
        y_true_tf = np.array([])
        y_pred_tf = np.array([])
        y_true_tf_unscaled = np.array([])
        y_pred_tf_unscaled = np.array([])
        mse_tf = np.nan
        mae_tf = np.nan
        rmse_tf = np.nan
        accuracy_tf = np.nan
        accuracy_tf_abs = np.nan
        accuracy_tf_sum = np.nan
    else:
        # Create dataset and dataloader
        pred_dataset = ForecastDataset(X_pred, y_actual, cat_pred)
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=config.training['test_batch_size'],
            shuffle=False
        )

        # If there are no prediction windows (e.g., January 2025 with large input_size/horizon),
        # skip teacher-forcing evaluation to avoid division-by-zero in the trainer.
        if len(pred_dataset) == 0:
            print("  [WARNING] No prediction windows available for Teacher Forcing. Skipping Mode 1 evaluation.")
            y_true_tf = np.array([])
            y_pred_tf = np.array([])
            y_true_tf_unscaled = np.array([])
            y_pred_tf_unscaled = np.array([])
            mse_tf = np.nan
            mae_tf = np.nan
            rmse_tf = np.nan
            accuracy_tf = np.nan
            # Also initialize detailed accuracy variants to avoid UnboundLocalError
            accuracy_tf_abs = np.nan
            accuracy_tf_sum = np.nan
        else:
            # Create trainer for prediction
            trainer = Trainer(
                model=model,
                criterion=nn.MSELoss(),
                optimizer=torch.optim.Adam(model.parameters()),
                device=device
            )

            y_true_tf, y_pred_tf = trainer.predict(pred_loader)

            print(f"  - Predictions made: {len(y_pred_tf)} samples")
            if len(pred_dates) > 0:
                print(f"  - Prediction date range: {pred_dates.min()} to {pred_dates.max()}")
            else:
                print("  - Prediction date range: N/A (no prediction windows)")

        # Inverse transform predictions and actuals if scaler is available
        # Only do this if we have actual predictions (arrays are not empty)
        if len(y_true_tf) > 0 and len(y_pred_tf) > 0:
            if scaler is not None:
                print("  - Inverse transforming scaled predictions to original scale...")
                # y_true_tf_unscaled = inverse_transform_scaling(y_true_tf.flatten(), scaler)
                # y_pred_tf_unscaled = inverse_transform_scaling(y_pred_tf.flatten(), scaler)
                y_true_tf_unscaled = scaler.inverse_transform_y(y_true_tf, cat_pred)
                y_pred_tf_unscaled = scaler.inverse_transform_y(y_pred_tf, cat_pred)
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
            
            # Apply Sunday-to-Monday carryover rule to teacher-forcing predictions
            if len(pred_dates) > 0 and len(y_pred_tf_unscaled) == len(pred_dates):
                print("  - Applying Sunday-to-Monday demand carryover to teacher-forcing predictions...")
                # Create temporary DataFrame for processing
                tf_pred_df = pd.DataFrame({
                    'date': pred_dates,
                    'predicted': y_pred_tf_unscaled
                })
                # Apply carryover (grouped by category if cat_pred is available)
                if len(cat_pred) == len(tf_pred_df):
                    tf_pred_df['category_id'] = cat_pred
                    def apply_carryover_per_cat_tf(group):
                        return apply_sunday_to_monday_carryover_predictions(
                            group,
                            date_col='date',
                            pred_col='predicted'
                        )
                    tf_pred_df = tf_pred_df.groupby('category_id', group_keys=False).apply(
                        apply_carryover_per_cat_tf
                    ).reset_index(drop=True)
                else:
                    tf_pred_df = apply_sunday_to_monday_carryover_predictions(
                        tf_pred_df,
                        date_col='date',
                        pred_col='predicted'
                    )
                y_pred_tf_unscaled = tf_pred_df['predicted'].values
                print("    - Sunday predictions set to 0, Sunday values added to following Monday")

            # Calculate metrics on unscaled values
            mse_tf = np.mean((y_true_tf_unscaled - y_pred_tf_unscaled) ** 2)
            mae_tf = np.mean(np.abs(y_true_tf_unscaled - y_pred_tf_unscaled))
            rmse_tf = np.sqrt(mse_tf)
            # Two accuracy views:
            #  - accuracy_tf_abs: Σ|error| / Σ|actual|  (daily abs before sum)
            #  - accuracy_tf_sum: |Σ error| / Σ|actual| (sum before abs)
            accuracy_tf_abs = calculate_accuracy(y_true_tf_unscaled, y_pred_tf_unscaled)
            accuracy_tf_sum = calculate_accuracy_sum_before_abs(y_true_tf_unscaled, y_pred_tf_unscaled)
        # If arrays are empty, metrics are already set to np.nan above
        # Preserve existing variable name for downstream compatibility
        accuracy_tf = accuracy_tf_abs
        
        print(f"\n  - MSE:  {mse_tf:.4f}")
        print(f"  - MAE:  {mae_tf:.4f}")
        print(f"  - RMSE: {rmse_tf:.4f}")
        if not np.isnan(accuracy_tf_abs):
            print(f"  - Accuracy (Σ|err|):       {accuracy_tf_abs:.2f}%")
        if not np.isnan(accuracy_tf_sum):
            print(f"  - Accuracy (|Σ err|):      {accuracy_tf_sum:.2f}%")
    
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
    # Use trained_cat2id if available, otherwise fall back to cat2id
    mapping_to_use = trained_cat2id if trained_cat2id is not None else cat2id
    if category_mode == 'single':
        cat_id = mapping_to_use.get(categories_to_predict[0])
        if cat_id is None:
            raise ValueError(f"Category '{categories_to_predict[0]}' not found in category mapping")
        process_categories = [categories_to_predict[0]]
    elif category_mode == 'each':
        # For "each" mode, process all categories from the original list
        # Each will use its own model loaded separately
        process_categories = categories_to_predict
        print(f"  - 'each' mode: Will process {len(process_categories)} categories with separate models")
    else:
        # For "all" mode, start from categories_to_predict but restrict
        # to only those the trained model actually knows about.
        known_categories = set(mapping_to_use.keys())
        process_categories = [c for c in categories_to_predict if c in known_categories]
        skipped_categories = [c for c in categories_to_predict if c not in known_categories]
        if skipped_categories:
            print(
                f"  [INFO] Skipping categories not present in trained model mapping: "
                f"{skipped_categories}. Model mapping keys: {sorted(known_categories)}"
            )
        if not process_categories:
            raise ValueError(
                "No overlapping categories between categories_to_predict and trained model mapping. "
                f"categories_to_predict={categories_to_predict}, trained_mapping_keys={sorted(known_categories)}"
            )

    recursive_results_list = []

    for current_category in process_categories:
        # For "each" mode, load this category's specific model and prepare data independently
        if category_mode == 'each':
            print(f"\n{'=' * 80}")
            print(f"PROCESSING CATEGORY: {current_category}")
            print(f"{'=' * 80}")
            print(f"  - Loading model for category: {current_category}")
            model_suffix_cat = f"_{current_category}"
            model_dir_path_cat = Path(f"{base_output_dir}{model_suffix_cat}/models")
            model_path_cat = model_dir_path_cat / "best_model.pth"
            
            if not model_path_cat.exists():
                print(f"    [WARNING] Model not found for {current_category} at {model_path_cat}, skipping...")
                continue
            
            model_cat, device_cat, scaler_cat, trained_cat_filter, trained_cat2id_cat = load_model_for_prediction(str(model_path_cat), config)
            # Use this category's model and mappings
            model = model_cat
            device = device_cat
            scaler = scaler_cat
            trained_category_filter = trained_cat_filter
            mapping_to_use = trained_cat2id_cat if trained_cat2id_cat is not None else cat2id
            
            # Prepare data independently for this category using its own scaler/mapping
            print(f"  - Preparing data for {current_category} using its own scaler/mapping...")
            historical_data_prepared_cat = prepare_prediction_data(ref_data.copy(), config, cat2id, scaler_cat, trained_cat2id_cat)
            
            # Get historical window for this category
            historical_window_cat = get_historical_window_data(
                historical_data_prepared_cat,
                end_date=date(historical_year, 12, 31),
                config=config,
                num_days=config.window['input_size']
            )
            print(f"  - Historical window for {current_category}: {len(historical_window_cat)} samples")
            
            # Filter historical window to this category only
            cat_id = mapping_to_use.get(current_category)
            if cat_id is None:
                raise ValueError(f"Category '{current_category}' not found in category mapping")
            historical_window_filtered = historical_window_cat[
                historical_window_cat[data_config['cat_id_col']] == cat_id
            ].copy()
            
            # Prepare actuals for this category
            data_2025_unscaled_cat = prepare_prediction_data(
                data_2025.copy(), config, cat2id, scaler=None, trained_cat2id=trained_cat2id_cat
            )
            data_2025_unscaled_cat['date'] = pd.to_datetime(data_2025_unscaled_cat[time_col]).dt.date
            actuals_by_date_cat = data_2025_unscaled_cat.groupby(
                ['date', data_config['cat_col']]
            )[data_config['target_col']].sum().reset_index()
            actuals_by_date_cat = actuals_by_date_cat.rename(columns={data_config['target_col']: 'actual'})
        else:
            # For non-"each" mode, use the global prepared data
            # Build per-(date, category) actuals on ORIGINAL scale once, then reuse
            if current_category == process_categories[0]:  # Only do this once for first category
                data_2025_unscaled = prepare_prediction_data(
                    data_2025.copy(), config, cat2id, scaler=None, trained_cat2id=trained_cat2id
                )
                data_2025_unscaled['date'] = pd.to_datetime(data_2025_unscaled[time_col]).dt.date
                actuals_by_date_cat = data_2025_unscaled.groupby(
                    ['date', data_config['cat_col']]
                )[data_config['target_col']].sum().reset_index()
                actuals_by_date_cat = actuals_by_date_cat.rename(columns={data_config['target_col']: 'actual'})
            
            cat_id = mapping_to_use.get(current_category)
            if cat_id is None:
                raise ValueError(f"Category '{current_category}' not found in category mapping")

            # Filter historical window to this category
            historical_window_filtered = historical_window[
                historical_window[data_config['cat_id_col']] == cat_id
            ].copy()

        if len(historical_window_filtered) < config.window['input_size']:
            # If not enough data for this category, use available data and duplicate
            print(
                f"  [WARNING] Only {len(historical_window_filtered)} samples for category "
                f"{current_category} (ID={cat_id}), need {config.window['input_size']}"
            )
            if len(historical_window_filtered) > 0:
                last_row = historical_window_filtered.iloc[-1:].copy()
                while len(historical_window_filtered) < config.window['input_size']:
                    historical_window_filtered = pd.concat(
                        [historical_window_filtered, last_row], ignore_index=True
                    )
            else:
                raise ValueError(f"No historical data found for category ID {cat_id}")

        # Get last input_size samples
        historical_window_filtered = historical_window_filtered.tail(
            config.window['input_size']
        ).copy()
        historical_window_filtered = historical_window_filtered.sort_values(
            time_col
        ).reset_index(drop=True)

        # Run direct multi-step prediction over configured prediction window
        # This addresses exposure bias by predicting entire horizon at once
        # If date range is longer than horizon, use rolling prediction
        horizon_days = config.window.get('horizon', 30)
        requested_days = (prediction_end_date - prediction_start_date).days + 1
        
        if requested_days > horizon_days:
            # Use rolling prediction for long date ranges
            print(f"  - Date range ({requested_days} days) exceeds horizon ({horizon_days} days)")
            print(f"  - Using rolling prediction to cover full date range")
            recursive_preds = predict_direct_multistep_rolling(
                model=model,
                device=device,
                initial_window_data=historical_window_filtered,
                start_date=prediction_start_date,
                end_date=prediction_end_date,
                config=config,
                cat_id=cat_id
            )
        else:
            # Use single-shot prediction for short date ranges
            recursive_preds = predict_direct_multistep(
                model=model,
                device=device,
                initial_window_data=historical_window_filtered,
                start_date=prediction_start_date,
                end_date=prediction_end_date,
                config=config,
                cat_id=cat_id
            )
        # Attach category label so downstream metrics/CSVs can group by category
        recursive_preds[data_config['cat_col']] = current_category

        # Ensure date columns have compatible types for merging
        # Convert both to date objects to avoid type mismatch
        recursive_preds['date'] = pd.to_datetime(recursive_preds['date']).dt.date
        actuals_by_date_cat['date'] = pd.to_datetime(actuals_by_date_cat['date']).dt.date

        # Merge with actuals for this (date, category) pair
        recursive_results_cat = recursive_preds.merge(
            actuals_by_date_cat,
            left_on=['date', data_config['cat_col']],
            right_on=['date', data_config['cat_col']],
            how='left'
        )
        # For dates with no actuals, treat actual as 0 so that
        # error and abs_error are still computed instead of NaN.
        recursive_results_cat['actual'] = recursive_results_cat['actual'].fillna(0.0)

        recursive_results_list.append(recursive_results_cat)

    # Concatenate all categories
    if len(recursive_results_list) > 0:
        recursive_results = pd.concat(recursive_results_list, ignore_index=True)
    else:
        recursive_results = pd.DataFrame(columns=['date', 'predicted', 'actual', data_config['cat_col']])

    # Inverse transform predictions if scaler is available
    if scaler is not None and len(recursive_results) > 0:
        print("  - Inverse transforming recursive predictions to original scale...")
        # recursive_results['predicted_unscaled'] = inverse_transform_scaling(
        #     recursive_results['predicted'].values, scaler
        # )
        recursive_results['predicted_unscaled'] = scaler.inverse_transform_y(recursive_results['predicted'].values,)
    else:
        recursive_results['predicted_unscaled'] = recursive_results['predicted'] if len(recursive_results) > 0 else []

    # Clip negative predictions to 0 (QTY/CBM cannot be negative)
    if len(recursive_results) > 0:
        negative_count = np.sum(recursive_results['predicted_unscaled'] < 0)
        if negative_count > 0:
            print(f"  [WARNING] Clipping {negative_count} negative recursive predictions to 0 (QTY must be >= 0)")
            recursive_results['predicted_unscaled'] = np.maximum(recursive_results['predicted_unscaled'], 0.0)

    # Holiday Zero‑Constraint in ORIGINAL scale:
    # Force predicted_unscaled to 0 on Vietnam warehouse day‑off dates,
    # while preserving the internal (scaled) series used for recursion.
    if len(recursive_results) > 0:
        rec_min_date = recursive_results['date'].min()
        rec_max_date = recursive_results['date'].max()
        rec_holidays = set(get_vietnam_holidays(rec_min_date, rec_max_date))
        holiday_mask_rec = recursive_results['date'].isin(rec_holidays)
        holiday_count = holiday_mask_rec.sum()
        if holiday_count > 0:
            print(f"  - Applying zero‑volume holiday mask to {holiday_count} recursive prediction day(s).")
            recursive_results.loc[holiday_mask_rec, 'predicted_unscaled'] = 0.0
    
    # Apply Sunday-to-Monday carryover rule to predictions (same as training data processing)
    # This ensures Sunday predictions = 0 and Sunday's value is added to following Monday
    if len(recursive_results) > 0:
        print("  - Applying Sunday-to-Monday demand carryover to predictions...")
        # Group by category to apply carryover per category independently
        def apply_carryover_per_cat(group):
            return apply_sunday_to_monday_carryover_predictions(
                group,
                date_col='date',
                pred_col='predicted_unscaled'
            )
        
        recursive_results = recursive_results.groupby(
            data_config['cat_col'], group_keys=False
        ).apply(apply_carryover_per_cat).reset_index(drop=True)
        print("    - Sunday predictions set to 0, Sunday values added to following Monday")

    # Calculate metrics using unscaled values (include all dates;
    # actuals that were originally missing are treated as 0).
    recursive_results_with_actuals = recursive_results.copy()
    monthly_results_by_cat = None

    if len(recursive_results_with_actuals) > 0:
        # --------------------------------------------------------------------
        # TOTAL metrics across all categories (for backward compatibility)
        # We aggregate per day by summing across categories.
        # --------------------------------------------------------------------
        totals_by_day = recursive_results_with_actuals.groupby('date').agg(
            actual=('actual', 'sum'),
            predicted_unscaled=('predicted_unscaled', 'sum')
        ).reset_index()

        mse_rec = np.mean(
            (totals_by_day['actual'] - totals_by_day['predicted_unscaled']) ** 2
        )
        mae_rec = np.mean(
            np.abs(totals_by_day['actual'] - totals_by_day['predicted_unscaled'])
        )
        rmse_rec = np.sqrt(mse_rec)

        # --------------------------------------------------------------------
        # TOTAL accuracy across all months, using MONTHLY aggregates
        # (matches manual calculation from monthly Total_Actual / Total_Pred)
        # --------------------------------------------------------------------
        totals_by_day['date'] = pd.to_datetime(totals_by_day['date'])
        totals_by_day['month'] = totals_by_day['date'].dt.to_period('M')

        totals_by_month = totals_by_day.groupby('month').agg(
            actual=('actual', 'sum'),
            predicted_unscaled=('predicted_unscaled', 'sum'),
        ).reset_index()

        accuracy_rec_total_abs = calculate_accuracy(
            totals_by_month['actual'].values,
            totals_by_month['predicted_unscaled'].values,
        )
        accuracy_rec_total_sum = calculate_accuracy_sum_before_abs(
            totals_by_month['actual'].values,
            totals_by_month['predicted_unscaled'].values,
        )
        # Preserve legacy variable name (backwards compatibility)
        accuracy_rec = accuracy_rec_total_abs

        print(f"\n  - Predictions made: {len(recursive_results_with_actuals)} samples (all categories)")
        print(f"  - Unique dates with predictions: {totals_by_day['date'].nunique()}")
        print(f"  - MSE (total):  {mse_rec:.4f}")
        print(f"  - MAE (total): {mae_rec:.4f}")
        print(f"  - RMSE (total): {rmse_rec:.4f}")
        if not np.isnan(accuracy_rec_total_abs):
            print(f"  - Accuracy (total, Σ|err|):  {accuracy_rec_total_abs:.2f}%")
        if not np.isnan(accuracy_rec_total_sum):
            print(f"  - Accuracy (total, |Σ err|): {accuracy_rec_total_sum:.2f}%")

        # --------------------------------------------------------------------
        # Accuracy BY MONTH (TOTAL across all categories)
        # --------------------------------------------------------------------

        print("\n  - Monthly accuracy (recursive, unscaled):")
        for month, group in totals_by_day.groupby('month'):
            month_accuracy_abs = calculate_accuracy(
                group['actual'].values,
                group['predicted_unscaled'].values,
            )
            month_accuracy_sum = calculate_accuracy_sum_before_abs(
                group['actual'].values,
                group['predicted_unscaled'].values,
            )

            # Skip months where both cannot be computed
            if np.isnan(month_accuracy_abs) and np.isnan(month_accuracy_sum):
                continue

            month_total_actual = np.sum(np.abs(group['actual'].values))
            month_total_pred = np.sum(np.abs(group['predicted_unscaled'].values))

            line = (
                f"    {month}: "
                f"Acc(Σ|err|)={month_accuracy_abs:.2f}% | "
                f"Acc(|Σ err|)={month_accuracy_sum:.2f}% | "
                f"Total_Actual={month_total_actual:.2f} | "
                f"Total_Pred={month_total_pred:.2f}"
            )
            print(line)

        # --------------------------------------------------------------------
        # Accuracy BY MONTH AND CATEGORY + monthly totals for export
        # --------------------------------------------------------------------
        # Ensure `date` is a proper datetime for grouping
        recursive_results_with_actuals['date'] = pd.to_datetime(
            recursive_results_with_actuals['date']
        )
        recursive_results_with_actuals['month'] = recursive_results_with_actuals['date'].dt.to_period('M')

        print("\n  - Monthly accuracy by category (recursive, unscaled):")
        for (cat_value, month), group in recursive_results_with_actuals.groupby(
            [data_config['cat_col'], 'month']
        ):
            month_accuracy_abs = calculate_accuracy(
                group['actual'].values,
                group['predicted_unscaled'].values,
            )
            month_accuracy_sum = calculate_accuracy_sum_before_abs(
                group['actual'].values,
                group['predicted_unscaled'].values,
            )

            # Skip months where both cannot be computed
            if np.isnan(month_accuracy_abs) and np.isnan(month_accuracy_sum):
                continue

            month_total_actual = np.sum(np.abs(group['actual'].values))
            month_total_pred = np.sum(np.abs(group['predicted_unscaled'].values))

            line = (
                f"    Category={cat_value} | {month}: "
                f"Acc(Σ|err|)={month_accuracy_abs:.2f}% | "
                f"Acc(|Σ err|)={month_accuracy_sum:.2f}% | "
                f"Total_Actual={month_total_actual:.2f} | "
                f"Total_Pred={month_total_pred:.2f}"
            )
            print(line)

        # Build monthly totals by Year / CATEGORY for Google Sheets "ResultByMonth"
        recursive_results_with_actuals['Year'] = recursive_results_with_actuals['date'].dt.year
        recursive_results_with_actuals['Month'] = recursive_results_with_actuals['date'].dt.month
        monthly_group = recursive_results_with_actuals.groupby(
            ['Year', data_config['cat_col'], 'Month'],
            as_index=False,
        ).agg(
            predicted=('predicted_unscaled', 'sum'),
            actual=('actual', 'sum'),
        )
        monthly_group['error'] = monthly_group['actual'] - monthly_group['predicted']
        monthly_group['abs_error'] = np.abs(monthly_group['error'])
        # Reorder columns: Year, CATEGORY, Month, predicted, actual, error, abs_error
        ordered_cols = [
            'Year',
            data_config['cat_col'],
            'Month',
            'predicted',
            'actual',
            'error',
            'abs_error',
        ]
        monthly_results_by_cat = monthly_group[ordered_cols]

        # Build a separate SummaryByMonth-style table with SAME format as
        # combine_data.create_monthly_summary: Year, CATEGORY, Total QTY, Total CBM, Month.
        # Here, Total QTY is left blank (NaN) and Total CBM uses predicted volume.
        summary_by_month_df = pd.DataFrame({
            'Year': np.nan,
            data_config['cat_col']: monthly_group[data_config['cat_col']].values,
            'Total QTY': np.nan,
            'Total CBM': monthly_group['predicted'].values,
            'Month': monthly_group['Month'].values,
        })
    else:
        print(f"\n  - Predictions made: 0 samples")
        print(f"  - No actual values available for comparison")
        mse_rec = mae_rec = rmse_rec = accuracy_rec_total_abs = accuracy_rec_total_sum = accuracy_rec = np.nan
        monthly_results_by_cat = None
        summary_by_month_df = None
    
    # ========================================================================
    # COMPARISON AND SAVING
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: TEACHER FORCING vs RECURSIVE")
    print("=" * 80)
    
    # Prepare teacher forcing results for comparison
    # Ensure all arrays have the same length
    len_pred = len(y_pred_tf_unscaled) if hasattr(y_pred_tf_unscaled, '__len__') and len(y_pred_tf_unscaled) > 0 else 0
    len_actual = len(y_true_tf_unscaled) if hasattr(y_true_tf_unscaled, '__len__') and len(y_true_tf_unscaled) > 0 else 0
    len_dates = len(pred_dates) if hasattr(pred_dates, '__len__') and len(pred_dates) > 0 else 0
    
    # Find the minimum length to ensure all arrays match
    if len_pred > 0 and len_actual > 0 and len_dates > 0:
        min_len = min(len_pred, len_actual, len_dates)
        if len_pred != len_actual or len_pred != len_dates or len_actual != len_dates:
            print(f"  [WARNING] Array length mismatch detected: predictions={len_pred}, actuals={len_actual}, dates={len_dates}")
            print(f"  [WARNING] Trimming all arrays to minimum length: {min_len}")
    else:
        min_len = 0
    
    if min_len == 0:
        # If any array is empty, create empty DataFrame
        print("  [WARNING] One or more arrays are empty. Creating empty DataFrame for teacher forcing results.")
        tf_results = pd.DataFrame(columns=['date', 'actual', 'predicted'])
    else:
        # Trim all arrays to the same length
        y_pred_tf_trimmed = y_pred_tf_unscaled[:min_len] if hasattr(y_pred_tf_unscaled, '__getitem__') else y_pred_tf_unscaled
        y_true_tf_trimmed = y_true_tf_unscaled[:min_len] if hasattr(y_true_tf_unscaled, '__getitem__') else y_true_tf_unscaled
        
        # Handle pred_dates - could be list, array, Series, or Index
        if hasattr(pred_dates, '__getitem__'):
            pred_dates_trimmed = pred_dates[:min_len]
        else:
            pred_dates_trimmed = pred_dates
        
        # Normalize pred_dates to plain Python date objects
        if len(pred_dates_trimmed) > 0:
            # Convert to pandas Series first, then extract date
            try:
                tf_dates_series = pd.to_datetime(pred_dates_trimmed)
                tf_dates = [d.date() for d in tf_dates_series]
            except Exception as e:
                print(f"  [WARNING] Error converting dates: {e}. Using original dates.")
                tf_dates = pred_dates_trimmed if isinstance(pred_dates_trimmed, list) else list(pred_dates_trimmed)
        else:
            tf_dates = []
        
        # Ensure all arrays have exactly the same length
        if len(tf_dates) != min_len:
            tf_dates = tf_dates[:min_len] if len(tf_dates) > min_len else tf_dates
        if len(y_pred_tf_trimmed) != min_len:
            y_pred_tf_trimmed = y_pred_tf_trimmed[:min_len] if hasattr(y_pred_tf_trimmed, '__getitem__') else y_pred_tf_trimmed
        if len(y_true_tf_trimmed) != min_len:
            y_true_tf_trimmed = y_true_tf_trimmed[:min_len] if hasattr(y_true_tf_trimmed, '__getitem__') else y_true_tf_trimmed
        
        tf_results = pd.DataFrame({
            'date': tf_dates,
            'actual': y_true_tf_trimmed,
            'predicted': y_pred_tf_trimmed
        })
    # Holiday Zero‑Constraint for Teacher Forcing as well:
    # mask predicted volume to 0 on Vietnam warehouse day‑off dates so that
    # evaluation and exported CSVs reflect the operational constraint.
    if len(tf_results) > 0:
        tf_min_date = tf_results['date'].min()
        tf_max_date = tf_results['date'].max()
        tf_holidays = set(get_vietnam_holidays(tf_min_date, tf_max_date))
        holiday_mask_tf = tf_results['date'].isin(tf_holidays)
        tf_results.loc[holiday_mask_tf, 'predicted'] = 0.0

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
        
        # Calculate error increase percentages for upload
        mae_error_increase_pct = (mae_rec / mae_tf - 1) * 100 if not np.isnan(mae_tf) and mae_tf != 0 else None
        rmse_error_increase_pct = (rmse_rec / rmse_tf - 1) * 100 if not np.isnan(rmse_tf) and rmse_tf != 0 else None
    else:
        print(f"{'MAE':<15} {mae_tf:<20.4f} {'N/A':<20}")
        print(f"{'RMSE':<15} {rmse_tf:<20.4f} {'N/A':<20}")
        if not np.isnan(accuracy_tf):
            print(f"{'Accuracy':<15} {accuracy_tf:<20.2f}% {'N/A':<20}")
        
        # Set error increase to None if recursive metrics are not available
        mae_error_increase_pct = None
        rmse_error_increase_pct = None
    
    # Upload history prediction results to Google Sheets
    if GSPREAD_AVAILABLE:
        # Calculate number of months for prediction
        # Calculate the difference in months between start and end dates
        months_diff = (prediction_end_date.year - prediction_start_date.year) * 12 + \
                      (prediction_end_date.month - prediction_start_date.month) + 1
        # Add 1 to include both start and end months
        
        # Determine category label
        if len(categories_to_predict) == 1:
            category_label = categories_to_predict[0]
        elif len(categories_to_predict) == len(available_categories):
            category_label = "ALL"
        else:
            # Multiple but not all categories - join them
            category_label = ", ".join(sorted(categories_to_predict))
        
        spreadsheet_id = os.getenv(
            "GOOGLE_SHEET_ID",
            "1I8JEqZbWGZNOsebzOBfeHKJ7Z7jA1zcfX8JhjqGSowE",  # same default as combine_data.py
        )
        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "key.json")
        
        print(f"\n[Google Sheets] Uploading history prediction results...")
        upload_history_prediction(
            spreadsheet_id=spreadsheet_id,
            category=category_label,
            prediction_year=prediction_start_date.year,
            num_months=months_diff,
            tf_mse=mse_tf if not np.isnan(mse_tf) else None,
            tf_mae=mae_tf if not np.isnan(mae_tf) else None,
            tf_rmse=rmse_tf if not np.isnan(rmse_tf) else None,
            tf_accuracy_abs=accuracy_tf_abs if not np.isnan(accuracy_tf_abs) else None,
            tf_accuracy_sum=accuracy_tf_sum if not np.isnan(accuracy_tf_sum) else None,
            rec_mse=mse_rec if not np.isnan(mse_rec) else None,
            rec_mae=mae_rec if not np.isnan(mae_rec) else None,
            rec_rmse=rmse_rec if not np.isnan(rmse_rec) else None,
            rec_accuracy_abs=accuracy_rec_total_abs if not np.isnan(accuracy_rec_total_abs) else None,
            rec_accuracy_sum=accuracy_rec_total_sum if not np.isnan(accuracy_rec_total_sum) else None,
            mae_error_increase_pct=mae_error_increase_pct,
            rmse_error_increase_pct=rmse_error_increase_pct,
            credentials_path=credentials_path,
            sheet_name="History_prediction"
        )
    else:
        print("\n[Google Sheets] gspread not available; skipping history prediction upload.")
    
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

    # Add UpdateTime column (same for all rows in this run)
    update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # For daily Result sheet
    if 'UpdateTime' not in recursive_results.columns:
        recursive_results['UpdateTime'] = update_time
    # For monthly ResultByMonth sheet (may be None if no monthly results)
    if monthly_results_by_cat is not None and 'UpdateTime' not in monthly_results_by_cat.columns:
        monthly_results_by_cat['UpdateTime'] = update_time
    
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
    # If any actuals are missing for TF, treat them as 0 so error still computed
    tf_results['actual'] = tf_results['actual'].fillna(0.0)
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
    # Ensure actual has no NaNs so error and abs_error are always computed
    recursive_results['actual'] = recursive_results['actual'].fillna(0.0)
    recursive_results['error'] = recursive_results['actual'] - recursive_results['predicted']
    recursive_results['abs_error'] = np.abs(recursive_results['error'])
    recursive_results = recursive_results.sort_values('date')
    # Drop intermediate columns for cleaner output (but keep category column and UpdateTime if present)
    output_cols = ['date', data_config['cat_col'], 'predicted', 'actual', 'error', 'abs_error', 'UpdateTime']
    recursive_results = recursive_results[[c for c in output_cols if c in recursive_results.columns]]
    recursive_results.to_csv(rec_output_path, index=False)
    print(f"  - Recursive results: {rec_output_path}")

    # Optionally upload recursive results to Google Sheets "Result" and "ResultByMonth" sheets
    # Uses the upload_to_google_sheets helper from src.utils.google_sheets
    if GSPREAD_AVAILABLE:
        spreadsheet_id = os.getenv(
            "GOOGLE_SHEET_ID",
            "1I8JEqZbWGZNOsebzOBfeHKJ7Z7jA1zcfX8JhjqGSowE",  # same default as combine_data.py
        )
        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "key.json")

        print(f"  - Uploading recursive results to Google Sheets sheet 'Result' (spreadsheet_id={spreadsheet_id}, update mode)")
        upload_to_google_sheets(
            saved_files={},  # not used when data_df is provided
            spreadsheet_id=spreadsheet_id,
            sheet_name="Result",
            credentials_path=credentials_path,
            data_df=recursive_results,
            update_mode=True,  # Update instead of overwrite to preserve data from multiple category runs
        )

        if monthly_results_by_cat is not None and not monthly_results_by_cat.empty:
            print(f"  - Uploading monthly recursive results to Google Sheets sheet 'ResultByMonth' (update mode)")
            upload_to_google_sheets(
                saved_files={},
                spreadsheet_id=spreadsheet_id,
                sheet_name="ResultByMonth",
                credentials_path=credentials_path,
                data_df=monthly_results_by_cat,
                update_mode=True,  # Update instead of overwrite to preserve data from multiple category runs
            )

        # Upload SummaryByMonth sheet derived from ResultByMonth-style data
        if 'summary_by_month_df' in locals() and summary_by_month_df is not None and not summary_by_month_df.empty:
            print(f"  - Uploading summary results to Google Sheets sheet 'SummaryByMonth' (update mode)")
            upload_to_google_sheets(
                saved_files={},
                spreadsheet_id=spreadsheet_id,
                sheet_name="SummaryByMonth",
                credentials_path=credentials_path,
                data_df=summary_by_month_df,
                update_mode=True,
            )
        else:
            print("  - No monthly results to upload to 'ResultByMonth'.")
    else:
        print("  - gspread not available; skipping Google Sheets upload.")
    
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
        # Use a generic title so it works for any configured window
        f.write("Prediction Comparison: Teacher Forcing vs Recursive\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Run ID: run_{run_timestamp}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        category_label = 'ALL categories' if category_mode == 'all' else f'{categories_to_predict[0]} category only'
        category_info = (
            f"{prediction_start_date} to {prediction_end_date}, {category_label}\n"
        )
        f.write(f"Data: {category_info}")
        f.write(f"Data source: dataset/test/data_2025.csv\n\n")
        
        f.write("Teacher Forcing Mode (Test Evaluation):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Description: Uses actual ground truth {data_config['target_col']} values from 2025 as features\n")
        f.write(f"  Suitable for: Model evaluation on test set\n")
        f.write(f"  Number of predictions: {len(y_pred_tf)}\n")
        if len(pred_dates) > 0:
            f.write(f"  Date range: {pred_dates.min()} to {pred_dates.max()}\n")
        else:
            f.write("  Date range: N/A (no prediction windows available)\n")
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
        # Use actual recursive prediction date range (January 2025)
        if len(recursive_preds) > 0:
            rec_start_date = min(recursive_preds['date'])
            rec_end_date = max(recursive_preds['date'])
            f.write(f"  Date range: {rec_start_date} to {rec_end_date}\n")
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

