"""MVP Test Script for MDLZ Warehouse Prediction System.

This script verifies the entire pipeline using a small subset of data:
- Loads 2023 and 2024 data
- Filters to DRY category only
- Trains for 20 epochs with spike_aware_mse loss
- Uses Vietnamese holidays (Tet, Mid-Autumn Festival, etc.)
- Generates test predictions and plots
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
import pickle
from datetime import date, timedelta
from typing import List, Tuple

from config import load_config, load_holidays
from src.data import (
    DataReader,
    ForecastDataset,
    slicing_window_category,
    encode_categories,
    split_data,
    add_temporal_features,
    aggregate_daily,
    add_cbm_density_features,
    add_year_over_year_volume_features,
    fit_scaler,
    apply_scaling,
    inverse_transform_scaling
)
from src.data.preprocessing import (
    add_day_of_week_cyclical_features,
    add_eom_features,
    add_mid_month_peak_features,
    add_early_month_low_volume_features,
    add_high_volume_month_features,
    add_pre_holiday_surge_features,
    add_weekday_volume_tier_features,
    add_is_monday_feature,
    apply_sunday_to_monday_carryover,
    add_operational_status_flags,
    add_seasonal_active_window_features
)
from src.models import RNNWithCategory
from src.training import Trainer
from src.utils import plot_difference, spike_aware_mse, quantile_loss, QuantileLoss, quantile_coverage, calculate_forecast_metrics


###############################################################################
# Helper Functions
###############################################################################

def get_category_params(category_specific_params: dict, category: str = None) -> dict:
    """
    Get category-specific parameters with default fallback.
    
    Priority:
    1. category_specific_params[category] if category exists and has specific config
    2. category_specific_params['default'] as fallback
    3. Empty dict if neither exists
    
    Args:
        category_specific_params: Dictionary of category-specific parameters
        category: Category name (e.g., "DRY", "TET")
    
    Returns:
        Dictionary of parameters for the category or default
    """
    if category and category in category_specific_params:
        return category_specific_params[category]
    elif 'default' in category_specific_params:
        return category_specific_params['default']
    else:
        return {}


###############################################################################
# Holiday and Lunar Calendar Utilities
###############################################################################

# NOTE:
# We keep a single source of truth for Vietnamese holidays (including Tet)
# so that both the discrete holiday indicators and the continuous
# "days-to-lunar-event" features stay perfectly aligned.
# Holidays are now loaded from config/holidays.yaml for easier maintenance.
VIETNAM_HOLIDAYS_BY_YEAR = load_holidays(holiday_type="model")


def get_vietnam_holidays(start_date: date, end_date: date) -> List[date]:
    """
    Get list of Vietnamese holidays between start_date and end_date.
    
    Includes:
    - Lunar New Year (Tet): 2023 (Jan 20-26), 2024 (Feb 8-14), 2025 (Jan 27 - Feb 2)
    - Mid-Autumn Festival: 2023 (Sep 29), 2024 (Sep 17), 2025 (Oct 6)
    - Independence Day (Sep 2)
    - Labor Day (Apr 30 - May 1)
    
    Args:
        start_date: Start date for holiday range.
        end_date: End date for holiday range.
    
    Returns:
        List of holiday dates.
    """
    holidays = []

    # Collect all holidays in the date range
    current = start_date
    while current <= end_date:
        year = current.year
        if year in VIETNAM_HOLIDAYS_BY_YEAR:
            year_holidays = VIETNAM_HOLIDAYS_BY_YEAR[year]
            # Use .get() to handle optional keys gracefully
            holidays.extend(year_holidays.get("tet", []))
            holidays.extend(year_holidays.get("mid_autumn", []))
            holidays.extend(year_holidays.get("independence", []))
            holidays.extend(year_holidays.get("labor", []))
            holidays.extend(year_holidays.get("hung_kings", []))
        current = date(year + 1, 1, 1)
    
    # Filter to date range and remove duplicates
    holidays = [h for h in holidays if start_date <= h <= end_date]
    holidays = sorted(list(set(holidays)))
    
    return holidays


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


def get_mid_autumn_dates(start_year: int, end_year: int) -> List[date]:
    """
    Get Mid-Autumn Festival dates for a year range.

    These are the anchor points for the "days_to_mid_autumn" continuous feature,
    representing the peak event for MOONCAKE category (Lunar Month 8, Day 15).
    The peak occurs 30-45 days before this date.
    """
    mid_autumn_dates: List[date] = []
    for year in range(start_year, end_year + 1):
        if year in VIETNAM_HOLIDAYS_BY_YEAR:
            mid_autumn_list = VIETNAM_HOLIDAYS_BY_YEAR[year].get("mid_autumn", [])
            if mid_autumn_list:
                # Use the first day of Mid-Autumn Festival as the peak event
                mid_autumn_dates.append(mid_autumn_list[0])

    # Remove duplicates and sort
    mid_autumn_dates = sorted(list(set(mid_autumn_dates)))
    return mid_autumn_dates


def add_holiday_features_vietnam(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    holiday_indicator_col: str = "holiday_indicator",
    days_until_holiday_col: str = "days_until_next_holiday",
    days_since_holiday_col: str = "days_since_holiday"
) -> pd.DataFrame:
    """
    Add Vietnamese holiday-related features to DataFrame.
    
    Creates:
    - holiday_indicator: Binary (0 or 1) indicating if date is a Vietnamese holiday
    - days_until_next_holiday: Number of days until the next Vietnamese holiday
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        holiday_indicator_col: Name for holiday indicator column.
        days_until_holiday_col: Name for days until next holiday column.
    
    Returns:
        DataFrame with added holiday features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Get date range from data
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    
    # Extend range slightly to ensure we have next holidays for all dates
    extended_max = max_date + timedelta(days=365)
    
    # Get Vietnamese holidays
    holidays = get_vietnam_holidays(min_date, extended_max)
    holiday_set = set(holidays)
    
    # Initialize columns
    df[holiday_indicator_col] = 0
    df[days_until_holiday_col] = np.nan
    df[days_since_holiday_col] = np.nan
    
    # Process each row
    for idx, row in df.iterrows():
        current_date = row[time_col].date()
        
        # Set holiday indicator
        if current_date in holiday_set:
            df.at[idx, holiday_indicator_col] = 1
        
        # Calculate days until next holiday
        next_holiday = None
        for holiday in holidays:
            if holiday > current_date:
                next_holiday = holiday
                break
        
        if next_holiday:
            days_until = (next_holiday - current_date).days
            df.at[idx, days_until_holiday_col] = days_until
        else:
            # If no holiday found (shouldn't happen with extended range), set to large value
            df.at[idx, days_until_holiday_col] = 365
        
        # Calculate days since last holiday
        last_holiday = None
        for holiday in sorted(holidays, reverse=True):
            if holiday <= current_date:
                last_holiday = holiday
                break
        
        if last_holiday:
            days_since = (current_date - last_holiday).days
            df.at[idx, days_since_holiday_col] = days_since
        else:
            # If no holiday found, set to large value
            df.at[idx, days_since_holiday_col] = 365
    
    # Fill any remaining NaN values (shouldn't happen, but safety check)
    df[days_until_holiday_col] = df[days_until_holiday_col].fillna(365)
    df[days_since_holiday_col] = df[days_since_holiday_col].fillna(365)
    
    return df


def solar_to_lunar_date(solar_date: date) -> Tuple[int, int]:
    """
    Convert solar (Gregorian) date to lunar (Vietnamese) date using anchor points.
    
    Uses known Mid-Autumn Festival dates (Lunar Month 8, Day 15) as anchor points:
    - 2023: Sep 29 = Lunar 08-15
    - 2024: Sep 17 = Lunar 08-15
    - 2025: Oct 6 = Lunar 08-15
    - 2026: Sep 25 = Lunar 08-15
    
    This provides accurate lunar date conversion for MOONCAKE forecasting.
    
    Args:
        solar_date: Gregorian date.
    
    Returns:
        Tuple of (lunar_month, lunar_day) where lunar_month is 1-12.
    """
    # Anchor points: Mid-Autumn Festival dates (Lunar Month 8, Day 15)
    mid_autumn_anchors = {
        2023: date(2023, 9, 29),   # Lunar 08-15
        2024: date(2024, 9, 17),   # Lunar 08-15
        2025: date(2025, 10, 6),   # Lunar 08-15
        2026: date(2026, 9, 25),   # Lunar 08-15
    }
    
    # Find the closest Mid-Autumn anchor (before or after)
    year = solar_date.year
    if year not in mid_autumn_anchors:
        # For years outside our anchor range, use the closest year
        if year < 2023:
            anchor_year = 2023
        elif year > 2026:
            anchor_year = 2026
        else:
            anchor_year = year
    else:
        anchor_year = year
    
    anchor_date = mid_autumn_anchors[anchor_year]
    days_diff = (solar_date - anchor_date).days
    
    # Calculate lunar date relative to anchor (Lunar Month 8, Day 15)
    # Approximate: 1 lunar month ≈ 29.5 solar days
    # Lunar months: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    # We're at Month 8, Day 15 when days_diff = 0
    
    # Convert days difference to lunar month/day offset
    # Positive days_diff = after Mid-Autumn (moving forward in lunar calendar)
    # Negative days_diff = before Mid-Autumn (moving backward in lunar calendar)
    
    # Start from Lunar Month 8, Day 15
    lunar_month = 8
    lunar_day = 15
    
    # Adjust by days difference (approximate: 29.5 days per lunar month)
    if days_diff > 0:
        # After Mid-Autumn: move forward
        lunar_day += days_diff
        while lunar_day > 30:
            lunar_day -= 30
            lunar_month += 1
            if lunar_month > 12:
                lunar_month = 1
    else:
        # Before Mid-Autumn: move backward
        lunar_day += days_diff
        while lunar_day < 1:
            lunar_day += 30
            lunar_month -= 1
            if lunar_month < 1:
                lunar_month = 12
    
    # Clamp to valid ranges
    lunar_month = max(1, min(12, lunar_month))
    lunar_day = max(1, min(30, lunar_day))
    
    return lunar_month, lunar_day


def add_weekend_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    is_weekend_col: str = "is_weekend",
    day_of_week_col: str = "day_of_week"
) -> pd.DataFrame:
    """
    Add weekend and day-of-week features to DataFrame.
    
    Creates:
    - is_weekend: Binary (1 for Saturday/Sunday, 0 otherwise)
    - day_of_week: Integer (0=Monday, 1=Tuesday, ..., 6=Sunday)
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column.
        is_weekend_col: Name for is_weekend column.
        day_of_week_col: Name for day_of_week column.
    
    Returns:
        DataFrame with added weekend features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Get day of week (0=Monday, 6=Sunday)
    df[day_of_week_col] = df[time_col].dt.dayofweek
    
    # Weekend indicator: Saturday (5) or Sunday (6)
    df[is_weekend_col] = (df[day_of_week_col] >= 5).astype(int)
    
    return df


def add_lunar_calendar_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    lunar_month_col: str = "lunar_month",
    lunar_day_col: str = "lunar_day"
) -> pd.DataFrame:
    """
    Add lunar calendar features for Vietnamese holiday prediction.
    
    Creates:
    - lunar_month: Lunar month (1-12)
    - lunar_day: Lunar day (1-30)
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column.
        lunar_month_col: Name for lunar_month column.
        lunar_day_col: Name for lunar_day column.
    
    Returns:
        DataFrame with added lunar features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Convert each date to lunar
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

    This turns the discrete lunar month/day into smooth, high-frequency signals
    so the model can learn periodic spikes like Tet without relying on
    fixed Gregorian dates.
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


def add_rolling_and_momentum_features(
    df: pd.DataFrame,
    target_col: str = "QTY",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    rolling_7_col: str = "rolling_mean_7d",
    rolling_30_col: str = "rolling_mean_30d",
    momentum_col: str = "momentum_3d_vs_14d"
) -> pd.DataFrame:
    """
    Add rolling mean and momentum features to reduce model inertia.
    
    Creates:
    - rolling_mean_7d: 7-day rolling average of QTY
    - rolling_mean_30d: 30-day rolling average of QTY
    - momentum_3d_vs_14d: Difference between 3-day and 14-day rolling means
    
    These features help the model see "pace" rather than just yesterday's value.
    
    Args:
        df: DataFrame with QTY and time columns (must be sorted by time).
        target_col: Name of target column (e.g., "QTY").
        time_col: Name of time column.
        cat_col: Name of category column (rolling calculated per category).
        rolling_7_col: Name for 7-day rolling mean column.
        rolling_30_col: Name for 30-day rolling mean column.
        momentum_col: Name for momentum column.
    
    Returns:
        DataFrame with added rolling and momentum features.
    """
    df = df.copy()
    
    # Ensure time column is datetime and sort
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    df = df.sort_values([cat_col, time_col]).reset_index(drop=True)
    
    # Initialize new columns
    df[rolling_7_col] = np.nan
    df[rolling_30_col] = np.nan
    df[momentum_col] = np.nan
    
    # Calculate rolling features per category
    for cat, group in df.groupby(cat_col, sort=False):
        cat_mask = df[cat_col] == cat
        cat_indices = df[cat_mask].index
        
        # Calculate rolling means
        rolling_7 = group[target_col].rolling(window=7, min_periods=1).mean()
        rolling_30 = group[target_col].rolling(window=30, min_periods=1).mean()
        rolling_3 = group[target_col].rolling(window=3, min_periods=1).mean()
        rolling_14 = group[target_col].rolling(window=14, min_periods=1).mean()
        
        # Assign values back to main dataframe
        df.loc[cat_indices, rolling_7_col] = rolling_7.values
        df.loc[cat_indices, rolling_30_col] = rolling_30.values
        
        # Momentum: difference between short-term and long-term averages
        momentum = rolling_3 - rolling_14
        df.loc[cat_indices, momentum_col] = momentum.values
    
    # Fill any remaining NaN with forward fill then backward fill
    for col in [rolling_7_col, rolling_30_col, momentum_col]:
        df[col] = df[col].ffill().bfill().fillna(0)
    
    return df


def add_days_since_holiday(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    days_since_holiday_col: str = "days_since_holiday"
) -> pd.DataFrame:
    """
    Add days since last holiday feature.
    
    This complements days_until_next_holiday to help model understand
    post-holiday patterns (e.g., demand drops after Tet).
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column.
        days_since_holiday_col: Name for days_since_holiday column.
    
    Returns:
        DataFrame with added days_since_holiday feature.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Get date range from data
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    
    # Get Vietnamese holidays
    holidays = get_vietnam_holidays(min_date - timedelta(days=365), max_date)
    holiday_set = set(holidays)
    
    # Initialize column
    df[days_since_holiday_col] = np.nan
    
    # Process each row
    for idx, row in df.iterrows():
        current_date = row[time_col].date()
        
        # Find last holiday before or on current_date
        last_holiday = None
        for holiday in sorted(holidays, reverse=True):
            if holiday <= current_date:
                last_holiday = holiday
                break
        
        if last_holiday:
            days_since = (current_date - last_holiday).days
            df.at[idx, days_since_holiday_col] = days_since
        else:
            # If no holiday found, set to large value
            df.at[idx, days_since_holiday_col] = 365
    
    # Fill any remaining NaN values
    df[days_since_holiday_col] = df[days_since_holiday_col].fillna(365)
    
    return df


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
        df[time_col] = pd.to_datetime(df[time_col])

    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()

    # Extend a bit so all dates have a "next Tet"
    extended_max = max_date + timedelta(days=365)
    tet_start_dates = get_tet_start_dates(min_date.year, extended_max.year)

    if not tet_start_dates:
        # Fallback: no Tet dates configured, set large constant
        df[days_to_tet_col] = 365
        return df

    tet_start_dates = sorted(tet_start_dates)

    df[days_to_tet_col] = np.nan

    for idx, row in df.iterrows():
        current_date = row[time_col].date()

        # Find the next Tet start on or after current_date
        next_tet = None
        for tet_date in tet_start_dates:
            if tet_date >= current_date:
                next_tet = tet_date
                break

        if next_tet is None:
            # If we're beyond the last configured Tet, use a large value
            df.at[idx, days_to_tet_col] = 365
        else:
            df.at[idx, days_to_tet_col] = (next_tet - current_date).days

    df[days_to_tet_col] = df[days_to_tet_col].fillna(365)
    return df


def add_days_to_mid_autumn_feature(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    lunar_month_col: str = "lunar_month",
    lunar_day_col: str = "lunar_day",
    days_to_mid_autumn_col: str = "days_to_mid_autumn",
) -> pd.DataFrame:
    """
    Add a continuous "days_to_mid_autumn" feature for MOONCAKE category.
    
    This feature calculates the number of days until the Mid-Autumn Festival
    (Lunar Month 8, Day 15), which is the peak event for MOONCAKE.
    The model must recognize that the peak occurs exactly 30-45 days before
    the 15th of Lunar Month 8.
    
    For each date, this feature:
    - If in Lunar Month 8: calculates days until Day 15
    - If before Lunar Month 8: calculates days until next Mid-Autumn Festival
    - If after Lunar Month 8: calculates days until next year's Mid-Autumn Festival
    
    Args:
        df: DataFrame with time and lunar calendar columns
        time_col: Name of time column
        lunar_month_col: Name of lunar month column
        lunar_day_col: Name of lunar day column
        days_to_mid_autumn_col: Name for days_to_mid_autumn column
    
    Returns:
        DataFrame with added days_to_mid_autumn feature
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Check if lunar columns exist
    if lunar_month_col not in df.columns or lunar_day_col not in df.columns:
        raise ValueError(f"Lunar calendar columns '{lunar_month_col}' and '{lunar_day_col}' must exist. "
                        f"Call add_lunar_calendar_features first.")
    
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    
    # Get Mid-Autumn Festival dates
    extended_max = max_date + timedelta(days=365)
    mid_autumn_dates = get_mid_autumn_dates(min_date.year, extended_max.year)
    
    # Initialize column
    df[days_to_mid_autumn_col] = np.nan
    
    for idx, row in df.iterrows():
        current_date = row[time_col].date()
        lunar_month = int(row[lunar_month_col])
        lunar_day = int(row[lunar_day_col])
        
        # Calculate days until Mid-Autumn Festival (Lunar Month 8, Day 15)
        if lunar_month == 8:
            # In Lunar Month 8: countdown to Day 15
            if lunar_day <= 15:
                days_to_peak = 15 - lunar_day
            else:
                # Past Day 15, find next year's Mid-Autumn Festival
                days_to_peak = (30 - lunar_day) + 15  # Days to end of month + 15 days into next month
                # Add days until next Mid-Autumn Festival date
                next_mid_autumn = None
                for ma_date in mid_autumn_dates:
                    if ma_date > current_date:
                        next_mid_autumn = ma_date
                        break
                if next_mid_autumn:
                    # Approximate: use solar date difference
                    days_to_peak = (next_mid_autumn - current_date).days
                else:
                    days_to_peak = 365  # Fallback
        elif lunar_month < 8:
            # Before Lunar Month 8: calculate days until Day 15 of Month 8
            months_until = 8 - lunar_month
            days_until_month_8 = months_until * 30  # Approximate 30 days per lunar month
            days_until_day_15 = days_until_month_8 + (15 - lunar_day)
            days_to_peak = days_until_day_15
        else:
            # After Lunar Month 8: find next year's Mid-Autumn Festival
            months_until_next = (12 - lunar_month) + 8  # Months until next Month 8
            days_until_month_8 = months_until_next * 30
            days_until_day_15 = days_until_month_8 + (15 - lunar_day)
            days_to_peak = days_until_day_15
            
            # Also try to use actual Mid-Autumn Festival dates for accuracy
            next_mid_autumn = None
            for ma_date in mid_autumn_dates:
                if ma_date > current_date:
                    next_mid_autumn = ma_date
                    break
            if next_mid_autumn:
                # Use actual date difference for better accuracy
                days_to_peak = (next_mid_autumn - current_date).days
        
        df.at[idx, days_to_mid_autumn_col] = days_to_peak
    
    # Fill any remaining NaN values
    df[days_to_mid_autumn_col] = df[days_to_mid_autumn_col].fillna(365)
    
    return df


def train_single_model(data, config, category_filter, output_suffix=""):
    """
    Train a single model for a specific category.
    
    Args:
        data: Full DataFrame with all data
        config: Configuration object (should be category-specific config)
        category_filter: Category name to filter (required - system now operates in category-specific mode)
        output_suffix: Suffix for output directories (typically empty as directory is category-specific)
    
    Returns:
        Dictionary with training results and metadata
    """
    if category_filter is None:
        raise ValueError(
            "category_filter is required. The system now operates exclusively in "
            "Category-Specific Mode. Each category must be trained independently."
        )
    
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL FOR CATEGORY: {category_filter}")
    print("=" * 80)
    
    data_config = config.data
    cat_col = data_config['cat_col']
    time_col = data_config['time_col']
    target_col_name = data_config['target_col']
    major_categories = data_config.get("major_categories", ["DRY", "FRESH"])
    minor_categories = data_config.get("minor_categories", [])
    
    # Get category-specific parameters from config (needed early for batch_size and other params)
    category_specific_params = config.category_specific_params

    # Get current feature list from config
    # IMPORTANT: If this is a category-specific config (config_DRY.yaml, config_FRESH.yaml, etc.),
    # we should respect the feature_cols defined there and NOT add extra features automatically
    current_features = list(data_config["feature_cols"])
    
    # Only add extra lunar/seasonal features if NOT already defined in category config
    # Check if config has category-specific features by looking for early_month or mid_month features
    has_category_specific_features = any(
        feat in current_features 
        for feat in ["early_month_low_tier", "is_early_month_low", "mid_month_peak_tier", "is_mid_month_peak"]
    )
    
    if not has_category_specific_features:
        # This is likely using base config, so add standard extra features
        print(f"  [INFO] Using base config features, adding standard lunar/seasonal features...")
        extra_features = [
            "lunar_month_sin",
            "lunar_month_cos",
            "lunar_day_sin",
            "lunar_day_cos",
            "days_to_tet",
            "days_to_mid_autumn",  # Days-to-Mid-Autumn countdown for MOONCAKE
            "is_active_season",  # Seasonal active-window masking
            "days_until_peak",  # Countdown to peak event
            "is_golden_window",  # Golden Window indicator (Lunar Months 6.15 to 8.01 for MOONCAKE)
            "is_peak_loss_window",  # Peak Loss Window indicator (Lunar Months 7.15 to 8.15 for MOONCAKE - critical peak period)
            # Structural prior on payload density (CBM per QTY)
            "cbm_per_qty",
            "cbm_per_qty_last_year",
        ]
        for feat in extra_features:
            if feat not in current_features:
                current_features.append(feat)
        # Update config so downstream components see the full feature list
        data_config["feature_cols"] = current_features
        config.set("data.feature_cols", current_features)
    else:
        # This is a category-specific config (e.g., config_DRY.yaml)
        # Respect the feature_cols defined there
        print(f"  [INFO] Using category-specific config features ({len(current_features)} features)")
        print(f"  [INFO] Respecting feature_cols from config_{category_filter}.yaml")
    
    
    # Create a copy of data for this training run
    filtered_data = data.copy()

    # Apply category filter (required in category-specific mode)
    print(f"\n[4/8] Filtering data to category: {category_filter}...")
    samples_before = len(filtered_data)

    if cat_col not in filtered_data.columns:
        raise ValueError(
            f"Category column '{cat_col}' not found in data. "
            f"Available columns: {list(filtered_data.columns)}"
        )

    filtered_data = filtered_data[filtered_data[cat_col] == category_filter].copy()
    samples_after = len(filtered_data)

    print(f"  - Samples before filtering: {samples_before}")
    print(f"  - Samples after filtering (CATEGORY == '{category_filter}'): {samples_after}")

    if samples_after == 0:
        raise ValueError(
            f"No samples found with CATEGORY == '{category_filter}'. Please check your data."
        )
    
    # Set output directories (should already be set in main() for category-specific paths)
    # But ensure they exist and are properly configured
    mvp_output_dir = config.output.get('output_dir', os.path.join('outputs', category_filter))
    mvp_models_dir = config.output.get('model_dir', os.path.join(mvp_output_dir, 'models'))
    
    os.makedirs(mvp_output_dir, exist_ok=True)
    os.makedirs(mvp_models_dir, exist_ok=True)
    config.set('output.output_dir', mvp_output_dir)
    config.set('output.model_dir', mvp_models_dir)
    
    # Ensure time column is datetime and sort by time
    if not pd.api.types.is_datetime64_any_dtype(filtered_data[time_col]):
        filtered_data[time_col] = pd.to_datetime(filtered_data[time_col])
    filtered_data = filtered_data.sort_values(time_col).reset_index(drop=True)
    
    # Feature engineering: Add temporal features
    print("\n[5/8] Adding temporal features...")
    filtered_data = add_temporal_features(
        filtered_data,
        time_col=time_col,
        month_sin_col="month_sin",
        month_cos_col="month_cos",
        dayofmonth_sin_col="dayofmonth_sin",
        dayofmonth_cos_col="dayofmonth_cos"
    )
    
    # Feature engineering: Add weekend features
    print("  - Adding weekend features (is_weekend, day_of_week)...")
    filtered_data = add_weekend_features(
        filtered_data,
        time_col=time_col,
        is_weekend_col="is_weekend",
        day_of_week_col="day_of_week"
    )
    
    # Feature engineering: Add cyclical day-of-week encoding (sin/cos)
    print("  - Adding cyclical day-of-week features (day_of_week_sin, day_of_week_cos)...")
    filtered_data = add_day_of_week_cyclical_features(
        filtered_data,
        time_col=time_col,
        day_of_week_sin_col="day_of_week_sin",
        day_of_week_cos_col="day_of_week_cos"
    )
    
    # Feature engineering: Add weekday volume tier features (category-specific)
    # Determine weekday pattern based on category
    weekday_pattern = "fresh" if category_filter == "FRESH" else "default"
    weekday_desc = "Mon/Wed/Fri high" if weekday_pattern == "fresh" else "Wed/Fri high"
    print(f"  - Adding weekday volume tier features ({weekday_desc}) for {category_filter} category...")
    filtered_data = add_weekday_volume_tier_features(
        filtered_data,
        time_col=time_col,
        weekday_volume_tier_col="weekday_volume_tier",
        is_high_volume_weekday_col="is_high_volume_weekday",
        weekday_pattern=weekday_pattern  # Category-specific pattern
    )
    
    # Feature engineering: Add Is_Monday feature to help model learn Monday peak patterns
    print("  - Adding Is_Monday feature...")
    filtered_data = add_is_monday_feature(
        filtered_data,
        time_col=time_col,
        is_monday_col="Is_Monday"
    )
    
    # Feature engineering: Add End-of-Month (EOM) surge features
    print("  - Adding EOM features (is_EOM, days_until_month_end)...")
    filtered_data = add_eom_features(
        filtered_data,
        time_col=time_col,
        is_eom_col="is_EOM",
        days_until_month_end_col="days_until_month_end",
        eom_window_days=3
    )
    
    # Feature engineering: Add mid-month peak features (category-specific)
    # Determine peak pattern based on category
    peak_pattern = "fresh" if category_filter == "FRESH" else "default"
    peak_desc = "8th-15th surge" if peak_pattern == "fresh" else "24th-25th surge"
    print(f"  - Adding mid-month peak features ({peak_desc}) for {category_filter} category...")
    filtered_data = add_mid_month_peak_features(
        filtered_data,
        time_col=time_col,
        mid_month_peak_tier_col="mid_month_peak_tier",
        is_mid_month_peak_col="is_mid_month_peak",
        days_to_peak_col="days_to_mid_month_peak",
        peak_pattern=peak_pattern  # Category-specific pattern
    )
    
    # Feature engineering: Add early month low volume features (1st-3rd lowest)
    print("  - Adding early month low volume features (early_month_low_tier, is_early_month_low, days_from_month_start)...")
    filtered_data = add_early_month_low_volume_features(
        filtered_data,
        time_col=time_col,
        early_month_low_tier_col="early_month_low_tier",
        is_early_month_low_col="is_early_month_low",
        days_from_month_start_col="days_from_month_start"
    )
    
    # Feature engineering: Add lunar calendar features (MUST be before high_volume_month_features)
    print("  - Adding lunar calendar features (lunar_month, lunar_day)...")
    filtered_data = add_lunar_calendar_features(
        filtered_data,
        time_col=time_col,
        lunar_month_col="lunar_month",
        lunar_day_col="lunar_day"
    )
    
    # Feature engineering: Add volume month features (High: Gregorian Dec + Lunar July/Aug, Low: Lunar Dec)
    print("  - Adding volume month features (high_volume_month_tier, is_high_volume_month, is_low_volume_month)...")
    filtered_data = add_high_volume_month_features(
        filtered_data,
        time_col=time_col,
        high_volume_month_tier_col="high_volume_month_tier",
        is_high_volume_month_col="is_high_volume_month",
        is_low_volume_month_col="is_low_volume_month",
        month_col="month",
        lunar_month_col="lunar_month"
    )
    
    # Feature engineering: Add holiday impact features (category-specific behavior)
    # Determine holiday pattern based on category
    holiday_pattern = "fresh" if category_filter == "FRESH" else "default"
    holiday_desc = "post-holiday surge" if holiday_pattern == "fresh" else "pre-holiday surge"
    print(f"  - Adding holiday features ({holiday_desc}) for {category_filter} category...")
    filtered_data = add_pre_holiday_surge_features(
        filtered_data,
        time_col=time_col,
        pre_holiday_surge_tier_col="pre_holiday_surge_tier",
        is_pre_holiday_surge_col="is_pre_holiday_surge",
        days_before_surge=10,
        holiday_pattern=holiday_pattern  # Category-specific pattern
    )
    
    # Feature engineering: Lunar cyclical encodings (sine/cosine)
    print("  - Adding lunar cyclical features (sine/cosine)...")
    filtered_data = add_lunar_cyclical_features(
        filtered_data,
        lunar_month_col="lunar_month",
        lunar_day_col="lunar_day",
        lunar_month_sin_col="lunar_month_sin",
        lunar_month_cos_col="lunar_month_cos",
        lunar_day_sin_col="lunar_day_sin",
        lunar_day_cos_col="lunar_day_cos",
    )

    # Feature engineering: Vietnamese holiday features (with days_since_holiday)
    print("  - Adding Vietnamese holiday features...")
    filtered_data = add_holiday_features_vietnam(
        filtered_data,
        time_col=time_col,
        holiday_indicator_col="holiday_indicator",
        days_until_holiday_col="days_until_next_holiday",
        days_since_holiday_col="days_since_holiday"
    )

    # Feature engineering: continuous countdown to Tet (lunar event)
    print("  - Adding Tet countdown feature (days_to_tet)...")
    filtered_data = add_days_to_tet_feature(
        filtered_data,
        time_col=time_col,
        days_to_tet_col="days_to_tet",
    )
    
    # Feature engineering: Days-to-Mid-Autumn countdown for MOONCAKE category
    # This recognizes that peak occurs 30-45 days before the 15th of Lunar Month 8
    # Always create this feature (set to default for non-MOONCAKE categories)
    if category_filter == "MOONCAKE":
        print("  - Adding Days-to-Mid-Autumn countdown feature (days_to_mid_autumn)...")
        filtered_data = add_days_to_mid_autumn_feature(
            filtered_data,
            time_col=time_col,
            lunar_month_col="lunar_month",
            lunar_day_col="lunar_day",
            days_to_mid_autumn_col="days_to_mid_autumn",
        )
    else:
        # For non-MOONCAKE categories, create the column with default value
        if "days_to_mid_autumn" not in filtered_data.columns:
            filtered_data["days_to_mid_autumn"] = 365  # Default: far from Mid-Autumn Festival
    
    # Feature engineering: Seasonal active-window masking for seasonal categories
    print("  - Adding seasonal active-window features (is_active_season, days_until_peak, is_golden_window)...")
    filtered_data = add_seasonal_active_window_features(
        filtered_data,
        time_col=time_col,
        cat_col=cat_col,
        lunar_month_col="lunar_month",
        days_to_tet_col="days_to_tet",
        is_active_season_col="is_active_season",
        days_until_peak_col="days_until_peak",
        is_golden_window_col="is_golden_window",
    )
    
    # NEW: Add Gregorian-Anchored Peak Alignment features for MOONCAKE
    if category_filter == "MOONCAKE":
        print("  - Adding Gregorian-Anchored Peak Alignment features (days_until_lunar_08_01, is_august)...")
        from src.data.preprocessing import add_days_until_lunar_08_01_feature, add_is_august_feature
        
        # Add countdown to Lunar 08-01 (replaces raw lunar_month dependency)
        filtered_data = add_days_until_lunar_08_01_feature(
            filtered_data,
            time_col=time_col,
            lunar_month_col="lunar_month",
            lunar_day_col="lunar_day",
            days_until_lunar_08_01_col="days_until_lunar_08_01"
        )
        
        # Add is_august feature (Gregorian month == 8) to reinforce August as strong signal
        filtered_data = add_is_august_feature(
            filtered_data,
            time_col=time_col,
            is_august_col="is_august"
        )
    
    # Daily aggregation: Group by date and category, sum QTY
    # This ensures the model learns daily demand patterns, not individual transaction sizes
    print("\n[5.5/8] Aggregating to daily totals by category...")
    samples_before_agg = len(filtered_data)
    filtered_data = aggregate_daily(
        filtered_data,
        time_col=time_col,
        cat_col=cat_col,
        target_col=target_col_name
    )
    samples_after_agg = len(filtered_data)
    print(f"  - Samples before aggregation: {samples_before_agg}")
    print(f"  - Samples after aggregation: {samples_after_agg} (one row per date per category)")
    
    # Re-add date-level features after aggregation (they might be lost during aggregation)
    # This ensures they're present even if they were lost during aggregation
    if "Is_Monday" not in filtered_data.columns:
        print("  - Re-adding Is_Monday feature after aggregation...")
        filtered_data = add_is_monday_feature(
            filtered_data,
            time_col=time_col,
            is_monday_col="Is_Monday"
        )
    
    # SOLUTION 1 & 2: Re-add early month features after aggregation (critical for early month over-prediction fix)
    if "early_month_low_tier" not in filtered_data.columns or "post_peak_signal" not in filtered_data.columns:
        print("  - Re-adding early month features (post_peak_signal, is_first_3_days, etc.) after aggregation...")
        filtered_data = add_early_month_low_volume_features(
            filtered_data,
            time_col=time_col,
            early_month_low_tier_col="early_month_low_tier",
            is_early_month_low_col="is_early_month_low",
            days_from_month_start_col="days_from_month_start"
        )
    
    # Ensure days_to_mid_autumn exists after aggregation
    if "days_to_mid_autumn" not in filtered_data.columns:
        print("  - Re-adding days_to_mid_autumn feature after aggregation...")
        if category_filter == "MOONCAKE":
            filtered_data = add_days_to_mid_autumn_feature(
                filtered_data,
                time_col=time_col,
                lunar_month_col="lunar_month",
                lunar_day_col="lunar_day",
                days_to_mid_autumn_col="days_to_mid_autumn",
            )
        else:
            filtered_data["days_to_mid_autumn"] = 365  # Default for non-MOONCAKE
    
    # Ensure is_golden_window and is_peak_loss_window exist after aggregation
    if "is_golden_window" not in filtered_data.columns or "is_peak_loss_window" not in filtered_data.columns:
        print("  - Re-adding is_golden_window and is_peak_loss_window features after aggregation...")
        filtered_data = add_seasonal_active_window_features(
            filtered_data,
            time_col=time_col,
            cat_col=cat_col,
            lunar_month_col="lunar_month",
            days_to_tet_col="days_to_tet",
            is_active_season_col="is_active_season",
            days_until_peak_col="days_until_peak",
            is_golden_window_col="is_golden_window",
        )
        # is_peak_loss_window is created inside add_seasonal_active_window_features
    
    # Ensure new Gregorian-Anchored Peak Alignment features exist after aggregation (for MOONCAKE)
    if category_filter == "MOONCAKE":
        if "days_until_lunar_08_01" not in filtered_data.columns:
            print("  - Re-adding days_until_lunar_08_01 feature after aggregation...")
            from src.data.preprocessing import add_days_until_lunar_08_01_feature
            filtered_data = add_days_until_lunar_08_01_feature(
                filtered_data,
                time_col=time_col,
                lunar_month_col="lunar_month",
                lunar_day_col="lunar_day",
                days_until_lunar_08_01_col="days_until_lunar_08_01"
            )
        if "is_august" not in filtered_data.columns:
            print("  - Re-adding is_august feature after aggregation...")
            from src.data.preprocessing import add_is_august_feature
            filtered_data = add_is_august_feature(
                filtered_data,
                time_col=time_col,
                is_august_col="is_august"
            )
    
    # Context-aware operational status flags (holiday/off, Sunday downtime, anomalies)
    # NOTE: This runs on the daily aggregated series. If you need strict calendar
    # reindexing (to include dates with no records at all), call a reindexing
    # helper before this step and then invoke add_operational_status_flags.
    print("  - Tagging Holiday_OFF, Weekend_Downtime, and Operational_Anomalies...")
    filtered_data = add_operational_status_flags(
        filtered_data,
        time_col=time_col,
        target_col=target_col_name,
        status_col="operational_status",
        expected_zero_flag_col="is_expected_zero",
        anomaly_flag_col="is_operational_anomaly",
    )
    
    # Apply Sunday-to-Monday demand carryover to capture backlog accumulation
    print("  - Applying Sunday-to-Monday demand carryover...")
    filtered_data = apply_sunday_to_monday_carryover(
        filtered_data,
        time_col=time_col,
        cat_col=cat_col,
        target_col=target_col_name,
        actual_col=target_col_name
    )
    print("    - Monday's target now includes Sunday's demand (captures backlog accumulation)")
    
    # Feature engineering: Add CBM/QTY density features, including last-year prior
    print("  - Adding CBM density features (cbm_per_qty, cbm_per_qty_last_year)...")
    filtered_data = add_cbm_density_features(
        filtered_data,
        cbm_col=target_col_name,   # e.g., "Total CBM"
        qty_col="Total QTY",
        time_col=time_col,
        cat_col=cat_col,
        density_col="cbm_per_qty",
        density_last_year_col="cbm_per_qty_last_year",
    )
    
    # CRITICAL: Add year-over-year volume features for seasonal products (especially MOONCAKE)
    # For highly seasonal products, same period from previous years is more predictive than recent days
    if category_filter == "MOONCAKE":
        print("  - Adding year-over-year volume features for MOONCAKE (cbm_last_year, cbm_2_years_ago)...")
        print("    - MOONCAKE is highly seasonal - same LUNAR period from previous years is more predictive")
        print("    - Using LUNAR date matching (not calendar date) to handle Mid-Autumn Festival shift")
        filtered_data = add_year_over_year_volume_features(
            filtered_data,
            target_col=target_col_name,
            time_col=time_col,
            cat_col=cat_col,
            yoy_1y_col="cbm_last_year",
            yoy_2y_col="cbm_2_years_ago",
            use_lunar_matching=True,  # CRITICAL: Match by lunar date, not calendar date
            lunar_month_col="lunar_month",
            lunar_day_col="lunar_day",
        )
        # Add these features to feature_cols
        if "cbm_last_year" not in data_config["feature_cols"]:
            data_config["feature_cols"].append("cbm_last_year")
        if "cbm_2_years_ago" not in data_config["feature_cols"]:
            data_config["feature_cols"].append("cbm_2_years_ago")
        config.set("data.feature_cols", data_config["feature_cols"])
        print(f"    - Added year-over-year features: cbm_last_year, cbm_2_years_ago")

    # Feature engineering: Add rolling means and momentum features (after aggregation)
    print("  - Adding rolling mean and momentum features (7d, 30d, momentum)...")
    filtered_data = add_rolling_and_momentum_features(
        filtered_data,
        target_col=target_col_name,
        time_col=time_col,
        cat_col=cat_col,
        rolling_7_col="rolling_mean_7d",
        rolling_30_col="rolling_mean_30d",
        momentum_col="momentum_3d_vs_14d"
    )
    
    # CRITICAL for MOONCAKE: Add trend deviation feature - compares recent trend vs year-over-year baseline
    # This allows the model to adjust the year-over-year baseline based on recent patterns
    # Example: If last year same period was 100 CBM, but recent 21 days average is 120 CBM,
    # the model should predict closer to 120 (trend-adjusted) rather than blindly using 100
    if category_filter == "MOONCAKE" and "cbm_last_year" in filtered_data.columns:
        print("  - Adding trend deviation feature for MOONCAKE (recent_trend_vs_yoy)...")
        print("    - Compares recent 21-day trend vs year-over-year baseline to detect deviations")
        
        # Calculate 21-day rolling mean (matches input_size window)
        filtered_data = filtered_data.sort_values([cat_col, time_col]).reset_index(drop=True)
        rolling_21d = filtered_data.groupby(cat_col)[target_col_name].transform(
            lambda x: x.rolling(window=21, min_periods=1).mean()
        )
        filtered_data["rolling_mean_21d"] = rolling_21d
        
        # Calculate trend deviation: ratio of recent trend to year-over-year baseline
        # If ratio > 1.0: recent trend is higher than last year (upward deviation)
        # If ratio < 1.0: recent trend is lower than last year (downward deviation)
        # If ratio = 1.0: recent trend matches last year (no deviation)
        cbm_last_year = filtered_data["cbm_last_year"].replace(0, np.nan)  # Avoid division by zero
        filtered_data["trend_vs_yoy_ratio"] = np.where(
            cbm_last_year > 0,
            rolling_21d / cbm_last_year,
            1.0  # If no last year data, assume no deviation
        )
        filtered_data["trend_vs_yoy_ratio"] = filtered_data["trend_vs_yoy_ratio"].fillna(1.0)
        
        # Also add absolute difference for cases where ratio might be misleading
        filtered_data["trend_vs_yoy_diff"] = rolling_21d - filtered_data["cbm_last_year"]
        
        # Add these features to feature_cols
        for feat in ["rolling_mean_21d", "trend_vs_yoy_ratio", "trend_vs_yoy_diff"]:
            if feat not in data_config["feature_cols"]:
                data_config["feature_cols"].append(feat)
        config.set("data.feature_cols", data_config["feature_cols"])
        print(f"    - Added trend deviation features: rolling_mean_21d, trend_vs_yoy_ratio, trend_vs_yoy_diff")
        print(f"    - Model will learn to adjust year-over-year baseline based on recent 21-day trend")
    
    # Ensure is_peak_loss_window is in feature_cols for MOONCAKE
    if category_filter == "MOONCAKE" and "is_peak_loss_window" not in data_config["feature_cols"]:
        data_config["feature_cols"].append("is_peak_loss_window")
        config.set("data.feature_cols", data_config["feature_cols"])
        print(f"  - Added is_peak_loss_window to feature_cols (Lunar Months 7.15 to 8.15)")
    
    # Residual learning: define causal baseline and residual target (optional)
    use_residual = data_config.get("use_residual_target", False)
    baseline_source_col = data_config.get("baseline_source_col", "rolling_mean_30d")
    baseline_col_name = data_config.get("baseline_col", "baseline_for_target")
    residual_col_name = data_config.get("residual_col", "target_residual")

    if use_residual:
        print("\n[5.8/8] Configuring residual target with causal baseline...")
        if baseline_source_col not in filtered_data.columns:
            raise ValueError(
                f"Baseline source column '{baseline_source_col}' not found in data. "
                f"Available columns: {list(filtered_data.columns)}"
            )

        # Build a *causal* baseline: for each category and date t, the baseline
        # used for predicting y_t is the rolling statistic from t-1 and earlier.
        # We start from the configured baseline_source_col (e.g., rolling_mean_30d)
        # and shift it by 1 day within each category.
        filtered_data[baseline_col_name] = (
            filtered_data
            .groupby(cat_col)[baseline_source_col]
            .shift(1)
        )

        # For the very first observations where the shift produces NaNs,
        # fall back to the unshifted baseline source to avoid dropping rows.
        filtered_data[baseline_col_name] = filtered_data[baseline_col_name].fillna(
            filtered_data[baseline_source_col]
        )

        # Residual target: y_resid_t = y_t - baseline_t
        filtered_data[residual_col_name] = (
            filtered_data[target_col_name] - filtered_data[baseline_col_name]
        )
        target_col_for_model = residual_col_name
        print(f"  - Using residual target column: '{residual_col_name}'")
        print(f"  - Baseline source column: '{baseline_source_col}' -> causal baseline column: '{baseline_col_name}'")
    else:
        target_col_for_model = target_col_name
        baseline_col_name = None
        print("\n[5.8/8] Residual target disabled. Model will predict absolute values.")

    # Encode categories
    print("  - Encoding categories...")
    filtered_data, cat2id, num_categories = encode_categories(filtered_data, cat_col)
    cat_id_col = data_config['cat_id_col']
    
    # Update config with num_categories for model
    config.set('model.num_categories', num_categories)
    print(f"  - Number of categories: {num_categories}")
    print(f"  - Category mapping: {cat2id}")
    
    # Split data
    print("\n[6/8] Splitting data...")
    train_data, val_data, test_data = split_data(
        filtered_data,
        train_size=data_config['train_size'],
        val_size=data_config['val_size'],
        test_size=data_config['test_size'],
        temporal=True
    )
    print(f"  - Train samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - Test samples: {len(test_data)}")
    
    # DEBUG: Check test data target values BEFORE scaling
    test_target_before_scaling = test_data[target_col_for_model].values
    print(f"\n  - DEBUG: Test data target BEFORE scaling (column='{target_col_for_model}'):")
    print(f"    - Min: {test_target_before_scaling.min():.4f}")
    print(f"    - Max: {test_target_before_scaling.max():.4f}")
    print(f"    - Mean: {test_target_before_scaling.mean():.4f}")
    print(f"    - Non-zero count: {np.sum(test_target_before_scaling != 0)} / {len(test_target_before_scaling)}")
    print(f"    - Zero count: {np.sum(test_target_before_scaling == 0)} / {len(test_target_before_scaling)}")
    
    # Fit scaler on training data and apply to all splits
    # CRITICAL: Scaler is category-specific because train_data is already filtered to category_filter
    # This ensures FRESH data is scaled independently from DRY/TET, preserving signal intensity
    print(f"\n[6.5/8] Scaling target values in column '{target_col_for_model}'...")
    print(f"  - Scaler will be fitted exclusively on {category_filter or 'all categories'} data")
    scaler = fit_scaler(train_data, target_col=target_col_for_model)
    print(f"  - Scaler fitted on training data (category-specific):")
    print(f"    Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")
    if category_filter:
        print(f"  - [OK] Verified: Scaler is category-specific for '{category_filter}' (isolated scaling)")
    
    train_data = apply_scaling(train_data, scaler, target_col=target_col_for_model)
    val_data = apply_scaling(val_data, scaler, target_col=target_col_for_model)
    test_data = apply_scaling(test_data, scaler, target_col=target_col_for_model)
    print("  - Scaling applied to train, validation, and test sets")
    
    # Create windows using slicing_window_category
    print("\n[7/8] Creating sliding windows...")
    window_config = config.window
    feature_cols = data_config['feature_cols']
    
    # CRITICAL: Filter feature_cols to only include columns that actually exist in the dataframe
    # This prevents KeyError when features are conditionally created (e.g., YoY features only for MOONCAKE)
    available_cols = set(train_data.columns)
    feature_cols = [col for col in feature_cols if col in available_cols]
    
    # Warn if any expected features are missing
    missing_features = set(data_config['feature_cols']) - set(feature_cols)
    if missing_features:
        print(f"  - WARNING: Some features in config are not available in data and will be skipped: {missing_features}")
    
    # Update config with filtered feature list
    data_config['feature_cols'] = feature_cols
    config.set("data.feature_cols", feature_cols)
    
    # The model's supervised target column (may be residual or absolute)
    target_col = [target_col_for_model]
    cat_col_list = [cat_id_col]
    time_col_list = [time_col]
    input_size = window_config['input_size']
    horizon = window_config['horizon']

    # Ensure model input_dim matches the final feature count (including new lunar encodings)
    num_features = len(feature_cols)
    config.set("model.input_dim", num_features)
    # Ensure model output_dim matches horizon for direct multi-step prediction
    config.set("model.output_dim", horizon)
    model_config = config.model

    print(f"  - Input size: {input_size}")
    print(f"  - Horizon: {horizon}")
    print(f"  - Feature columns: {feature_cols}")
    
    print("  - Creating training windows...")
    X_train, y_train, cat_train = slicing_window_category(
        train_data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=cat_col_list,
        time_col=time_col_list
    )
    
    print("  - Creating validation windows...")
    X_val, y_val, cat_val = slicing_window_category(
        val_data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=cat_col_list,
        time_col=time_col_list
    )
    
    print("  - Creating test windows...")
    X_test, y_test, cat_test = slicing_window_category(
        test_data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=cat_col_list,
        time_col=time_col_list
    )
    
    print(f"  - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"  - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # If using residual targets, also create aligned baseline windows for test set
    # so that we can reconstruct absolute Total CBM after inverse-scaling.
    y_test_baseline = None
    if use_residual and baseline_col_name is not None:
        print("  - Creating baseline windows for test set (for residual reconstruction)...")
        _, y_test_baseline, _ = slicing_window_category(
            test_data,
            input_size,
            horizon,
            feature_cols=feature_cols,
            target_col=[baseline_col_name],
            cat_col=cat_col_list,
            time_col=time_col_list
        )
        print(f"  - y_test_baseline shape: {y_test_baseline.shape}")
    
    # Create datasets
    train_dataset = ForecastDataset(X_train, y_train, cat_train)
    val_dataset = ForecastDataset(X_val, y_val, cat_val)
    test_dataset = ForecastDataset(X_test, y_test, cat_test)
    
    # Create data loaders
    training_config = config.training
    
    # Get category-specific batch_size if specified
    train_batch_size = training_config['batch_size']
    cat_params = get_category_params(category_specific_params, category_filter)
    if 'batch_size' in cat_params:
        train_batch_size = cat_params['batch_size']
        print(f"  - Using category-specific batch_size: {train_batch_size} (for {category_filter or 'default'})")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['val_batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['test_batch_size'],
        shuffle=False
    )
    
    # Build model
    print("\n[8/8] Building model and trainer...")
    
    # Determine model hyperparameters (from category-specific config or fallback to category_specific_params)
    # Category-specific configs should already have these values loaded, but we check both sources
    model_hidden_size = model_config.get('hidden_size', 128)
    model_n_layers = model_config.get('n_layers', 2)
    model_dropout = model_config.get('dropout_prob', 0.0)
    
    # Check category_specific_params as fallback (for backward compatibility)
    cat_params = get_category_params(category_specific_params, category_filter)
    if 'hidden_size' in cat_params:
        model_hidden_size = cat_params['hidden_size']
        print(f"  - Using category-specific hidden_size from category_specific_params: {model_hidden_size} (for {category_filter or 'default'})")
    if 'dropout_prob' in cat_params:
        model_dropout = cat_params['dropout_prob']
        print(f"  - Using category-specific dropout_prob from category_specific_params: {model_dropout} (for {category_filter or 'default'})")
    
    # Ensure numeric types (YAML may parse values as strings)
    model_hidden_size = int(model_hidden_size)
    model_n_layers = int(model_n_layers)
    model_dropout = float(model_dropout)
    
    # Auto-calculate cat_emb_dim if not explicitly set in config
    # Default: use a fraction of hidden_size (common practice: hidden_size // 4 to hidden_size // 2)
    # This ensures the embedding dimension scales appropriately with model capacity
    if 'cat_emb_dim' in model_config:
        cat_emb_dim = model_config['cat_emb_dim']
    else:
        # Auto-calculate: use hidden_size // 4 as default (e.g., 128 -> 32, 64 -> 16)
        # But ensure minimum of 4 and maximum of hidden_size
        cat_emb_dim = max(4, min(model_hidden_size // 4, model_hidden_size))
        print(f"  - Auto-calculated cat_emb_dim={cat_emb_dim} from hidden_size={model_hidden_size}")
    
    # Log the final model architecture
    print(f"  - Model architecture: hidden_size={model_hidden_size}, n_layers={model_n_layers}, dropout={model_dropout}, cat_emb_dim={cat_emb_dim}")
    
    model = RNNWithCategory(
        num_categories=num_categories,
        cat_emb_dim=cat_emb_dim,
        input_dim=model_config['input_dim'],
        hidden_size=model_hidden_size,
        n_layers=model_n_layers,
        output_dim=horizon,  # output_dim must match horizon for direct multi-step prediction
        use_layer_norm=model_config.get('use_layer_norm', True),
        dropout_prob=model_dropout,
        category_specific_params=category_specific_params,
    )
    print(f"  - Model: {model_config['name']}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Transfer Learning: Initialize MOONCAKE weights from pre-trained TET model
    transfer_config = config.get('transfer_learning', {})
    if transfer_config.get('enabled', False) and category_filter == "MOONCAKE":
        source_category = transfer_config.get('source_category', 'TET')
        source_model_path = transfer_config.get('source_model_path')
        
        if source_model_path is None:
            # Auto-detect TET model path
            base_output_dir = config.output.get('output_dir', '../outputs')
            source_model_path = Path(base_output_dir) / source_category / 'models' / 'best_model.pth'
        
        source_model_path = Path(source_model_path)
        
        if source_model_path.exists():
            print(f"  - Transfer Learning: Loading weights from {source_category} model at {source_model_path}")
            try:
                checkpoint = torch.load(source_model_path, map_location='cpu')
                source_state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Get current model state dict
                current_state_dict = model.state_dict()
                
                # Filter compatible layers (same architecture components)
                # Only transfer weights if layer names and shapes match
                transferred_layers = []
                skipped_layers = []
                
                for name, param in source_state_dict.items():
                    if name in current_state_dict:
                        if current_state_dict[name].shape == param.shape:
                            current_state_dict[name] = param
                            transferred_layers.append(name)
                        else:
                            skipped_layers.append(f"{name} (shape mismatch: {param.shape} vs {current_state_dict[name].shape})")
                    else:
                        skipped_layers.append(f"{name} (not in target model)")
                
                # Load the filtered state dict
                model.load_state_dict(current_state_dict, strict=False)
                
                print(f"  - Transfer Learning: Transferred {len(transferred_layers)} layers from {source_category}")
                if skipped_layers:
                    print(f"  - Transfer Learning: Skipped {len(skipped_layers)} incompatible layers")
                
                # Freeze layers if specified
                freeze_layers = transfer_config.get('freeze_layers', [])
                if freeze_layers:
                    for name, param in model.named_parameters():
                        if any(freeze_pattern in name for freeze_pattern in freeze_layers):
                            param.requires_grad = False
                            print(f"  - Transfer Learning: Frozen layer {name}")
            except Exception as e:
                print(f"  - Transfer Learning: Failed to load weights from {source_category} model: {e}")
                print(f"  - Continuing with random initialization...")
        else:
            print(f"  - Transfer Learning: Source model not found at {source_model_path}, using random initialization")
    
    # Build loss function - check if quantile loss is requested
    loss_function_name = training_config.get('loss', 'spike_aware_mse')
    quantile_value = training_config.get('quantile', 0.9)  # Default to 0.9 for P90
    
    if loss_function_name == "quantile":
        # Use Quantile Loss (Pinball Loss) for P90 prediction
        print(f"  - Configuring Quantile Loss (Pinball Loss) with quantile={quantile_value} (P90)...")
        print(f"  - Quantile Loss provides asymmetric penalty (9x more for under-forecasting):")
        print(f"    - Higher penalty when actual > prediction (under-forecasting: stock-outs) = {quantile_value} * error")
        print(f"    - Lower penalty when actual < prediction (over-forecasting: safety stock) = {1-quantile_value} * error")
        print(f"    - Penalty ratio: {quantile_value}/{1-quantile_value:.1f} = {quantile_value/(1-quantile_value):.1f}x more severe for under-forecasting")
        
        # Get active season weight from category-specific params
        active_season_weight = 1.0  # Default (no additional weighting)
        peak_loss_window_weight = 1.0  # Default (no additional weighting)
        august_boost_weight = 1.0  # Default (no additional weighting)
        cat_params = get_category_params(category_specific_params, category_filter)
        if 'active_season_weight' in cat_params:
            active_season_weight = float(cat_params['active_season_weight'])
            print(f"  - Active season weight: {active_season_weight}x (applies to entire active season, 70+ days)")
        if 'peak_loss_window_weight' in cat_params:
            peak_loss_window_weight = float(cat_params['peak_loss_window_weight'])
            print(f"  - Peak Loss Window weight: {peak_loss_window_weight}x (Lunar Months 7.15 to 8.15, critical peak period)")
        if 'august_boost_weight' in cat_params:
            august_boost_weight = float(cat_params['august_boost_weight'])
            print(f"  - August boost weight: {august_boost_weight}x (Gregorian August, for {category_filter or 'default'})")
        
        # Find is_active_season feature index for extracting from inputs
        is_active_season_idx = None
        if 'is_active_season' in feature_cols:
            is_active_season_idx = feature_cols.index('is_active_season')
            print(f"  - is_active_season feature found at index {is_active_season_idx}")
        
        # Find is_peak_loss_window feature index for extracting from inputs
        is_peak_loss_window_idx = None
        if 'is_peak_loss_window' in feature_cols:
            is_peak_loss_window_idx = feature_cols.index('is_peak_loss_window')
            print(f"  - is_peak_loss_window feature found at index {is_peak_loss_window_idx}")
        
        # Find is_august feature index for extracting from inputs
        is_august_idx = None
        if 'is_august' in feature_cols:
            is_august_idx = feature_cols.index('is_august')
            print(f"  - is_august feature found at index {is_august_idx}")
        
        # Create QuantileLoss instance with active season, peak loss window, and August boost weighting
        quantile_loss_instance = QuantileLoss(
            quantile=quantile_value, 
            reduction='mean', 
            active_season_weight=active_season_weight,
            peak_loss_window_weight=peak_loss_window_weight,
            august_boost_weight=august_boost_weight
        )
        
        # Create wrapper function to extract is_active_season, is_peak_loss_window, and is_august from inputs
        def create_quantile_criterion(quantile_loss_fn, is_active_season_feature_idx, is_peak_loss_window_feature_idx, is_august_feature_idx):
            def criterion_fn(y_pred, y_true, category_ids=None, inputs=None):
                # Extract is_active_season for the prediction horizon
                is_active_season_horizon = None
                if is_active_season_feature_idx is not None and inputs is not None and active_season_weight > 1.0:
                    # Extract is_active_season from the last timestep of input
                    last_day_is_active = inputs[:, -1, is_active_season_feature_idx]  # (batch,) or (batch, 1)
                    if last_day_is_active.ndim > 1:
                        last_day_is_active = last_day_is_active.squeeze()
                    
                    if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                        batch_size = y_pred.shape[0]
                        horizon = y_pred.shape[1]
                        # Expand to match horizon shape
                        is_active_season_horizon = last_day_is_active.unsqueeze(-1).expand(batch_size, horizon)
                    else:  # Single-step
                        is_active_season_horizon = last_day_is_active
                
                # Extract is_peak_loss_window for the prediction horizon
                is_peak_loss_window_horizon = None
                if is_peak_loss_window_feature_idx is not None and inputs is not None and peak_loss_window_weight > 1.0:
                    # Extract is_peak_loss_window from the last timestep of input
                    last_day_is_peak = inputs[:, -1, is_peak_loss_window_feature_idx]  # (batch,) or (batch, 1)
                    if last_day_is_peak.ndim > 1:
                        last_day_is_peak = last_day_is_peak.squeeze()
                    
                    if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                        batch_size = y_pred.shape[0]
                        horizon = y_pred.shape[1]
                        # Expand to match horizon shape
                        is_peak_loss_window_horizon = last_day_is_peak.unsqueeze(-1).expand(batch_size, horizon)
                    else:  # Single-step
                        is_peak_loss_window_horizon = last_day_is_peak
                
                # Extract is_august for the prediction horizon (for MOONCAKE category - Gregorian August)
                is_august_horizon = None
                if is_august_feature_idx is not None and inputs is not None and august_boost_weight > 1.0:
                    # Extract is_august from the last timestep of input
                    last_day_is_august = inputs[:, -1, is_august_feature_idx]  # (batch,) or (batch, 1)
                    if last_day_is_august.ndim > 1:
                        last_day_is_august = last_day_is_august.squeeze()
                    
                    if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                        batch_size = y_pred.shape[0]
                        horizon = y_pred.shape[1]
                        # Expand to match horizon shape
                        is_august_horizon = last_day_is_august.unsqueeze(-1).expand(batch_size, horizon)
                    else:  # Single-step
                        is_august_horizon = last_day_is_august
                
                return quantile_loss_fn(
                    y_pred, y_true, 
                    category_ids=category_ids, 
                    inputs=inputs, 
                    is_active_season=is_active_season_horizon,
                    is_peak_loss_window=is_peak_loss_window_horizon,
                    is_august=is_august_horizon
                )
            return criterion_fn
        
        criterion = create_quantile_criterion(quantile_loss_instance, is_active_season_idx, is_peak_loss_window_idx, is_august_idx)
        print(f"  - [OK] Quantile Loss configured for {category_filter or 'all categories'}")
        print(f"  - Target: At least {quantile_value*100:.0f}% of actual values should fall below P90 prediction")
    else:
        # Use spike_aware_mse loss (default behavior)
        print(f"  - Configuring spike_aware_mse loss with category-specific weights...")
        
        # Build category loss weights mapping (category_id -> loss_weight)
        category_loss_weights = {}
        cat_params = get_category_params(category_specific_params, category_filter)
        if 'loss_weight' in cat_params:
            # Map category name to category_id
            if category_filter and category_filter in cat2id:
                category_loss_weights[cat2id[category_filter]] = cat_params['loss_weight']
                print(f"  - Loss weight for {category_filter}: {cat_params['loss_weight']}")
        
        # Get Monday/Wednesday/Friday loss weights from config (for FRESH category)
        monday_loss_weight = 3.0  # Default
        wednesday_loss_weight = 1.0  # Default (no additional weighting)
        friday_loss_weight = 1.0  # Default (no additional weighting)
        early_month_loss_weight = 1.0  # Default (no additional weighting for DRY category)
        use_dynamic_early_month_weight = False  # Default: use static weighting
        cat_params = get_category_params(category_specific_params, category_filter)
        if 'monday_loss_weight' in cat_params:
            monday_loss_weight = float(cat_params['monday_loss_weight'])
            print(f"  - Monday loss weight: {monday_loss_weight}x (for {category_filter or 'default'})")
        if 'wednesday_loss_weight' in cat_params:
            wednesday_loss_weight = float(cat_params['wednesday_loss_weight'])
            print(f"  - Wednesday loss weight: {wednesday_loss_weight}x (for {category_filter or 'default'})")
        if 'friday_loss_weight' in cat_params:
            friday_loss_weight = float(cat_params['friday_loss_weight'])
            print(f"  - Friday loss weight: {friday_loss_weight}x (for {category_filter or 'default'})")
        if 'early_month_loss_weight' in cat_params:
            early_month_loss_weight = float(cat_params['early_month_loss_weight'])
            print(f"  - Early Month loss weight (static): {early_month_loss_weight}x (days 1-10, for {category_filter or 'default'})")
        if 'use_dynamic_early_month_weight' in cat_params:
            use_dynamic_early_month_weight = bool(cat_params['use_dynamic_early_month_weight'])
            if use_dynamic_early_month_weight:
                print(f"  - SOLUTION 2: Dynamic Early Month Weighting ENABLED (Days 1-3: 20x, Days 4-10: linear decay, for {category_filter or 'default'})")
            else:
                print(f"  - Using static early month weighting (for {category_filter or 'default'})")
        
        # Get Golden Window loss weight from config (for MOONCAKE category)
        golden_window_weight = 1.0  # Default (no additional weighting)
        peak_loss_window_weight = 1.0  # Default (no additional weighting)
        august_boost_weight = 1.0  # Default (no additional weighting)
        use_smooth_l1 = False
        smooth_l1_beta = 1.0
        # Balanced Distribution parameters (new)
        use_asymmetric_penalty = False
        over_pred_penalty = 2.0
        under_pred_penalty = 1.0
        apply_mean_error_constraint = False
        mean_error_weight = 0.1
        
        if 'golden_window_weight' in cat_params:
            golden_window_weight = float(cat_params['golden_window_weight'])
            print(f"  - Golden Window loss weight: {golden_window_weight}x (for {category_filter or 'default'})")
        if 'peak_loss_window_weight' in cat_params:
            peak_loss_window_weight = float(cat_params['peak_loss_window_weight'])
            print(f"  - Peak Loss Window weight: {peak_loss_window_weight}x (Lunar Months 7.15 to 8.15, for {category_filter or 'default'})")
        if 'august_boost_weight' in cat_params:
            august_boost_weight = float(cat_params['august_boost_weight'])
            print(f"  - August boost weight: {august_boost_weight}x (Gregorian August, for {category_filter or 'default'})")
        if 'use_smooth_l1' in cat_params:
            use_smooth_l1 = bool(cat_params['use_smooth_l1'])
            print(f"  - Using SmoothL1Loss: {use_smooth_l1} (for {category_filter or 'default'})")
        if 'smooth_l1_beta' in cat_params:
            smooth_l1_beta = float(cat_params['smooth_l1_beta'])
            print(f"  - SmoothL1Loss beta: {smooth_l1_beta} (for {category_filter or 'default'})")
        
        # Balanced Distribution Mode parameters
        if 'use_asymmetric_penalty' in cat_params:
            use_asymmetric_penalty = bool(cat_params['use_asymmetric_penalty'])
            if use_asymmetric_penalty:
                print(f"  - BALANCED DISTRIBUTION MODE ENABLED: Asymmetric penalties active (for {category_filter or 'default'})")
        if 'over_pred_penalty' in cat_params:
            over_pred_penalty = float(cat_params['over_pred_penalty'])
            if use_asymmetric_penalty:
                print(f"  - Over-prediction penalty: {over_pred_penalty}x in non-peak periods (eliminates upward bias)")
        if 'under_pred_penalty' in cat_params:
            under_pred_penalty = float(cat_params['under_pred_penalty'])
            if use_asymmetric_penalty:
                print(f"  - Under-prediction penalty: {under_pred_penalty}x in non-peak periods")
        if 'apply_mean_error_constraint' in cat_params:
            apply_mean_error_constraint = bool(cat_params['apply_mean_error_constraint'])
            if apply_mean_error_constraint:
                print(f"  - Mean Error Constraint ENABLED: Forces monthly predictions to align with historical averages")
        if 'mean_error_weight' in cat_params:
            mean_error_weight = float(cat_params['mean_error_weight'])
            if apply_mean_error_constraint:
                print(f"  - Mean error weight: {mean_error_weight} (pulls down inflated predictions)")
        
        # Find feature indices for extracting from inputs
        is_monday_idx = None
        day_of_week_sin_idx = None
        day_of_week_cos_idx = None
        is_early_month_low_idx = None
        days_from_month_start_idx = None  # For dynamic early month weighting
        dayofmonth_sin_idx = None  # Fallback for reconstructing day_of_month
        dayofmonth_cos_idx = None  # Fallback for reconstructing day_of_month
        is_golden_window_idx = None
        if 'Is_Monday' in feature_cols:
            is_monday_idx = feature_cols.index('Is_Monday')
            print(f"  - Is_Monday feature found at index {is_monday_idx}")
        if 'day_of_week_sin' in feature_cols and 'day_of_week_cos' in feature_cols:
            day_of_week_sin_idx = feature_cols.index('day_of_week_sin')
            day_of_week_cos_idx = feature_cols.index('day_of_week_cos')
            print(f"  - day_of_week_sin/cos features found at indices {day_of_week_sin_idx}/{day_of_week_cos_idx}")
        if 'is_early_month_low' in feature_cols:
            is_early_month_low_idx = feature_cols.index('is_early_month_low')
            print(f"  - is_early_month_low feature found at index {is_early_month_low_idx}")
        if 'days_from_month_start' in feature_cols:
            days_from_month_start_idx = feature_cols.index('days_from_month_start')
            print(f"  - days_from_month_start feature found at index {days_from_month_start_idx} (for dynamic early month weighting)")
        if 'dayofmonth_sin' in feature_cols and 'dayofmonth_cos' in feature_cols:
            dayofmonth_sin_idx = feature_cols.index('dayofmonth_sin')
            dayofmonth_cos_idx = feature_cols.index('dayofmonth_cos')
            print(f"  - dayofmonth_sin/cos features found at indices {dayofmonth_sin_idx}/{dayofmonth_cos_idx} (fallback for day reconstruction)")
        if 'is_golden_window' in feature_cols:
            is_golden_window_idx = feature_cols.index('is_golden_window')
            print(f"  - is_golden_window feature found at index {is_golden_window_idx}")
        
        is_peak_loss_window_idx = None
        if 'is_peak_loss_window' in feature_cols:
            is_peak_loss_window_idx = feature_cols.index('is_peak_loss_window')
            print(f"  - is_peak_loss_window feature found at index {is_peak_loss_window_idx}")
        
        is_august_idx = None
        if 'is_august' in feature_cols:
            is_august_idx = feature_cols.index('is_august')
            print(f"  - is_august feature found at index {is_august_idx}")
        
        # Create a partial function that includes category loss weights, Mon/Wed/Fri weighting, Early Month weighting (static or dynamic), Golden Window weighting, and Balanced Distribution parameters
        def create_criterion(cat_loss_weights, monday_weight, wednesday_weight, friday_weight, early_month_weight, use_dyn_early_month, golden_window_weight, peak_loss_window_weight, august_boost_weight, use_smooth_l1, smooth_l1_beta,
                            use_asym_penalty, over_penalty, under_penalty, apply_mean_constraint, mean_constraint_weight,
                            is_monday_feature_idx, day_of_week_sin_idx, day_of_week_cos_idx, is_early_month_low_idx, days_from_month_start_idx, dayofmonth_sin_idx, dayofmonth_cos_idx, is_golden_window_idx, is_peak_loss_window_idx, is_august_idx, horizon_days):
            def criterion_fn(y_pred, y_true, category_ids=None, inputs=None):
                # Extract day of week for the prediction horizon (for Monday/Wednesday/Friday weighting)
                is_monday_horizon = None
                is_wednesday_horizon = None
                is_friday_horizon = None
                
                # Determine if we need to compute day of week (if any weight > 1.0)
                need_dow = (monday_weight > 1.0 or wednesday_weight > 1.0 or friday_weight > 1.0)
                
                if (is_monday_feature_idx is not None or (day_of_week_sin_idx is not None and day_of_week_cos_idx is not None)) and inputs is not None and need_dow:
                    # inputs shape: (batch, input_size, features)
                    batch_size = inputs.shape[0]
                    
                    # Compute day of week for the last input day
                    # Method 1: Use Is_Monday if available (most direct)
                    if is_monday_feature_idx is not None:
                        last_day_is_monday = inputs[:, -1, is_monday_feature_idx]  # (batch,) or (batch, 1)
                        # Ensure 1D
                        if last_day_is_monday.ndim > 1:
                            last_day_is_monday = last_day_is_monday.squeeze()
                        # If Is_Monday == 1, then day_of_week = 0 (Monday)
                        # For non-Monday days, we set to -1 (unknown) and skip Monday weighting for those
                        last_day_dow = torch.where(
                            last_day_is_monday > 0.5, 
                            torch.tensor(0.0, device=inputs.device), 
                            torch.tensor(-1.0, device=inputs.device)
                        )
                        # Ensure 1D shape
                        if last_day_dow.ndim > 1:
                            last_day_dow = last_day_dow.squeeze()
                    # Method 2: Compute from day_of_week_sin/cos (more accurate)
                    elif day_of_week_sin_idx is not None and day_of_week_cos_idx is not None:
                        last_day_sin = inputs[:, -1, day_of_week_sin_idx]  # (batch,) or (batch, 1)
                        last_day_cos = inputs[:, -1, day_of_week_cos_idx]  # (batch,) or (batch, 1)
                        # Ensure 1D
                        if last_day_sin.ndim > 1:
                            last_day_sin = last_day_sin.squeeze()
                        if last_day_cos.ndim > 1:
                            last_day_cos = last_day_cos.squeeze()
                        # Recover day_of_week from sin/cos: atan2(sin, cos) * 7 / (2*pi)
                        # Monday (0) -> sin=0, cos=1
                        # We'll use a simpler approach: check if values are close to Monday's encoding
                        # Monday: sin ≈ 0, cos ≈ 1
                        last_day_is_monday = (torch.abs(last_day_sin) < 0.1) & (last_day_cos > 0.9)
                        last_day_dow = torch.where(
                            last_day_is_monday, 
                            torch.tensor(0.0, device=inputs.device), 
                            torch.tensor(-1.0, device=inputs.device)
                        )
                        # Ensure 1D shape
                        if last_day_dow.ndim > 1:
                            last_day_dow = last_day_dow.squeeze()
                    else:
                        last_day_is_monday = None
                        last_day_dow = None
                    
                    if last_day_dow is not None:
                        # Ensure last_day_dow is 1D: (batch_size,)
                        # Squeeze all dimensions of size 1
                        while last_day_dow.ndim > 1:
                            last_day_dow = last_day_dow.squeeze()
                        # If still not 1D, flatten it
                        if last_day_dow.ndim > 1:
                            last_day_dow = last_day_dow.flatten()
                        
                        if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                            horizon = y_pred.shape[1]
                            is_monday_horizon = torch.zeros((batch_size, horizon), device=y_pred.device)
                            is_wednesday_horizon = torch.zeros((batch_size, horizon), device=y_pred.device)
                            is_friday_horizon = torch.zeros((batch_size, horizon), device=y_pred.device)
                            
                            # Compute which days in horizon are Mon/Wed/Fri
                            # If last input day is Monday (dow=0), then:
                            # - Horizon day 0 (tomorrow) is Tuesday (dow=1)
                            # - Horizon day 6 is Monday (dow=0)
                            # - Horizon day 13 is Monday (dow=0)
                            # General: horizon day H has dow = (last_day_dow + H + 1) % 7
                            for h in range(horizon):
                                # Compute day of week for horizon day h
                                # Last input day is day -1, first horizon day is day 0
                                # last_day_dow is (batch_size,), we add h+1 and take mod 7
                                # Only compute if we know the day of week (last_day_dow >= 0)
                                known_dow_mask = last_day_dow >= 0  # True where we know the day of week
                                horizon_day_dow = torch.where(
                                    known_dow_mask,
                                    (last_day_dow + h + 1) % 7,
                                    torch.tensor(-1.0, device=inputs.device)  # Unknown
                                )
                                # Ensure result is 1D
                                if horizon_day_dow.ndim > 1:
                                    horizon_day_dow = horizon_day_dow.squeeze()
                                # Mark Monday (0), Wednesday (2), Friday (4)
                                is_monday_horizon[:, h] = ((horizon_day_dow == 0) & known_dow_mask).float()
                                is_wednesday_horizon[:, h] = ((horizon_day_dow == 2) & known_dow_mask).float()
                                is_friday_horizon[:, h] = ((horizon_day_dow == 4) & known_dow_mask).float()
                        else:  # Single-step
                            # For single-step, check if the prediction day is Mon/Wed/Fri
                            # If last input day is Monday, next day is Tuesday (not Monday)
                            # We need to check if next day is Mon/Wed/Fri: (last_day_dow + 1) % 7
                            next_day_dow = (last_day_dow + 1) % 7
                            if next_day_dow.ndim > 1:
                                next_day_dow = next_day_dow.squeeze()
                            is_monday_horizon = (next_day_dow == 0).float()
                            is_wednesday_horizon = (next_day_dow == 2).float()
                            is_friday_horizon = (next_day_dow == 4).float()
                    else:
                        # If we can't determine day of week, set all to zero (no weekday weighting)
                        if y_pred.ndim == 2:
                            is_monday_horizon = torch.zeros((batch_size, y_pred.shape[1]), device=y_pred.device)
                            is_wednesday_horizon = torch.zeros((batch_size, y_pred.shape[1]), device=y_pred.device)
                            is_friday_horizon = torch.zeros((batch_size, y_pred.shape[1]), device=y_pred.device)
                        else:
                            is_monday_horizon = torch.zeros_like(y_pred, device=y_pred.device)
                            is_wednesday_horizon = torch.zeros_like(y_pred, device=y_pred.device)
                            is_friday_horizon = torch.zeros_like(y_pred, device=y_pred.device)
                
                # Extract is_early_month_low for the prediction horizon (for DRY category - static weighting)
                is_early_month_horizon = None
                if is_early_month_low_idx is not None and inputs is not None and early_month_weight > 1.0 and not use_dyn_early_month:
                    # Extract is_early_month_low from the last timestep of input
                    # For multi-step forecasting, we use the last input day's early month status
                    # and apply it to all horizon days (simplified approach)
                    last_day_is_early = inputs[:, -1, is_early_month_low_idx]  # (batch,) or (batch, 1)
                    if last_day_is_early.ndim > 1:
                        last_day_is_early = last_day_is_early.squeeze()
                    
                    if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                        batch_size = y_pred.shape[0]
                        horizon = y_pred.shape[1]
                        # Expand to match horizon shape
                        is_early_month_horizon = last_day_is_early.unsqueeze(-1).expand(batch_size, horizon)
                    else:  # Single-step
                        is_early_month_horizon = last_day_is_early
                
                # SOLUTION 2: Extract day_of_month for dynamic early month weighting (for DRY category)
                day_of_month_horizon = None
                if use_dyn_early_month and inputs is not None:
                    # Method 1: Use days_from_month_start if available (most direct)
                    if days_from_month_start_idx is not None:
                        # days_from_month_start is 0-indexed (0 on 1st, 1 on 2nd, etc.)
                        # Convert to 1-indexed day_of_month (1-31)
                        last_day_from_start = inputs[:, -1, days_from_month_start_idx]  # (batch,) or (batch, 1)
                        if last_day_from_start.ndim > 1:
                            last_day_from_start = last_day_from_start.squeeze()
                        day_of_month_last = last_day_from_start + 1  # Convert to 1-indexed
                        
                        if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                            batch_size = y_pred.shape[0]
                            horizon = y_pred.shape[1]
                            # For multi-step, we need to project forward
                            # Simplified: assume the forecast horizon starts from day_of_month_last + 1
                            # and increments by 1 for each horizon step (wrapping handled by loss function)
                            day_of_month_horizon = day_of_month_last.unsqueeze(-1).expand(batch_size, horizon) + torch.arange(1, horizon + 1, device=day_of_month_last.device).unsqueeze(0)
                            # Clip to valid day range (1-31, though month length varies)
                            day_of_month_horizon = torch.clamp(day_of_month_horizon, 1, 31)
                        else:  # Single-step
                            day_of_month_horizon = day_of_month_last + 1  # Next day
                            day_of_month_horizon = torch.clamp(day_of_month_horizon, 1, 31)
                    # Method 2: Reconstruct from dayofmonth_sin/cos (fallback)
                    elif dayofmonth_sin_idx is not None and dayofmonth_cos_idx is not None:
                        # Reconstruct day_of_month from cyclical features
                        # Formula: day = arctan2(sin, cos) * 31 / (2π) + 1
                        last_sin = inputs[:, -1, dayofmonth_sin_idx]  # (batch,)
                        last_cos = inputs[:, -1, dayofmonth_cos_idx]  # (batch,)
                        if last_sin.ndim > 1:
                            last_sin = last_sin.squeeze()
                        if last_cos.ndim > 1:
                            last_cos = last_cos.squeeze()
                        
                        # Reconstruct day_of_month (1-31)
                        angle = torch.atan2(last_sin, last_cos)  # Range: [-π, π]
                        day_of_month_last = (angle / (2 * np.pi)) * 31 + 1  # Convert to 1-31
                        day_of_month_last = torch.clamp(torch.round(day_of_month_last), 1, 31).long()
                        
                        if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                            batch_size = y_pred.shape[0]
                            horizon = y_pred.shape[1]
                            # Project forward
                            day_of_month_horizon = day_of_month_last.unsqueeze(-1).expand(batch_size, horizon) + torch.arange(1, horizon + 1, device=day_of_month_last.device).unsqueeze(0)
                            day_of_month_horizon = torch.clamp(day_of_month_horizon, 1, 31)
                        else:  # Single-step
                            day_of_month_horizon = day_of_month_last + 1
                            day_of_month_horizon = torch.clamp(day_of_month_horizon, 1, 31)
                
                # Extract is_golden_window for the prediction horizon (for MOONCAKE category)
                is_golden_window_horizon = None
                if is_golden_window_idx is not None and inputs is not None and golden_window_weight > 1.0:
                    # Extract is_golden_window from the last timestep of input
                    # For multi-step forecasting, we use the last input day's golden window status
                    # and apply it to all horizon days (simplified approach)
                    last_day_is_golden = inputs[:, -1, is_golden_window_idx]  # (batch,) or (batch, 1)
                    if last_day_is_golden.ndim > 1:
                        last_day_is_golden = last_day_is_golden.squeeze()
                    
                    if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                        batch_size = y_pred.shape[0]
                        horizon = y_pred.shape[1]
                        # Expand to match horizon shape
                        is_golden_window_horizon = last_day_is_golden.unsqueeze(-1).expand(batch_size, horizon)
                    else:  # Single-step
                        is_golden_window_horizon = last_day_is_golden
                
                # Extract is_peak_loss_window for the prediction horizon (for MOONCAKE category - Lunar Months 7.15 to 8.15)
                is_peak_loss_window_horizon = None
                if is_peak_loss_window_idx is not None and inputs is not None and peak_loss_window_weight > 1.0:
                    # Extract is_peak_loss_window from the last timestep of input
                    last_day_is_peak = inputs[:, -1, is_peak_loss_window_idx]  # (batch,) or (batch, 1)
                    if last_day_is_peak.ndim > 1:
                        last_day_is_peak = last_day_is_peak.squeeze()
                    
                    if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                        batch_size = y_pred.shape[0]
                        horizon = y_pred.shape[1]
                        # Expand to match horizon shape
                        is_peak_loss_window_horizon = last_day_is_peak.unsqueeze(-1).expand(batch_size, horizon)
                    else:  # Single-step
                        is_peak_loss_window_horizon = last_day_is_peak
                
                # Extract is_august for the prediction horizon (for MOONCAKE category - Gregorian August)
                is_august_horizon = None
                if is_august_idx is not None and inputs is not None and august_boost_weight > 1.0:
                    # Extract is_august from the last timestep of input
                    last_day_is_august = inputs[:, -1, is_august_idx]  # (batch,) or (batch, 1)
                    if last_day_is_august.ndim > 1:
                        last_day_is_august = last_day_is_august.squeeze()
                    
                    if y_pred.ndim == 2:  # Multi-step: (batch, horizon)
                        batch_size = y_pred.shape[0]
                        horizon = y_pred.shape[1]
                        # Expand to match horizon shape
                        is_august_horizon = last_day_is_august.unsqueeze(-1).expand(batch_size, horizon)
                    else:  # Single-step
                        is_august_horizon = last_day_is_august
                
                return spike_aware_mse(
                    y_pred, 
                    y_true, 
                    category_ids=category_ids,
                    category_loss_weights=cat_loss_weights if cat_loss_weights else None,
                    is_monday=is_monday_horizon,
                    monday_loss_weight=monday_weight,
                    is_wednesday=is_wednesday_horizon,
                    wednesday_loss_weight=wednesday_weight,
                    is_friday=is_friday_horizon,
                    friday_loss_weight=friday_weight,
                    is_early_month=is_early_month_horizon,
                    early_month_loss_weight=early_month_weight,
                    day_of_month=day_of_month_horizon,
                    use_dynamic_early_month_weight=use_dyn_early_month,
                    is_golden_window=is_golden_window_horizon,
                    golden_window_weight=golden_window_weight,
                    is_peak_loss_window=is_peak_loss_window_horizon,
                    peak_loss_window_weight=peak_loss_window_weight,
                    is_august=is_august_horizon,
                    august_boost_weight=august_boost_weight,
                    use_smooth_l1=use_smooth_l1,
                    smooth_l1_beta=smooth_l1_beta,
                    use_asymmetric_penalty=use_asym_penalty,
                    over_pred_penalty=over_penalty,
                    under_pred_penalty=under_penalty,
                    apply_mean_error_constraint=apply_mean_constraint,
                    mean_error_weight=mean_constraint_weight
                )
            return criterion_fn
        
        criterion = create_criterion(
            category_loss_weights, 
            monday_loss_weight,
            wednesday_loss_weight,
            friday_loss_weight,
            early_month_loss_weight,
            use_dynamic_early_month_weight,
            golden_window_weight,
            peak_loss_window_weight,
            august_boost_weight,
            use_smooth_l1,
            smooth_l1_beta,
            use_asymmetric_penalty,
            over_pred_penalty,
            under_pred_penalty,
            apply_mean_error_constraint,
            mean_error_weight,
            is_monday_idx, 
            day_of_week_sin_idx, 
            day_of_week_cos_idx,
            is_early_month_low_idx,
            days_from_month_start_idx,
            dayofmonth_sin_idx,
            dayofmonth_cos_idx,
            is_golden_window_idx,
            is_peak_loss_window_idx,
            is_august_idx,
            horizon
        )
    
    # Build optimizer with category-specific learning rate
    # Priority: category-specific config > category_specific_params > base config
    optimizer_lr = training_config.get('learning_rate', 0.001)
    optimizer_weight_decay = training_config.get('weight_decay', 0.0)
    
    # Check category_specific_params as fallback (for backward compatibility)
    cat_params = get_category_params(category_specific_params, category_filter)
    if 'learning_rate' in cat_params:
        optimizer_lr = cat_params['learning_rate']
        print(f"  - Using category-specific learning_rate from category_specific_params: {optimizer_lr} (for {category_filter or 'default'})")
    if 'weight_decay' in cat_params:
        optimizer_weight_decay = cat_params['weight_decay']
        print(f"  - Using category-specific weight_decay from category_specific_params: {optimizer_weight_decay} (for {category_filter or 'default'})")
    
    # Ensure numeric types (YAML may parse scientific notation as strings)
    optimizer_lr = float(optimizer_lr)
    optimizer_weight_decay = float(optimizer_weight_decay)
    
    print(f"  - Optimizer: Adam(lr={optimizer_lr}, weight_decay={optimizer_weight_decay})")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_lr,
        weight_decay=optimizer_weight_decay
    )
    
    # Build scheduler with category-specific parameters
    # Priority: category-specific config > category_specific_params > base config
    scheduler_config = training_config.get('scheduler', {})
    scheduler_name = scheduler_config.get('name', 'ReduceLROnPlateau')
    
    scheduler_factor = scheduler_config.get('factor', 0.5)
    scheduler_patience = scheduler_config.get('patience', 3)
    
    # Check category_specific_params as fallback (for backward compatibility)
    cat_params = get_category_params(category_specific_params, category_filter)
    if 'scheduler_factor' in cat_params:
        scheduler_factor = cat_params['scheduler_factor']
        print(f"  - Using category-specific scheduler_factor from category_specific_params: {scheduler_factor} (for {category_filter or 'default'})")
    if 'patience' in cat_params:
        scheduler_patience = cat_params['patience']
        print(f"  - Using category-specific scheduler patience from category_specific_params: {scheduler_patience} (for {category_filter or 'default'})")
    
    # Ensure numeric types (YAML may parse values as strings)
    scheduler_factor = float(scheduler_factor)
    scheduler_patience = int(scheduler_patience)
    
    if scheduler_name == 'ReduceLROnPlateau':
        min_lr = scheduler_config.get('min_lr', 1e-5)
        min_lr = float(min_lr)  # Ensure min_lr is float
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr
        )
    else:
        scheduler = None
    
    # Setup device
    device = torch.device(training_config['device'])
    print(f"  - Device: {device}")
    
    # Create trainer
    save_dir = config.output.get('model_dir')
    print(f"  - Model save directory: {save_dir}")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        log_interval=training_config['log_interval'],
        save_dir=save_dir
    )
    
    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    # Convert config to dictionary for saving in checkpoint and metadata
    config_dict = config._config.copy()
    
    # Get category-specific epochs
    # Priority: category-specific config > category_specific_params > base config
    training_epochs = training_config.get('epochs', 20)
    
    # Check category_specific_params as fallback (for backward compatibility)
    cat_params = get_category_params(category_specific_params, category_filter)
    if 'epochs' in cat_params:
        training_epochs = cat_params['epochs']
        print(f"  - Using category-specific epochs from category_specific_params: {training_epochs} (for {category_filter or 'default'})")
    
    # Ensure numeric type (YAML may parse values as strings)
    training_epochs = int(training_epochs)
    
    print(f"  - Training epochs: {training_epochs}")
    
    train_losses, val_losses = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_epochs,
        save_best=True,
        verbose=True,
        config=config_dict
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save scaler for use in prediction
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  - Scaler saved to: {scaler_path}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    test_loss, y_true, y_pred = trainer.evaluate(
        test_loader,
        return_predictions=True
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test samples: {len(y_true)}")
    
    # Calculate quantile coverage if using quantile loss (primary metric for P90)
    if loss_function_name == "quantile":
        print(f"\n  - Calculating Quantile Coverage (Primary Metric for P90)...")
        quantile_val = training_config.get('quantile', 0.9)
        
        # Calculate coverage on original scale (after inverse transform)
        # We'll calculate this after inverse scaling below
        use_quantile_evaluation = True
    else:
        use_quantile_evaluation = False
    
    # Apply seasonal active-window masking (hard-zero constraint) for seasonal categories
    # This enforces that predictions are zero during off-season periods
    # Check if category-specific config requires seasonal masking
    apply_seasonal_mask = data_config.get('apply_seasonal_mask', False)
    seasonal_active_window = data_config.get('seasonal_active_window', None)
    seasonal_categories = ["TET", "MOONCAKE"]  # Known seasonal categories
    
    if apply_seasonal_mask or category_filter in seasonal_categories:
        print(f"  - Applying seasonal active-window masking for {category_filter}...")
        print(f"  - Seasonal active window: {seasonal_active_window}")
        
        # Find is_active_season feature index
        feature_cols = data_config['feature_cols']
        if 'is_active_season' in feature_cols:
            is_active_idx = feature_cols.index('is_active_season')
            print(f"  - Found is_active_season at feature index {is_active_idx}")
            
            # Extract is_active_season from test data windows
            # Note: This is a simplified approach using the last timestep's value
            # For production, compute is_active_season for each future timestep in horizon
            test_dataset = test_loader.dataset
            if hasattr(test_dataset, 'X'):
                # Get is_active_season from the last timestep of input windows
                is_active_season_values = test_dataset.X[:, -1, is_active_idx]  # (N_samples,)
                
                # Expand to match prediction shape (N_samples, horizon)
                if y_pred.ndim == 2:
                    is_active_mask = np.tile(is_active_season_values[:, np.newaxis], (1, y_pred.shape[1]))
                else:
                    is_active_mask = is_active_season_values
                
                # Apply hard-zero constraint: set predictions to 0 when not in active season
                # This is a strict enforcement - any prediction outside the active window must be zero
                zero_mask = (is_active_mask > 0.5).astype(float)
                y_pred = y_pred * zero_mask
                
                num_active = np.sum(zero_mask > 0.5)
                num_total = zero_mask.size
                print(f"  - Applied seasonal masking: {num_active}/{num_total} timesteps in active season")
                print(f"  - Zero-enforced predictions: {num_total - num_active} timesteps forced to zero")
                
                # Additional check for MOONCAKE: ensure lunar_months_6_9 is strictly enforced
                if category_filter == "MOONCAKE" and seasonal_active_window == "lunar_months_6_9":
                    # Verify that predictions outside the window are indeed zero
                    num_nonzero_outside = np.sum((y_pred > 1e-6) & (zero_mask < 0.5))
                    if num_nonzero_outside > 0:
                        print(f"  - WARNING: Found {num_nonzero_outside} non-zero predictions outside active window!")
                        print(f"  - Forcing all off-season predictions to zero...")
                        y_pred = y_pred * zero_mask  # Re-apply mask to ensure strict enforcement
        else:
            print(f"  - Warning: is_active_season feature not found in feature_cols. Seasonal masking skipped.")
    
    # DEBUG: Check y_true values (should be in scaled space of target_col_for_model)
    print("  - DEBUG: Checking y_true values (scaled):")
    print(f"    - Min: {y_true.min():.4f}")
    print(f"    - Max: {y_true.max():.4f}")
    print(f"    - Mean: {y_true.mean():.4f}")
    print(f"    - Non-zero count: {np.sum(y_true != 0)} / {len(y_true)}")
    print(f"    - Zero count: {np.sum(y_true == 0)} / {len(y_true)}")
    
    # CRITICAL: Inverse transform predictions and true values back to original scale
    # For direct multi-step forecasting, y_true and y_pred have shape (N_samples, horizon)
    # We need to flatten them for inverse transform, then reshape back if needed
    print("  - Inverse transforming predictions to supervised target scale...")
    print(f"  - y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    
    # Flatten for inverse transform (StandardScaler expects 1D or 2D with single feature)
    y_true_flat = y_true.reshape(-1, 1) if y_true.ndim > 1 else y_true.reshape(-1, 1)
    y_pred_flat = y_pred.reshape(-1, 1) if y_pred.ndim > 1 else y_pred.reshape(-1, 1)
    
    y_true_supervised_flat = inverse_transform_scaling(y_true_flat, scaler, target_col=target_col_for_model)
    y_pred_supervised_flat = inverse_transform_scaling(y_pred_flat, scaler, target_col=target_col_for_model)
    
    # Reshape back to (N_samples, horizon) if multi-step
    if y_true.ndim > 1:
        y_true_supervised = y_true_supervised_flat.reshape(y_true.shape)
        y_pred_supervised = y_pred_supervised_flat.reshape(y_pred.shape)
    else:
        y_true_supervised = y_true_supervised_flat
        y_pred_supervised = y_pred_supervised_flat

    # If residual learning is enabled, reconstruct absolute Total CBM by
    # adding back the baseline that was subtracted during target creation.
    if use_residual and y_test_baseline is not None:
        print("  - Reconstructing absolute Total CBM from residuals + baseline...")
        baseline_flat = y_test_baseline.reshape(-1)
        y_true_resid_flat = y_true_supervised.reshape(-1)
        y_pred_resid_flat = y_pred_supervised.reshape(-1)

        y_true_original = y_true_resid_flat + baseline_flat
        y_pred_original = y_pred_resid_flat + baseline_flat
    else:
        # No residuals: supervised target is already the absolute series
        # Flatten for plotting/evaluation
        y_true_original = y_true_supervised.reshape(-1)
        y_pred_original = y_pred_supervised.reshape(-1)
    
    # DEBUG: Check y_true_original values (should be in original scale)
    print("  - DEBUG: Checking y_true_original values (after inverse transform):")
    print(f"    - Min: {y_true_original.min():.4f}")
    print(f"    - Max: {y_true_original.max():.4f}")
    print(f"    - Mean: {y_true_original.mean():.4f}")
    print(f"    - Non-zero count: {np.sum(y_true_original != 0)} / {len(y_true_original)}")
    print(f"    - Zero count: {np.sum(y_true_original == 0)} / {len(y_true_original)}")
    print(f"    - First 10 values: {y_true_original[:10]}")
    
    # Calculate quantile coverage (primary metric for P90 quantile regression)
    training_results = {}  # Initialize training results dict
    if use_quantile_evaluation:
        print(f"\n  - Quantile Coverage Evaluation (P90):")
        quantile_val = training_config.get('quantile', 0.9)
        
        # Reshape to match original shapes if needed
        y_true_eval = y_true_original.reshape(-1) if y_true_original.ndim > 1 else y_true_original
        y_pred_eval = y_pred_original.reshape(-1) if y_pred_original.ndim > 1 else y_pred_original
        
        # Calculate comprehensive metrics including quantile coverage
        metrics = calculate_forecast_metrics(
            y_true_eval,
            y_pred_eval,
            quantile=quantile_val,
            mask=None  # Could add mask to exclude off-season zeros if needed
        )
        
        coverage_rate = metrics['quantile_coverage']
        coverage_details = metrics['quantile_coverage_details']
        
        print(f"    - Quantile Coverage: {coverage_rate:.2%} (Target: {quantile_val*100:.0f}%)")
        print(f"    - Coverage Error: {coverage_details['coverage_error']:+.2%} (actual - target)")
        print(f"    - Covered samples: {coverage_details['over_coverage']} / {len(y_true_eval)}")
        print(f"    - Violations (under-coverage): {coverage_details['under_coverage']} samples")
        if coverage_details['under_coverage'] > 0:
            print(f"    - Mean Excess (violations): {coverage_details['mean_excess']:.2f} CBM")
        
        # Success criterion: at least 90% coverage for P90
        if coverage_rate >= quantile_val:
            print(f"    - [SUCCESS] Coverage ({coverage_rate:.2%}) meets target ({quantile_val*100:.0f}%)")
        else:
            print(f"    - ⚠ WARNING: Coverage ({coverage_rate:.2%}) below target ({quantile_val*100:.0f}%)")
        
        # Also print traditional metrics for reference
        print(f"    - MAE: {metrics['mae']:.2f} CBM (reference)")
        print(f"    - RMSE: {metrics['rmse']:.2f} CBM (reference)")
        
        # Store quantile coverage in metadata (convert numpy types to native Python types for JSON)
        training_results['quantile_coverage'] = float(coverage_rate)
        training_results['quantile_coverage_details'] = {
            'coverage_rate': float(coverage_details['coverage_rate']),
            'target_coverage': float(coverage_details['target_coverage']),
            'coverage_error': float(coverage_details['coverage_error']),
            'under_coverage': int(coverage_details['under_coverage']),
            'over_coverage': int(coverage_details['over_coverage']),
            'mean_excess': float(coverage_details['mean_excess'])
        }
        training_results['mae'] = float(metrics['mae'])
        training_results['rmse'] = float(metrics['rmse'])
    else:
        # For non-quantile loss, calculate traditional metrics
        y_true_eval = y_true_original.reshape(-1) if y_true_original.ndim > 1 else y_true_original
        y_pred_eval = y_pred_original.reshape(-1) if y_pred_original.ndim > 1 else y_pred_original
        metrics = calculate_forecast_metrics(y_true_eval, y_pred_eval)
        print(f"\n  - Traditional Metrics:")
        print(f"    - MAE: {metrics['mae']:.2f} CBM")
        print(f"    - RMSE: {metrics['rmse']:.2f} CBM")
        training_results['mae'] = float(metrics['mae'])
        training_results['rmse'] = float(metrics['rmse'])
    
    # Save training metadata (including a compact text summary of this log)
    print("\n[9/9] Saving training metadata...")
    log_summary_lines = [
        f"Category mode: {data_config.get('category_mode', 'single')}",
        f"Category filter: {category_filter if category_filter else 'ALL CATEGORIES'}",
        f"Number of categories: {num_categories}",
        f"Category mapping: {cat2id}",
        f"Best validation loss: {trainer.best_val_loss:.4f}",
        f"Test loss: {test_loss:.4f}",
        f"Training time (seconds): {training_time:.2f}",
        f"Test samples: {len(y_true)}",
    ]
    metadata = {
        'training_config': {
            'epochs': training_config['epochs'],
            'batch_size': training_config['batch_size'],
            'learning_rate': training_config['learning_rate'],
            'loss_function': training_config['loss'],
            'device': training_config['device']
        },
        'model_config': dict(model_config),
        'data_config': {
            'years': data_config['years'],
            'cat_col': data_config['cat_col'],
            'category_filter': category_filter,  # Record which category was used (None means all)
            'feature_cols': data_config['feature_cols'],
            'target_col': data_config['target_col'],
            'daily_aggregation': True,  # Flag indicating daily aggregation was used
            'scaling': {
                'method': 'StandardScaler',
                'scaler_mean': float(scaler.mean_[0]),
                'scaler_scale': float(scaler.scale_[0])
            }
        },
        'window_config': dict(window_config),
        'training_results': {
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': trainer.best_val_loss,
            'test_loss': float(test_loss),
            'training_time_seconds': training_time,
            **training_results  # Include quantile coverage or traditional metrics
        },
        'log_summary': "\n".join(log_summary_lines),
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  - Metadata saved to: {metadata_path}")
    
    # Generate prediction plot
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    output_dir = config.output['output_dir']
    
    # Save test predictions to CSV for detailed analysis (including dates if available)
    print("\n[9.5/9] Saving test predictions to CSV...")
    predictions_csv_path = os.path.join(output_dir, 'test_predictions.csv')
    
    # Get test dates from the original filtered_data
    # Calculate test start index: train_data + val_data
    test_start_idx = len(train_data) + len(val_data)
    test_end_idx = test_start_idx + len(test_dataset)
    test_data_subset = filtered_data.iloc[test_start_idx:test_end_idx]
    
    # Flatten predictions for CSV export (each row = one prediction)
    y_true_flat = y_true_original.reshape(-1) if y_true_original.ndim > 1 else y_true_original
    y_pred_flat = y_pred_original.reshape(-1) if y_pred_original.ndim > 1 else y_pred_original
    
    # Create prediction dataframe
    # Note: For multi-step predictions, we need to handle the horizon
    if y_true_original.ndim == 2:  # Multi-step: (n_windows, horizon)
        n_windows, horizon = y_true_original.shape
        pred_records = []
        
        for i in range(n_windows):
            # Get the date of the last input day for this window
            if i < len(test_data_subset):
                window_start_date = test_data_subset.iloc[i][time_col]
                # Each prediction in the horizon is for the next 1, 2, ..., horizon days
                for h in range(horizon):
                    pred_date = window_start_date + pd.Timedelta(days=h+1)
                    pred_records.append({
                        'window_idx': i,
                        'horizon_step': h + 1,
                        'date': pred_date,
                        'day_of_month': pred_date.day,
                        'actual': y_true_original[i, h],
                        'predicted': y_pred_original[i, h],
                        'error': y_pred_original[i, h] - y_true_original[i, h],
                        'abs_error': abs(y_pred_original[i, h] - y_true_original[i, h]),
                    })
        
        predictions_df = pd.DataFrame(pred_records)
    else:  # Single-step: (n_windows,)
        pred_records = []
        for i in range(len(y_true_original)):
            if i < len(test_data_subset):
                pred_date = test_data_subset.iloc[i][time_col]
                pred_records.append({
                    'date': pred_date,
                    'day_of_month': pred_date.day,
                    'actual': y_true_original[i],
                    'predicted': y_pred_original[i],
                    'error': y_pred_original[i] - y_true_original[i],
                    'abs_error': abs(y_pred_original[i] - y_true_original[i]),
                })
        
        predictions_df = pd.DataFrame(pred_records)
    
    # Save to CSV
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"  - Test predictions saved to: {predictions_csv_path}")
    print(f"  - Total prediction records: {len(predictions_df)}")
    
    plot_path = os.path.join(output_dir, "test_predictions.png")
    
    # Use a reasonable number of samples for plotting
    n_samples = min(100, len(y_true_original))
    
    # DEBUG: Verify values being passed to plot_difference
    y_true_plot = y_true_original[:n_samples]
    y_pred_plot = y_pred_original[:n_samples]
    print(f"  - DEBUG: Values being passed to plot_difference:")
    print(f"    - y_true_plot shape: {y_true_plot.shape}")
    print(f"    - y_true_plot min: {y_true_plot.min():.4f}, max: {y_true_plot.max():.4f}")
    print(f"    - y_true_plot first 5: {y_true_plot[:5]}")
    print(f"    - y_true_plot non-zero: {np.sum(y_true_plot != 0)} / {len(y_true_plot)}")
    
    plot_difference(
        y_true_plot,
        y_pred_plot,
        save_path=plot_path,
        show=False
    )
    print(f"  - Prediction plot saved to: {plot_path}")
    
    result = {
        'output_dir': output_dir,
        'model_dir': save_dir,
        'plot_path': plot_path,
        'training_time': training_time,
        'test_loss': test_loss,
        'category_filter': category_filter
    }
    
    return result


def main():
    """
    Main execution function for Category-Specific Independent Training Pipeline.
    
    The system now operates exclusively in Category-Specific Mode, where each category
    (DRY, TET, FRESH, etc.) is treated as a standalone task with its own unique
    architectural and training parameters loaded from category-specific config files.
    """
    print("=" * 80)
    print("MDLZ FORECASTING: Category-Specific Independent Training Pipeline")
    print("=" * 80)
    
    # Load base configuration
    print("\n[1/8] Loading base configuration...")
    base_config = load_config()
    
    # Override configuration for training
    print("\n[2/8] Applying training overrides...")
    # Only force spike_aware_mse if loss is not explicitly set to "quantile"
    # This allows category-specific configs to use quantile loss (e.g., MOONCAKE)
    if base_config.training.get('loss', 'spike_aware_mse') != 'quantile':
        base_config.set('training.loss', 'spike_aware_mse')  # Force spike_aware_mse loss for other categories
    base_config.set('output.save_model', True)  # Ensure model saving is enabled
    
    print(f"  - Data years: {base_config.data['years']}")
    print(f"  - Loss function: {base_config.training.get('loss', 'spike_aware_mse')}")
    
    # Load data
    print("\n[3/8] Loading data...")
    data_config = base_config.data
    data_reader = DataReader(
        data_dir=data_config['data_dir'],
        file_pattern=data_config['file_pattern']
    )
    
    # Load data
    try:
        data = data_reader.load(years=data_config['years'])
    except FileNotFoundError:
        print("[WARNING] Combined year files not found, trying pattern-based loading...")
        file_prefix = data_config.get('file_prefix', 'Outboundreports')
        data = data_reader.load_by_file_pattern(
            years=data_config['years'],
            file_prefix=file_prefix
        )
    
    print(f"  - Loaded {len(data)} samples before filtering")
    
    # Fix DtypeWarning: Cast columns (0, 4) to string/category to resolve mixed types
    if len(data.columns) > 0:
        col_0 = data.columns[0]
        if col_0 in data.columns:
            data[col_0] = data[col_0].astype(str)
    if len(data.columns) > 4:
        col_4 = data.columns[4]
        if col_4 in data.columns:
            data[col_4] = data[col_4].astype(str)
    print("  - Fixed DtypeWarning by casting columns 0 and 4 to string")
    
    # Get available categories
    cat_col = data_config['cat_col']
    if cat_col not in data.columns:
        raise ValueError(f"Category column '{cat_col}' not found in data. Available columns: {list(data.columns)}")
    
    available_categories = sorted(data[cat_col].unique().tolist())
    print(f"  - Available categories in data: {available_categories}")
    
    # Get categories to train from config (major_categories or all available)
    major_categories = data_config.get("major_categories", [])
    if major_categories:
        # Only train on major categories that exist in data
        categories_to_train = [cat for cat in major_categories if cat in available_categories]
        print(f"  - Training on major_categories: {categories_to_train}")
        if not categories_to_train:
            raise ValueError(
                f"None of the specified major_categories {major_categories} found in data. "
                f"Available: {available_categories}"
            )
    else:
        # Train on all available categories
        categories_to_train = available_categories
        print(f"  - Training on all available categories: {categories_to_train}")
    
    print(f"\n[SUMMARY] Will train {len(categories_to_train)} independent model(s):")
    for i, cat in enumerate(categories_to_train, 1):
        print(f"  {i}. {cat} -> outputs/{cat}/")
    
    # Execute training tasks - each category is independent
    results = []
    for task_idx, category in enumerate(categories_to_train, 1):
        print(f"\n{'=' * 80}")
        print(f"TASK {task_idx}/{len(categories_to_train)}: {category}")
        print(f"{'=' * 80}")
        
        try:
            # Load category-specific config (merges with base config)
            print(f"\n[Loading category-specific config for {category}...]")
            category_config = load_config(category=category)
            
            # Only override training loss if not explicitly set to "quantile"
            # This allows MOONCAKE to use quantile loss while other categories use spike_aware_mse
            if category_config.training.get('loss', 'spike_aware_mse') != 'quantile':
                category_config.set('training.loss', 'spike_aware_mse')
            
            # Create isolated output directory for this category
            category_output_dir = os.path.join("outputs", category)
            category_models_dir = os.path.join(category_output_dir, "models")
            os.makedirs(category_output_dir, exist_ok=True)
            os.makedirs(category_models_dir, exist_ok=True)
            category_config.set('output.output_dir', category_output_dir)
            category_config.set('output.model_dir', category_models_dir)
            
            # Train model for this category
            result = train_single_model(
                data, 
                category_config, 
                category_filter=category, 
                output_suffix=""  # No suffix needed, directory is category-specific
            )
            results.append(result)
            
            print(f"\n[SUCCESS] Task {task_idx} ({category}) completed successfully")
            print(f"  - Results saved to: {result['output_dir']}")
            print(f"  - Training time: {result['training_time']:.2f} seconds ({result['training_time']/60:.2f} minutes)")
        except Exception as e:
            print(f"\n[FAILED] Task {task_idx} ({category}) failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue with other categories even if one fails
            print(f"  - Continuing with remaining categories...")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("CATEGORY-SPECIFIC TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total tasks completed: {len(results)}/{len(categories_to_train)}")
    for i, result in enumerate(results, 1):
        cat_name = result['category_filter']
        print(f"\n{i}. {cat_name}:")
        print(f"   - Output directory: {result['output_dir']}")
        print(f"   - Model checkpoint: {os.path.join(result['model_dir'], 'best_model.pth')}")
        print(f"   - Scaler: {os.path.join(result['model_dir'], 'scaler.pkl')}")
        print(f"   - Metadata: {os.path.join(result['model_dir'], 'metadata.json')}")
        print(f"   - Test predictions plot: {result['plot_path']}")
        print(f"   - Test loss: {result['test_loss']:.4f}")
        print(f"   - Training time: {result['training_time']:.2f} seconds ({result['training_time']/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()

