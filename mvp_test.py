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

from config import load_config
from src.data import (
    DataReader,
    ForecastDataset,
    slicing_window_category,
    encode_categories,
    split_data,
    add_temporal_features,
    aggregate_daily,
    fit_scaler,
    apply_scaling,
    inverse_transform_scaling
)
from src.models import RNNWithCategory
from src.training import Trainer
from src.utils import plot_difference, spike_aware_mse


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
    
    # Define Vietnamese holidays by year
    vietnam_holidays = {
        2023: {
            'tet': [date(2023, 1, 20), date(2023, 1, 21), date(2023, 1, 22), 
                   date(2023, 1, 23), date(2023, 1, 24), date(2023, 1, 25), date(2023, 1, 26)],
            'mid_autumn': [date(2023, 9, 29)],
            'independence': [date(2023, 9, 2)],
            'labor': [date(2023, 4, 30), date(2023, 5, 1)]
        },
        2024: {
            'tet': [date(2024, 2, 8), date(2024, 2, 9), date(2024, 2, 10),
                   date(2024, 2, 11), date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14)],
            'mid_autumn': [date(2024, 9, 17)],
            'independence': [date(2024, 9, 2)],
            'labor': [date(2024, 4, 30), date(2024, 5, 1)]
        },
        2025: {
            'tet': [date(2025, 1, 27), date(2025, 1, 28), date(2025, 1, 29),
                   date(2025, 1, 30), date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2)],
            'mid_autumn': [date(2025, 10, 6)],
            'independence': [date(2025, 9, 2)],
            'labor': [date(2025, 4, 30), date(2025, 5, 1)]
        }
    }
    
    # Collect all holidays in the date range
    current = start_date
    while current <= end_date:
        year = current.year
        if year in vietnam_holidays:
            year_holidays = vietnam_holidays[year]
            holidays.extend(year_holidays['tet'])
            holidays.extend(year_holidays['mid_autumn'])
            holidays.extend(year_holidays['independence'])
            holidays.extend(year_holidays['labor'])
        current = date(year + 1, 1, 1)
    
    # Filter to date range and remove duplicates
    holidays = [h for h in holidays if start_date <= h <= end_date]
    holidays = sorted(list(set(holidays)))
    
    return holidays


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


def train_single_model(data, config, category_filter=None, output_suffix=""):
    """
    Train a single model with optional category filtering.
    
    Args:
        data: Full DataFrame with all data
        config: Configuration object
        category_filter: Optional category name to filter (None means all categories)
        output_suffix: Suffix for output directories (e.g., "_DRY", "_all", etc.)
    
    Returns:
        Dictionary with training results and metadata
    """
    print("\n" + "=" * 80)
    if category_filter:
        print(f"TRAINING MODEL FOR CATEGORY: {category_filter}")
    else:
        print("TRAINING MODEL FOR ALL CATEGORIES")
    print("=" * 80)
    
    data_config = config.data
    cat_col = data_config['cat_col']
    time_col = data_config['time_col']
    
    # Create a copy of data for this training run
    filtered_data = data.copy()
    
    # Apply category filter if specified
    if category_filter:
        print(f"\n[4/8] Filtering data to category: {category_filter}...")
        samples_before = len(filtered_data)
        
        if cat_col not in filtered_data.columns:
            raise ValueError(f"Category column '{cat_col}' not found in data. Available columns: {list(filtered_data.columns)}")
        
        filtered_data = filtered_data[filtered_data[cat_col] == category_filter].copy()
        samples_after = len(filtered_data)
        
        print(f"  - Samples before filtering: {samples_before}")
        print(f"  - Samples after filtering (CATEGORY == '{category_filter}'): {samples_after}")
        
        if samples_after == 0:
            raise ValueError(f"No samples found with CATEGORY == '{category_filter}'. Please check your data.")
    else:
        print("\n[4/8] Using all categories (no filtering)...")
        print(f"  - Total samples: {len(filtered_data)}")
    
    # Update output directories with suffix
    base_output_dir = config.output.get('output_dir', 'outputs/mvp_test')
    base_models_dir = config.output.get('model_dir', os.path.join(base_output_dir, 'models'))
    
    if output_suffix:
        mvp_output_dir = f"{base_output_dir}{output_suffix}"
        mvp_models_dir = os.path.join(mvp_output_dir, "models")
    else:
        mvp_output_dir = base_output_dir
        mvp_models_dir = base_models_dir
    
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
    
    # Feature engineering: Add lunar calendar features
    print("  - Adding lunar calendar features (lunar_month, lunar_day)...")
    filtered_data = add_lunar_calendar_features(
        filtered_data,
        time_col=time_col,
        lunar_month_col="lunar_month",
        lunar_day_col="lunar_day"
    )
    
    # Feature engineering: Add Vietnamese holiday features (with days_since_holiday)
    print("  - Adding Vietnamese holiday features...")
    filtered_data = add_holiday_features_vietnam(
        filtered_data,
        time_col=time_col,
        holiday_indicator_col="holiday_indicator",
        days_until_holiday_col="days_until_next_holiday",
        days_since_holiday_col="days_since_holiday"
    )
    
    # Daily aggregation: Group by date and category, sum QTY
    # This ensures the model learns daily demand patterns, not individual transaction sizes
    print("\n[5.5/8] Aggregating to daily totals by category...")
    samples_before_agg = len(filtered_data)
    filtered_data = aggregate_daily(
        filtered_data,
        time_col=time_col,
        cat_col=cat_col,
        target_col=data_config['target_col']
    )
    samples_after_agg = len(filtered_data)
    print(f"  - Samples before aggregation: {samples_before_agg}")
    print(f"  - Samples after aggregation: {samples_after_agg} (one row per date per category)")
    
    # Feature engineering: Add rolling means and momentum features (after aggregation)
    print("  - Adding rolling mean and momentum features (7d, 30d, momentum)...")
    filtered_data = add_rolling_and_momentum_features(
        filtered_data,
        target_col=data_config['target_col'],
        time_col=time_col,
        cat_col=cat_col,
        rolling_7_col="rolling_mean_7d",
        rolling_30_col="rolling_mean_30d",
        momentum_col="momentum_3d_vs_14d"
    )
    
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
    
    # DEBUG: Check test data QTY values BEFORE scaling
    test_qty_before_scaling = test_data[data_config['target_col']].values
    print(f"\n  - DEBUG: Test data QTY BEFORE scaling:")
    print(f"    - Min: {test_qty_before_scaling.min():.4f}")
    print(f"    - Max: {test_qty_before_scaling.max():.4f}")
    print(f"    - Mean: {test_qty_before_scaling.mean():.4f}")
    print(f"    - Non-zero count: {np.sum(test_qty_before_scaling != 0)} / {len(test_qty_before_scaling)}")
    print(f"    - Zero count: {np.sum(test_qty_before_scaling == 0)} / {len(test_qty_before_scaling)}")
    
    # Fit scaler on training data and apply to all splits
    print(f"\n[6.5/8] Scaling {data_config['target_col']} values...")
    scaler = fit_scaler(train_data, target_col=data_config['target_col'])
    print(f"  - Scaler fitted on training data:")
    print(f"    Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")
    
    train_data = apply_scaling(train_data, scaler, target_col=data_config['target_col'])
    val_data = apply_scaling(val_data, scaler, target_col=data_config['target_col'])
    test_data = apply_scaling(test_data, scaler, target_col=data_config['target_col'])
    print("  - Scaling applied to train, validation, and test sets")
    
    # Create windows using slicing_window_category
    print("\n[7/8] Creating sliding windows...")
    window_config = config.window
    feature_cols = data_config['feature_cols']
    target_col = [data_config['target_col']]
    cat_col_list = [cat_id_col]
    time_col_list = [time_col]
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    
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
    
    # Create datasets
    train_dataset = ForecastDataset(X_train, y_train, cat_train)
    val_dataset = ForecastDataset(X_val, y_val, cat_val)
    test_dataset = ForecastDataset(X_test, y_test, cat_test)
    
    # Create data loaders
    training_config = config.training
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
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
    model_config = config.model
    model = RNNWithCategory(
        num_categories=num_categories,
        cat_emb_dim=model_config['cat_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        output_dim=model_config['output_dim']
    )
    print(f"  - Model: {model_config['name']}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build loss, optimizer, scheduler
    # Force spike_aware_mse loss function for Vietnamese market demand spikes
    print("  - Using spike_aware_mse loss (3x weight for top 20% QTY values)...")
    criterion = spike_aware_mse
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate']
    )
    
    scheduler_config = training_config.get('scheduler', {})
    scheduler_name = scheduler_config.get('name')
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 3),
            min_lr=scheduler_config.get('min_lr', 1e-5)
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
    
    train_losses, val_losses = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config['epochs'],
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
    
    # Save training metadata
    print("\n[9/9] Saving training metadata...")
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
            'training_time_seconds': training_time
        },
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  - Metadata saved to: {metadata_path}")
    
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
    
    # DEBUG: Check y_true values (should be in scaled space)
    print("  - DEBUG: Checking y_true values (scaled):")
    print(f"    - Min: {y_true.min():.4f}")
    print(f"    - Max: {y_true.max():.4f}")
    print(f"    - Mean: {y_true.mean():.4f}")
    print(f"    - Non-zero count: {np.sum(y_true != 0)} / {len(y_true)}")
    print(f"    - Zero count: {np.sum(y_true == 0)} / {len(y_true)}")
    
    # CRITICAL: Inverse transform predictions and true values back to original scale
    print("  - Inverse transforming predictions to original scale...")
    y_true_original = inverse_transform_scaling(y_true, scaler, target_col=data_config['target_col'])
    y_pred_original = inverse_transform_scaling(y_pred, scaler, target_col=data_config['target_col'])
    
    # DEBUG: Check y_true_original values (should be in original scale)
    print("  - DEBUG: Checking y_true_original values (after inverse transform):")
    print(f"    - Min: {y_true_original.min():.4f}")
    print(f"    - Max: {y_true_original.max():.4f}")
    print(f"    - Mean: {y_true_original.mean():.4f}")
    print(f"    - Non-zero count: {np.sum(y_true_original != 0)} / {len(y_true_original)}")
    print(f"    - Zero count: {np.sum(y_true_original == 0)} / {len(y_true_original)}")
    print(f"    - First 10 values: {y_true_original[:10]}")
    
    # Generate prediction plot
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    output_dir = config.output['output_dir']
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
    """Main execution function for MVP test."""
    print("=" * 80)
    print("MVP TEST: MDLZ Warehouse Prediction System")
    print("=" * 80)
    
    # Load default configuration
    print("\n[1/8] Loading configuration...")
    config = load_config()
    
    # Override configuration for MVP test
    print("\n[2/8] Applying MVP test overrides...")
    config.set('data.years', [2023, 2024])
    config.set('training.epochs', 20)  # Increased to 20 epochs for better learning
    config.set('training.loss', 'spike_aware_mse')  # Force spike_aware_mse loss
    
    # Set base output directory
    mvp_output_dir = "outputs/mvp_test"
    mvp_models_dir = os.path.join(mvp_output_dir, "models")
    os.makedirs(mvp_output_dir, exist_ok=True)
    os.makedirs(mvp_models_dir, exist_ok=True)
    config.set('output.output_dir', mvp_output_dir)
    config.set('output.model_dir', mvp_models_dir)
    config.set('output.save_model', True)  # Ensure model saving is enabled
    
    # Get category mode from config
    data_config = config.data
    category_mode = data_config.get('category_mode', 'single')  # Default to 'single' for backward compatibility
    category_filter = data_config.get('category_filter', 'DRY')  # Default to 'DRY' for backward compatibility
    
    print(f"  - Data years: {config.data['years']}")
    print(f"  - Training epochs: {config.training['epochs']}")
    print(f"  - Loss function: spike_aware_mse (forced)")
    print(f"  - Category mode: {category_mode}")
    if category_mode == 'single':
        print(f"  - Category filter: {category_filter}")
    print(f"  - Base output directory: {mvp_output_dir}")
    
    # Load data
    print("\n[3/8] Loading data...")
    data_reader = DataReader(
        data_dir=data_config['data_dir'],
        file_pattern=data_config['file_pattern']
    )
    
    # Load 2023 and 2024 data
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
    
    # Determine training tasks based on category_mode
    training_tasks = []
    
    if category_mode == 'all':
        # Train on all categories
        training_tasks.append((None, "_all"))
    elif category_mode == 'single':
        # Train on single category
        if category_filter not in available_categories:
            raise ValueError(f"Category '{category_filter}' not found in data. Available categories: {available_categories}")
        training_tasks.append((category_filter, f"_{category_filter}"))
    elif category_mode == 'both':
        # Train on all categories first
        training_tasks.append((None, "_all"))
        # Then train on each category separately
        for cat in available_categories:
            training_tasks.append((cat, f"_{cat}"))
    else:
        raise ValueError(f"Invalid category_mode: {category_mode}. Must be 'all', 'single', or 'both'")
    
    print(f"\n[SUMMARY] Will train {len(training_tasks)} model(s):")
    for i, (cat, suffix) in enumerate(training_tasks, 1):
        cat_name = cat if cat else "ALL CATEGORIES"
        print(f"  {i}. {cat_name} -> suffix: {suffix}")
    
    # Execute training tasks
    results = []
    for task_idx, (category_filter, suffix) in enumerate(training_tasks, 1):
        print(f"\n{'=' * 80}")
        print(f"TASK {task_idx}/{len(training_tasks)}")
        print(f"{'=' * 80}")
        
        try:
            result = train_single_model(data, config, category_filter=category_filter, output_suffix=suffix)
            results.append(result)
            
            print(f"\n✓ Task {task_idx} completed successfully")
            print(f"  - Results saved to: {result['output_dir']}")
            print(f"  - Training time: {result['training_time']:.2f} seconds ({result['training_time']/60:.2f} minutes)")
        except Exception as e:
            print(f"\n✗ Task {task_idx} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            if category_mode == 'single':
                # If single mode fails, we should raise
                raise
    
    # Print final summary
    print("\n" + "=" * 80)
    print("MVP TEST COMPLETE!")
    print("=" * 80)
    print(f"Total tasks completed: {len(results)}/{len(training_tasks)}")
    for i, result in enumerate(results, 1):
        cat_name = result['category_filter'] if result['category_filter'] else "ALL CATEGORIES"
        print(f"\n{i}. {cat_name}:")
        print(f"   - Output directory: {result['output_dir']}")
        print(f"   - Model checkpoint: {os.path.join(result['model_dir'], 'best_model.pth')}")
        print(f"   - Test predictions plot: {result['plot_path']}")
        print(f"   - Test loss: {result['test_loss']:.4f}")
        print(f"   - Training time: {result['training_time']:.2f} seconds ({result['training_time']/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()

