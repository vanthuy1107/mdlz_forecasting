"""Data preparation utilities for prediction.

This module handles preparing data for prediction, ensuring consistent
category ID mappings and feature engineering that matches training time.
"""
import pandas as pd
import numpy as np

from src.data import (
    add_temporal_features,
    aggregate_daily,
    add_cbm_density_features,
    add_year_over_year_volume_features,
    apply_scaling,
)
from src.data.preprocessing import (
    add_day_of_week_cyclical_features,
    add_eom_features,
    add_weekday_volume_tier_features,
    add_is_monday_feature,
    apply_sunday_to_monday_carryover,
    add_operational_status_flags,
    add_seasonal_active_window_features,
)


def prepare_prediction_data(
    data: pd.DataFrame,
    config,
    cat2id: dict,
    scaler=None,
    trained_cat2id: dict = None,
    current_category: str = None,
    historical_data: pd.DataFrame = None
):
    """
    Prepare data for prediction using the same preprocessing as training.
    
    CRITICAL: This function ensures consistent category ID mapping by:
    1. Always using trained_cat2id when available (from model metadata)
    2. For category-specific models (num_categories=1), ensuring cat_id=0
    3. Filtering data to the current category BEFORE feature engineering
       to ensure rolling features are computed correctly per category
    
    CRITICAL for MOONCAKE: historical_data must be provided to calculate year-over-year
    features (cbm_last_year, cbm_2_years_ago). Without historical data, these features
    will be 0.0 for all prediction dates, preventing the model from learning from
    same period last year patterns.
    
    Args:
        data: DataFrame with raw data (should already be filtered to current_category if provided)
        config: Configuration object
        cat2id: Category to ID mapping from prediction data (fallback only)
        scaler: Optional StandardScaler for target values (if None, no scaling applied)
        trained_cat2id: Training-time category mapping (PRIORITY - always use if available)
        current_category: Current category being processed (for logging)
        historical_data: Optional DataFrame with historical data (required for MOONCAKE YoY features)
    
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
    data = _add_weekend_features(
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
    
    # Add Is_Monday feature to help model learn Monday peak patterns
    print("  - Adding Is_Monday feature...")
    data = add_is_monday_feature(
        data,
        time_col=time_col,
        is_monday_col="Is_Monday"
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
    data = _add_lunar_calendar_features(
        data,
        time_col=time_col,
        lunar_month_col="lunar_month",
        lunar_day_col="lunar_day"
    )

    # Lunar cyclical encodings (sine/cosine) to mirror training-time features
    print("  - Adding lunar cyclical features (sine/cosine)...")
    data = _add_lunar_cyclical_features(
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
    data = _add_holiday_features_vietnam(
        data,
        time_col=time_col,
        holiday_indicator_col="holiday_indicator",
        days_until_holiday_col="days_until_next_holiday",
        days_since_holiday_col="days_since_holiday"
    )
    
    # Feature engineering: continuous countdown to Tet (lunar event)
    print("  - Adding Tet countdown feature (days_to_tet)...")
    data = _add_days_to_tet_feature(
        data,
        time_col=time_col,
        days_to_tet_col="days_to_tet",
    )
    
    # Feature engineering: Days-to-Mid-Autumn countdown for MOONCAKE category
    # This recognizes that peak occurs 30-45 days before the 15th of Lunar Month 8
    if current_category == "MOONCAKE":
        print("  - Adding Days-to-Mid-Autumn countdown feature (days_to_mid_autumn)...")
        data = _add_days_to_mid_autumn_feature(
            data,
            time_col=time_col,
            lunar_month_col="lunar_month",
            lunar_day_col="lunar_day",
            days_to_mid_autumn_col="days_to_mid_autumn",
        )
    else:
        # For non-MOONCAKE categories, create the column with default value
        if "days_to_mid_autumn" not in data.columns:
            data["days_to_mid_autumn"] = 365  # Default: far from Mid-Autumn Festival
    
    # Feature engineering: Seasonal active-window masking for seasonal categories
    print("  - Adding seasonal active-window features (is_active_season, days_until_peak, is_golden_window)...")
    data = add_seasonal_active_window_features(
        data,
        time_col=time_col,
        cat_col=cat_col,
        lunar_month_col="lunar_month",
        days_to_tet_col="days_to_tet",
        is_active_season_col="is_active_season",
        days_until_peak_col="days_until_peak",
        is_golden_window_col="is_golden_window",
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
    
    # CRITICAL: Add year-over-year volume features for seasonal products (especially MOONCAKE)
    # For highly seasonal products, same period from previous years is more predictive than recent days
    if current_category == "MOONCAKE":
        print("  - Adding year-over-year volume features for MOONCAKE (cbm_last_year, cbm_2_years_ago)...")
        print("    - MOONCAKE is highly seasonal - same period from previous years is more predictive")
        
        # CRITICAL FIX: Combine historical data with prediction data before calculating YoY features
        # The add_year_over_year_volume_features function needs historical data in the same DataFrame
        # to find matching dates from previous years. Without this, cbm_last_year and cbm_2_years_ago
        # will be 0.0 for all prediction dates, preventing the model from learning historical patterns.
        if historical_data is not None and len(historical_data) > 0:
            print("    - Combining historical data with prediction data for YoY feature calculation...")
            # Historical data should already be aggregated (from prepare_prediction_data call)
            # But ensure it has the required columns and is in the right format
            historical_prepared = historical_data.copy()
            
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(historical_prepared[time_col]):
                historical_prepared[time_col] = pd.to_datetime(historical_prepared[time_col], dayfirst=True, errors='coerce')
            
            # Ensure we have the target column (should already be there from aggregation)
            if data_config['target_col'] not in historical_prepared.columns:
                print(f"    - WARNING: Historical data missing target column {data_config['target_col']}")
                historical_prepared = None
            
            if historical_prepared is not None:
                # Ensure category column exists
                if cat_col not in historical_prepared.columns:
                    # If historical data was filtered to single category, add it back
                    if current_category:
                        historical_prepared[cat_col] = current_category
                
                # Combine historical + prediction data for YoY calculation
                # Only keep essential columns to avoid merge issues
                essential_cols = [time_col, cat_col, data_config['target_col']]
                historical_for_merge = historical_prepared[essential_cols].copy()
                data_for_merge = data[essential_cols].copy()
                
                combined_for_yoy = pd.concat([historical_for_merge, data_for_merge], ignore_index=True)
                combined_for_yoy = combined_for_yoy.sort_values(time_col).reset_index(drop=True)
                
                # Calculate YoY features on combined data
                combined_with_yoy = add_year_over_year_volume_features(
                    combined_for_yoy,
                    target_col=data_config['target_col'],
                    time_col=time_col,
                    cat_col=cat_col,
                    yoy_1y_col="cbm_last_year",
                    yoy_2y_col="cbm_2_years_ago",
                )
                
                # Extract only prediction data rows (with YoY features populated)
                # Filter to dates that are in the original prediction data
                pred_dates = pd.to_datetime(data[time_col]).dt.date
                pred_mask = pd.to_datetime(combined_with_yoy[time_col]).dt.date.isin(pred_dates)
                
                # Get YoY features for prediction dates
                yoy_features = combined_with_yoy.loc[pred_mask, [time_col, cat_col, 'cbm_last_year', 'cbm_2_years_ago']].copy()
                
                # Merge YoY features back to original data (preserving all other columns)
                # Merge on date and category to ensure correct alignment
                data = data.merge(
                    yoy_features[[time_col, cat_col, 'cbm_last_year', 'cbm_2_years_ago']],
                    on=[time_col, cat_col],
                    how='left',
                    suffixes=('', '_yoy')
                )
                
                # If merge created duplicate columns, use the _yoy version
                if 'cbm_last_year_yoy' in data.columns:
                    data['cbm_last_year'] = data['cbm_last_year_yoy'].fillna(data.get('cbm_last_year', 0.0))
                    data = data.drop(columns=['cbm_last_year_yoy'])
                if 'cbm_2_years_ago_yoy' in data.columns:
                    data['cbm_2_years_ago'] = data['cbm_2_years_ago_yoy'].fillna(data.get('cbm_2_years_ago', 0.0))
                    data = data.drop(columns=['cbm_2_years_ago_yoy'])
                
                # Fill NaN with 0.0 if merge didn't find matches
                data['cbm_last_year'] = data['cbm_last_year'].fillna(0.0)
                data['cbm_2_years_ago'] = data['cbm_2_years_ago'].fillna(0.0)
                
                # Check if YoY features are populated
                non_zero_last_year = (data['cbm_last_year'] > 0).sum() if 'cbm_last_year' in data.columns else 0
                non_zero_2y = (data['cbm_2_years_ago'] > 0).sum() if 'cbm_2_years_ago' in data.columns else 0
                print(f"    - YoY features calculated: {non_zero_last_year}/{len(data)} dates have non-zero cbm_last_year")
                print(f"    - YoY features calculated: {non_zero_2y}/{len(data)} dates have non-zero cbm_2_years_ago")
                
                # Log sample values for debugging
                if len(data) > 0 and non_zero_last_year > 0:
                    sample = data[data['cbm_last_year'] > 0].iloc[0]
                    print(f"    - Sample: {sample[time_col]} -> cbm_last_year={sample['cbm_last_year']:.2f}, "
                          f"cbm_2_years_ago={sample.get('cbm_2_years_ago', 0):.2f}, current={sample[data_config['target_col']]:.2f}")
            else:
                print("    - WARNING: Could not prepare historical data for YoY calculation")
                data = add_year_over_year_volume_features(
                    data,
                    target_col=data_config['target_col'],
                    time_col=time_col,
                    cat_col=cat_col,
                    yoy_1y_col="cbm_last_year",
                    yoy_2y_col="cbm_2_years_ago",
                )
        else:
            print("    - WARNING: No historical data provided! YoY features will be 0.0 for all dates.")
            print("    - This will prevent the model from learning from same period last year patterns.")
            data = add_year_over_year_volume_features(
                data,
                target_col=data_config['target_col'],
                time_col=time_col,
                cat_col=cat_col,
                yoy_1y_col="cbm_last_year",
                yoy_2y_col="cbm_2_years_ago",
            )
        
        print(f"    - Added year-over-year features: cbm_last_year, cbm_2_years_ago")
    
    # Add rolling mean and momentum features (after aggregation)
    print("  - Adding rolling mean and momentum features (7d, 30d, momentum)...")
    data = _add_rolling_and_momentum_features(
        data,
        target_col=data_config['target_col'],
        time_col=time_col,
        cat_col=cat_col,
        rolling_7_col="rolling_mean_7d",
        rolling_30_col="rolling_mean_30d",
        momentum_col="momentum_3d_vs_14d"
    )
    
    # CRITICAL for MOONCAKE: Add trend deviation feature - compares recent trend vs year-over-year baseline
    # This allows the model to adjust the year-over-year baseline based on recent patterns
    if current_category == "MOONCAKE" and "cbm_last_year" in data.columns:
        print("  - Adding trend deviation feature for MOONCAKE (recent_trend_vs_yoy)...")
        print("    - Compares recent 21-day trend vs year-over-year baseline to detect deviations")
        
        # Calculate 21-day rolling mean (matches input_size window)
        data = data.sort_values([cat_col, time_col]).reset_index(drop=True)
        rolling_21d = data.groupby(cat_col)[data_config['target_col']].transform(
            lambda x: x.rolling(window=21, min_periods=1).mean()
        )
        data["rolling_mean_21d"] = rolling_21d
        
        # Calculate trend deviation: ratio of recent trend to year-over-year baseline
        cbm_last_year = data["cbm_last_year"].replace(0, np.nan)  # Avoid division by zero
        data["trend_vs_yoy_ratio"] = np.where(
            cbm_last_year > 0,
            rolling_21d / cbm_last_year,
            1.0  # If no last year data, assume no deviation
        )
        data["trend_vs_yoy_ratio"] = data["trend_vs_yoy_ratio"].fillna(1.0)
        
        # Also add absolute difference for cases where ratio might be misleading
        data["trend_vs_yoy_diff"] = rolling_21d - data["cbm_last_year"]
        
        print(f"    - Added trend deviation features: rolling_mean_21d, trend_vs_yoy_ratio, trend_vs_yoy_diff")
        print(f"    - Model will learn to adjust year-over-year baseline based on recent 21-day trend")
    
    # CRITICAL: Encode categories using training-time mapping
    # This ensures consistency regardless of which categories are processed together
    print("  - Encoding categories...")
    data = data.copy()
    
    # PRIORITY 1: Always use trained_cat2id if available (from model metadata)
    if trained_cat2id is not None:
        print(f"  - Using training-time category mapping: {trained_cat2id}")
        if current_category:
            print(f"  - Processing category: {current_category}")
        
        # Only keep categories that the model was trained on
        trained_categories = set(trained_cat2id.keys())
        data_before = len(data)
        data = data[data[cat_col].isin(trained_categories)].copy()
        data_after = len(data)
        if data_before > data_after:
            print(f"  - Filtered out {data_before - data_after} samples with categories not in training data")
        
        # CRITICAL FIX: For category-specific models (num_categories=1),
        # always use cat_id=0 regardless of what trained_cat2id says.
        # This ensures DRY always gets the same ID whether run alone or with FRESH.
        if len(trained_cat2id) == 1:
            # Category-specific model: always use cat_id = 0
            print(f"  - Category-specific model detected (num_categories=1)")
            print(f"  - Setting all category IDs to 0 for consistency")
            data[cat_id_col] = 0
        else:
            # Multi-category model: use training-time mapping
            data[cat_id_col] = data[cat_col].map(trained_cat2id)
    else:
        # FALLBACK: Use prediction-time mapping (should not happen in production)
        print(f"  [WARNING] No trained_cat2id available, using prediction-time mapping: {cat2id}")
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


# Helper functions (copied from mvp_predict.py for self-contained module)
def _add_weekend_features(
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


def _solar_to_lunar_date(solar_date) -> tuple:
    """Convert solar (Gregorian) date to lunar (Vietnamese) date approximation."""
    from datetime import date
    if solar_date.month == 1 and solar_date.day >= 20:
        lunar_month = 1
        lunar_day = solar_date.day - 19
    elif solar_date.month == 2:
        if solar_date.day <= 10:
            lunar_month = 1
            lunar_day = solar_date.day + 12
        else:
            lunar_month = 2
            lunar_day = solar_date.day - 10
    else:
        lunar_month = solar_date.month
        lunar_day = solar_date.day
    
    lunar_month = max(1, min(12, lunar_month))
    lunar_day = max(1, min(30, lunar_day))
    return lunar_month, lunar_day


def _add_lunar_calendar_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    lunar_month_col: str = "lunar_month",
    lunar_day_col: str = "lunar_day"
) -> pd.DataFrame:
    """Add lunar calendar features for Vietnamese holiday prediction."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    lunar_dates = df[time_col].dt.date.apply(_solar_to_lunar_date)
    df[lunar_month_col] = [ld[0] for ld in lunar_dates]
    df[lunar_day_col] = [ld[1] for ld in lunar_dates]
    return df


def _add_lunar_cyclical_features(
    df: pd.DataFrame,
    lunar_month_col: str = "lunar_month",
    lunar_day_col: str = "lunar_day",
    lunar_month_sin_col: str = "lunar_month_sin",
    lunar_month_cos_col: str = "lunar_month_cos",
    lunar_day_sin_col: str = "lunar_day_sin",
    lunar_day_cos_col: str = "lunar_day_cos",
) -> pd.DataFrame:
    """Add sine/cosine cyclical encodings for the lunar calendar."""
    df = df.copy()
    if lunar_month_col not in df.columns or lunar_day_col not in df.columns:
        raise ValueError("Lunar calendar columns not found.")
    df[lunar_month_sin_col] = np.sin(2 * np.pi * (df[lunar_month_col] - 1) / 12.0)
    df[lunar_month_cos_col] = np.cos(2 * np.pi * (df[lunar_month_col] - 1) / 12.0)
    df[lunar_day_sin_col] = np.sin(2 * np.pi * (df[lunar_day_col] - 1) / 30.0)
    df[lunar_day_cos_col] = np.cos(2 * np.pi * (df[lunar_day_col] - 1) / 30.0)
    return df


def _add_holiday_features_vietnam(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    holiday_indicator_col: str = "holiday_indicator",
    days_until_holiday_col: str = "days_until_next_holiday",
    days_since_holiday_col: str = "days_since_holiday"
) -> pd.DataFrame:
    """Add Vietnamese holiday-related features to DataFrame."""
    from config import load_holidays
    from datetime import timedelta
    
    VIETNAM_HOLIDAYS_BY_YEAR = load_holidays(holiday_type="model")
    
    def get_vietnam_holidays(start_date, end_date):
        """Get all Vietnamese holidays in the date range."""
        holidays = []
        for year in range(start_date.year, end_date.year + 2):
            if year in VIETNAM_HOLIDAYS_BY_YEAR:
                year_holidays = VIETNAM_HOLIDAYS_BY_YEAR[year]
                for holiday_type, dates in year_holidays.items():
                    if isinstance(dates, list):
                        holidays.extend(dates)
                    elif isinstance(dates, dict) and 'dates' in dates:
                        holidays.extend(dates['dates'])
        return sorted(list(set(holidays)))
    
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    
    valid_mask = df[time_col].notna()
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
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
                continue
        df.at[idx, days_until_holiday_col] = (next_holiday - current_date).days if next_holiday else 365
        
        last_holiday = None
        for holiday in sorted(holidays, reverse=True):
            try:
                if holiday <= current_date:
                    last_holiday = holiday
                    break
            except (TypeError, ValueError):
                continue
        df.at[idx, days_since_holiday_col] = (current_date - last_holiday).days if last_holiday else 365
    
    df[days_until_holiday_col] = df[days_until_holiday_col].fillna(365)
    df[days_since_holiday_col] = df[days_since_holiday_col].fillna(365)
    return df


def _add_days_to_tet_feature(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    days_to_tet_col: str = "days_to_tet",
) -> pd.DataFrame:
    """Add a continuous 'days_to_tet' feature based on the lunar Tet window."""
    from config import load_holidays
    from datetime import timedelta
    
    VIETNAM_HOLIDAYS_BY_YEAR = load_holidays(holiday_type="model")
    
    def get_tet_start_dates(start_year: int, end_year: int):
        tet_dates = []
        for year in range(start_year, end_year + 1):
            if year in VIETNAM_HOLIDAYS_BY_YEAR:
                tet_window = VIETNAM_HOLIDAYS_BY_YEAR[year].get("tet", [])
                if tet_window:
                    tet_dates.append(tet_window[0])
        return sorted(list(set(tet_dates)))
    
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    
    valid_mask = df[time_col].notna()
    df_valid = df[valid_mask].copy()
    
    df[days_to_tet_col] = 365
    
    if len(df_valid) == 0:
        return df
    
    min_date = df_valid[time_col].min().date()
    max_date = df_valid[time_col].max().date()
    extended_max = max_date + timedelta(days=365)
    tet_start_dates = get_tet_start_dates(min_date.year, extended_max.year)
    
    if not tet_start_dates:
        return df
    
    tet_start_dates = sorted(tet_start_dates)
    
    for idx, row in df_valid.iterrows():
        try:
            current_date = row[time_col].date()
        except (AttributeError, ValueError):
            continue
        
        next_tet = None
        for tet_date in tet_start_dates:
            try:
                if tet_date >= current_date:
                    next_tet = tet_date
                    break
            except (TypeError, ValueError):
                continue
        
        if next_tet is None:
            df.at[idx, days_to_tet_col] = 365
        else:
            try:
                df.at[idx, days_to_tet_col] = (next_tet - current_date).days
            except (TypeError, ValueError):
                df.at[idx, days_to_tet_col] = 365
    
    df[days_to_tet_col] = df[days_to_tet_col].fillna(365)
    return df


def _add_days_to_mid_autumn_feature(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    lunar_month_col: str = "lunar_month",
    lunar_day_col: str = "lunar_day",
    days_to_mid_autumn_col: str = "days_to_mid_autumn",
) -> pd.DataFrame:
    """Add a continuous 'days_to_mid_autumn' feature for MOONCAKE category."""
    from config import load_holidays
    from datetime import timedelta, date
    
    VIETNAM_HOLIDAYS_BY_YEAR = load_holidays(holiday_type="model")
    
    def get_mid_autumn_dates(start_year: int, end_year: int):
        mid_autumn_dates = []
        for year in range(start_year, end_year + 1):
            if year in VIETNAM_HOLIDAYS_BY_YEAR:
                mid_autumn_list = VIETNAM_HOLIDAYS_BY_YEAR[year].get("mid_autumn", [])
                if mid_autumn_list:
                    mid_autumn_dates.append(mid_autumn_list[0])
        return sorted(list(set(mid_autumn_dates)))
    
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    
    # Check if lunar columns exist
    if lunar_month_col not in df.columns or lunar_day_col not in df.columns:
        # If lunar columns don't exist, set default value
        df[days_to_mid_autumn_col] = 365
        return df
    
    valid_mask = df[time_col].notna()
    df_valid = df[valid_mask].copy()
    
    df[days_to_mid_autumn_col] = 365
    
    if len(df_valid) == 0:
        return df
    
    min_date = df_valid[time_col].min().date()
    max_date = df_valid[time_col].max().date()
    extended_max = max_date + timedelta(days=365)
    mid_autumn_dates = get_mid_autumn_dates(min_date.year, extended_max.year)
    
    for idx, row in df_valid.iterrows():
        try:
            current_date = row[time_col].date()
            lunar_month = int(row[lunar_month_col])
            lunar_day = int(row[lunar_day_col])
        except (AttributeError, ValueError, TypeError):
            continue
        
        # Calculate days until Mid-Autumn Festival (Lunar Month 8, Day 15)
        if lunar_month == 8:
            if lunar_day <= 15:
                days_to_peak = 15 - lunar_day
            else:
                # Past Day 15, find next year's Mid-Autumn Festival
                days_to_peak = (30 - lunar_day) + 15
                next_mid_autumn = None
                for ma_date in mid_autumn_dates:
                    if ma_date > current_date:
                        next_mid_autumn = ma_date
                        break
                if next_mid_autumn:
                    days_to_peak = (next_mid_autumn - current_date).days
                else:
                    days_to_peak = 365
        elif lunar_month < 8:
            # Before Lunar Month 8: calculate days until Day 15 of Month 8
            months_until = 8 - lunar_month
            days_until_month_8 = months_until * 30
            days_until_day_15 = days_until_month_8 + (15 - lunar_day)
            days_to_peak = days_until_day_15
        else:
            # After Lunar Month 8: find next year's Mid-Autumn Festival
            months_until_next = (12 - lunar_month) + 8
            days_until_month_8 = months_until_next * 30
            days_until_day_15 = days_until_month_8 + (15 - lunar_day)
            days_to_peak = days_until_day_15
            
            # Use actual Mid-Autumn Festival dates for accuracy
            next_mid_autumn = None
            for ma_date in mid_autumn_dates:
                if ma_date > current_date:
                    next_mid_autumn = ma_date
                    break
            if next_mid_autumn:
                days_to_peak = (next_mid_autumn - current_date).days
        
        try:
            df.at[idx, days_to_mid_autumn_col] = days_to_peak
        except (TypeError, ValueError):
            df.at[idx, days_to_mid_autumn_col] = 365
    
    df[days_to_mid_autumn_col] = df[days_to_mid_autumn_col].fillna(365)
    return df


def _add_rolling_and_momentum_features(
    df: pd.DataFrame,
    target_col: str = "Total CBM",
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
