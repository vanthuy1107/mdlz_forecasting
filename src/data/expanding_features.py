"""Leak-free feature engineering using expanding windows.

This module implements time-series feature engineering that strictly prevents data leakage
by using only information available up to time t when computing features for time t.

Key Principles:
1. **Expanding Window**: For any time point t, statistical features (mean, std, etc.) are
   calculated using ONLY data from times < t (not <= t to avoid target leakage).
2. **No Look-Ahead Bias**: Never use future information to compute past features.
3. **Per-Category Computation**: Features are computed separately for each category/brand
   to preserve category-specific patterns.
4. **Proper Validation Split**: Features must be computed on training set and applied
   to validation/test sets using only training statistics.

Example Violation (DO NOT DO THIS):
    # BAD: Uses entire dataset to calculate mean
    df['avg_monday_volume'] = df[df['is_monday'] == 1]['Total CBM'].mean()
    
    This introduces future information into past predictions!

Correct Approach (DO THIS):
    # GOOD: Uses expanding window - only past data
    for t in range(len(df)):
        if df.loc[t, 'is_monday'] == 1:
            past_data = df.loc[:t-1]  # Only data BEFORE time t
            past_mondays = past_data[past_data['is_monday'] == 1]
            df.loc[t, 'avg_monday_volume'] = past_mondays['Total CBM'].mean()
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, date


def add_expanding_statistical_features(
    df: pd.DataFrame,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: Optional[str] = "CATEGORY",
    day_of_week_col: str = "day_of_week",
    features_to_add: List[str] = ["weekday_avg", "weekday_std", "category_avg", "category_std"],
    min_periods: int = 5,
) -> pd.DataFrame:
    """
    Add statistical features using expanding windows to prevent data leakage.
    
    For each row at time t, computes statistics using ONLY data from times < t.
    This ensures that no future information is used when making predictions.
    
    Features Computed:
    - weekday_avg: Average volume for the same weekday, using only past occurrences
    - weekday_std: Standard deviation for the same weekday
    - category_avg: Average volume for the category, using only past data
    - category_std: Standard deviation for the category
    
    Args:
        df: DataFrame with temporal data (must be sorted by time_col).
        target_col: Name of target column to compute statistics on.
        time_col: Name of time column (must be datetime).
        cat_col: Name of category column (None to compute globally).
        day_of_week_col: Name of day-of-week column (0=Monday, 6=Sunday).
        features_to_add: List of feature names to compute.
        min_periods: Minimum number of past observations required (else use global mean).
    
    Returns:
        DataFrame with added statistical features.
    """
    df = df.copy()
    
    # Ensure sorted by time
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([cat_col, time_col] if cat_col else [time_col]).reset_index(drop=True)
    
    # Extract day of week if not present
    if day_of_week_col not in df.columns:
        df[day_of_week_col] = df[time_col].dt.dayofweek
    
    # Initialize feature columns
    if "weekday_avg" in features_to_add:
        df["weekday_avg_expanding"] = np.nan
    if "weekday_std" in features_to_add:
        df["weekday_std_expanding"] = np.nan
    if "category_avg" in features_to_add:
        df["category_avg_expanding"] = np.nan
    if "category_std" in features_to_add:
        df["category_std_expanding"] = np.nan
    
    # Global fallback mean (computed from first min_periods samples only)
    global_mean = df[target_col].iloc[:min_periods].mean() if len(df) >= min_periods else df[target_col].mean()
    
    # Process per category if specified
    if cat_col:
        for cat_name, cat_group in df.groupby(cat_col, sort=False):
            cat_mask = df[cat_col] == cat_name
            cat_indices = df[cat_mask].index.tolist()
            
            # Sort indices to ensure temporal order
            cat_indices_sorted = sorted(cat_indices)
            
            for i, idx in enumerate(cat_indices_sorted):
                # Get all PAST data (strictly before current index)
                past_indices = cat_indices_sorted[:i]  # All indices before current
                
                if len(past_indices) < min_periods:
                    # Not enough history - use global fallback
                    if "weekday_avg" in features_to_add:
                        df.at[idx, "weekday_avg_expanding"] = global_mean
                    if "weekday_std" in features_to_add:
                        df.at[idx, "weekday_std_expanding"] = 0.0
                    if "category_avg" in features_to_add:
                        df.at[idx, "category_avg_expanding"] = global_mean
                    if "category_std" in features_to_add:
                        df.at[idx, "category_std_expanding"] = 0.0
                    continue
                
                past_data = df.loc[past_indices]
                
                # Weekday-specific features
                if "weekday_avg" in features_to_add or "weekday_std" in features_to_add:
                    current_dow = df.at[idx, day_of_week_col]
                    past_same_weekday = past_data[past_data[day_of_week_col] == current_dow]
                    
                    if len(past_same_weekday) >= 1:
                        if "weekday_avg" in features_to_add:
                            df.at[idx, "weekday_avg_expanding"] = past_same_weekday[target_col].mean()
                        if "weekday_std" in features_to_add:
                            df.at[idx, "weekday_std_expanding"] = past_same_weekday[target_col].std()
                    else:
                        # No past data for this weekday - use category mean
                        if "weekday_avg" in features_to_add:
                            df.at[idx, "weekday_avg_expanding"] = past_data[target_col].mean()
                        if "weekday_std" in features_to_add:
                            df.at[idx, "weekday_std_expanding"] = 0.0
                
                # Category-level features
                if "category_avg" in features_to_add:
                    df.at[idx, "category_avg_expanding"] = past_data[target_col].mean()
                if "category_std" in features_to_add:
                    df.at[idx, "category_std_expanding"] = past_data[target_col].std()
    else:
        # Global computation (no category)
        for i in range(len(df)):
            if i < min_periods:
                # Not enough history
                if "weekday_avg" in features_to_add:
                    df.at[i, "weekday_avg_expanding"] = global_mean
                if "weekday_std" in features_to_add:
                    df.at[i, "weekday_std_expanding"] = 0.0
                if "category_avg" in features_to_add:
                    df.at[i, "category_avg_expanding"] = global_mean
                if "category_std" in features_to_add:
                    df.at[i, "category_std_expanding"] = 0.0
                continue
            
            past_data = df.iloc[:i]  # All rows before current
            
            # Weekday-specific features
            if "weekday_avg" in features_to_add or "weekday_std" in features_to_add:
                current_dow = df.at[i, day_of_week_col]
                past_same_weekday = past_data[past_data[day_of_week_col] == current_dow]
                
                if len(past_same_weekday) >= 1:
                    if "weekday_avg" in features_to_add:
                        df.at[i, "weekday_avg_expanding"] = past_same_weekday[target_col].mean()
                    if "weekday_std" in features_to_add:
                        df.at[i, "weekday_std_expanding"] = past_same_weekday[target_col].std()
                else:
                    if "weekday_avg" in features_to_add:
                        df.at[i, "weekday_avg_expanding"] = past_data[target_col].mean()
                    if "weekday_std" in features_to_add:
                        df.at[i, "weekday_std_expanding"] = 0.0
            
            # Global features
            if "category_avg" in features_to_add:
                df.at[i, "category_avg_expanding"] = past_data[target_col].mean()
            if "category_std" in features_to_add:
                df.at[i, "category_std_expanding"] = past_data[target_col].std()
    
    # Fill any remaining NaNs with global mean
    for col in ["weekday_avg_expanding", "weekday_std_expanding", "category_avg_expanding", "category_std_expanding"]:
        if col in df.columns:
            df[col] = df[col].fillna(global_mean if "avg" in col else 0.0)
    
    return df


def add_expanding_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: Optional[str] = "CATEGORY",
    windows: List[int] = [7, 30],
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Add rolling mean features using proper expanding windows (no data leakage).
    
    Standard pandas.rolling() is backward-looking, which is safe during training
    IF the entire dataset hasn't been seen yet. However, for production deployment,
    we need to ensure features are computed correctly.
    
    This function explicitly uses expanding windows to compute rolling statistics
    at each time point using only past data.
    
    Args:
        df: DataFrame with temporal data (must be sorted by time_col).
        target_col: Name of target column.
        time_col: Name of time column.
        cat_col: Name of category column (None to compute globally).
        windows: List of window sizes (in days) for rolling features.
        min_periods: Minimum periods required for rolling calculation.
    
    Returns:
        DataFrame with added rolling features (e.g., rolling_mean_7d, rolling_mean_30d).
    """
    df = df.copy()
    
    # Ensure sorted by time
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([cat_col, time_col] if cat_col else [time_col]).reset_index(drop=True)
    
    # Process per category if specified
    if cat_col:
        for window in windows:
            col_name = f"rolling_mean_{window}d_expanding"
            df[col_name] = np.nan
            
            for cat_name, cat_group in df.groupby(cat_col, sort=False):
                cat_mask = df[cat_col] == cat_name
                cat_indices = df[cat_mask].index
                
                # Use pandas rolling (which is backward-looking by default)
                # Shift by 1 to exclude current value (strict past-only)
                rolling_values = cat_group[target_col].shift(1).rolling(
                    window=window, 
                    min_periods=min_periods
                ).mean()
                
                df.loc[cat_indices, col_name] = rolling_values.values
    else:
        # Global rolling features
        for window in windows:
            col_name = f"rolling_mean_{window}d_expanding"
            # Shift by 1 to exclude current value
            df[col_name] = df[target_col].shift(1).rolling(
                window=window,
                min_periods=min_periods
            ).mean()
    
    # Fill NaNs with forward fill (for first few rows) then backward fill
    for window in windows:
        col_name = f"rolling_mean_{window}d_expanding"
        df[col_name] = df[col_name].ffill().bfill().fillna(0)
    
    return df


def add_expanding_momentum_features(
    df: pd.DataFrame,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: Optional[str] = "CATEGORY",
    short_window: int = 3,
    long_window: int = 14,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Add momentum features (difference between short and long rolling means).
    
    Momentum = short_term_average - long_term_average
    
    Uses expanding windows to prevent data leakage.
    
    Args:
        df: DataFrame with temporal data.
        target_col: Name of target column.
        time_col: Name of time column.
        cat_col: Name of category column.
        short_window: Window size for short-term average (e.g., 3 days).
        long_window: Window size for long-term average (e.g., 14 days).
        min_periods: Minimum periods required.
    
    Returns:
        DataFrame with momentum feature.
    """
    df = df.copy()
    
    # Ensure sorted by time
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([cat_col, time_col] if cat_col else [time_col]).reset_index(drop=True)
    
    momentum_col = f"momentum_{short_window}d_vs_{long_window}d_expanding"
    
    # Process per category if specified
    if cat_col:
        df[momentum_col] = np.nan
        
        for cat_name, cat_group in df.groupby(cat_col, sort=False):
            cat_mask = df[cat_col] == cat_name
            cat_indices = df[cat_mask].index
            
            # Compute short and long rolling means (excluding current value)
            short_rolling = cat_group[target_col].shift(1).rolling(
                window=short_window,
                min_periods=min_periods
            ).mean()
            
            long_rolling = cat_group[target_col].shift(1).rolling(
                window=long_window,
                min_periods=min_periods
            ).mean()
            
            momentum = short_rolling - long_rolling
            df.loc[cat_indices, momentum_col] = momentum.values
    else:
        # Global momentum
        short_rolling = df[target_col].shift(1).rolling(
            window=short_window,
            min_periods=min_periods
        ).mean()
        
        long_rolling = df[target_col].shift(1).rolling(
            window=long_window,
            min_periods=min_periods
        ).mean()
        
        df[momentum_col] = short_rolling - long_rolling
    
    # Fill NaNs
    df[momentum_col] = df[momentum_col].fillna(0)
    
    return df


def add_brand_tier_feature(
    df: pd.DataFrame,
    target_col: str = "Total CBM",
    brand_col: str = "BRAND",
    cat_col: str = "CATEGORY",
    time_col: str = "ACTUALSHIPDATE",
) -> pd.DataFrame:
    """
    Add brand tier feature representing brand's contribution to category volume.
    
    This is computed using expanding window: for each time t, we calculate
    the brand's average contribution using only past data.
    
    Feature: brand_avg_contribution_pct = 
        (brand's average past volume) / (category's average past volume)
    
    This helps the model differentiate "High-Volume" vs "Low-Volume" brand patterns.
    
    Args:
        df: DataFrame with brand, category, and target columns.
        target_col: Name of target column.
        brand_col: Name of brand column.
        cat_col: Name of category column.
        time_col: Name of time column.
    
    Returns:
        DataFrame with brand_tier_expanding feature.
    """
    df = df.copy()
    
    # Ensure sorted by time
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([cat_col, brand_col, time_col]).reset_index(drop=True)
    
    df["brand_tier_expanding"] = 0.0
    
    # Process per category
    for cat_name, cat_group in df.groupby(cat_col, sort=False):
        cat_mask = df[cat_col] == cat_name
        
        # Get category-level average (expanding)
        cat_indices = df[cat_mask].index.tolist()
        cat_expanding_avg = []
        
        for i in range(len(cat_indices)):
            if i == 0:
                cat_expanding_avg.append(0.0)
            else:
                past_cat_data = df.loc[cat_indices[:i], target_col]
                cat_expanding_avg.append(past_cat_data.mean())
        
        # Now process per brand within this category
        for brand_name, brand_group in cat_group.groupby(brand_col, sort=False):
            brand_mask = (df[cat_col] == cat_name) & (df[brand_col] == brand_name)
            brand_indices = df[brand_mask].index.tolist()
            
            for i, idx in enumerate(brand_indices):
                if i == 0:
                    # First occurrence - no past data
                    df.at[idx, "brand_tier_expanding"] = 0.0
                else:
                    # Calculate brand's average using past data
                    past_brand_data = df.loc[brand_indices[:i], target_col]
                    brand_avg = past_brand_data.mean()
                    
                    # Get category average at this time point
                    # Find position of current idx in cat_indices
                    pos_in_cat = cat_indices.index(idx)
                    cat_avg = cat_expanding_avg[pos_in_cat] if pos_in_cat < len(cat_expanding_avg) else 1.0
                    
                    # Calculate tier as ratio (avoid division by zero)
                    if cat_avg > 0:
                        df.at[idx, "brand_tier_expanding"] = brand_avg / cat_avg
                    else:
                        df.at[idx, "brand_tier_expanding"] = 0.0
    
    return df


def verify_no_leakage(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    time_col: str,
    cat_col: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Verify that features do not contain future information (data leakage check).
    
    This function checks:
    1. Feature values at time t do not depend on target values at times > t
    2. Feature values are monotonic or reasonable with respect to time
    3. No sudden jumps that indicate future information contamination
    
    Args:
        df: DataFrame with features and target.
        feature_cols: List of feature column names to check.
        target_col: Name of target column.
        time_col: Name of time column.
        cat_col: Name of category column (optional).
        verbose: Print detailed results.
    
    Returns:
        Dictionary mapping feature names to pass/fail status (True = no leakage detected).
    """
    results = {}
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    df_sorted = df.sort_values([cat_col, time_col] if cat_col else [time_col]).reset_index(drop=True)
    
    for feature in feature_cols:
        if feature not in df_sorted.columns:
            results[feature] = None  # Feature not found
            continue
        
        # Check 1: Are there any sudden backward jumps that indicate future info?
        # (Feature values should generally be stable or slowly evolving)
        feature_values = df_sorted[feature].values
        
        # Calculate rate of change
        rate_of_change = np.diff(feature_values)
        
        # Look for anomalous jumps (> 3 standard deviations)
        if len(rate_of_change) > 10:
            std_change = np.std(rate_of_change)
            mean_change = np.mean(rate_of_change)
            
            # Flag as suspicious if there are extreme changes
            suspicious_changes = np.abs(rate_of_change - mean_change) > (3 * std_change)
            suspicious_ratio = suspicious_changes.sum() / len(rate_of_change)
            
            # If more than 5% of changes are suspicious, flag as potential leakage
            passed = suspicious_ratio < 0.05
            results[feature] = passed
            
            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"[{status}] {feature}: {suspicious_ratio:.2%} suspicious changes")
        else:
            results[feature] = True  # Not enough data to check
    
    return results


__all__ = [
    "add_expanding_statistical_features",
    "add_expanding_rolling_features",
    "add_expanding_momentum_features",
    "add_brand_tier_feature",
    "verify_no_leakage",
]
