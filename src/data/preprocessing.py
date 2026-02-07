"""Data preprocessing and window slicing utilities."""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
import pickle
import sys
from pathlib import Path

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import load_holidays


def slicing_window(
    df: Union[pd.DataFrame, np.ndarray],
    df_start_idx: int,
    df_end_idx: Optional[int],
    input_size: int,
    label_name: str,
    label_size: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from time series data (legacy function).
    
    Args:
        df: DataFrame or array containing time series data.
        df_start_idx: Starting index for window creation.
        df_end_idx: Ending index for window creation (None = end of data).
        input_size: Size of input window.
        label_name: Name of label column (if DataFrame) or index (if array).
        label_size: Size of label window.
    
    Returns:
        Tuple of (features, labels) arrays.
    """
    features = []
    labels = []
    
    window_size = input_size + label_size
    
    if df_end_idx is None:
        df_end_idx = len(df) - window_size
    
    for idx in range(df_start_idx, df_end_idx):
        feature = df[idx : idx + input_size]
        label = df[label_name][idx + window_size - label_size : idx + window_size]
        
        features.append(feature)
        labels.append(label)
    
    features = np.expand_dims(np.array(features), -1)
    labels = np.array(labels)
    return features, labels


def slicing_window_multivariate(
    data: np.ndarray,
    target_col_idx: int,
    input_size: int,
    label_size: int = 1,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from multivariate time series data.
    
    Args:
        data: Array of shape (N, D) containing multivariate time series.
        target_col_idx: Index of target column in data.
        input_size: Size of input window.
        label_size: Size of label window.
        start_idx: Starting index for window creation.
        end_idx: Ending index for window creation (None = end of data).
    
    Returns:
        Tuple of (X, y) where:
            X: Array of shape (N_samples, input_size, D)
            y: Array of shape (N_samples, label_size)
    """
    data = np.asarray(data)
    N, D = data.shape
    
    window_size = input_size + label_size
    
    if end_idx is None:
        end_idx = N - window_size
    
    X, y = [], []
    
    for i in range(start_idx, end_idx):
        X.append(data[i : i + input_size, :])  # (T, D)
        y.append(
            data[i + input_size : i + input_size + label_size, target_col_idx]
        )
    
    X = np.array(X)  # (N_samples, T, D)
    y = np.array(y)  # (N_samples, label_size)
    
    return X, y


def slicing_window_category(
    df: pd.DataFrame,
    input_size: int,
    horizon: int,
    feature_cols: List[str],
    target_col: List[str],
    cat_col: List[str],
    time_col: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows from time series data grouped by category.
    
    Args:
        df: DataFrame containing time series data with categories.
        input_size: Size of input window.
        horizon: Prediction horizon (number of steps ahead to predict).
        feature_cols: List of feature column names.
        target_col: List of target column names.
        cat_col: List of category column names.
        time_col: List of time column names for sorting.
    
    Returns:
        Tuple of (X, y, cats) where:
            X: Array of shape (N_samples, input_size, n_features)
            y: Array of shape (N_samples, horizon)
            cats: Array of shape (N_samples,) containing category IDs
    """
    X, y, cats = [], [], []
    
    # Get category column name
    cat_col_name = cat_col[0] if isinstance(cat_col, list) else cat_col
    time_col_name = time_col[0] if isinstance(time_col, list) else time_col
    target_col_name = target_col[0] if isinstance(target_col, list) else target_col
    
    for cat, g in df.groupby(cat_col_name, sort=False):
        g = g.sort_values(time_col_name)
        
        X_data = g[feature_cols].values
        y_data = g[target_col_name].values
        
        for i in range(len(g) - input_size - horizon + 1):
            X.append(X_data[i:i+input_size])
            y.append(y_data[i+input_size:i+input_size+horizon])
            cats.append(cat)
    
    return np.array(X), np.array(y), np.array(cats)


def get_us_holidays(start_date: date, end_date: date) -> List[date]:
    """
    Get list of US holidays between start_date and end_date.
    
    Includes major US holidays that typically affect shipping patterns:
    - New Year's Day
    - Martin Luther King Jr. Day (3rd Monday of January)
    - Presidents' Day (3rd Monday of February)
    - Memorial Day (last Monday of May)
    - Independence Day
    - Labor Day (1st Monday of September)
    - Columbus Day (2nd Monday of October)
    - Veterans Day
    - Thanksgiving (4th Thursday of November)
    - Christmas Day
    
    Args:
        start_date: Start date for holiday range.
        end_date: End date for holiday range.
    
    Returns:
        List of holiday dates.
    """
    holidays = []
    current = start_date
    
    while current <= end_date:
        year = current.year
        
        # Fixed date holidays
        holidays.append(date(year, 1, 1))   # New Year's Day
        holidays.append(date(year, 7, 4))   # Independence Day
        holidays.append(date(year, 11, 11))  # Veterans Day
        holidays.append(date(year, 12, 25))  # Christmas Day
        
        # Helper function to get nth weekday of a month
        def get_nth_weekday(year, month, weekday, n):
            """Get nth occurrence of weekday in month (0=Monday, 6=Sunday)."""
            first_day = date(year, month, 1)
            days_until_weekday = (weekday - first_day.weekday()) % 7
            if days_until_weekday == 0 and first_day.weekday() != weekday:
                days_until_weekday = 7
            return first_day + timedelta(days=days_until_weekday + (n - 1) * 7)
        
        # MLK Day (3rd Monday of January)
        holidays.append(get_nth_weekday(year, 1, 0, 3))
        
        # Presidents' Day (3rd Monday of February)
        holidays.append(get_nth_weekday(year, 2, 0, 3))
        
        # Memorial Day (last Monday of May)
        last_day = date(year, 5, 31)
        days_since_monday = last_day.weekday()
        mem = last_day - timedelta(days=days_since_monday)
        holidays.append(mem)
        
        # Labor Day (1st Monday of September)
        holidays.append(get_nth_weekday(year, 9, 0, 1))
        
        # Columbus Day (2nd Monday of October)
        holidays.append(get_nth_weekday(year, 10, 0, 2))
        
        # Thanksgiving (4th Thursday of November)
        holidays.append(get_nth_weekday(year, 11, 3, 4))
        
        current = date(year + 1, 1, 1)
    
    # Filter to date range and remove duplicates
    holidays = [h for h in holidays if start_date <= h <= end_date]
    holidays = sorted(list(set(holidays)))
    
    return holidays


def get_vietnam_holidays(start_date: date, end_date: date) -> List[date]:
    """
    Get list of Vietnamese holidays between start_date and end_date.
    
    Includes the official Vietnam day‑off calendar for:
    - New Year
    - Tet (Lunar New Year)
    - Hung Kings / Reunification / Labor
    - Independence Day
    
    Args:
        start_date: Start date for holiday range.
        end_date: End date for holiday range.
    
    Returns:
        List of holiday dates.
    """
    holidays: List[date] = []
    
    # Canonical Vietnam holiday calendar (days off) aligned with business rules.
    # NOTE: These dates are now loaded from config/holidays.yaml for easier maintenance.
    vietnam_holidays = load_holidays(holiday_type="business")
    
    # Collect all holidays in the date range
    current = start_date
    while current <= end_date:
        year = current.year
        if year in vietnam_holidays:
            for dates in vietnam_holidays[year].values():
                holidays.extend(dates)
        # Jump to next year boundary
        current = date(year + 1, 1, 1)
    
    # Filter to date range and remove duplicates
    holidays = [h for h in holidays if start_date <= h <= end_date]
    holidays = sorted(list(set(holidays)))
    
    return holidays


def add_holiday_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    holiday_indicator_col: str = "holiday_indicator",
    days_until_holiday_col: str = "days_until_next_holiday"
) -> pd.DataFrame:
    """
    Add holiday-related features to DataFrame.
    
    Creates:
    - holiday_indicator: Binary (0 or 1) indicating if date is a holiday
    - days_until_next_holiday: Number of days until the next holiday
    
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
    
    # Get holidays
    holidays = get_us_holidays(min_date, extended_max)
    holiday_set = set(holidays)
    
    # Initialize columns
    df[holiday_indicator_col] = 0
    df[days_until_holiday_col] = np.nan
    
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
    
    # Fill any remaining NaN values (shouldn't happen, but safety check)
    df[days_until_holiday_col] = df[days_until_holiday_col].fillna(365)
    
    return df


def add_temporal_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    month_sin_col: str = "month_sin",
    month_cos_col: str = "month_cos",
    dayofmonth_sin_col: str = "dayofmonth_sin",
    dayofmonth_cos_col: str = "dayofmonth_cos"
) -> pd.DataFrame:
    """
    Add cyclical temporal features from datetime column.
    
    Creates:
    - month_sin: sin(2π × month / 12)
    - month_cos: cos(2π × month / 12)
    - dayofmonth_sin: sin(2π × day / 31)
    - dayofmonth_cos: cos(2π × day / 31)
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        month_sin_col: Name for month_sin column.
        month_cos_col: Name for month_cos column.
        dayofmonth_sin_col: Name for dayofmonth_sin column.
        dayofmonth_cos_col: Name for dayofmonth_cos column.
    
    Returns:
        DataFrame with added temporal features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract month (1-12) and day of month (1-31)
    month = df[time_col].dt.month
    dayofmonth = df[time_col].dt.day
    
    # Create cyclical encoding for month (0-11 for sin/cos, so subtract 1)
    # Normalize to [0, 1] range, then apply sin/cos
    df[month_sin_col] = np.sin(2 * np.pi * (month - 1) / 12)
    df[month_cos_col] = np.cos(2 * np.pi * (month - 1) / 12)
    
    # Create cyclical encoding for day of month (0-30, so subtract 1)
    # Normalize to [0, 1] range, then apply sin/cos
    df[dayofmonth_sin_col] = np.sin(2 * np.pi * (dayofmonth - 1) / 31)
    df[dayofmonth_cos_col] = np.cos(2 * np.pi * (dayofmonth - 1) / 31)
    
    return df


def add_day_of_week_cyclical_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    day_of_week_sin_col: str = "day_of_week_sin",
    day_of_week_cos_col: str = "day_of_week_cos"
) -> pd.DataFrame:
    """
    Add cyclical day-of-week features using sine/cosine transformations.
    
    This encoding ensures the model understands that Sunday (6) is adjacent to Monday (0),
    capturing the 7-day cyclicality of weekly demand patterns.
    
    Creates:
    - day_of_week_sin: sin(2π × day_of_week / 7)
    - day_of_week_cos: cos(2π × day_of_week / 7)
    
    Where day_of_week is 0=Monday, 1=Tuesday, ..., 6=Sunday.
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        day_of_week_sin_col: Name for day_of_week_sin column.
        day_of_week_cos_col: Name for day_of_week_cos column.
    
    Returns:
        DataFrame with added cyclical day-of-week features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract day of week (0=Monday, 6=Sunday)
    day_of_week = df[time_col].dt.dayofweek
    
    # Create cyclical encoding for day of week (0-6)
    # Normalize to [0, 1] range, then apply sin/cos
    # This ensures Sunday (6) and Monday (0) are close in the encoding space
    df[day_of_week_sin_col] = np.sin(2 * np.pi * day_of_week / 7)
    df[day_of_week_cos_col] = np.cos(2 * np.pi * day_of_week / 7)
    
    return df


def add_is_monday_feature(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    is_monday_col: str = "Is_Monday"
) -> pd.DataFrame:
    """
    Add binary Is_Monday feature to help model learn Monday peak patterns.
    
    For FRESH category, Monday is the consistent peak day (~208 CBM), so this
    binary flag helps the model explicitly identify and learn this pattern.
    
    Creates:
    - Is_Monday: Binary (1 for Monday, 0 otherwise)
    
    Where day_of_week is 0=Monday, 1=Tuesday, ..., 6=Sunday.
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        is_monday_col: Name for Is_Monday column.
    
    Returns:
        DataFrame with added Is_Monday feature.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract day of week (0=Monday, 6=Sunday)
    day_of_week = df[time_col].dt.dayofweek
    
    # Create binary Is_Monday feature (1 for Monday, 0 otherwise)
    df[is_monday_col] = (day_of_week == 0).astype(int)
    
    return df


def add_weekday_volume_tier_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    weekday_volume_tier_col: str = "weekday_volume_tier",
    is_high_volume_weekday_col: str = "is_high_volume_weekday",
    weekday_pattern: str = "default"
) -> pd.DataFrame:
    """
    Add weekday volume tier features to capture weekly demand patterns.
    
    Supports two patterns:
    
    DEFAULT pattern:
    - Wednesday (day_of_week=2) and Friday (day_of_week=4) have higher volume
    - Tuesday (day_of_week=1) and Thursday (day_of_week=3) have lower volume
    - Among high-volume days, Wednesday is highest, Friday is high
    
    FRESH pattern:
    - Monday (day_of_week=0), Wednesday (day_of_week=2), and Friday (day_of_week=4) have 25-50% higher volume
    - Tuesday (day_of_week=1) and Thursday (day_of_week=3) are baseline/normal volume
    - Saturday and Sunday have minimal/zero volume
    
    Creates:
    - weekday_volume_tier: Numeric tier representing the volume pattern
      * Default: 2=Wednesday highest, 1=Friday high, 0=Tuesday/Thursday low, -1=Monday/Saturday/Sunday neutral
      * Fresh: 5=Mon/Wed/Fri (25-50% higher - STRONG), 0=Tue/Thu (baseline), -3=Sat/Sun (minimal)
    - is_high_volume_weekday: Binary flag for high-volume weekdays
      * Default: 1 for Wednesday/Friday, 0 otherwise
      * Fresh: 1 for Monday/Wednesday/Friday, 0 otherwise
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        weekday_volume_tier_col: Name for weekday_volume_tier column.
        is_high_volume_weekday_col: Name for is_high_volume_weekday column.
        weekday_pattern: Pattern type - "default" (Wed/Fri high) or "fresh" (Mon/Wed/Fri high).
    
    Returns:
        DataFrame with added weekday volume tier features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract day of week (0=Monday, 6=Sunday)
    day_of_week = df[time_col].dt.dayofweek
    
    if weekday_pattern.lower() == "fresh":
        # FRESH pattern: Monday, Wednesday, Friday have 25-50% higher volume than normal days
        # Using stronger tier values to emphasize the weekday pattern for FRESH
        # 5 = Monday, Wednesday, Friday (25-50% higher volume - STRONG SIGNAL)
        # 0 = Tuesday, Thursday (baseline/normal volume)
        # -3 = Saturday, Sunday (minimal/zero volume)
        df[weekday_volume_tier_col] = -3  # Default for Saturday, Sunday (strong negative)
        df.loc[day_of_week == 0, weekday_volume_tier_col] = 6  # Monday (STRONG: 25-50% higher)
        df.loc[day_of_week == 2, weekday_volume_tier_col] = 5  # Wednesday (STRONG: 25-50% higher)
        df.loc[day_of_week == 4, weekday_volume_tier_col] = 5  # Friday (STRONG: 25-50% higher)
        df.loc[day_of_week == 1, weekday_volume_tier_col] = 0  # Tuesday (baseline)
        df.loc[day_of_week == 3, weekday_volume_tier_col] = 0  # Thursday (baseline)
        
        # Binary indicator for high volume weekdays (Monday, Wednesday, Friday)
        df[is_high_volume_weekday_col] = ((day_of_week == 0) | (day_of_week == 2) | (day_of_week == 4)).astype(int)
        
    else:
        # DEFAULT pattern: Wednesday and Friday have higher volume
        # 2 = Wednesday (highest volume)
        # 1 = Friday (high volume, but lower than Wednesday)
        # 0 = Tuesday, Thursday (low volume)
        # -1 = Saturday (lower than Tuesday/Thursday), Monday, Sunday (neutral/other)
        df[weekday_volume_tier_col] = -1  # Default for Monday, Saturday, Sunday
        df.loc[day_of_week == 2, weekday_volume_tier_col] = 2  # Wednesday (highest)
        df.loc[day_of_week == 4, weekday_volume_tier_col] = 1  # Friday (high)
        df.loc[day_of_week == 1, weekday_volume_tier_col] = 0  # Tuesday (low)
        df.loc[day_of_week == 3, weekday_volume_tier_col] = 0  # Thursday (low)
        
        # Binary indicator for high volume weekdays (Wednesday and Friday)
        df[is_high_volume_weekday_col] = ((day_of_week == 2) | (day_of_week == 4)).astype(int)
    
    return df


def add_eom_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    is_eom_col: str = "is_EOM",
    days_until_month_end_col: str = "days_until_month_end",
    eom_window_days: int = 3
) -> pd.DataFrame:
    """
    Add End-of-Month (EOM) surge features to DataFrame.
    
    This captures the KPI-driven pattern where volume spikes in the last few days
    of the month as distributors or sales teams try to hit monthly targets.
    
    Pattern: Average CBM in the last 3 days of the month is approximately 42% higher
    than the rest of the month.
    
    Creates:
    - is_EOM: Binary flag (1 if date is in last N days of month, 0 otherwise)
    - days_until_month_end: Countdown feature (days until the end of the month)
    
    The countdown feature provides a smooth gradient that neural networks can learn
    more easily than a sudden binary switch, helping the model anticipate EOM surges.
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        is_eom_col: Name for is_EOM binary flag column.
        days_until_month_end_col: Name for days_until_month_end countdown column.
        eom_window_days: Number of days at end of month to flag as EOM (default: 3).
    
    Returns:
        DataFrame with added EOM features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Get the day of month and calculate days in the current month
    day_of_month = df[time_col].dt.day
    
    # Calculate the last day of each month
    # Using pandas offset to get the last day of the month
    last_day_of_month = df[time_col] + pd.offsets.MonthEnd(0)
    days_in_month = last_day_of_month.dt.day
    
    # Calculate days until month end (countdown)
    # This gives a smooth gradient: 0 on the last day, 1 on second-to-last, etc.
    df[days_until_month_end_col] = days_in_month - day_of_month
    
    # Binary flag: 1 if date is in last N days of month, 0 otherwise
    df[is_eom_col] = (df[days_until_month_end_col] < eom_window_days).astype(int)
    
    return df


def add_mid_month_peak_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    mid_month_peak_tier_col: str = "mid_month_peak_tier",
    is_mid_month_peak_col: str = "is_mid_month_peak",
    days_to_peak_col: str = "days_to_mid_month_peak",
    peak_pattern: str = "default"
) -> pd.DataFrame:
    """
    Add mid-month peak features to capture volume surge patterns.
    
    Supports two patterns:
    
    DEFAULT pattern (19th-25th surge):
    - Volume starts building up from the 19th
    - Volume gradually increases from 19th through 23rd
    - Volume peaks on 24th and 25th (highest volume days)
    - Volume declines after the 25th
    
    FRESH pattern (8th-15th surge):
    - Volume starts building up from the 8th
    - Volume gradually increases from 8th through 9th
    - Volume peaks on 10th, 11th, 12th (highest volume days)
    - Volume remains strong through 13th-15th
    - Volume declines from 16th onwards
    
    Creates:
    - mid_month_peak_tier: Numeric tier representing the volume pattern
      * 4 = peak days (highest volume)
      * 3 = strong build-up to peak
      * 2 = moderate build-up
      * 1 = early build-up and declining after peak
      * 0 = other days (neutral)
    - is_mid_month_peak: Binary flag (1 for peak days, 0 otherwise)
    - days_to_mid_month_peak: Distance to the primary peak day (negative before, positive after)
      * Helps model learn the gradient leading to and from the peak
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        mid_month_peak_tier_col: Name for mid_month_peak_tier column.
        is_mid_month_peak_col: Name for is_mid_month_peak binary flag column.
        days_to_peak_col: Name for days_to_mid_month_peak column.
        peak_pattern: Pattern type - "default" (24th-25th peak) or "fresh" (8th-15th peak).
    
    Returns:
        DataFrame with added mid-month peak features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract day of month (1-31)
    day_of_month = df[time_col].dt.day
    
    # Initialize with zeros
    df[mid_month_peak_tier_col] = 0
    
    if peak_pattern.lower() == "fresh":
        # FRESH pattern: Peak from 8th-15th, with center on 11th
        # 4 = 10th-12th (peak days, highest volume)
        # 3 = 9th and 13th-15th (strong volume, build-up and sustained high volume)
        # 2 = 8th (moderate build-up)
        # 1 = 16th-18th (declining after peak)
        # 0 = other days (neutral)
        df.loc[day_of_month == 8, mid_month_peak_tier_col] = 2  # Moderate build-up
        df.loc[day_of_month == 9, mid_month_peak_tier_col] = 3  # Strong build-up
        df.loc[(day_of_month >= 10) & (day_of_month <= 12), mid_month_peak_tier_col] = 4  # Peak (10th, 11th, 12th)
        df.loc[(day_of_month >= 13) & (day_of_month <= 15), mid_month_peak_tier_col] = 3  # Strong sustained volume
        df.loc[(day_of_month >= 16) & (day_of_month <= 18), mid_month_peak_tier_col] = 1  # Declining
        
        # Binary indicator for peak days (10th, 11th, 12th)
        df[is_mid_month_peak_col] = ((day_of_month >= 10) & (day_of_month <= 12)).astype(int)
        
        # Distance to peak (distance to 11th - center of peak period)
        df[days_to_peak_col] = day_of_month - 11
        
    else:
        # DEFAULT pattern: Peak from 19th-25th, with center on 24th-25th
        # 4 = 24th-25th (peak days)
        # 3 = 23rd (strong build-up)
        # 2 = 21st-22nd (moderate build-up)
        # 1 = 19th-20th (early build-up) and 26th-30th (declining)
        # 0 = other days (neutral)
        df.loc[(day_of_month == 19) | (day_of_month == 20), mid_month_peak_tier_col] = 1  # Early build-up
        df.loc[(day_of_month == 21) | (day_of_month == 22), mid_month_peak_tier_col] = 2  # Moderate build-up
        df.loc[day_of_month == 23, mid_month_peak_tier_col] = 3  # Strong build-up
        df.loc[(day_of_month == 24) | (day_of_month == 25), mid_month_peak_tier_col] = 4  # Peak
        df.loc[(day_of_month >= 26) & (day_of_month <= 30), mid_month_peak_tier_col] = 1  # Declining
        
        # Binary indicator for peak days (24th and 25th)
        df[is_mid_month_peak_col] = ((day_of_month == 24) | (day_of_month == 25)).astype(int)
        
        # Distance to peak (distance to 24th)
        df[days_to_peak_col] = day_of_month - 24
    
    return df


def add_early_month_low_volume_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    early_month_low_tier_col: str = "early_month_low_tier",
    is_early_month_low_col: str = "is_early_month_low",
    days_from_month_start_col: str = "days_from_month_start"
) -> pd.DataFrame:
    """
    Add early month low volume features to capture beginning-of-month patterns.
    
    Based on observed patterns:
    - 1st through 10th of each month have the lowest volume
    - This captures the monthly cycle where volume is low at the start
    
    Creates:
    - early_month_low_tier: Numeric tier representing the volume pattern
      * 0 = 1st-5th (very low volume days - critical for DRY)
      * 1 = 6th-10th (transitioning low volume)
      * 2 = Other days (normal volume)
    - is_early_month_low: Binary flag (1 for 1st-10th, 0 otherwise)
    - days_from_month_start: Days from the start of the month (0 on 1st, 1 on 2nd, etc.)
      * Helps model learn the gradient from low volume at start
    - is_high_vol_weekday_AND_early_month: EXPLICIT INTERACTION FEATURE
      * Binary flag (1 when BOTH is_high_volume_weekday==1 AND day<=10, 0 otherwise)
      * Prevents the model from applying Monday/Wed/Fri boost during early month period
      * Addresses the "Logic Collision" between weekday signal and early month signal
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        early_month_low_tier_col: Name for early_month_low_tier column.
        is_early_month_low_col: Name for is_early_month_low binary flag column.
        days_from_month_start_col: Name for days_from_month_start column.
    
    Returns:
        DataFrame with added early month low volume features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract day of month (1-31)
    day_of_month = df[time_col].dt.day
    
    # Create early_month_low_tier feature with 3 tiers for better granularity
    # STRONGER SIGNAL: Using more extreme negative values for days 1-5 to override weekday_volume_tier
    # -10 = 1st-5th (EXTREME low volume - CRITICAL pattern, MUST override weekday signals)
    # 1 = 6th-10th (transitioning low volume)
    # 2 = Other days (normal)
    df[early_month_low_tier_col] = 2  # Default for normal days
    df.loc[(day_of_month >= 6) & (day_of_month <= 10), early_month_low_tier_col] = 1  # Transitioning
    df.loc[day_of_month <= 5, early_month_low_tier_col] = -10  # EXTREME low volume (1st-5th) - STRONGER SIGNAL
    
    # Create binary indicator for early month low volume days (1st-10th)
    df[is_early_month_low_col] = (day_of_month <= 10).astype(int)
    
    # Create binary indicator for VERY EARLY days (1st-5th) - CRITICAL for DRY severe drop
    df['is_first_5_days'] = (day_of_month <= 5).astype(int)
    
    # Create binary indicator for FIRST 3 DAYS (1st-3rd) - MAXIMUM PENALTY period for early month fix
    df['is_first_3_days'] = (day_of_month <= 3).astype(int)
    
    # Create days from month start feature (0-based: 0 on 1st, 1 on 2nd, etc.)
    df[days_from_month_start_col] = day_of_month - 1
    
    # SOLUTION 1: Post-Peak Decay Feature
    # This feature represents the "exhaustion" of demand after an EOM peak event
    # It helps break the LSTM's "momentum" from the previous month's high volume
    # Formula: V = exp(-lambda * t), where t = day of month (0-indexed)
    # Day 1: 1.0 (maximum risk of over-prediction)
    # Day 2: 0.74, Day 5: 0.30, Day 10: 0.06
    # This allows the LSTM Forget Gate to learn to suppress high-volume hidden states
    # carried over from the previous month's EOM spike
    lambda_decay = 0.3  # Decay rate
    df['post_peak_signal'] = np.exp(-lambda_decay * df[days_from_month_start_col])
    
    # SOLUTION 3: Explicit Feature Interaction - is_high_vol_weekday_AND_early_month
    # This binary flag explicitly tells the model: "This is a Monday/Wed/Fri, BUT it's early month, so IGNORE the weekday boost"
    # Instead of letting the model figure out the math (which it's failing at), we give it a pre-computed flag
    # This breaks the "Logic Collision" where Is_Monday coefficient overpowers the early month features
    # STRENGTHENED: Now using NEGATIVE value (-1) to actively suppress weekday boost, not just flag it
    # If is_high_volume_weekday doesn't exist yet, we'll create it based on weekday
    if 'is_high_volume_weekday' not in df.columns:
        # Create is_high_volume_weekday on the fly (Monday=0, Wednesday=2, Friday=4)
        weekday = df[time_col].dt.weekday
        df['is_high_volume_weekday'] = weekday.isin([0, 2, 4]).astype(int)
    
    # Now create the explicit interaction feature with STRONGER negative signal for days 1-5
    # Days 1-5: -2 (STRONG suppression of weekday effect)
    # Days 6-10: -1 (moderate suppression of weekday effect)
    # Other days: 0 (no suppression)
    df['is_high_vol_weekday_AND_early_month'] = 0  # Default: no suppression
    df.loc[(df['is_high_volume_weekday'] == 1) & (day_of_month >= 6) & (day_of_month <= 10), 'is_high_vol_weekday_AND_early_month'] = -1  # Moderate suppression (days 6-10)
    df.loc[(df['is_high_volume_weekday'] == 1) & (day_of_month <= 5), 'is_high_vol_weekday_AND_early_month'] = -2  # STRONG suppression (days 1-5)
    
    return df


def add_high_volume_month_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    high_volume_month_tier_col: str = "high_volume_month_tier",
    is_high_volume_month_col: str = "is_high_volume_month",
    is_low_volume_month_col: str = "is_low_volume_month",
    month_col: str = "month",
    lunar_month_col: str = "lunar_month"
) -> pd.DataFrame:
    """
    Add volume month features to capture seasonal volume patterns.
    
    Based on observed patterns:
    - Gregorian December has higher volume
    - Lunar July and August have higher volume
    - Lunar December has lowest volume
    - This captures annual seasonality trends
    
    Creates:
    - high_volume_month_tier: Numeric tier representing volume level by month
      * 2 = Gregorian December OR Lunar July/August (high volume months)
      * 1 = Other months (normal volume)
      * 0 = Lunar December (low volume month)
    - is_high_volume_month: Binary flag (1 for high volume months, 0 otherwise)
    - is_low_volume_month: Binary flag (1 for Lunar December, 0 otherwise)
    - month: Gregorian month number (1-12) extracted from time column
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        high_volume_month_tier_col: Name for high_volume_month_tier column.
        is_high_volume_month_col: Name for is_high_volume_month binary flag column.
        is_low_volume_month_col: Name for is_low_volume_month binary flag column.
        month_col: Name for Gregorian month column (1-12).
        lunar_month_col: Name of lunar month column (must exist in df).
    
    Returns:
        DataFrame with added volume month features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract Gregorian month (1-12)
    df[month_col] = df[time_col].dt.month
    
    # Check if lunar_month column exists
    if lunar_month_col not in df.columns:
        raise ValueError(
            f"Lunar month column '{lunar_month_col}' not found in DataFrame. "
            "Please ensure add_lunar_calendar_features() is called before add_high_volume_month_features()."
        )
    
    # Identify high volume months:
    # - Gregorian December (month == 12)
    # - Lunar July (lunar_month == 7)
    # - Lunar August (lunar_month == 8)
    is_gregorian_december = (df[month_col] == 12)
    is_lunar_july_august = df[lunar_month_col].isin([7, 8])
    is_high_volume = is_gregorian_december | is_lunar_july_august
    
    # Identify low volume months:
    # - Lunar December (lunar_month == 12)
    is_lunar_december = (df[lunar_month_col] == 12)
    is_low_volume = is_lunar_december
    
    # Create high_volume_month_tier feature
    # 2 = High volume months (Gregorian Dec OR Lunar July/Aug)
    # 1 = Normal volume (default)
    # 0 = Low volume months (Lunar Dec)
    df[high_volume_month_tier_col] = 1  # Default for normal months
    df.loc[is_high_volume, high_volume_month_tier_col] = 2  # High volume months
    df.loc[is_low_volume, high_volume_month_tier_col] = 0  # Low volume months
    
    # Create binary indicator for high volume months
    df[is_high_volume_month_col] = is_high_volume.astype(int)
    
    # Create binary indicator for low volume months
    df[is_low_volume_month_col] = is_low_volume.astype(int)
    
    return df


def add_pre_holiday_surge_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    pre_holiday_surge_tier_col: str = "pre_holiday_surge_tier",
    is_pre_holiday_surge_col: str = "is_pre_holiday_surge",
    days_before_surge: int = 7,
    holiday_pattern: str = "default"
) -> pd.DataFrame:
    """
    Add holiday-related volume features (category-specific behavior).
    
    Supports two patterns:
    
    DEFAULT pattern (DRY goods):
    - Volume INCREASES before holidays (Tet, Mid-Autumn) as people stock up
    - Positive tiers for pre-holiday period
    
    FRESH pattern (perishable goods):
    - Volume DECREASES before holidays (people travel, less ordering)
    - Volume INCREASES after holidays (people return, restock)
    - Negative tiers for pre-holiday, positive for post-holiday
    
    Creates:
    - pre_holiday_surge_tier: Numeric tier representing holiday proximity impact
      * Default: +3 (1-3d before), +2 (4-7d before), +1 (8-10d before), 0 (other)
      * Fresh: -3 (1-3d before), -2 (4-7d before), -1 (8-10d before), +2 (1-3d after), +1 (4-7d after), 0 (other)
    - is_pre_holiday_surge: Binary flag (1 if within holiday impact period, 0 otherwise)
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        pre_holiday_surge_tier_col: Name for pre_holiday_surge_tier column.
        is_pre_holiday_surge_col: Name for is_pre_holiday_surge binary flag column.
        days_before_surge: Number of days before holiday to consider as surge period (default: 7).
        holiday_pattern: Pattern type - "default" (pre-holiday surge) or "fresh" (post-holiday surge).
    
    Returns:
        DataFrame with added holiday impact features.
    """
    from datetime import date, timedelta
    from config import load_holidays
    
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Get date range from data
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    
    # Load holidays (model_holidays has Tet and Mid-Autumn)
    holidays_data = load_holidays(holiday_type="model")
    
    # Collect major holidays (Tet and Mid-Autumn only)
    major_holidays = []
    for year, year_holidays in holidays_data.items():
        if 'tet' in year_holidays:
            major_holidays.extend(year_holidays['tet'])
        if 'mid_autumn' in year_holidays:
            major_holidays.extend(year_holidays['mid_autumn'])
    
    # Initialize columns
    df[pre_holiday_surge_tier_col] = 0  # Default: no surge
    df[is_pre_holiday_surge_col] = 0
    
    # For each row, calculate distance to nearest major holiday
    for idx, row in df.iterrows():
        # Skip rows with NaT dates
        if pd.isna(row[time_col]):
            continue
            
        current_date = row[time_col].date()
        
        # Find nearest major holiday (past or future) for FRESH pattern
        all_holidays = []
        for h in major_holidays:
            if pd.notna(h):
                try:
                    # Convert to date object if it's a Timestamp
                    h_date = h.date() if isinstance(h, pd.Timestamp) else h
                    # Additional check: ensure h_date is not None or NaT after conversion
                    if h_date is not None and pd.notna(h_date):
                        all_holidays.append(h_date)
                except (AttributeError, TypeError):
                    # Skip any problematic holiday dates
                    continue
        
        if len(all_holidays) > 0:
            # Find nearest holiday (before or after)
            nearest_holiday = min(all_holidays, key=lambda h: abs((h - current_date).days))
            days_diff = (nearest_holiday - current_date).days  # Negative if holiday passed, positive if upcoming
            
            if holiday_pattern.lower() == "fresh":
                # FRESH pattern: Decline before, surge after holidays
                if 1 <= days_diff <= 3:
                    # 1-3 days BEFORE: Strong decline (people traveling)
                    df.at[idx, pre_holiday_surge_tier_col] = -3
                    df.at[idx, is_pre_holiday_surge_col] = 1
                elif 4 <= days_diff <= 7:
                    # 4-7 days BEFORE: Moderate decline
                    df.at[idx, pre_holiday_surge_tier_col] = -2
                    df.at[idx, is_pre_holiday_surge_col] = 1
                elif 8 <= days_diff <= 10:
                    # 8-10 days BEFORE: Early decline
                    df.at[idx, pre_holiday_surge_tier_col] = -1
                    df.at[idx, is_pre_holiday_surge_col] = 1
                elif -3 <= days_diff <= -1:
                    # 1-3 days AFTER: Strong surge (people restocking)
                    df.at[idx, pre_holiday_surge_tier_col] = 3
                    df.at[idx, is_pre_holiday_surge_col] = 1
                elif -7 <= days_diff <= -4:
                    # 4-7 days AFTER: Moderate surge
                    df.at[idx, pre_holiday_surge_tier_col] = 2
                    df.at[idx, is_pre_holiday_surge_col] = 1
                elif -10 <= days_diff <= -8:
                    # 8-10 days AFTER: Early surge
                    df.at[idx, pre_holiday_surge_tier_col] = 1
                    df.at[idx, is_pre_holiday_surge_col] = 1
            else:
                # DEFAULT pattern: Surge before holidays (DRY goods)
                if days_diff > 0:  # Only for upcoming holidays
                    if 1 <= days_diff <= 3:
                        # 1-3 days before: highest surge
                        df.at[idx, pre_holiday_surge_tier_col] = 3
                        df.at[idx, is_pre_holiday_surge_col] = 1
                    elif 4 <= days_diff <= 7:
                        # 4-7 days before: moderate surge
                        df.at[idx, pre_holiday_surge_tier_col] = 2
                        df.at[idx, is_pre_holiday_surge_col] = 1
                    elif 8 <= days_diff <= 10:
                        # 8-10 days before: early surge
                        df.at[idx, pre_holiday_surge_tier_col] = 1
                        df.at[idx, is_pre_holiday_surge_col] = 1
    
    return df


def encode_categories(df: pd.DataFrame, cat_col: str = "CATEGORY") -> Tuple[pd.DataFrame, dict, int]:
    """
    Encode categorical column to integer IDs.
    
    Args:
        df: DataFrame containing category column.
        cat_col: Name of category column.
    
    Returns:
        Tuple of (df_with_cat_id, cat2id_dict, num_categories)
    """
    categories = sorted(df[cat_col].unique())
    cat2id = {cat: i for i, cat in enumerate(categories)}
    cat_id_col = f"{cat_col}_ID"
    df = df.copy()
    df[cat_id_col] = df[cat_col].map(cat2id)
    num_categories = len(categories)
    
    return df, cat2id, num_categories


def split_data(
    data: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    temporal: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: DataFrame to split.
        train_size: Proportion of data for training.
        val_size: Proportion of data for validation.
        test_size: Proportion of data for testing.
        temporal: If True, use temporal split (sequential). If False, use random split.
    
    Returns:
        Tuple of (train_data, val_data, test_data).
    
    Raises:
        ValueError: If sizes don't sum to 1.0.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    
    N = len(data)
    
    if temporal:
        train_end = int(train_size * N)
        val_end = train_end + int(val_size * N)
        
        train_data = data[:train_end].copy()
        val_data = data[train_end:val_end].copy()
        test_data = data[val_end:].copy()
    else:
        # Random split (not recommended for time series)
        from sklearn.model_selection import train_test_split
        
        train_data, temp_data = train_test_split(
            data, test_size=(val_size + test_size), shuffle=True, random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=(test_size / (val_size + test_size)), shuffle=True, random_state=42
        )
    
    return train_data, val_data, test_data


def aggregate_daily(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    target_col: str = "QTY",
    keep_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate transaction-level data to daily totals by category.
    
    Groups data by date and category, summing QTY values. Preserves temporal
    and holiday features by taking the first value for each date (since they
    should be the same for all transactions on the same day).
    
    Args:
        df: DataFrame with transaction-level data.
        time_col: Name of time column.
        cat_col: Name of category column.
        target_col: Name of target column to sum (e.g., "QTY").
        keep_cols: Additional columns to keep (first value taken per group).
    
    Returns:
        DataFrame with daily aggregated data (one row per date per category).
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Normalize to date only (remove time component)
    df['date_only'] = pd.to_datetime(df[time_col]).dt.normalize()
    
    # Columns to aggregate - sum target_col, optionally sum additional numeric totals
    agg_dict = {target_col: 'sum'}

    # If a separate total quantity column exists (e.g. "Total QTY" when predicting "Total CBM"),
    # aggregate it as well so downstream features can use both volume and quantity.
    if "Total QTY" in df.columns and "Total QTY" != target_col:
        agg_dict["Total QTY"] = "sum"
    
    # For temporal/holiday/weekend features, take first value (should be same for all rows on same date)
    feature_cols_to_keep = [
        'month_sin', 'month_cos', 'dayofmonth_sin', 'dayofmonth_cos',
        'holiday_indicator', 'days_until_next_holiday', 'days_since_holiday',
        'is_weekend', 'day_of_week', 'day_of_week_sin', 'day_of_week_cos',
        'weekday_volume_tier', 'is_high_volume_weekday',
        'Is_Monday',  # Binary flag for Monday peak patterns (FRESH category)
        'lunar_month', 'lunar_day',
        'lunar_month_sin', 'lunar_month_cos', 'lunar_day_sin', 'lunar_day_cos',
        'days_to_tet', 'days_to_mid_autumn',  # Tet and Mid-Autumn countdown features
        'is_active_season', 'days_until_peak', 'is_golden_window',  # Seasonal active-window features
        # Lunar cyclical encodings and Tet countdown should also persist after aggregation
        'lunar_month_sin', 'lunar_month_cos',
        'lunar_day_sin', 'lunar_day_cos',
        'days_to_tet',
        # EOM (End-of-Month) surge features
        'is_EOM', 'days_until_month_end',
        # Seasonal active-window features
        'is_active_season', 'days_until_peak',
        # NEW: Gregorian-Anchored Peak Alignment features for MOONCAKE
        'days_until_lunar_08_01',  # Countdown to Lunar 08-01
        'is_august',  # Binary feature for Gregorian August (month == 8)
    ]
    
    for col in feature_cols_to_keep:
        if col in df.columns and col not in agg_dict:
            agg_dict[col] = 'first'
    
    # Add any explicitly requested columns
    if keep_cols:
        for col in keep_cols:
            if col in df.columns and col not in agg_dict:
                agg_dict[col] = 'first'
    
    # Group by date and category
    grouped = df.groupby(['date_only', cat_col], as_index=False).agg(agg_dict)
    
    # Rename date_only back to time_col
    grouped = grouped.rename(columns={'date_only': time_col})
    
    # Sort by category and date
    grouped = grouped.sort_values([cat_col, time_col]).reset_index(drop=True)
    
    return grouped


def add_operational_status_flags(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    target_col: str = "QTY",
    status_col: str = "operational_status",
    expected_zero_flag_col: str = "is_expected_zero",
    anomaly_flag_col: str = "is_operational_anomaly",
    holiday_label: str = "Holiday_OFF",
    weekend_label: str = "Weekend_Downtime",
    anomaly_label: str = "Operational_Anomaly",
    normal_label: str = "Business_Day_Normal",
) -> pd.DataFrame:
    """
    Context-aware data imputation and anomaly tracking for daily time-series.

    This function processes each calendar date as follows:

    1) Holiday-driven downtime (expected zero):
       - Cross-references each date against the canonical Vietnam business
         holiday calendar (from config/holidays.yaml via get_vietnam_holidays).
       - If a date is a recognized public holiday or scheduled facility closure,
         it is labeled as `holiday_label` (default: "Holiday_OFF").
       - Volume on these days is treated as an *expected external constraint*.

    2) Standard weekend downtime (Sunday low-volume):
       - All Sundays (day_of_week == 6) are labeled as `weekend_label`
         (default: "Weekend_Downtime").
       - Given the historical pattern of zero/negligible volume on Sundays,
         these are treated as *expected zeros* and decoupled from
         business-day demand signals.

    3) Unexpected operational gaps (critical tracking):
       - For any date that is neither a holiday nor a Sunday but contains
         missing or zero volume, the system:
           * Imputes the volume to 0 to maintain time-series continuity
             for downstream LSTM layers.
           * Flags the date as `anomaly_label`
             (default: "Operational_Anomaly").
           * Marks it with a binary `anomaly_flag_col` so that baseline /
             trend components can explicitly *exclude* it from their
             reference windows.

    In addition, a binary `expected_zero_flag_col` is provided to indicate
    Holiday_OFF and Weekend_Downtime cases, which are structurally expected
    zeros and should not be treated as demand collapse.

    This logic improves:
    - Model interpretability (clear separation of calendar-driven zeros vs
      unexplained operational gaps).
    - Statistical baselines (by excluding Operational_Anomalies from
      standard trend estimation).
    """
    df = df.copy()

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Ensure target column exists; if not, nothing to do
    if target_col not in df.columns:
        return df

    # Normalize to date (drop time component)
    dates = df[time_col].dt.date

    # Build Vietnam business holiday calendar over the data range
    if len(df) > 0:
        min_date = dates.min()
        max_date = dates.max()
        holiday_list = get_vietnam_holidays(min_date, max_date)
        holiday_set = set(holiday_list)
    else:
        holiday_set = set()

    # Day-of-week (0=Mon, 6=Sun)
    day_of_week = df[time_col].dt.dayofweek

    # Treat missing values in the target as zero for classification/imputation
    vol = pd.to_numeric(df[target_col], errors="coerce")
    is_missing_or_zero = vol.isna() | (vol == 0)

    # Initialize status labels
    status = np.full(len(df), normal_label, dtype=object)

    is_holiday = dates.isin(holiday_set)
    is_weekend_sun = day_of_week == 6

    # 1) Holidays
    status[is_holiday] = holiday_label

    # 2) Sunday downtime
    # Do not override holidays that also fall on Sunday
    weekend_only = (~is_holiday) & is_weekend_sun
    status[weekend_only] = weekend_label

    # 3) Operational anomalies:
    #    Non-holiday, non-Sunday business days with missing/zero volume.
    anomaly_mask = (~is_holiday) & (~is_weekend_sun) & is_missing_or_zero

    # Impute to 0 to maintain continuity
    vol_imputed = vol.copy()
    vol_imputed[anomaly_mask] = 0.0
    df[target_col] = vol_imputed.fillna(0.0)

    # Label anomalies
    status[anomaly_mask] = anomaly_label

    # Write status + flags back to dataframe
    df[status_col] = status
    df[expected_zero_flag_col] = ((status == holiday_label) | (status == weekend_label)).astype(int)
    df[anomaly_flag_col] = (status == anomaly_label).astype(int)

    return df


def add_seasonal_active_window_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    lunar_month_col: str = "lunar_month",
    days_to_tet_col: str = "days_to_tet",
    is_active_season_col: str = "is_active_season",
    days_until_peak_col: str = "days_until_peak",
    is_golden_window_col: str = "is_golden_window",
) -> pd.DataFrame:
    """
    Add seasonal active-window masking features for seasonal categories (TET, MOONCAKE).
    
    Creates:
    - is_active_season: Binary feature (1 if date is in active season, 0 otherwise)
      - For MOONCAKE: Active only between Lunar Months 7-9 AND Gregorian months 7-9 (July-September)
      - For TET: Active only 45 days prior to the Lunar New Year
    - days_until_peak: Continuous countdown feature to the peak event
      - For MOONCAKE: Days until Mid-Autumn Festival (lunar month 8, day 15)
      - For TET: Days until Tet start (from days_to_tet)
    - is_golden_window: Binary feature (1 if date is in Golden Window, 0 otherwise)
      - For MOONCAKE: Golden Window is Lunar Months 6.15 to 8.01 (peak buildup period)
      - For TET: Not applicable (set to 0)
    
    This feature helps the model learn when seasonal categories are "active" and
    provides a gradient signal for approaching peak demand periods.
    
    Args:
        df: DataFrame with time, category, and lunar calendar columns.
        time_col: Name of time column.
        cat_col: Name of category column.
        lunar_month_col: Name of lunar month column.
        days_to_tet_col: Name of days_to_tet column (for TET category).
        is_active_season_col: Name for is_active_season column.
        days_until_peak_col: Name for days_until_peak column.
        is_golden_window_col: Name for is_golden_window column.
    
    Returns:
        DataFrame with added seasonal active-window features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Initialize columns
    df[is_active_season_col] = 0
    df[days_until_peak_col] = np.nan
    df[is_golden_window_col] = 0  # Initialize Golden Window feature
    
    # Process each category
    for category in df[cat_col].unique():
        cat_mask = df[cat_col] == category
        
        if category == "MOONCAKE":
            # MOONCAKE: Active between Lunar Months 7 and 9 (narrowed from 6-9 to prevent early predictions)
            # Peak is Mid-Autumn Festival (Lunar Month 8, Day 15)
            # CRITICAL: Start at Lunar Month 7 to prevent predictions in June (Lunar Month 6 starts too early)
            if lunar_month_col not in df.columns:
                raise ValueError(f"Lunar month column '{lunar_month_col}' not found. Call add_lunar_calendar_features first.")
            
            # Ensure time column is datetime for Gregorian month check
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            
            # Active season: Lunar months 7-9 AND Gregorian months 7-9 (July-September)
            # This prevents early predictions in June and late predictions in October
            lunar_month = df.loc[cat_mask, lunar_month_col]
            lunar_day = df.loc[cat_mask, "lunar_day"] if "lunar_day" in df.columns else None
            gregorian_month = df.loc[cat_mask, time_col].dt.month
            
            # Narrow active season: Lunar months 7-9 AND Gregorian months 7-9 (July-September only)
            is_active = ((lunar_month >= 7) & (lunar_month <= 9)) & ((gregorian_month >= 7) & (gregorian_month <= 9))
            df.loc[cat_mask & is_active, is_active_season_col] = 1
            
            # Golden Window: GREGORIAN AUGUST (August 1-31) - Anchored to Gregorian calendar
            # This overrides the lunar-based window to prevent phase shift errors.
            # The model must treat Gregorian Month == 8 as a high-probability trigger,
            # regardless of the corresponding Lunar month.
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            
            # Golden Window: Gregorian August (month == 8)
            gregorian_month = df.loc[cat_mask, time_col].dt.month
            is_golden = (gregorian_month == 8)
            df.loc[cat_mask & is_golden, is_golden_window_col] = 1
            
            # Peak Loss Window: Lunar Months 7.15 to 8.15 (for loss weighting - more focused than golden window)
            # This is the critical peak period where errors must be heavily penalized
            if lunar_day is not None:
                is_peak_loss_window = (
                    ((lunar_month == 7) & (lunar_day >= 15)) |
                    ((lunar_month == 8) & (lunar_day <= 15))
                )
                # Store as a separate feature for loss weighting
                if 'is_peak_loss_window' not in df.columns:
                    df['is_peak_loss_window'] = 0
                df.loc[cat_mask & is_peak_loss_window, 'is_peak_loss_window'] = 1
            
            # Calculate days until peak (Mid-Autumn Festival: Lunar Month 8, Day 15)
            mooncake_mask = cat_mask
            if lunar_day is not None:
                # If in lunar month 8, countdown to day 15
                in_month_8 = lunar_month == 8
                days_to_peak = np.where(
                    in_month_8,
                    np.abs(lunar_day - 15),
                    np.where(
                        lunar_month < 8,
                        (8 - lunar_month) * 30 + (15 - lunar_day),  # Approximate
                        (lunar_month - 8) * 30 + lunar_day - 15  # Past peak
                    )
                )
                df.loc[mooncake_mask, days_until_peak_col] = days_to_peak
            else:
                # Fallback: use lunar month distance
                days_to_peak = np.abs((lunar_month - 8) * 30)
                df.loc[mooncake_mask, days_until_peak_col] = days_to_peak
        
        elif category == "TET":
            # TET: Active 45 days prior to Lunar New Year
            if days_to_tet_col not in df.columns:
                raise ValueError(f"Days to Tet column '{days_to_tet_col}' not found. Call add_days_to_tet_feature first.")
            
            # Active season: 45 days before Tet (days_to_tet <= 45)
            days_to_tet = df.loc[cat_mask, days_to_tet_col]
            is_active = days_to_tet <= 45
            df.loc[cat_mask & is_active, is_active_season_col] = 1
            
            # days_until_peak for TET is simply days_to_tet
            df.loc[cat_mask, days_until_peak_col] = days_to_tet.values
            
            # TET does not have a Golden Window (set to 0)
            df.loc[cat_mask, is_golden_window_col] = 0
        
        # For any other categories not explicitly handled, ensure columns exist with defaults
        # (is_golden_window already initialized to 0 for all rows at line 901)
        # (days_until_peak will be filled with NaN and then 365 below)
    
    # Fill NaN values with large number (outside active season)
    df[days_until_peak_col] = df[days_until_peak_col].fillna(365)
    
    # Ensure is_golden_window exists for all rows (should already be initialized, but double-check)
    if is_golden_window_col not in df.columns:
        df[is_golden_window_col] = 0
    
    return df


def add_days_until_lunar_08_01_feature(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    lunar_month_col: str = "lunar_month",
    lunar_day_col: str = "lunar_day",
    days_until_lunar_08_01_col: str = "days_until_lunar_08_01"
) -> pd.DataFrame:
    """
    Add countdown feature to Lunar 08-01 (the 1st day of the 8th Lunar Month).
    
    This feature replaces the raw lunar_month dependency with a consistent,
    non-drifting countdown signal. Lunar 08-01 is the "Golden Deadline" where
    warehouses must be near-empty of Mooncake stock to ensure it reaches store
    shelves for the month of August (Lunar).
    
    ENHANCED: Now uses lunar_utils for precise countdown computation.
    
    Creates:
    - days_until_lunar_08_01: Number of days until the next occurrence of
      Lunar Month 8, Day 1. This provides a smooth countdown gradient that
      helps the model anticipate the peak period.
    
    Args:
        df: DataFrame with time, lunar_month, and lunar_day columns.
        time_col: Name of time column.
        lunar_month_col: Name of lunar month column.
        lunar_day_col: Name of lunar day column.
        days_until_lunar_08_01_col: Name for days_until_lunar_08_01 column.
    
    Returns:
        DataFrame with added days_until_lunar_08_01 feature.
    """
    from src.utils.lunar_utils import compute_days_until_lunar_08_01
    
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Initialize column
    df[days_until_lunar_08_01_col] = 365  # Default fallback
    
    # Compute for each row using enhanced lunar_utils
    for idx in df.index:
        row_date = df.loc[idx, time_col]
        
        # Skip NaN/None dates
        if pd.isna(row_date):
            continue
        
        try:
            if hasattr(row_date, 'date'):
                row_date = row_date.date()
            else:
                row_date = pd.to_datetime(row_date).date()
            
            # Use enhanced lunar_utils function
            days_until = compute_days_until_lunar_08_01(row_date)
            df.loc[idx, days_until_lunar_08_01_col] = days_until
        except (ValueError, TypeError, AttributeError) as e:
            # If date conversion fails, keep default value (365)
            continue
    
    return df


def add_is_august_feature(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    is_august_col: str = "is_august"
) -> pd.DataFrame:
    """
    Add binary is_august feature to reinforce August (Gregorian month 8) as a strong signal.
    
    For MOONCAKE category, August is the "Sell-in" month for retail distribution.
    Even if the Mid-Autumn Festival is late (October), the warehouse must clear stock
    in August to fill the distribution channels. This feature anchors the model to
    treat Gregorian Month == 8 as a high-probability trigger for MOONCAKE volume,
    regardless of the corresponding Lunar month.
    
    Creates:
    - is_august: Binary feature (1 if Gregorian month == 8, 0 otherwise)
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        is_august_col: Name for is_august column.
    
    Returns:
        DataFrame with added is_august feature.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract Gregorian month (1-12)
    month = df[time_col].dt.month
    
    # Create binary is_august feature (1 for August, 0 otherwise)
    df[is_august_col] = (month == 8).astype(int)
    
    return df


def add_cbm_density_features(
    df: pd.DataFrame,
    cbm_col: str = "Total CBM",
    qty_col: str = "Total QTY",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    density_col: str = "cbm_per_qty",
    density_last_year_col: str = "cbm_per_qty_last_year",
    eps: float = 1e-3,
) -> pd.DataFrame:
    """
    Add volume–quantity density features, including a "last-year" structural prior.

    Creates:
    - cbm_per_qty:      Current-day density (CBM / QTY) per (date, category)
    - cbm_per_qty_last_year: Density observed on the same calendar date one year earlier
                              for the same category.

    This gives the model an explicit prior about how "bulky" shipments were in the
    same seasonal window last year, which is especially useful around Tet / peak seasons.

    Assumptions:
    - df is already aggregated at daily granularity per category.
    - cbm_col and qty_col exist in the dataframe.
    """
    df = df.copy()

    # Basic checks
    if cbm_col not in df.columns or qty_col not in df.columns:
        # Nothing to do if we don't have both volume and quantity
        return df

    # Ensure time column is datetime for DateOffset operations
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Coerce CBM and QTY to numeric to avoid string arithmetic issues coming from CSV dtypes
    cbm_numeric = pd.to_numeric(df[cbm_col], errors="coerce")
    qty_numeric = pd.to_numeric(df[qty_col], errors="coerce")

    # Safety handling:
    # - Replace NaN/zero quantities with small epsilon to avoid division by zero
    # - Replace NaN CBM with 0.0 (no volume recorded)
    qty_safe = qty_numeric.replace(0, eps).fillna(eps)
    cbm_safe = cbm_numeric.fillna(0.0)

    # Current-day density: CBM / max(QTY, eps)
    df[density_col] = cbm_safe / qty_safe
    # Optional stability cap: prevent extreme densities from dominating
    df[density_col] = df[density_col].clip(lower=0.0, upper=1.0)

    # Build a mapping from (category, date_minus_1y) -> density.
    # We construct an auxiliary dataframe where we shift dates by +1 year,
    # then left-join back on (CATEGORY, ACTUALSHIPDATE).
    aux = df[[time_col, cat_col, density_col]].copy()
    aux[time_col] = aux[time_col] + pd.DateOffset(years=1)
    aux = aux.rename(columns={density_col: density_last_year_col})

    # Merge back to align each row with the density from exactly one year before
    df = df.merge(
        aux[[time_col, cat_col, density_last_year_col]],
        on=[time_col, cat_col],
        how="left",
    )

    # For dates where we don't have data from a full prior year (e.g., the first year),
    # fall back to a stable per-category statistic to avoid NaNs.
    if df[density_last_year_col].isna().any():
        # Category-level median density as a robust fallback
        cat_median = (
            df.groupby(cat_col)[density_col]
            .transform("median")
            .fillna(df[density_col].median())
        )
        df[density_last_year_col] = df[density_last_year_col].fillna(cat_median)

    return df


def add_year_over_year_volume_features(
    df: pd.DataFrame,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    yoy_1y_col: str = "cbm_last_year",
    yoy_2y_col: str = "cbm_2_years_ago",
    use_lunar_matching: bool = False,
    lunar_month_col: str = "lunar_month",
    lunar_day_col: str = "lunar_day",
) -> pd.DataFrame:
    """
    Add year-over-year volume features for seasonal products like MOONCAKE.
    
    For highly seasonal products that follow annual patterns (e.g., Mid-Autumn Festival),
    the same period from previous years is more predictive than recent days.
    
    CRITICAL FOR MOONCAKE: When use_lunar_matching=True, matches by LUNAR date instead of
    calendar date. This is essential because Mid-Autumn Festival shifts 10-20 days every year
    on the solar calendar. For example:
    - 2025-08-15 (solar) might be Lunar 07-15, which should match 2024-08-XX (solar) that is also Lunar 07-15
    - NOT 2024-08-15 (solar), which would be a different lunar date
    
    Creates:
    - cbm_last_year: Total CBM from the same LUNAR date one year earlier (if use_lunar_matching=True)
                     OR same calendar date one year earlier (if use_lunar_matching=False)
    - cbm_2_years_ago: Total CBM from the same LUNAR date two years earlier (if use_lunar_matching=True)
                      OR same calendar date two years earlier (if use_lunar_matching=False)
    
    ENHANCED: Now uses the lunar_utils module for precise Lunar-to-Lunar date mapping.
    
    Args:
        df: DataFrame with daily aggregated data (one row per date per category).
        target_col: Name of target volume column (e.g., "Total CBM").
        time_col: Name of time column.
        cat_col: Name of category column.
        yoy_1y_col: Name for 1-year-ago volume feature.
        yoy_2y_col: Name for 2-years-ago volume feature.
        use_lunar_matching: If True, match by lunar date instead of calendar date (for MOONCAKE).
        lunar_month_col: Name of lunar month column (required if use_lunar_matching=True).
        lunar_day_col: Name of lunar day column (required if use_lunar_matching=True).
    
    Returns:
        DataFrame with added year-over-year volume features.
    """
    df = df.copy()
    
    # Basic checks
    if target_col not in df.columns:
        return df
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Coerce target to numeric
    target_numeric = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    
    if use_lunar_matching:
        # ENHANCED: Use lunar_utils for precise Lunar-to-Lunar date mapping
        from src.utils.lunar_utils import find_lunar_aligned_date_from_previous_year
        
        if lunar_month_col not in df.columns or lunar_day_col not in df.columns:
            raise ValueError(
                f"Lunar columns '{lunar_month_col}' and '{lunar_day_col}' must exist "
                "when use_lunar_matching=True. Call add_lunar_calendar_features first."
            )
        
        # Initialize YoY columns
        df[yoy_1y_col] = 0.0
        df[yoy_2y_col] = 0.0
        
        # Process each category separately
        for category in df[cat_col].unique():
            cat_mask = df[cat_col] == category
            cat_data = df[cat_mask].copy()
            
            # For each row in this category, find lunar-aligned historical dates
            for idx in cat_data.index:
                current_date = cat_data.loc[idx, time_col]
                
                # Skip NaN/None dates
                if pd.isna(current_date):
                    continue
                
                try:
                    if hasattr(current_date, 'date'):
                        current_date = current_date.date()
                    else:
                        current_date = pd.to_datetime(current_date).date()
                except (ValueError, TypeError, AttributeError):
                    # If date conversion fails, skip this row
                    continue
                
                # Find lunar-aligned date from 1 year ago
                date_1y_ago = find_lunar_aligned_date_from_previous_year(
                    current_date,
                    cat_data,
                    time_col=time_col,
                    years_back=1
                )
                
                if date_1y_ago is not None:
                    # Fetch CBM value from that date
                    mask_1y = cat_data[time_col].dt.date == date_1y_ago
                    if mask_1y.any():
                        cbm_val = cat_data.loc[mask_1y, target_col].iloc[0]
                        if pd.notna(cbm_val):
                            df.at[idx, yoy_1y_col] = cbm_val
                
                # Find lunar-aligned date from 2 years ago
                date_2y_ago = find_lunar_aligned_date_from_previous_year(
                    current_date,
                    cat_data,
                    time_col=time_col,
                    years_back=2
                )
                
                if date_2y_ago is not None:
                    # Fetch CBM value from that date
                    mask_2y = cat_data[time_col].dt.date == date_2y_ago
                    if mask_2y.any():
                        cbm_val = cat_data.loc[mask_2y, target_col].iloc[0]
                        if pd.notna(cbm_val):
                            df.at[idx, yoy_2y_col] = cbm_val
        
    else:
        # Original calendar date matching
        # Build mapping for 1 year ago: (category, date+1year) -> volume
        aux_1y = df[[time_col, cat_col, target_col]].copy()
        aux_1y[time_col] = aux_1y[time_col] + pd.DateOffset(years=1)
        aux_1y = aux_1y.rename(columns={target_col: yoy_1y_col})
        
        # Merge back to align each row with volume from exactly one year before
        df = df.merge(
            aux_1y[[time_col, cat_col, yoy_1y_col]],
            on=[time_col, cat_col],
            how="left",
        )
        
        # Build mapping for 2 years ago: (category, date+2years) -> volume
        aux_2y = df[[time_col, cat_col, target_col]].copy()
        aux_2y[time_col] = aux_2y[time_col] + pd.DateOffset(years=2)
        aux_2y = aux_2y.rename(columns={target_col: yoy_2y_col})
        
        # Merge back to align each row with volume from exactly two years before
        df = df.merge(
            aux_2y[[time_col, cat_col, yoy_2y_col]],
            on=[time_col, cat_col],
            how="left",
        )
    
    # For dates where we don't have data from prior years, fall back to 0.0
    # (off-season periods should be 0, which is correct for MOONCAKE)
    df[yoy_1y_col] = df[yoy_1y_col].fillna(0.0)
    df[yoy_2y_col] = df[yoy_2y_col].fillna(0.0)
    
    return df


def fit_scaler(
    train_data: pd.DataFrame,
    target_col: str = "QTY"
) -> StandardScaler:
    """
    Fit a StandardScaler on training data QTY values.
    
    CRITICAL: Root Cause #2 - Scaling Strategy
    ==========================================
    This scaler is ONLY applied to the TARGET COLUMN (Total CBM), NOT to features.
    
    Why this matters:
    - Binary penalty features (is_first_5_days, is_early_month_low, etc.) remain as 0/1
    - Tier features (early_month_low_tier with values like -10, 1, 2) remain unscaled
    - This preserves their sharp "on/off" impact and prevents signal erosion
    
    However, the target column IN THE FEATURE WINDOW is scaled, which means:
    - Historical volume values fed to the LSTM are standardized (mean=0, std=1)
    - This helps with gradient stability and convergence
    - But it means penalty features must be strong enough to overcome scaled momentum
    
    Args:
        train_data: Training DataFrame.
        target_col: Name of target column to scale.
    
    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    # Fit on a NumPy array rather than a pandas DataFrame so that the
    # scaler does not store feature_names_in_. This avoids sklearn
    # warnings when we later call transform/inverse_transform with
    # plain NumPy arrays (which is what we do throughout the pipeline).
    values = train_data[[target_col]].to_numpy()
    scaler.fit(values)
    return scaler


def apply_scaling(
    df: pd.DataFrame,
    scaler: StandardScaler,
    target_col: str = "QTY",
    scale_feature: bool = True
) -> pd.DataFrame:
    """
    Apply scaling to QTY column in DataFrame.
    
    Args:
        df: DataFrame to scale.
        scaler: Fitted StandardScaler.
        target_col: Name of target column to scale.
        scale_feature: If True, also scale QTY in feature columns (for input windows).
                       If False, only scale the target column.
    
    Returns:
        DataFrame with scaled QTY values.
    """
    df = df.copy()
    
    if target_col not in df.columns:
        return df
    
    # Scale target column
    # NOTE: We intentionally pass a NumPy array to the scaler to avoid
    # strict feature-name checks (feature_names_in_) that can differ
    # between training and inference (e.g., residual vs absolute target
    # column names). The scaler statistics (mean_, scale_) are still
    # applied consistently.
    values = df[[target_col]].to_numpy()
    scaled = scaler.transform(values)
    df[target_col] = scaled.flatten()
    
    return df


def inverse_transform_scaling(
    values: np.ndarray,
    scaler: StandardScaler,
    target_col: str = "QTY"
) -> np.ndarray:
    """
    Inverse transform scaled values back to original scale.
    
    Args:
        values: Scaled values (can be 1D array or 2D with shape (n_samples, 1)).
        scaler: Fitted StandardScaler used for scaling.
        target_col: Name of target column (for compatibility, not used).
    
    Returns:
        Unscaled values in original scale.
    """
    # Ensure values are 2D for scaler
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    
    return scaler.inverse_transform(values).flatten()


def prepare_data(
    df: pd.DataFrame,
    cat_col: str = "CATEGORY",
    feature_cols: Optional[List[str]] = None,
    target_col: str = "QTY",
    time_col: str = "ACTUALSHIPDATE",
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2
) -> dict:
    """
    Complete data preparation pipeline: encoding, splitting, and window creation.
    
    Args:
        df: Raw DataFrame.
        cat_col: Name of category column.
        feature_cols: List of feature column names.
        target_col: Name of target column.
        time_col: Name of time column.
        train_size: Proportion for training.
        val_size: Proportion for validation.
        test_size: Proportion for testing.
    
    Returns:
        Dictionary containing prepared data and metadata.
    """
    # Encode categories
    df, cat2id, num_categories = encode_categories(df, cat_col)
    
    # Split data
    train_data, val_data, test_data = split_data(
        df, train_size, val_size, test_size, temporal=True
    )
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'cat2id': cat2id,
        'num_categories': num_categories,
        'cat_id_col': f"{cat_col}_ID"
    }


def apply_sunday_to_monday_carryover(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    target_col: str = "Total CBM",
    actual_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply Sunday-to-Monday demand carryover to capture backlog accumulation.
    
    Logic: Sundays are non-operational (Total CBM = 0). This function implements
    Sunday-to-Monday demand carryover where Monday's Target = (Actual Monday + Actual Sunday).
    
    Purpose: Capture backlog accumulation and eliminate misleading zero-demand
    patterns in the time-series that occur when Sunday demand is deferred to Monday.
    
    Args:
        df: DataFrame with daily aggregated data (one row per date per category).
        time_col: Name of time column (should be datetime or date).
        cat_col: Name of category column.
        target_col: Name of target column to adjust (e.g., "Total CBM").
        actual_col: Optional name of actual column to use for carryover calculation.
                   If None, uses target_col for both Sunday and Monday values.
    
    Returns:
        DataFrame with adjusted target values where Monday's target includes Sunday's demand.
    
    Note:
        - This function modifies the target_col in-place for Monday rows.
        - Sunday rows remain unchanged (they keep their original values, typically 0).
        - The adjustment is applied per category independently.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Use target_col for actual values if actual_col not specified
    if actual_col is None:
        actual_col = target_col
    
    # Ensure both columns exist
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if actual_col not in df.columns:
        raise ValueError(f"Actual column '{actual_col}' not found in DataFrame.")
    
    # Extract day of week (0=Monday, 6=Sunday)
    df['_day_of_week'] = df[time_col].dt.dayofweek
    
    # Sort by category and date to ensure proper ordering
    df = df.sort_values([cat_col, time_col]).reset_index(drop=True)
    
    # For each category, apply carryover logic
    def apply_carryover_per_category(group):
        """Apply Sunday-to-Monday carryover for a single category."""
        group = group.copy()
        
        # Identify Mondays (day_of_week == 0)
        monday_mask = group['_day_of_week'] == 0
        monday_indices = group.index[monday_mask]
        
        # For each Monday, find the previous Sunday and add its value
        for monday_idx in monday_indices:
            # Find position of Monday in the group
            monday_pos = group.index.get_loc(monday_idx)
            
            # Look backwards for the previous Sunday (day_of_week == 6)
            if monday_pos > 0:
                prev_days = group.iloc[:monday_pos]
                sunday_in_prev = prev_days[prev_days['_day_of_week'] == 6]
                
                if len(sunday_in_prev) > 0:
                    # Get the most recent Sunday before this Monday
                    sunday_idx = sunday_in_prev.index[-1]
                    sunday_value = group.loc[sunday_idx, actual_col]
                    
                    # Apply carryover: Monday's target = Monday actual + Sunday actual
                    if pd.notna(sunday_value):
                        monday_actual = group.loc[monday_idx, actual_col]
                        group.loc[monday_idx, target_col] = monday_actual + sunday_value
        
        return group
    
    # Apply carryover per category
    df = df.groupby(cat_col, group_keys=False).apply(apply_carryover_per_category)
    
    # Clean up temporary columns
    df = df.drop(columns=['_day_of_week'])
    
    return df


def moving_average_forecast_by_category(
    df: pd.DataFrame,
    cat_col: str = "CATEGORY",
    time_col: str = "ACTUALSHIPDATE",
    target_col: str = "QTY",
    window: int = 7,
) -> pd.DataFrame:
    """
    Simple heuristic forecaster: category-wise moving average baseline.

    This is intended for *minor* / low-volume categories (e.g., POSM, OTHER)
    whose noisy behavior should not be backpropagated through the LSTM.

    The function computes, for each (category, date), a trailing moving average
    of the target over the previous `window` days and stores it in a new column
    `<target_col>_MA{window}`. This can be used as a lightweight statistical
    head for minor entities during inference.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    df = df.sort_values([cat_col, time_col])
    ma_col = f"{target_col}_MA{window}"

    # If operational anomaly flags are available, exclude those days from the
    # baseline trend calculation. This prevents unexplained zero-volume gaps
    # from biasing the moving-average forecast for minor categories.
    anomaly_flag_col = "is_operational_anomaly"

    if anomaly_flag_col in df.columns:
        def masked_ma(group: pd.Series, mask: pd.Series) -> pd.Series:
            # Compute rolling mean over non-anomalous days only.
            # We use a masked series where anomalies are dropped from the
            # rolling window rather than set to zero.
            values = group.copy()
            values_masked = values.where(~mask, np.nan)
            return values_masked.rolling(window=window, min_periods=1).mean()

        df[ma_col] = (
            df.groupby(cat_col, group_keys=False).apply(
                lambda g: masked_ma(
                    g[target_col],
                    g[anomaly_flag_col].astype(bool)
                )
            )
        )
    else:
        df[ma_col] = (
            df.groupby(cat_col)[target_col]
            .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        )

    return df

