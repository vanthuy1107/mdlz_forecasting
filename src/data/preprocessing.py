"""Data preprocessing and window slicing utilities."""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import date, timedelta
import sys
from pathlib import Path

from src.utils.date import solar_to_lunar_date, get_tet_start_dates

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import load_holidays


def slicing_window_category(
    df,
    input_size,
    horizon,
    feature_cols,
    target_col,
    cat_col,
    time_col,
    label_start_date=None,
    label_end_date=None,
    return_dates=False
):
    """
    Tạo sliding windows từ dữ liệu time series.
    
    Args:
        df: DataFrame chứa dữ liệu
        input_size: kích thước cửa sổ input
        horizon: số ngày dự đoán
        feature_cols: danh sách tên cột features
        target_col: danh sách tên cột target
        cat_col: tên cột category
        time_col: tên cột thời gian
        label_start_date: ngày bắt đầu cho label
        label_end_date: ngày kết thúc cho label
        return_dates: nếu True, trả về thêm mảng dates tương ứng với label_date
    
    Returns:
        Nếu return_dates=False: (X, y, cats)
        Nếu return_dates=True: (X, y, cats, dates)
    """

    X, y, cats, dates = [], [], [], []

    for cat, g in df.groupby(cat_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)

        X_data = g[feature_cols].values
        Q = g[target_col].values.squeeze()
        time_vals = g[time_col].values

        for i in range(len(g) - input_size - horizon + 1):

            label_date = time_vals[i + input_size]

            # rolling-origin / test split
            # Keep windows where label_date (start of prediction) >= label_start_date
            # AND prediction_end_date (label_date + horizon - 1) <= label_end_date
            if label_start_date and label_date < label_start_date:
                continue
            if label_end_date and (label_date + horizon - 1) > label_end_date:
                continue

            # input window
            X_seq = X_data[i:i + input_size]

            # future label
            y_future = Q[i + input_size: i + input_size + horizon]

            # collect
            X.append(X_seq)
            y.append(y_future)
            cats.append(cat)
            dates.append(label_date)

    X = np.array(X)
    y = np.array(y)
    cats = np.array(cats)
    dates = np.array(dates) if return_dates else None
    
    if return_dates:
        return X, y, cats, dates
    else:
        return X, y, cats


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


def add_weekday_volume_tier_features(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    weekday_volume_tier_col: str = "weekday_volume_tier",
    is_high_volume_weekday_col: str = "is_high_volume_weekday"
) -> pd.DataFrame:
    """
    Add weekday volume tier features to capture weekly demand patterns.
    
    Based on observed patterns:
    - Wednesday (day_of_week=2) and Friday (day_of_week=4) have higher volume
    - Tuesday (day_of_week=1) and Thursday (day_of_week=3) have lower volume
    - Among high-volume days, Friday is lower than Wednesday
    
    Creates:
    - weekday_volume_tier: Numeric tier (2=Wednesday highest, 1=Friday high, 
      0=Tuesday/Thursday low, -1=Monday/Saturday/Sunday neutral)
    - is_high_volume_weekday: Binary (1 for Wednesday/Friday, 0 otherwise)
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        weekday_volume_tier_col: Name for weekday_volume_tier column.
        is_high_volume_weekday_col: Name for is_high_volume_weekday column.
    
    Returns:
        DataFrame with added weekday volume tier features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract day of week (0=Monday, 6=Sunday)
    day_of_week = df[time_col].dt.dayofweek
    
    # Create weekday_volume_tier feature
    # 2 = Wednesday (highest volume)
    # 1 = Friday (high volume, but lower than Wednesday)
    # 0 = Tuesday, Thursday (low volume)
    # -1 = Saturday (lower than Tuesday/Thursday), Monday, Sunday (neutral/other)
    df[weekday_volume_tier_col] = -1  # Default for Monday, Saturday, Sunday
    df.loc[day_of_week == 2, weekday_volume_tier_col] = 2  # Wednesday (highest)
    df.loc[day_of_week == 4, weekday_volume_tier_col] = 1  # Friday (high)
    df.loc[day_of_week == 1, weekday_volume_tier_col] = 0  # Tuesday (low)
    df.loc[day_of_week == 3, weekday_volume_tier_col] = 0  # Thursday (low)
    
    # Create binary indicator for high volume weekdays (Wednesday and Friday)
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
    df[cat_id_col] = df[cat_id_col].astype(int)
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
        'lunar_month', 'lunar_day',
        # Lunar cyclical encodings and Tet countdown should also persist after aggregation
        'lunar_month_sin', 'lunar_month_cos',
        'lunar_day_sin', 'lunar_day_cos',
        'days_to_tet',
        # EOM (End-of-Month) surge features
        'is_EOM', 'days_until_month_end',
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


# def apply_sunday_to_monday_carryover(
#     df: pd.DataFrame,
#     time_col: str = "ACTUALSHIPDATE",
#     cat_col: str = "CATEGORY",
#     target_col: str = "Total CBM",
#     actual_col: Optional[str] = None
# ) -> pd.DataFrame:
#     """
#     Apply Sunday-to-Monday demand carryover to capture backlog accumulation.
    
#     Logic: Sundays are non-operational (Total CBM = 0). This function implements
#     Sunday-to-Monday demand carryover where Monday's Target = (Actual Monday + Actual Sunday).
    
#     Purpose: Capture backlog accumulation and eliminate misleading zero-demand
#     patterns in the time-series that occur when Sunday demand is deferred to Monday.
    
#     Args:
#         df: DataFrame with daily aggregated data (one row per date per category).
#         time_col: Name of time column (should be datetime or date).
#         cat_col: Name of category column.
#         target_col: Name of target column to adjust (e.g., "Total CBM").
#         actual_col: Optional name of actual column to use for carryover calculation.
#                    If None, uses target_col for both Sunday and Monday values.
    
#     Returns:
#         DataFrame with adjusted target values where Monday's target includes Sunday's demand.
    
#     Note:
#         - This function modifies the target_col in-place for Monday rows.
#         - Sunday rows remain unchanged (they keep their original values, typically 0).
#         - The adjustment is applied per category independently.
#     """
#     df = df.copy()
    
#     # Ensure time column is datetime
#     if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
#         df[time_col] = pd.to_datetime(df[time_col])
    
#     # Use target_col for actual values if actual_col not specified
#     if actual_col is None:
#         actual_col = target_col
    
#     # Ensure both columns exist
#     if target_col not in df.columns:
#         raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
#     if actual_col not in df.columns:
#         raise ValueError(f"Actual column '{actual_col}' not found in DataFrame.")
    
#     # Extract day of week (0=Monday, 6=Sunday)
#     df['_day_of_week'] = df[time_col].dt.dayofweek
    
#     # Sort by category and date to ensure proper ordering
#     df = df.sort_values([cat_col, time_col]).reset_index(drop=True)
    
#     # For each category, apply carryover logic
#     def apply_carryover_per_category(group):
#         """Apply Sunday-to-Monday carryover for a single category."""
#         group[cat_col] = group.name
#         group = group.copy()
        
#         # Identify Mondays (day_of_week == 0)
#         monday_mask = group['_day_of_week'] == 0
#         monday_indices = group.index[monday_mask]
        
#         # For each Monday, find the previous Sunday and add its value
#         for monday_idx in monday_indices:
#             # Find position of Monday in the group
#             monday_pos = group.index.get_loc(monday_idx)
            
#             # Look backwards for the previous Sunday (day_of_week == 6)
#             if monday_pos > 0:
#                 prev_days = group.iloc[:monday_pos]
#                 sunday_in_prev = prev_days[prev_days['_day_of_week'] == 6]
                
#                 if len(sunday_in_prev) > 0:
#                     # Get the most recent Sunday before this Monday
#                     sunday_idx = sunday_in_prev.index[-1]
#                     sunday_value = group.loc[sunday_idx, actual_col]
                    
#                     # Apply carryover: Monday's target = Monday actual + Sunday actual
#                     if pd.notna(sunday_value):
#                         monday_actual = group.loc[monday_idx, actual_col]
#                         group.loc[monday_idx, target_col] = monday_actual + sunday_value
        
#         return group
    
#     # Apply carryover per category
#     df = df.groupby(cat_col, group_keys=False).apply(apply_carryover_per_category).reset_index(drop=False)
#     # Clean up temporary columns
#     df = df.drop(columns=['_day_of_week'])
    
#     return df


# def moving_average_forecast_by_category(
#     df: pd.DataFrame,
#     cat_col: str = "CATEGORY",
#     time_col: str = "ACTUALSHIPDATE",
#     target_col: str = "QTY",
#     window: int = 7,
# ) -> pd.DataFrame:
#     """
#     Simple heuristic forecaster: category-wise moving average baseline.

#     This is intended for *minor* / low-volume categories (e.g., POSM, OTHER)
#     whose noisy behavior should not be backpropagated through the LSTM.

#     The function computes, for each (category, date), a trailing moving average
#     of the target over the previous `window` days and stores it in a new column
#     `<target_col>_MA{window}`. This can be used as a lightweight statistical
#     head for minor entities during inference.
#     """
#     df = df.copy()

#     if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
#         df[time_col] = pd.to_datetime(df[time_col])

#     df = df.sort_values([cat_col, time_col])
#     ma_col = f"{target_col}_MA{window}"

#     # If operational anomaly flags are available, exclude those days from the
#     # baseline trend calculation. This prevents unexplained zero-volume gaps
#     # from biasing the moving-average forecast for minor categories.
#     anomaly_flag_col = "is_operational_anomaly"

#     if anomaly_flag_col in df.columns:
#         def masked_ma(group: pd.Series, mask: pd.Series) -> pd.Series:
#             # Compute rolling mean over non-anomalous days only.
#             # We use a masked series where anomalies are dropped from the
#             # rolling window rather than set to zero.
#             values = group.copy()
#             values_masked = values.where(~mask, np.nan)
#             return values_masked.rolling(window=window, min_periods=1).mean()

#         df[ma_col] = (
#             df.groupby(cat_col, group_keys=False).apply(
#                 lambda g: masked_ma(
#                     g[target_col],
#                     g[anomaly_flag_col].astype(bool)
#                 )
#             )
#         )
#     else:
#         df[ma_col] = (
#             df.groupby(cat_col)[target_col]
#             .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
#         )

#     return df

def add_feature(data : pd.DataFrame, time_col : str, cat_col : str, target_col : str) -> pd.DataFrame:
    """
    Add comprehensive feature set to the DataFrame for time-series forecasting.

    Args:
        data: Input DataFrame with raw data.
        time_col: Name of the time column (should be datetime).
        cat_col: Name of the category column.
        target_col: Name of the target column (e.g., "Total CBM").

    Returns:
        DataFrame enriched with engineered features.
    """    

    data = data.copy()

    # data = add_temporal_features(
    #     data,
    #     time_col=time_col,
    #     month_sin_col="month_sin",
    #     month_cos_col="month_cos",
    #     dayofmonth_sin_col="dayofmonth_sin",
    #     dayofmonth_cos_col="dayofmonth_cos"
    # )
    
    # # Feature engineering: Add weekend features
    # print("  - Adding weekend features (is_weekend, day_of_week)...")
    # data = add_weekend_features(
    #     data,
    #     time_col=time_col,
    #     is_weekend_col="is_weekend",
    #     day_of_week_col="day_of_week"
    # )
    
    # # Feature engineering: Add cyclical day-of-week encoding (sin/cos)
    # print("  - Adding cyclical day-of-week features (day_of_week_sin, day_of_week_cos)...")
    # data = add_day_of_week_cyclical_features(
    #     data,
    #     time_col=time_col,
    #     day_of_week_sin_col="day_of_week_sin",
    #     day_of_week_cos_col="day_of_week_cos"
    # )
    
    # # Feature engineering: Add weekday volume tier features
    # print("  - Adding weekday volume tier features (weekday_volume_tier, is_high_volume_weekday)...")
    # data = add_weekday_volume_tier_features(
    #     data,
    #     time_col=time_col,
    #     weekday_volume_tier_col="weekday_volume_tier",
    #     is_high_volume_weekday_col="is_high_volume_weekday"
    # )
    
    # # Feature engineering: Add End-of-Month (EOM) surge features
    # print("  - Adding EOM features (is_EOM, days_until_month_end)...")
    # data = add_eom_features(
    #     data,
    #     time_col=time_col,
    #     is_eom_col="is_EOM",
    #     days_until_month_end_col="days_until_month_end",
    #     eom_window_days=3
    # )
    
    # # Feature engineering: Add lunar calendar features
    # print("  - Adding lunar calendar features (lunar_month, lunar_day)...")
    # data = add_lunar_calendar_features(
    #     data,
    #     time_col=time_col,
    #     lunar_month_col="lunar_month",
    #     lunar_day_col="lunar_day"
    # )

    # # Feature engineering: Lunar cyclical encodings (sine/cosine)
    # print("  - Adding lunar cyclical features (sine/cosine)...")
    # data = add_lunar_cyclical_features(
    #     data,
    #     lunar_month_col="lunar_month",
    #     lunar_day_col="lunar_day",
    #     lunar_month_sin_col="lunar_month_sin",
    #     lunar_month_cos_col="lunar_month_cos",
    #     lunar_day_sin_col="lunar_day_sin",
    #     lunar_day_cos_col="lunar_day_cos",
    # )

    # # Feature engineering: Vietnamese holiday features (with days_since_holiday)
    # print("  - Adding Vietnamese holiday features...")
    # data = add_holiday_features_vietnam(
    #     data,
    #     time_col=time_col,
    #     holiday_indicator_col="holiday_indicator",
    #     days_until_holiday_col="days_until_next_holiday",
    #     days_since_holiday_col="days_since_holiday"
    # )

    # # Feature engineering: continuous countdown to Tet (lunar event)
    # print("  - Adding Tet countdown feature (days_to_tet)...")
    # data = add_days_to_tet_feature(
    #     data,
    #     time_col=time_col,
    #     days_to_tet_col="days_to_tet",
    # )
    
    # # Daily aggregation: Group by date and category, sum QTY
    # # This ensures the model learns daily demand patterns, not individual transaction sizes
    # print("\n[5.5/8] Aggregating to daily totals by category...")
    # samples_before_agg = len(data)
    # data = aggregate_daily(
    #     data,
    #     time_col=time_col,
    #     cat_col=cat_col,
    #     target_col=target_col
    # )
    # samples_after_agg = len(data)
    # print(f"  - Samples before aggregation: {samples_before_agg}")
    # print(f"  - Samples after aggregation: {samples_after_agg} (one row per date per category)")
    
    
    # # Feature engineering: Add CBM/QTY density features, including last-year prior
    # print("  - Adding CBM density features (cbm_per_qty, cbm_per_qty_last_year)...")
    
    # data = add_cbm_density_features(
    #     data,
    #     cbm_col=target_col,   # e.g., "Total CBM"
    #     qty_col="Total QTY",
    #     time_col=time_col,
    #     cat_col=cat_col,
    #     density_col="cbm_per_qty",
    #     density_last_year_col="cbm_per_qty_last_year",
    # )

    # # Feature engineering: Add rolling means and momentum features (after aggregation)
    # print("  - Adding rolling mean and momentum features (7d, 30d, momentum)...")
    # data = add_rolling_and_momentum_features(
    #     data,
    #     target_col=target_col,
    #     time_col=time_col,
    #     cat_col=cat_col,
    #     rolling_7_col="rolling_mean_7d",
    #     rolling_30_col="rolling_mean_30d",
    #     momentum_col="momentum_3d_vs_14d"
    # )

    return data

