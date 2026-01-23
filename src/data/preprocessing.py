"""Data preprocessing and window slicing utilities."""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
import pickle


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
    # NOTE: These dates must stay in sync with the prediction/training scripts.
    vietnam_holidays = {
        2023: {
            "new_year": [
                date(2023, 1, 1),
                date(2023, 1, 2),
            ],
            "tet": [
                date(2023, 1, 20),
                date(2023, 1, 21),
                date(2023, 1, 22),
                date(2023, 1, 23),
                date(2023, 1, 24),
                date(2023, 1, 25),
                date(2023, 1, 26),
            ],
            "hung_kings_reunification_labor": [
                date(2023, 4, 29),
                date(2023, 4, 30),
                date(2023, 5, 1),
                date(2023, 5, 2),
                date(2023, 5, 3),
            ],
            "independence_day": [
                date(2023, 9, 1),
                date(2023, 9, 2),
                date(2023, 9, 3),
                date(2023, 9, 4),
            ],
        },
        2024: {
            "new_year": [
                date(2024, 1, 1),
            ],
            "tet": [
                date(2024, 2, 8),
                date(2024, 2, 9),
                date(2024, 2, 10),
                date(2024, 2, 11),
                date(2024, 2, 12),
                date(2024, 2, 13),
                date(2024, 2, 14),
            ],
            "hung_kings": [
                date(2024, 4, 18),
            ],
            "reunification_labor": [
                date(2024, 4, 30),
                date(2024, 5, 1),
            ],
            "independence_day": [
                date(2024, 8, 31),
                date(2024, 9, 1),
                date(2024, 9, 2),
                date(2024, 9, 3),
            ],
        },
        2025: {
            "new_year": [
                date(2025, 1, 1),
            ],
            "tet": [
                date(2025, 1, 25),
                date(2025, 1, 26),
                date(2025, 1, 27),
                date(2025, 1, 28),
                date(2025, 1, 29),
                date(2025, 1, 30),
                date(2025, 1, 31),
                date(2025, 2, 1),
                date(2025, 2, 2),
            ],
            "hung_kings": [
                date(2025, 4, 7),
            ],
            "reunification_labor": [
                date(2025, 4, 30),
                date(2025, 5, 1),
            ],
            "independence_day": [
                date(2025, 8, 30),
                date(2025, 8, 31),
                date(2025, 9, 1),
                date(2025, 9, 2),
            ],
        },
        2026: {
            "new_year": [
                date(2026, 1, 1),
            ],
            "tet": [
                date(2026, 2, 14),
                date(2026, 2, 15),
                date(2026, 2, 16),
                date(2026, 2, 17),
                date(2026, 2, 18),
                date(2026, 2, 19),
                date(2026, 2, 20),
                date(2026, 2, 21),
                date(2026, 2, 22),
            ],
            "hung_kings": [
                date(2026, 4, 26),
                date(2026, 4, 27),
            ],
            "reunification_labor": [
                date(2026, 4, 30),
                date(2026, 5, 1),
            ],
            "independence_day": [
                date(2026, 9, 1),
                date(2026, 9, 2),
            ],
        },
    }
    
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


def fit_scaler(
    train_data: pd.DataFrame,
    target_col: str = "QTY"
) -> StandardScaler:
    """
    Fit a StandardScaler on training data QTY values.
    
    Args:
        train_data: Training DataFrame.
        target_col: Name of target column to scale.
    
    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_data[[target_col]])
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

    df[ma_col] = (
        df.groupby(cat_col)[target_col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )

    return df

