"""Data preprocessing and window slicing utilities."""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union
from datetime import datetime, date, timedelta


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

