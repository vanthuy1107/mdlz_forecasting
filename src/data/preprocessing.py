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


def slicing_window(
    df,
    input_size,
    horizon,
    feature_cols,
    target_col,
    baseline_col,
    brand_col,
    time_col,
    off_holiday_col=None,
    label_start_date=None,
    label_end_date=None,
    return_dates=False,
    return_off_holiday=False,
):
    X, y, baselines, brands, dates, off_flags = [], [], [], [], [], []

    for brand, g in df.groupby(brand_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)

        X_data = g[feature_cols].values
        y_data = g[target_col].values.squeeze()
        b_data = g[baseline_col].values
        time_vals = g[time_col].values

        for i in range(len(g) - input_size - horizon + 1):
            label_date = time_vals[i + input_size]

            if label_start_date and label_date < label_start_date:
                continue
            if label_end_date and label_date >= label_end_date:
                continue

            X.append(X_data[i : i + input_size])
            y.append(y_data[i + input_size : i + input_size + horizon])
            baselines.append(b_data[i + input_size : i + input_size + horizon])
            brands.append(brand)
            dates.append(label_date)

            if off_holiday_col is not None:
                off_flags.append(
                    g.loc[i + input_size, off_holiday_col]
                )

    X = np.array(X)
    y = np.array(y)
    baselines = np.array(baselines)
    brands = np.array(brands)
    dates = np.array(dates) if return_dates else None
    off_flags = np.array(off_flags) if return_off_holiday else None

    if return_dates and return_off_holiday:
        return X, y, baselines, brands, dates, off_flags
    elif return_dates:
        return X, y, baselines, brands, dates
    elif return_off_holiday:
        return X, y, baselines, brands, off_flags
    else:
        return X, y, baselines, brands



def add_holiday_features(
    df: pd.DataFrame,
    time_col: str = "DATE",
    off_holiday_col: str = "is_off_holiday",
    other_holiday_col: str = "is_other_holiday",
    holidays_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add Vietnam holiday flags to DataFrame.

    Creates:
    - is_off_holiday: 1 if date is an official day-off holiday
    - is_other_holiday: 1 if date is an 'other' holiday

    Args:
        df: DataFrame with time column.
        time_col: Name of time column (datetime or date).
        off_holiday_col: Column name for off-holiday flag.
        other_holiday_col: Column name for other-holiday flag.
        holidays_path: Optional path to holidays.yaml.

    Returns:
        DataFrame with added holiday indicator columns.
    """
    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Load holidays
    holidays = load_holidays(holidays_path)

    # Flatten holiday dates into sets
    off_holidays = set()
    other_holidays = set()

    for year_data in holidays.values():
        off_holidays.update(year_data.get("off", []))
        other_holidays.update(year_data.get("other", []))

    # Convert dataframe dates to date (not datetime)
    dates = df[time_col].dt.date

    # Binary flags
    df[off_holiday_col] = dates.isin(off_holidays).astype(int)
    df[other_holiday_col] = dates.isin(other_holidays).astype(int)

    return df


def add_temporal_features(
    df: pd.DataFrame,
    time_col: str = "DATE",
    month_sin_col: str = "month_sin",
    month_cos_col: str = "month_cos",
    dayofmonth_sin_col: str = "dayofmonth_sin",
    dayofmonth_cos_col: str = "dayofmonth_cos"
) -> pd.DataFrame:
    """
    Add cyclical temporal features using true month lengths.
    """
    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    month = df[time_col].dt.month
    day = df[time_col].dt.day
    days_in_month = df[time_col].dt.days_in_month

    # Month cycle (fixed 12)
    df[month_sin_col] = np.sin(2 * np.pi * (month - 1) / 12)
    df[month_cos_col] = np.cos(2 * np.pi * (month - 1) / 12)

    # Day-of-month cycle (true month length)
    phase = (day - 1) / days_in_month
    df[dayofmonth_sin_col] = np.sin(2 * np.pi * phase)
    df[dayofmonth_cos_col] = np.cos(2 * np.pi * phase)

    return df


def add_day_of_week_cyclical_features(
    df: pd.DataFrame,
    time_col: str = "DATE",
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
    time_col: str = "DATE",
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
    time_col: str = "DATE",
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
    time_col: str = "DATE",
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
    time_col: str = "DATE",
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


def add_baseline(
    df: pd.DataFrame,
    target_col: str = "CUBE_OUT",
    time_col: str = "DATE",
    brand_col: str = "BRAND",
    baseline_col: str = "rolling_mean_7d"
) -> pd.DataFrame:
    """
    Add rolling mean and momentum features to reduce model inertia.
    
    Creates:
    - rolling_mean_7d: 7-day rolling average of CBM
    
    These features help the model see "pace" rather than just yesterday's value.
    
    Args:
        df: DataFrame with QTY and time columns (must be sorted by time).
        target_col: Name of target column (e.g., "CBM").
        time_col: Name of time column.
        brand_col: Name of BRAND column (rolling calculated per BRAND).
        baseline_col: Name for baseline column.
    
    Returns:
        DataFrame with added baseline features.
    """
    df = df.copy()
    
    # Ensure time column is datetime and sort
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    df = df.sort_values([brand_col, time_col]).reset_index(drop=True)
    
    # Initialize new columns
    df[baseline_col] = np.nan
    
    # Calculate rolling features per BRAND
    for brand, group in df.groupby(brand_col, sort=False):
        brand_mask = df[brand_col] == brand
        brand_indices = df[brand_mask].index
        
        # Calculate rolling means
        rolling_7 = group[target_col].rolling(window=7, min_periods=1).mean().shift(1)
        
        # Assign values back to main dataframe
        df.loc[brand_indices, baseline_col] = rolling_7.values
    
    # Fill any remaining NaN with forward fill then backward fill
    df[baseline_col] = df.groupby(brand_col)[baseline_col].ffill().fillna(0)

    return df


def add_days_to_tet_feature(
    df: pd.DataFrame,
    time_col: str = "DATE",
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


def encode_brands(df: pd.DataFrame, brand_col: str = "BRAND") -> Tuple[pd.DataFrame, dict, int]:
    """
    Encode brand column to integer IDs.
    
    Args:
        df: DataFrame containing BRAND column.
        brand_col: Name of BRAND column.
    
    Returns:
        Tuple of (df_with_brand_id, brand2id_dict, num_brands)
    """
    categories = sorted(df[brand_col].unique())
    brand2id = {brand: i for i, brand in enumerate(categories)}
    brand_id_col = f"{brand_col}_ID"
    df = df.copy()
    df[brand_id_col] = df[brand_col].map(brand2id)
    df[brand_id_col] = df[brand_id_col].astype(int)
    num_categories = len(categories)

    return df, brand2id, num_categories


def add_cbm_density_features(
    df: pd.DataFrame,
    cbm_col: str = "Total CBM",
    qty_col: str = "Total QTY",
    time_col: str = "DATE",
    brand_col: str = "BRAND",
    density_col: str = "cbm_per_qty",
    density_last_year_col: str = "cbm_per_qty_last_year",
    eps: float = 1e-3,
) -> pd.DataFrame:
    """
    Add volume–quantity density features, including a "last-year" structural prior.

    Creates:
    - cbm_per_qty:      Current-day density (CBM / QTY) per (date, BRAND)
    - cbm_per_qty_last_year: Density observed on the same calendar date one year earlier
                              for the same BRAND.

    This gives the model an explicit prior about how "bulky" shipments were in the
    same seasonal window last year, which is especially useful around Tet / peak seasons.

    Assumptions:
    - df is already aggregated at daily granularity per BRAND.
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

    # Build a mapping from (BRAND, date_minus_1y) -> density.
    # We construct an auxiliary dataframe where we shift dates by +1 year,
    # then left-join back on (BRAND, DATE).
    aux = df[[time_col, brand_col, density_col]].copy()
    aux[time_col] = aux[time_col] + pd.DateOffset(years=1)
    aux = aux.rename(columns={density_col: density_last_year_col})

    # Merge back to align each row with the density from exactly one year before
    df = df.merge(
        aux[[time_col, brand_col, density_last_year_col]],
        on=[time_col, brand_col],
        how="left",
    )

    # For dates where we don't have data from a full prior year (e.g., the first year),
    # fall back to a stable per-BRAND statistic to avoid NaNs.
    if df[density_last_year_col].isna().any():
        # BRAND-level median density as a robust fallback
        cat_median = (
            df.groupby(brand_col)[density_col]
            .transform("median")
            .fillna(df[density_col].median())
        )
        df[density_last_year_col] = df[density_last_year_col].fillna(cat_median)

    return df



def add_features(data : pd.DataFrame, time_col : str, brand_col : str, target_col : str) -> pd.DataFrame:
    """
    Add comprehensive feature set to the DataFrame for time-series forecasting.

    Args:
        data: Input DataFrame with raw data.
        time_col: Name of the time column (should be datetime).
        brand_col: Name of the BRAND column.
        target_col: Name of the target column (e.g., "Total CBM").

    Returns:
        DataFrame enriched with engineered features.
    """    

    data = add_temporal_features(
        data,
        time_col=time_col,
        month_sin_col="month_sin",
        month_cos_col="month_cos",
        dayofmonth_sin_col="dayofmonth_sin",
        dayofmonth_cos_col="dayofmonth_cos"
    )
    
    # Feature engineering: Add weekend features
    print("  - Adding weekend features (is_weekend, day_of_week)...")
    data = add_weekend_features(
        data,
        time_col=time_col,
        is_weekend_col="is_weekend",
        day_of_week_col="dow"
    )
    
    # Feature engineering: Add cyclical day-of-week encoding (sin/cos)
    print("  - Adding cyclical day-of-week features (day_of_week_sin, day_of_week_cos)...")
    data = add_day_of_week_cyclical_features(
        data,
        time_col=time_col,
        day_of_week_sin_col="dow_sin",
        day_of_week_cos_col="dow_cos"
    )
    
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

    # Feature engineering: Vietnamese holiday features (with days_since_holiday)
    print("  - Adding Vietnamese holiday features...")
    data = add_holiday_features(
        data,
        time_col=time_col,
        off_holiday_col="is_off_holiday",
        other_holiday_col="is_other_holiday",
        holidays_path=None
    )

    # # Feature engineering: continuous countdown to Tet (lunar event)
    # print("  - Adding Tet countdown feature (days_to_tet)...")
    # data = add_days_to_tet_feature(
    #     data,
    #     time_col=time_col,
    #     days_to_tet_col="days_to_tet",
    # )
    
    # # Feature engineering: Add CBM/QTY density features, including last-year prior
    # print("  - Adding CBM density features (cbm_per_qty, cbm_per_qty_last_year)...")
    
    # data = add_cbm_density_features(
    #     data,
    #     cbm_col=target_col,   # e.g., "Total CBM"
    #     qty_col="Total QTY",
    #     time_col=time_col,
    #     brand_col=brand_col,
    #     density_col="cbm_per_qty",
    #     density_last_year_col="cbm_per_qty_last_year",
    # )

    # Feature engineering: Add Baseline (Rolling mean 7d)
    print("  - Adding baseline rolling mean 7d...")
    data = add_baseline(
        data,
        target_col=target_col,
        time_col=time_col,
        brand_col=brand_col,
        baseline_col="baseline",
    )

    # Add residual
    data["residual"] = data[target_col] - data["baseline"]
    target_col = "residual"

    return data

