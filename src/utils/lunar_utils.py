"""Lunar calendar utilities for MOONCAKE forecasting.

This module provides lunar date conversion and alignment functions to support
Lunar-Aligned Adaptive Forecasting. This is critical for seasonal products like
MOONCAKE where the peak occurs at a fixed lunar date (Mid-Autumn Festival = Lunar 08-15)
but shifts by 10-20 days annually on the Gregorian calendar.
"""
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Tuple, Optional, Dict


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
    
    # Validate input
    if solar_date is None or pd.isna(solar_date):
        return 8, 15  # Return default Mid-Autumn date
    
    # Find the closest Mid-Autumn anchor (before or after)
    try:
        year = solar_date.year
    except (AttributeError, TypeError):
        return 8, 15  # Return default Mid-Autumn date
    
    # Select appropriate anchor year
    if year not in mid_autumn_anchors:
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
    
    # Start from Lunar Month 8, Day 15
    lunar_month = 8
    lunar_day = 15
    
    # Adjust by days difference (approximate: 29.5 days per lunar month)
    if days_diff > 0:
        lunar_day += days_diff
        while lunar_day > 30:
            lunar_day -= 30
            lunar_month += 1
            if lunar_month > 12:
                lunar_month = 1
    else:
        lunar_day += days_diff
        while lunar_day < 1:
            lunar_day += 30
            lunar_month -= 1
            if lunar_month < 1:
                lunar_month = 12
    
    # Clamp to valid range
    lunar_month = max(1, min(12, lunar_month))
    lunar_day = max(1, min(30, lunar_day))
    
    return lunar_month, lunar_day


def find_gregorian_date_for_lunar_date(
    target_lunar_month: int,
    target_lunar_day: int,
    target_year: int
) -> Optional[date]:
    """
    Find the Gregorian date that corresponds to a specific lunar date in a given year.
    
    This is the inverse operation of solar_to_lunar_date. It searches for the Gregorian
    date in the target year that matches the specified lunar month and day.
    
    Args:
        target_lunar_month: Target lunar month (1-12).
        target_lunar_day: Target lunar day (1-30).
        target_year: Target Gregorian year.
    
    Returns:
        Gregorian date object, or None if not found.
    
    Example:
        >>> find_gregorian_date_for_lunar_date(8, 1, 2024)
        date(2024, 9, 3)  # Lunar 08-01 in 2024 is Sept 3
        >>> find_gregorian_date_for_lunar_date(8, 1, 2025)
        date(2025, 9, 22)  # Lunar 08-01 in 2025 is Sept 22
    """
    # Search within a reasonable range (entire year + buffer)
    start_date = date(target_year, 1, 1)
    end_date = date(target_year, 12, 31)
    
    # Binary search approach for efficiency
    current_date = start_date
    best_match = None
    min_diff = float('inf')
    
    # Linear search (could be optimized with binary search if needed)
    while current_date <= end_date:
        lunar_month, lunar_day = solar_to_lunar_date(current_date)
        
        # Exact match
        if lunar_month == target_lunar_month and lunar_day == target_lunar_day:
            return current_date
        
        # Track best match (in case exact match not found due to lunar month irregularities)
        day_diff = abs((lunar_month * 30 + lunar_day) - (target_lunar_month * 30 + target_lunar_day))
        if day_diff < min_diff:
            min_diff = day_diff
            best_match = current_date
        
        current_date += timedelta(days=1)
    
    # Return best match if exact match not found (within 2 days tolerance)
    if min_diff <= 2:
        return best_match
    
    return None


def find_lunar_aligned_date_from_previous_year(
    current_date: date,
    historical_data: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    years_back: int = 1
) -> Optional[date]:
    """
    Find the Gregorian date from N years ago that shares the same lunar date as current_date.
    
    This is the core function for Lunar-to-Lunar Mapping. It ensures that YoY features
    fetch data from the SAME LUNAR DATE in the previous year, not the same Gregorian date.
    
    For example:
    - Current: 2025-10-06 (Lunar 08-15, Mid-Autumn)
    - Returns: 2024-09-17 (also Lunar 08-15, Mid-Autumn)
    - NOT: 2024-10-06 (which would be a different lunar date)
    
    Args:
        current_date: Current prediction date (Gregorian).
        historical_data: DataFrame with historical data (to verify date exists).
        time_col: Name of time column in historical_data.
        years_back: Number of years to look back (1 = last year, 2 = 2 years ago).
    
    Returns:
        Gregorian date from N years ago with matching lunar date, or None if not found.
    """
    # Get lunar date for current date
    current_lunar_month, current_lunar_day = solar_to_lunar_date(current_date)
    
    # Target year for lookup
    target_year = current_date.year - years_back
    
    # Find Gregorian date in target year with same lunar date
    target_gregorian_date = find_gregorian_date_for_lunar_date(
        current_lunar_month,
        current_lunar_day,
        target_year
    )
    
    if target_gregorian_date is None:
        return None
    
    # Verify this date exists in historical data
    if historical_data is not None and len(historical_data) > 0:
        historical_dates = pd.to_datetime(historical_data[time_col]).dt.date
        if target_gregorian_date not in historical_dates.values:
            # Date not found in historical data, look for closest match within ±3 days
            for offset in range(1, 4):
                for sign in [-1, 1]:
                    candidate = target_gregorian_date + timedelta(days=sign * offset)
                    if candidate in historical_dates.values:
                        return candidate
            return None
    
    return target_gregorian_date


def get_lunar_aligned_yoy_lookup(
    current_date: date,
    historical_data: pd.DataFrame,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    category: str = "MOONCAKE"
) -> Tuple[float, float]:
    """
    Get year-over-year volume values using Lunar-to-Lunar Mapping.
    
    This function implements the core Lunar-Aligned YoY lookup:
    1. Convert current_date to lunar coordinates (month, day)
    2. Find the Gregorian date from 1 year ago with the SAME lunar coordinates
    3. Find the Gregorian date from 2 years ago with the SAME lunar coordinates
    4. Fetch actual CBM values from those dates
    
    This ensures the model sees the relevant historical peaks regardless of
    Gregorian calendar drift (10-20 days annually).
    
    Args:
        current_date: Current prediction date.
        historical_data: DataFrame with historical CBM data.
        target_col: Name of volume column (e.g., "Total CBM").
        time_col: Name of time column.
        cat_col: Name of category column.
        category: Category name (e.g., "MOONCAKE").
    
    Returns:
        Tuple of (cbm_last_year, cbm_2_years_ago).
        Returns (0.0, 0.0) if no matching dates found.
    
    Example:
        For 2025-10-06 (Lunar 08-15 = Mid-Autumn):
        - Finds 2024-09-17 (also Lunar 08-15)
        - Fetches cbm_last_year from 2024-09-17
        - NOT from 2024-10-06 (which would be wrong lunar date)
    """
    cbm_last_year = 0.0
    cbm_2_years_ago = 0.0
    
    # Filter to category
    if cat_col in historical_data.columns:
        cat_data = historical_data[historical_data[cat_col] == category].copy()
    else:
        cat_data = historical_data.copy()
    
    if len(cat_data) == 0:
        return cbm_last_year, cbm_2_years_ago
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(cat_data[time_col]):
        cat_data[time_col] = pd.to_datetime(cat_data[time_col])
    
    # Get lunar-aligned date from 1 year ago
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
            cbm_last_year = cat_data.loc[mask_1y, target_col].iloc[0]
            if pd.isna(cbm_last_year):
                cbm_last_year = 0.0
    
    # Get lunar-aligned date from 2 years ago
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
            cbm_2_years_ago = cat_data.loc[mask_2y, target_col].iloc[0]
            if pd.isna(cbm_2_years_ago):
                cbm_2_years_ago = 0.0
    
    return cbm_last_year, cbm_2_years_ago


def inject_lunar_aligned_warmup(
    input_window: np.ndarray,
    window_dates: list,
    historical_data: pd.DataFrame,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    category: str = "MOONCAKE",
    target_col_idx: int = 0
) -> np.ndarray:
    """
    Inject Lunar-Aligned historical patterns into a zero-filled input window.
    
    This implements the "Hidden State Warm-up" technique to prevent "Zero-locking".
    When the input window is all zeros (off-season), the RNN has no signal to
    anticipate the upcoming peak. This function injects the volume pattern from
    the corresponding Lunar window of the previous year.
    
    For example:
    - If predicting 2025-08-01 to 2025-08-30 (early peak season)
    - And the 21-day lookback window is all zeros (July = off-season)
    - Inject the volume pattern from 2024-07-20 to 2024-08-09 (same lunar dates)
    - This gives the RNN the "momentum" to recognize the peak is coming
    
    Args:
        input_window: Array of shape (window_size, n_features) with current input.
        window_dates: List of dates corresponding to input_window rows.
        historical_data: DataFrame with historical data for pattern injection.
        target_col: Name of volume column.
        time_col: Name of time column.
        cat_col: Name of category column.
        category: Category name.
        target_col_idx: Index of target column in input_window features.
    
    Returns:
        Modified input_window with injected historical patterns if needed.
    """
    # Check if window is all zeros (or near-zero)
    window_volume = input_window[:, target_col_idx]
    if np.sum(np.abs(window_volume)) > 1.0:  # Not a zero window
        return input_window
    
    # Zero window detected - inject lunar-aligned historical pattern
    print(f"  [WARM-UP] Zero input window detected. Injecting lunar-aligned historical pattern...")
    
    # Filter historical data to category
    if cat_col in historical_data.columns:
        cat_data = historical_data[historical_data[cat_col] == category].copy()
    else:
        cat_data = historical_data.copy()
    
    if len(cat_data) == 0:
        print(f"  [WARM-UP] No historical data available for {category}")
        return input_window
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(cat_data[time_col]):
        cat_data[time_col] = pd.to_datetime(cat_data[time_col])
    
    # For each date in window, find lunar-aligned date from previous year
    injected_count = 0
    for i, current_date in enumerate(window_dates):
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date).date()
        elif isinstance(current_date, pd.Timestamp):
            current_date = current_date.date()
        
        # Get lunar-aligned date from 1 year ago
        historical_date = find_lunar_aligned_date_from_previous_year(
            current_date,
            cat_data,
            time_col=time_col,
            years_back=1
        )
        
        if historical_date is not None:
            # Fetch volume from that date
            mask = cat_data[time_col].dt.date == historical_date
            if mask.any():
                historical_volume = cat_data.loc[mask, target_col].iloc[0]
                if pd.notna(historical_volume) and historical_volume > 0:
                    # Inject historical volume into input window
                    input_window[i, target_col_idx] = historical_volume
                    injected_count += 1
    
    if injected_count > 0:
        print(f"  [WARM-UP] Injected {injected_count}/{len(window_dates)} historical values")
        print(f"  [WARM-UP] Window volume range: [{np.min(window_volume):.2f}, {np.max(window_volume):.2f}]")
    else:
        print(f"  [WARM-UP] No historical values found for injection")
    
    return input_window


def compute_days_until_lunar_08_01(pred_date: date) -> int:
    """
    Compute the number of days until the next occurrence of Lunar 08-01.
    
    This is a critical temporal feature for MOONCAKE forecasting. The countdown
    acts as a "trigger" signal for the RNN to anticipate the peak season.
    
    Lunar 08-01 marks the start of the peak outbound period for MOONCAKE
    (approximately 2 weeks before Mid-Autumn Festival on Lunar 08-15).
    
    Args:
        pred_date: Prediction date (Gregorian).
    
    Returns:
        Number of days until Lunar 08-01 (0 if already on or past Lunar 08-01).
    """
    lunar_month, lunar_day = solar_to_lunar_date(pred_date)
    
    # If already at Lunar 08-01
    if lunar_month == 8 and lunar_day == 1:
        return 0
    
    # If before Lunar 08-01 in the same lunar year
    if lunar_month < 8 or (lunar_month == 8 and lunar_day < 1):
        # Calculate days remaining in current lunar month + full months until Lunar 08
        if lunar_month < 8:
            days_until = (8 - lunar_month) * 30 - lunar_day + 1
        else:
            days_until = 1 - lunar_day
        return max(0, days_until)
    
    # If past Lunar 08-01, calculate days until next year's Lunar 08-01
    # (approximately 12 - lunar_month) months + days
    days_until = (12 - lunar_month + 8) * 30 - lunar_day + 1
    return max(0, days_until)


def validate_lunar_yoy_mapping(
    test_date: date,
    historical_data: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    verbose: bool = True
) -> Dict[str, any]:
    """
    Validate that Lunar-to-Lunar YoY mapping is working correctly.
    
    This diagnostic function verifies that:
    1. Lunar date conversion is accurate
    2. Historical date lookup finds the correct matching lunar date
    3. The dates are approximately 365 days apart (±20 days for lunar drift)
    
    Args:
        test_date: Date to test (e.g., 2025-09-22 for Lunar 08-01)
        historical_data: DataFrame with historical data
        time_col: Name of time column
        verbose: Whether to print diagnostic info
    
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Get lunar date for test date
    lunar_month, lunar_day = solar_to_lunar_date(test_date)
    results['test_date'] = test_date
    results['test_lunar_date'] = f"{lunar_month:02d}-{lunar_day:02d}"
    
    # Find matching date from 1 year ago
    date_1y_ago = find_lunar_aligned_date_from_previous_year(
        test_date,
        historical_data,
        time_col=time_col,
        years_back=1
    )
    results['date_1y_ago'] = date_1y_ago
    
    if date_1y_ago:
        lunar_month_1y, lunar_day_1y = solar_to_lunar_date(date_1y_ago)
        results['date_1y_ago_lunar'] = f"{lunar_month_1y:02d}-{lunar_day_1y:02d}"
        results['days_diff_1y'] = (test_date - date_1y_ago).days
        results['lunar_match_1y'] = (lunar_month == lunar_month_1y and lunar_day == lunar_day_1y)
    
    # Find matching date from 2 years ago
    date_2y_ago = find_lunar_aligned_date_from_previous_year(
        test_date,
        historical_data,
        time_col=time_col,
        years_back=2
    )
    results['date_2y_ago'] = date_2y_ago
    
    if date_2y_ago:
        lunar_month_2y, lunar_day_2y = solar_to_lunar_date(date_2y_ago)
        results['date_2y_ago_lunar'] = f"{lunar_month_2y:02d}-{lunar_day_2y:02d}"
        results['days_diff_2y'] = (test_date - date_2y_ago).days
        results['lunar_match_2y'] = (lunar_month == lunar_month_2y and lunar_day == lunar_day_2y)
    
    if verbose:
        print(f"\n=== Lunar YoY Mapping Validation ===")
        print(f"Test Date: {test_date} (Lunar {results['test_lunar_date']})")
        print(f"\n1 Year Ago:")
        if date_1y_ago:
            print(f"  Gregorian: {date_1y_ago}")
            print(f"  Lunar: {results['date_1y_ago_lunar']}")
            print(f"  Days Difference: {results['days_diff_1y']} days")
            print(f"  Lunar Match: {'[YES]' if results['lunar_match_1y'] else '[NO]'}")
        else:
            print(f"  [NOT FOUND]")
        
        print(f"\n2 Years Ago:")
        if date_2y_ago:
            print(f"  Gregorian: {date_2y_ago}")
            print(f"  Lunar: {results['date_2y_ago_lunar']}")
            print(f"  Days Difference: {results['days_diff_2y']} days")
            print(f"  Lunar Match: {'[YES]' if results['lunar_match_2y'] else '[NO]'}")
        else:
            print(f"  [NOT FOUND]")
    
    return results
