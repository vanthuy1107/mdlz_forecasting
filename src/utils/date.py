from config import load_holidays
from datetime import date
from typing import List, Tuple
##############################################################################
# Holiday and Lunar Calendar Utilities
###############################################################################

# NOTE:
# We keep a single source of truth for Vietnamese holidays (including Tet)
# so that both the discrete holiday indicators and the continuous
# "days-to-lunar-event" features stay perfectly aligned.
# Holidays are now loaded from config/holidays.yaml for easier maintenance.
VIETNAM_HOLIDAYS_BY_YEAR = load_holidays()


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