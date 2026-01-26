"""
Diagnostic script to check why year-over-year features (cbm_last_year, cbm_2_years_ago)
are not being learned properly by the MOONCAKE model.

This script will:
1. Check if historical data is combined with prediction data before calculating YoY features
2. Verify the year-over-year feature calculation logic
3. Check if features are included in model input
4. Analyze feature values during prediction
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import sys

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import load_config
from src.data.preprocessing import add_year_over_year_volume_features

def check_yoy_feature_calculation():
    """Check how year-over-year features are calculated."""
    print("=" * 80)
    print("YEAR-OVER-YEAR FEATURE DIAGNOSTIC")
    print("=" * 80)
    
    # Load config
    config = load_config()
    data_config = config.data
    time_col = data_config['time_col']
    cat_col = data_config['cat_col']
    target_col = data_config['target_col']
    
    # Check what years are available in training data
    data_dir = Path(data_config['data_dir'])
    print(f"\n1. Checking available data files in {data_dir}:")
    available_years = []
    for year in [2022, 2023, 2024, 2025]:
        file_path = data_dir / f"data_{year}.csv"
        if file_path.exists():
            available_years.append(year)
            print(f"   ✓ Found: data_{year}.csv")
        else:
            print(f"   ✗ Missing: data_{year}.csv")
    
    if not available_years:
        print("   ERROR: No data files found!")
        return
    
    # Load a sample of historical data (2023, 2024) for MOONCAKE
    print(f"\n2. Loading historical data for MOONCAKE...")
    historical_data_list = []
    for year in [2023, 2024]:
        file_path = data_dir / f"data_{year}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            except:
                df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
            
            # Filter to MOONCAKE
            if cat_col in df.columns:
                mooncake_data = df[df[cat_col] == 'MOONCAKE'].copy()
                if len(mooncake_data) > 0:
                    historical_data_list.append(mooncake_data)
                    print(f"   ✓ Loaded {len(mooncake_data)} MOONCAKE samples from {year}")
    
    if not historical_data_list:
        print("   ERROR: No MOONCAKE data found in historical files!")
        return
    
    # Combine historical data
    historical_combined = pd.concat(historical_data_list, ignore_index=True)
    historical_combined[time_col] = pd.to_datetime(historical_combined[time_col], dayfirst=True, errors='coerce')
    historical_combined = historical_combined.sort_values(time_col).reset_index(drop=True)
    
    # Check date range
    print(f"\n3. Historical data date range:")
    print(f"   Min: {historical_combined[time_col].min()}")
    print(f"   Max: {historical_combined[time_col].max()}")
    
    # Aggregate to daily totals
    print(f"\n4. Aggregating to daily totals...")
    daily_agg = historical_combined.groupby([time_col, cat_col])[target_col].sum().reset_index()
    daily_agg[time_col] = pd.to_datetime(daily_agg[time_col])
    daily_agg = daily_agg.sort_values(time_col).reset_index(drop=True)
    print(f"   Daily samples: {len(daily_agg)}")
    
    # Check August-September data (peak season)
    print(f"\n5. Checking peak season data (August-September):")
    for year in [2023, 2024]:
        year_data = daily_agg[
            (daily_agg[time_col].dt.year == year) &
            (daily_agg[time_col].dt.month.isin([8, 9]))
        ]
        if len(year_data) > 0:
            total_cbm = year_data[target_col].sum()
            avg_cbm = year_data[target_col].mean()
            print(f"   {year} Aug-Sep: {len(year_data)} days, Total={total_cbm:.2f} CBM, Avg={avg_cbm:.2f} CBM/day")
        else:
            print(f"   {year} Aug-Sep: No data")
    
    # Test year-over-year feature calculation
    print(f"\n6. Testing year-over-year feature calculation...")
    test_data = daily_agg.copy()
    
    # Add YoY features
    test_data_with_yoy = add_year_over_year_volume_features(
        test_data,
        target_col=target_col,
        time_col=time_col,
        cat_col=cat_col,
        yoy_1y_col="cbm_last_year",
        yoy_2y_col="cbm_2_years_ago",
    )
    
    # Check if features were added
    if "cbm_last_year" in test_data_with_yoy.columns:
        print(f"   ✓ cbm_last_year feature added")
    else:
        print(f"   ✗ cbm_last_year feature NOT added!")
        return
    
    if "cbm_2_years_ago" in test_data_with_yoy.columns:
        print(f"   ✓ cbm_2_years_ago feature added")
    else:
        print(f"   ✗ cbm_2_years_ago feature NOT added!")
        return
    
    # Check feature values for 2024 August (should have 2023 August values)
    print(f"\n7. Checking feature values for 2024 August (should reference 2023 August):")
    aug_2024 = test_data_with_yoy[
        (test_data_with_yoy[time_col].dt.year == 2024) &
        (test_data_with_yoy[time_col].dt.month == 8)
    ].copy()
    
    if len(aug_2024) > 0:
        print(f"   Found {len(aug_2024)} days in August 2024")
        
        # Check if cbm_last_year has values (should reference 2023 August)
        non_zero_last_year = (aug_2024['cbm_last_year'] > 0).sum()
        zero_last_year = (aug_2024['cbm_last_year'] == 0).sum()
        print(f"   cbm_last_year: {non_zero_last_year} non-zero, {zero_last_year} zero")
        
        if non_zero_last_year > 0:
            print(f"   ✓ Year-over-year features are populated!")
            sample = aug_2024[aug_2024['cbm_last_year'] > 0].iloc[0]
            print(f"   Sample: {sample[time_col].date()} -> cbm_last_year={sample['cbm_last_year']:.2f}, current={sample[target_col]:.2f}")
        else:
            print(f"   ✗ WARNING: All cbm_last_year values are zero!")
            print(f"   This means the function cannot find matching dates from previous year.")
            print(f"   Possible causes:")
            print(f"   - Historical data doesn't have matching dates")
            print(f"   - Date format mismatch")
            print(f"   - Category mismatch")
    else:
        print(f"   ✗ No August 2024 data found!")
    
    # Check what happens when we try to predict 2025 (future dates)
    print(f"\n8. Simulating prediction scenario (2025 dates):")
    print(f"   Creating synthetic 2025 data...")
    
    # Create a few 2025 dates
    pred_dates = pd.date_range(start='2025-08-01', end='2025-08-15', freq='D')
    pred_data = pd.DataFrame({
        time_col: pred_dates,
        cat_col: 'MOONCAKE',
        target_col: 0.0  # Placeholder - we don't know actual values yet
    })
    
    # CRITICAL TEST: Can we calculate YoY features for 2025 if we combine with historical?
    print(f"   Testing: Can we calculate YoY features for 2025 if combined with historical data?")
    
    # Combine historical + prediction data
    combined_data = pd.concat([daily_agg, pred_data], ignore_index=True)
    combined_data = combined_data.sort_values(time_col).reset_index(drop=True)
    
    # Add YoY features to combined data
    combined_with_yoy = add_year_over_year_volume_features(
        combined_data,
        target_col=target_col,
        time_col=time_col,
        cat_col=cat_col,
        yoy_1y_col="cbm_last_year",
        yoy_2y_col="cbm_2_years_ago",
    )
    
    # Check 2025 dates
    aug_2025 = combined_with_yoy[
        (combined_with_yoy[time_col].dt.year == 2025) &
        (combined_with_yoy[time_col].dt.month == 8)
    ].copy()
    
    if len(aug_2025) > 0:
        non_zero_last_year = (aug_2025['cbm_last_year'] > 0).sum()
        zero_last_year = (aug_2025['cbm_last_year'] == 0).sum()
        print(f"   ✓ Combined data approach: {non_zero_last_year} non-zero, {zero_last_year} zero")
        
        if non_zero_last_year > 0:
            print(f"   ✓ SUCCESS: Year-over-year features CAN be calculated for 2025!")
            sample = aug_2025[aug_2025['cbm_last_year'] > 0].iloc[0]
            print(f"   Sample: {sample[time_col].date()} -> cbm_last_year={sample['cbm_last_year']:.2f}")
        else:
            print(f"   ✗ FAILED: Cannot calculate YoY features for 2025 even with combined data")
    else:
        print(f"   ✗ No 2025 data found!")
    
    # Check if prediction pipeline combines data
    print(f"\n9. Checking prediction pipeline logic:")
    print(f"   Question: Does mvp_predict.py combine historical + prediction data before calling add_year_over_year_volume_features?")
    print(f"   Answer: Need to check mvp_predict.py and src/predict/prepare.py")
    print(f"   ")
    print(f"   If NOT combined:")
    print(f"   - Prediction data (2025) alone cannot calculate YoY features")
    print(f"   - cbm_last_year and cbm_2_years_ago will be 0.0 for all 2025 dates")
    print(f"   - Model will not see historical patterns!")
    
    print(f"\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    check_yoy_feature_calculation()
