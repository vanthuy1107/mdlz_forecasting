"""
MOONCAKE Prediction Diagnostic Script

This script helps diagnose issues with MOONCAKE predictions by checking:
1. Model retraining status
2. Historical data availability
3. YoY feature computation
4. Feature columns configuration
"""
import os
import pandas as pd
import yaml
from pathlib import Path
from datetime import date, datetime
import sys

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import load_config
from src.data.preprocessing import add_year_over_year_volume_features
from src.predict.predictor import _solar_to_lunar_date as solar_to_lunar_date


def check_model_retraining():
    """Step 1: Verify Model Retraining Status"""
    print("=" * 80)
    print("STEP 1: Verify Model Retraining Status")
    print("=" * 80)
    
    model_path = Path("outputs/dow-anchored/MOONCAKE/models/best_model_dow-anchored.pth")
    improvements_doc = Path("MOONCAKE_PREDICTION_IMPROVEMENTS.md")
    
    if not model_path.exists():
        print("❌ ERROR: Model file not found!")
        print(f"   Expected: {model_path}")
        print("   Action: Run 'python mvp_train.py' to train the model")
        return False
    
    model_time = datetime.fromtimestamp(model_path.stat().st_mtime)
    
    if improvements_doc.exists():
        doc_time = datetime.fromtimestamp(improvements_doc.stat().st_mtime)
        if model_time < doc_time:
            print("⚠️  WARNING: Model is older than improvements document")
            print(f"   Model timestamp: {model_time}")
            print(f"   Improvements doc timestamp: {doc_time}")
            print("   Action: Retrain the model with 'python mvp_train.py'")
            return False
        else:
            print("✅ Model is newer than improvements document")
            print(f"   Model timestamp: {model_time}")
            print(f"   Improvements doc timestamp: {doc_time}")
    else:
        print(f"✅ Model exists: {model_path}")
        print(f"   Model timestamp: {model_time}")
    
    # Check for scaler and metadata
    scaler_path = Path("outputs/dow-anchored/MOONCAKE/models/scaler_dow-anchored.pkl")
    metadata_path = Path("outputs/dow-anchored/MOONCAKE/models/metadata_dow-anchored.json")
    
    if scaler_path.exists():
        print(f"✅ Scaler found: {scaler_path}")
    else:
        print(f"⚠️  Scaler not found: {scaler_path}")
    
    if metadata_path.exists():
        print(f"✅ Metadata found: {metadata_path}")
        # Try to load and check feature columns
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if 'feature_cols' in metadata:
                feature_cols = metadata['feature_cols']
                has_yoy = 'cbm_last_year' in feature_cols and 'cbm_2_years_ago' in feature_cols
                if has_yoy:
                    print(f"✅ YoY features found in model metadata: cbm_last_year, cbm_2_years_ago")
                else:
                    print(f"⚠️  YoY features NOT in model metadata")
                    print(f"   Available features: {feature_cols[:10]}...")
        except Exception as e:
            print(f"⚠️  Could not read metadata: {e}")
    else:
        print(f"⚠️  Metadata not found: {metadata_path}")
    
    return True


def check_historical_data():
    """Step 2: Check Historical Data Availability"""
    print("\n" + "=" * 80)
    print("STEP 2: Check Historical Data Availability")
    print("=" * 80)
    
    # Try to find historical data files
    data_paths = [
        Path("dataset/data_cat/data_2024.csv"),
        Path("dataset/train/data_2024.csv"),
        Path("dataset/data_2024.csv"),
        Path("data/data_2024.csv"),
    ]
    
    data_path = None
    for path in data_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print("❌ ERROR: Historical data file not found!")
        print("   Checked paths:")
        for path in data_paths:
            print(f"     - {path}")
        print("   Action: Ensure 2024 data file exists")
        return None
    
    print(f"✅ Found historical data: {data_path}")
    
    try:
        # Load data
        print("   Loading data...")
        ref_data = pd.read_csv(data_path)
        
        # Check if MOONCAKE category exists
        if 'CATEGORY' not in ref_data.columns:
            print("❌ ERROR: 'CATEGORY' column not found in data")
            return None
        
        mooncake_data = ref_data[ref_data['CATEGORY'] == 'MOONCAKE'].copy()
        
        if len(mooncake_data) == 0:
            print("❌ ERROR: No MOONCAKE data found in historical file")
            print(f"   Available categories: {ref_data['CATEGORY'].unique()}")
            return None
        
        print(f"✅ Found {len(mooncake_data)} MOONCAKE records")
        
        # Convert date column
        time_col = 'ACTUALSHIPDATE'
        if time_col not in mooncake_data.columns:
            print(f"❌ ERROR: '{time_col}' column not found")
            return None
        
        mooncake_data['date'] = pd.to_datetime(mooncake_data[time_col]).dt.date
        
        # Check target column
        target_col = 'Total CBM'
        if target_col not in mooncake_data.columns:
            print(f"⚠️  WARNING: '{target_col}' column not found")
            print(f"   Available columns: {mooncake_data.columns.tolist()}")
            target_col = None
        
        # Check August 2024
        print("\n   Checking August 2024...")
        aug_2024 = mooncake_data[
            (mooncake_data['date'] >= date(2024, 8, 1)) & 
            (mooncake_data['date'] <= date(2024, 8, 31))
        ]
        
        if len(aug_2024) == 0:
            print("   ❌ No data found for August 2024")
        else:
            print(f"   ✅ Found {len(aug_2024)} records for August 2024")
            if target_col:
                aug_total = aug_2024[target_col].sum()
                print(f"   Total CBM: {aug_total:.2f}")
                if aug_total < 1000:
                    print("   ⚠️  WARNING: Very low volume for August (expected ~6,000 CBM)")
                elif aug_total > 10000:
                    print("   ⚠️  WARNING: Very high volume for August (expected ~6,000 CBM)")
                else:
                    print("   ✅ Volume looks reasonable for August")
        
        # Check September 2024
        print("\n   Checking September 2024...")
        sep_2024 = mooncake_data[
            (mooncake_data['date'] >= date(2024, 9, 1)) & 
            (mooncake_data['date'] <= date(2024, 9, 30))
        ]
        
        if len(sep_2024) == 0:
            print("   ❌ No data found for September 2024")
        else:
            print(f"   ✅ Found {len(sep_2024)} records for September 2024")
            if target_col:
                sep_total = sep_2024[target_col].sum()
                print(f"   Total CBM: {sep_total:.2f}")
                if sep_total < 1000:
                    print("   ⚠️  WARNING: Very low volume for September (expected ~4,500 CBM)")
                elif sep_total > 10000:
                    print("   ⚠️  WARNING: Very high volume for September (expected ~4,500 CBM)")
                else:
                    print("   ✅ Volume looks reasonable for September")
        
        # Check date range
        min_date = mooncake_data['date'].min()
        max_date = mooncake_data['date'].max()
        print(f"\n   Date range: {min_date} to {max_date}")
        
        return mooncake_data
        
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_yoy_feature_computation(historical_data=None):
    """Step 3: Check YoY Feature Computation"""
    print("\n" + "=" * 80)
    print("STEP 3: Check YoY Feature Computation")
    print("=" * 80)
    
    if historical_data is None:
        print("⚠️  SKIPPED: No historical data available (run Step 2 first)")
        return None
    
    try:
        # Sample dates to test
        test_dates = [
            date(2025, 8, 15),  # Should match August 2024
            date(2025, 9, 15),  # Should match September 2024
        ]
        
        print("   Testing YoY feature computation for sample dates...")
        
        # Prepare a small test dataset
        test_data = []
        for test_date in test_dates:
            lunar_month, lunar_day = solar_to_lunar_date(test_date)
            test_data.append({
                'ACTUALSHIPDATE': pd.Timestamp(test_date),
                'CATEGORY': 'MOONCAKE',
                'Total CBM': 0.0,  # Placeholder
                'lunar_month': lunar_month,
                'lunar_day': lunar_day,
            })
        
        test_df = pd.DataFrame(test_data)
        
        # Combine with historical data
        hist_copy = historical_data.copy()
        hist_copy['ACTUALSHIPDATE'] = pd.to_datetime(hist_copy['ACTUALSHIPDATE'])
        
        # Ensure required columns exist
        if 'lunar_month' not in hist_copy.columns:
            print("   Computing lunar dates for historical data...")
            hist_copy['lunar_month'] = hist_copy['date'].apply(lambda d: solar_to_lunar_date(d)[0])
            hist_copy['lunar_day'] = hist_copy['date'].apply(lambda d: solar_to_lunar_date(d)[1])
        
        combined = pd.concat([hist_copy, test_df], ignore_index=True)
        
        # Compute YoY features
        print("   Computing YoY features with lunar matching...")
        combined_with_yoy = add_year_over_year_volume_features(
            combined.copy(),
            target_col='Total CBM',
            time_col='ACTUALSHIPDATE',
            cat_col='CATEGORY',
            use_lunar_matching=True,
            lunar_month_col='lunar_month',
            lunar_day_col='lunar_day'
        )
        
        # Check results for test dates
        print("\n   YoY Feature Results:")
        for test_date in test_dates:
            test_row = combined_with_yoy[
                pd.to_datetime(combined_with_yoy['ACTUALSHIPDATE']).dt.date == test_date
            ]
            if len(test_row) > 0:
                cbm_last_year = test_row.iloc[0].get('cbm_last_year', 0.0)
                cbm_2_years_ago = test_row.iloc[0].get('cbm_2_years_ago', 0.0)
                lunar_month = test_row.iloc[0].get('lunar_month', 0)
                lunar_day = test_row.iloc[0].get('lunar_day', 0)
                
                print(f"\n   Date: {test_date} (Lunar {lunar_month}-{lunar_day})")
                print(f"   cbm_last_year: {cbm_last_year:.2f}")
                print(f"   cbm_2_years_ago: {cbm_2_years_ago:.2f}")
                
                if cbm_last_year == 0.0:
                    print("   ⚠️  WARNING: cbm_last_year is zero - YoY lookup may have failed")
                elif cbm_last_year < 100:
                    print("   ⚠️  WARNING: cbm_last_year is very low (expected ~4,000-6,000)")
                else:
                    print("   ✅ cbm_last_year looks reasonable")
            else:
                print(f"   ❌ No data found for {test_date}")
        
        return combined_with_yoy
        
    except Exception as e:
        print(f"❌ ERROR computing YoY features: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_feature_columns():
    """Step 4: Verify Feature Columns Configuration"""
    print("\n" + "=" * 80)
    print("STEP 4: Verify Feature Columns Configuration")
    print("=" * 80)
    
    config_path = Path("config/config_MOONCAKE.yaml")
    
    if not config_path.exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        return False
    
    try:
        config = load_config(category="MOONCAKE")
        feature_cols = config.data.get('feature_cols', [])
        
        print(f"✅ Config file loaded: {config_path}")
        print(f"   Total features: {len(feature_cols)}")
        
        # Check for YoY features
        has_cbm_last_year = 'cbm_last_year' in feature_cols
        has_cbm_2_years_ago = 'cbm_2_years_ago' in feature_cols
        
        print(f"\n   YoY Features:")
        print(f"   - cbm_last_year: {'✅ Found' if has_cbm_last_year else '❌ Missing'}")
        print(f"   - cbm_2_years_ago: {'✅ Found' if has_cbm_2_years_ago else '❌ Missing'}")
        
        if not (has_cbm_last_year and has_cbm_2_years_ago):
            print("\n   ⚠️  WARNING: YoY features are missing from feature columns!")
            print("   Action: Add cbm_last_year and cbm_2_years_ago to config_MOONCAKE.yaml")
        
        # Check for other important features
        important_features = [
            'lunar_month', 'lunar_day',
            'is_active_season', 'is_golden_window', 'is_peak_loss_window',
            'rolling_mean_7d', 'rolling_mean_30d'
        ]
        
        print(f"\n   Other Important Features:")
        for feat in important_features:
            has_feat = feat in feature_cols
            status = '✅' if has_feat else '⚠️ '
            print(f"   {status} {feat}: {'Found' if has_feat else 'Missing'}")
        
        return has_cbm_last_year and has_cbm_2_years_ago
        
    except Exception as e:
        print(f"❌ ERROR loading config: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic steps"""
    print("\n" + "=" * 80)
    print("MOONCAKE PREDICTION DIAGNOSTICS")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Step 1: Model retraining
    results['model_retrained'] = check_model_retraining()
    
    # Step 2: Historical data
    historical_data = check_historical_data()
    results['historical_data_available'] = historical_data is not None
    
    # Step 3: YoY feature computation
    if historical_data is not None:
        yoy_results = check_yoy_feature_computation(historical_data)
        results['yoy_computation_works'] = yoy_results is not None
    else:
        results['yoy_computation_works'] = False
    
    # Step 4: Feature columns
    results['feature_columns_ok'] = check_feature_columns()
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    all_ok = all(results.values())
    
    for step, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {step}: {status}")
    
    if all_ok:
        print("\n✅ All checks passed! Model should be ready for prediction.")
    else:
        print("\n⚠️  Some checks failed. Please review the issues above.")
        print("\nRecommended Actions:")
        if not results['model_retrained']:
            print("  1. Retrain the model: python mvp_train.py")
        if not results['historical_data_available']:
            print("  2. Ensure historical data file exists (dataset/data_cat/data_2024.csv)")
        if not results['yoy_computation_works']:
            print("  3. Check YoY feature computation logic")
        if not results['feature_columns_ok']:
            print("  4. Add missing features to config_MOONCAKE.yaml")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
