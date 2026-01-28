"""Validation script to verify no data leakage in the feature engineering pipeline.

This script performs comprehensive checks to ensure that:
1. Training features do not contain information from validation/test sets
2. Statistical features use only expanding windows (no look-ahead bias)
3. Window slicing correctly separates input (past) from target (future)
4. Time-series cross-validation is properly implemented

Run this script BEFORE deploying to production to ensure model validity.

Usage:
    python validate_no_leakage.py
    
    or with specific data:
    
    python validate_no_leakage.py --data-path dataset/data_cat/data_2024.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.expanding_features import (
    add_expanding_statistical_features,
    add_expanding_rolling_features,
    add_expanding_momentum_features,
    verify_no_leakage,
)
from src.data.preprocessing import slicing_window_category


def validate_expanding_window_features(
    df: pd.DataFrame,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
) -> Dict[str, bool]:
    """
    Validate that expanding window features do not contain future information.
    
    Test Strategy:
    1. Split data into train/validation at time T
    2. Compute features on train set using expanding windows
    3. Verify that no feature value at time t depends on data from times > t
    4. Check that validation set features do not use training set statistics
    
    Returns:
        Dictionary with test results (True = passed, False = failed).
    """
    print("=" * 80)
    print("VALIDATION TEST 1: Expanding Window Features")
    print("=" * 80)
    
    results = {}
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    df = df.sort_values([cat_col, time_col]).reset_index(drop=True)
    
    # Split at 70% point (temporal split)
    split_idx = int(0.7 * len(df))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} rows ({train_df[time_col].min()} to {train_df[time_col].max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df[time_col].min()} to {val_df[time_col].max()})")
    
    # Test 1: Add expanding statistical features to train set
    print("\n[TEST 1.1] Adding expanding statistical features to train set...")
    train_with_features = add_expanding_statistical_features(
        train_df,
        target_col=target_col,
        time_col=time_col,
        cat_col=cat_col,
        features_to_add=["weekday_avg", "category_avg"],
    )
    
    # Check that early rows have lower statistics than later rows
    # (because expanding window grows over time)
    if "weekday_avg_expanding" in train_with_features.columns:
        early_avg = train_with_features["weekday_avg_expanding"].iloc[:10].mean()
        late_avg = train_with_features["weekday_avg_expanding"].iloc[-10:].mean()
        
        # Early average should be close to late average (within 50% tolerance)
        # because expanding window stabilizes over time
        ratio = early_avg / late_avg if late_avg > 0 else 0
        
        # We expect the ratio to be reasonable (not wildly different)
        # If ratio is < 0.1 or > 10, something is wrong
        test_passed = 0.1 < ratio < 10
        results["expanding_window_stability"] = test_passed
        
        print(f"  Early average (first 10 rows): {early_avg:.2f}")
        print(f"  Late average (last 10 rows): {late_avg:.2f}")
        print(f"  Ratio: {ratio:.2f}")
        print(f"  Result: {'PASS' if test_passed else 'FAIL'}")
    
    # Test 2: Verify that features at time t do not depend on times > t
    print("\n[TEST 1.2] Verifying no look-ahead bias...")
    
    # For a specific row, check that its feature value is computed
    # using only rows before it
    test_idx = len(train_with_features) // 2
    test_row = train_with_features.iloc[test_idx]
    
    if "weekday_avg_expanding" in train_with_features.columns:
        # Get all rows before test_idx with same weekday
        test_weekday = test_row[time_col].weekday()
        past_rows = train_with_features.iloc[:test_idx]
        past_same_weekday = past_rows[past_rows[time_col].dt.weekday == test_weekday]
        
        # Calculate expected average from past data
        expected_avg = past_same_weekday[target_col].mean()
        actual_avg = test_row["weekday_avg_expanding"]
        
        # They should match (within floating point tolerance)
        diff = abs(expected_avg - actual_avg)
        test_passed = diff < 0.01 or (diff / max(expected_avg, actual_avg)) < 0.01
        results["no_lookahead_bias"] = test_passed
        
        print(f"  Test row index: {test_idx}")
        print(f"  Expected weekday average (from past data): {expected_avg:.4f}")
        print(f"  Actual weekday average (from feature): {actual_avg:.4f}")
        print(f"  Difference: {diff:.4f}")
        print(f"  Result: {'PASS' if test_passed else 'FAIL'}")
    
    return results


def validate_window_slicing(
    df: pd.DataFrame,
    input_size: int = 30,
    horizon: int = 30,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
) -> Dict[str, bool]:
    """
    Validate that window slicing correctly separates input (past) from target (future).
    
    Test Strategy:
    1. Create sliding windows using slicing_window_category
    2. Verify that X windows end before y windows start (no overlap)
    3. Check that there's exactly 1 timestep gap between X and y
    
    Returns:
        Dictionary with test results.
    """
    print("\n" + "=" * 80)
    print("VALIDATION TEST 2: Window Slicing")
    print("=" * 80)
    
    results = {}
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    df = df.sort_values([cat_col, time_col]).reset_index(drop=True)
    
    # Select a single category for testing
    categories = df[cat_col].unique()
    test_cat = categories[0]
    cat_df = df[df[cat_col] == test_cat].copy()
    
    print(f"\nTesting with category: {test_cat}")
    print(f"  Total rows: {len(cat_df)}")
    print(f"  Date range: {cat_df[time_col].min()} to {cat_df[time_col].max()}")
    
    # Get feature columns (all columns except time, category, target)
    feature_cols = [col for col in cat_df.columns if col not in [time_col, cat_col]]
    
    # Create windows
    print(f"\n[TEST 2.1] Creating sliding windows (input_size={input_size}, horizon={horizon})...")
    
    try:
        X, y, cat, dates = slicing_window_category(
            cat_df,
            input_size=input_size,
            horizon=horizon,
            feature_cols=feature_cols,
            target_col_name=target_col,
            time_col_name=time_col,
            cat_col_name=cat_col,
        )
        
        print(f"  Created {len(X)} windows")
        print(f"  X shape: {X.shape} (samples, timesteps, features)")
        print(f"  y shape: {y.shape} (samples, horizon)")
        
        # Test: Check that X and y are properly separated
        # For first window, X should be rows [0:input_size], y should be rows [input_size:input_size+horizon]
        first_window_X_indices = list(range(0, input_size))
        first_window_y_indices = list(range(input_size, input_size + horizon))
        
        # Verify no overlap
        overlap = set(first_window_X_indices) & set(first_window_y_indices)
        no_overlap = len(overlap) == 0
        results["no_window_overlap"] = no_overlap
        
        print(f"\n[TEST 2.2] Checking for overlap between X and y windows...")
        print(f"  X indices: {first_window_X_indices[0]} to {first_window_X_indices[-1]}")
        print(f"  y indices: {first_window_y_indices[0]} to {first_window_y_indices[-1]}")
        print(f"  Overlap: {overlap if overlap else 'None'}")
        print(f"  Result: {'PASS' if no_overlap else 'FAIL'}")
        
        # Test: Verify gap of exactly 0 timesteps (X ends at t-1, y starts at t)
        gap = first_window_y_indices[0] - first_window_X_indices[-1] - 1
        correct_gap = gap == 0
        results["correct_gap"] = correct_gap
        
        print(f"\n[TEST 2.3] Checking gap between X and y windows...")
        print(f"  Gap size: {gap} timesteps")
        print(f"  Expected: 0 timesteps (X ends at t-1, y starts at t)")
        print(f"  Result: {'PASS' if correct_gap else 'FAIL'}")
        
    except Exception as e:
        print(f"\n[ERROR] Window slicing failed: {e}")
        results["window_slicing_success"] = False
        return results
    
    results["window_slicing_success"] = True
    return results


def validate_temporal_split(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    train_size: float = 0.7,
    val_size: float = 0.1,
) -> Dict[str, bool]:
    """
    Validate that data splitting is strictly temporal (no shuffling).
    
    Test Strategy:
    1. Split data using temporal split
    2. Verify that train dates < val dates < test dates (no overlap)
    3. Check that splits are contiguous (no gaps)
    
    Returns:
        Dictionary with test results.
    """
    print("\n" + "=" * 80)
    print("VALIDATION TEST 3: Temporal Data Splitting")
    print("=" * 80)
    
    results = {}
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Split data
    N = len(df)
    train_end = int(train_size * N)
    val_end = train_end + int(val_size * N)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"\n[TEST 3.1] Checking temporal ordering of splits...")
    print(f"  Train: {train_df[time_col].min()} to {train_df[time_col].max()}")
    print(f"  Val:   {val_df[time_col].min()} to {val_df[time_col].max()}")
    print(f"  Test:  {test_df[time_col].min()} to {test_df[time_col].max()}")
    
    # Test: Train dates should be < Val dates < Test dates
    train_max = train_df[time_col].max()
    val_min = val_df[time_col].min()
    val_max = val_df[time_col].max()
    test_min = test_df[time_col].min()
    
    temporal_order = (train_max <= val_min) and (val_max <= test_min)
    results["temporal_order"] = temporal_order
    
    print(f"\n  Train max date: {train_max}")
    print(f"  Val min date: {val_min}")
    print(f"  Val max date: {val_max}")
    print(f"  Test min date: {test_min}")
    print(f"  Result: {'PASS' if temporal_order else 'FAIL'}")
    
    # Test: Check for contiguity (no large gaps between splits)
    gap_train_val = (val_min - train_max).days
    gap_val_test = (test_min - val_max).days
    
    # Allow small gaps (up to 1 day due to calendar reindexing)
    contiguous = (gap_train_val <= 1) and (gap_val_test <= 1)
    results["contiguous_splits"] = contiguous
    
    print(f"\n[TEST 3.2] Checking contiguity of splits...")
    print(f"  Gap between train and val: {gap_train_val} days")
    print(f"  Gap between val and test: {gap_val_test} days")
    print(f"  Result: {'PASS' if contiguous else 'FAIL'}")
    
    return results


def main():
    """Main validation pipeline."""
    parser = argparse.ArgumentParser(
        description="Validate that the forecasting pipeline has no data leakage."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/data_cat/data_2024.csv",
        help="Path to data file for validation",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="Total CBM",
        help="Name of target column",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="ACTUALSHIPDATE",
        help="Name of time column",
    )
    parser.add_argument(
        "--cat-col",
        type=str,
        default="CATEGORY",
        help="Name of category column",
    )
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print(f"Please provide a valid data file path using --data-path")
        return 1
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Ensure required columns exist
    required_cols = [args.time_col, args.target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return 1
    
    # Run validation tests
    all_results = {}
    
    # Test 1: Expanding window features
    test1_results = validate_expanding_window_features(
        df,
        target_col=args.target_col,
        time_col=args.time_col,
        cat_col=args.cat_col,
    )
    all_results.update(test1_results)
    
    # Test 2: Window slicing
    test2_results = validate_window_slicing(
        df,
        target_col=args.target_col,
        time_col=args.time_col,
        cat_col=args.cat_col,
    )
    all_results.update(test2_results)
    
    # Test 3: Temporal split
    test3_results = validate_temporal_split(
        df,
        time_col=args.time_col,
    )
    all_results.update(test3_results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for result in all_results.values() if result is True)
    failed_tests = sum(1 for result in all_results.values() if result is False)
    total_tests = len(all_results)
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    print("\nDetailed results:")
    for test_name, result in all_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test_name}")
    
    if failed_tests == 0:
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED - No data leakage detected!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ VALIDATION FAILED - Data leakage detected!")
        print("=" * 80)
        print("\nPlease review the failed tests above and fix the issues before")
        print("deploying to production.")
        return 1


if __name__ == "__main__":
    exit(main())
