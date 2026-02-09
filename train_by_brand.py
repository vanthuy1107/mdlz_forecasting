"""Brand-Level Training Script for DRY Category.

This script trains separate models for each BRAND within the DRY category,
using the DRY category configuration (config_DRY.yaml) as the base.

Each brand gets its own:
- Trained model: outputs/spike-anchored/DRY_BRANDNAME/models/best_model.pth
- Training logs and plots: outputs/spike-anchored/DRY_BRANDNAME/
- Scaler: outputs/spike-anchored/DRY_BRANDNAME/scaler.pkl

Usage:
    python train_by_brand.py --category DRY --brands AFC COSY OREO
    python train_by_brand.py --category DRY  # Train all brands in DRY
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
import pickle
import argparse
from datetime import date, timedelta
from typing import List, Tuple, Optional

from config import load_config, load_holidays
from src.data import (
    DataReader,
    ForecastDataset,
    slicing_window_category,
    encode_categories,
    split_data,
    add_temporal_features,
    aggregate_daily,
    add_cbm_density_features,
    add_year_over_year_volume_features,
    fit_scaler,
    apply_scaling,
    inverse_transform_scaling
)
from src.data.preprocessing import (
    add_day_of_week_cyclical_features,
    add_eom_features,
    add_mid_month_peak_features,
    add_early_month_low_volume_features,
    add_high_volume_month_features,
    add_pre_holiday_surge_features,
    add_weekday_volume_tier_features,
    add_is_monday_feature,
    apply_sunday_to_monday_carryover,
    add_operational_status_flags,
    add_seasonal_active_window_features
)
from src.models import RNNWithCategory
from src.training import Trainer
from src.utils import plot_difference, spike_aware_mse, quantile_loss, QuantileLoss, quantile_coverage, calculate_forecast_metrics


# Import training function from mvp_train.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvp_train import train_single_model


def get_available_brands(data: pd.DataFrame, category: str, cat_col: str = "CATEGORY", brand_col: str = "BRAND") -> List[str]:
    """
    Get list of unique brands for a specific category.
    
    Args:
        data: DataFrame with all data
        category: Category name (e.g., "DRY")
        cat_col: Name of category column
        brand_col: Name of brand column
    
    Returns:
        Sorted list of brand names
    """
    if brand_col not in data.columns:
        raise ValueError(f"BRAND column '{brand_col}' not found in data. Available columns: {list(data.columns)}")
    
    # Filter to category
    cat_data = data[data[cat_col] == category].copy()
    
    if len(cat_data) == 0:
        raise ValueError(f"No data found for category '{category}'")
    
    # Get unique brands (excluding NaN)
    brands = cat_data[brand_col].dropna().unique().tolist()
    brands = [b for b in brands if str(b).strip() and str(b).upper() != 'NAN']
    
    return sorted(brands)


def filter_data_by_brand(data: pd.DataFrame, category: str, brand: str, 
                         cat_col: str = "CATEGORY", brand_col: str = "BRAND") -> pd.DataFrame:
    """
    Filter data to a specific category and brand.
    
    Args:
        data: DataFrame with all data
        category: Category name (e.g., "DRY")
        brand: Brand name (e.g., "AFC", "COSY")
        cat_col: Name of category column
        brand_col: Name of brand column
    
    Returns:
        Filtered DataFrame
    """
    filtered = data[(data[cat_col] == category) & (data[brand_col] == brand)].copy()
    
    if len(filtered) == 0:
        raise ValueError(f"No data found for category '{category}' and brand '{brand}'")
    
    return filtered


def train_brand_model(
    category: str,
    brand: str,
    data: pd.DataFrame,
    base_config,
    brand_col: str = "BRAND"
):
    """
    Train a model for a specific brand within a category.
    
    Uses the category's configuration (e.g., config_DRY.yaml) but filters data
    to a specific brand and saves outputs to outputs/{CATEGORY}_{BRAND}/
    
    Args:
        category: Category name (e.g., "DRY")
        brand: Brand name (e.g., "AFC")
        data: Full dataset (will be filtered to category and brand)
        base_config: Configuration object for the category
        brand_col: Name of brand column
    
    Returns:
        Dictionary with training results
    """
    print(f"\n{'=' * 80}")
    print(f"Training Brand Model: {category} - {brand}")
    print(f"{'=' * 80}")
    
    # Filter data to this brand
    print(f"\n[1/10] Filtering data to category '{category}' and brand '{brand}'...")
    cat_col = base_config.data['cat_col']
    brand_data = filter_data_by_brand(data, category, brand, cat_col, brand_col)
    print(f"  - Filtered to {len(brand_data)} samples for {category} - {brand}")
    
    # Create brand-specific output directory directly under the configured output root
    brand_output_name = f"{category}_{brand.replace(' ', '_').replace('/', '_')}"
    brand_output_dir = Path(base_config.output['output_dir']) / brand_output_name
    brand_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  - Brand output directory: {brand_output_dir}")
    
    # Save original output paths
    original_output_dir = base_config.output.get('output_dir')
    original_model_dir = base_config.output.get('model_dir')
    
    # Temporarily update config to use brand-specific output directory
    base_config.set('output.output_dir', str(brand_output_dir))
    base_config.set('output.model_dir', str(brand_output_dir / 'models'))
    
    # Use the category-specific training pipeline
    # We treat the brand as if it's a category (num_categories=1)
    print(f"\n[2/10] Processing {brand} data using {category} configuration...")
    
    try:
        # Call the existing train_single_model function from mvp_train.py
        # This handles all the feature engineering, scaling, and training
        result = train_single_model(
            data=brand_data,
            config=base_config,
            category_filter=category,  # Use actual category for filtering logic
            output_suffix=""  # Output dir already set in base_config
        )
    finally:
        # Restore original output paths
        if original_output_dir:
            base_config.set('output.output_dir', original_output_dir)
        if original_model_dir:
            base_config.set('output.model_dir', original_model_dir)
    
    print(f"\n[SUCCESS] Brand model training completed for {category} - {brand}")
    print(f"  - Model saved to: {brand_output_dir / 'models' / 'best_model.pth'}")
    print(f"  - Metrics: {result.get('metrics', {})}")
    
    return result


def main():
    """
    Main execution function for Brand-Level Training Pipeline.
    
    Trains separate models for each brand within a category, using the
    category's configuration as the base.
    """
    parser = argparse.ArgumentParser(
        description="Train separate models for each BRAND within a category"
    )
    parser.add_argument(
        '--category',
        type=str,
        required=False,
        default=None,
        help='Category to train brands for (e.g., DRY, FRESH, TET). If not specified, reads major_categories from config.yaml'
    )
    parser.add_argument(
        '--brands',
        type=str,
        nargs='*',
        help='Specific brands to train (if not specified, trains all brands in category)'
    )
    parser.add_argument(
        '--brand-col',
        type=str,
        default='BRAND',
        help='Name of BRAND column in data (default: BRAND)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip training for brands that already have a trained model'
    )
    
    args = parser.parse_args()
    
    # If no category specified, read from config.yaml
    if args.category is None:
        print("=" * 80)
        print(f"MDLZ FORECASTING: Brand-Level Training Pipeline")
        print("=" * 80)
        print("\n[INFO] No --category specified, reading from config.yaml...")
        
        base_config = load_config()
        major_categories = base_config.data.get('major_categories', [])
        
        if not major_categories:
            print("[ERROR] No major_categories found in config.yaml")
            print("        Please either:")
            print("        1. Set major_categories in config/config.yaml, or")
            print("        2. Run with: python train_by_brand.py --category CATEGORY_NAME")
            return 1
        
        print(f"[INFO] Found major_categories: {major_categories}\n")
        
        # Train each major category
        overall_results = []
        for category in major_categories:
            print("\n" + "=" * 80)
            print(f"CATEGORY: {category}")
            print("=" * 80 + "\n")
            
            # Create a mock args object with the category set
            class CategoryArgs:
                pass
            cat_args = CategoryArgs()
            cat_args.category = category
            cat_args.brands = args.brands
            cat_args.brand_col = args.brand_col
            cat_args.skip_existing = args.skip_existing
            
            # Call training logic for this category
            result = train_category(cat_args)
            overall_results.append((category, result))
        
        # Print overall summary
        print("\n" + "=" * 80)
        print("OVERALL TRAINING SUMMARY")
        print("=" * 80)
        successful_cats = sum(1 for _, r in overall_results if r == 0)
        failed_cats = sum(1 for _, r in overall_results if r != 0)
        print(f"Categories trained: {len(overall_results)}")
        print(f"Successful: {successful_cats}")
        print(f"Failed: {failed_cats}")
        
        for category, result in overall_results:
            status = "✓" if result == 0 else "✗"
            print(f"  {status} {category}")
        
        print("=" * 80)
        return 0 if failed_cats == 0 else 1
    
    # If category is specified, train just that category
    return train_category(args)


def train_category(args):
    """Train brand models for a single category."""
    print("=" * 80)
    print(f"MDLZ FORECASTING: Brand-Level Training Pipeline")
    print(f"Category: {args.category}")
    print("=" * 80)
    
    # Load category-specific configuration
    print(f"\n[1/5] Loading configuration for category '{args.category}'...")
    try:
        category_config = load_config(category=args.category)
    except FileNotFoundError:
        print(f"[WARNING] No category-specific config found for '{args.category}'")
        print(f"         Expected: config/config_{args.category}.yaml")
        print(f"         Falling back to base config...")
        category_config = load_config()
    
    # Force spike_aware_mse loss if not quantile
    if category_config.training.get('loss', 'spike_aware_mse') != 'quantile':
        category_config.set('training.loss', 'spike_aware_mse')
    category_config.set('output.save_model', True)
    
    print(f"  - Data years: {category_config.data['years']}")
    print(f"  - Loss function: {category_config.training.get('loss', 'spike_aware_mse')}")
    print(f"  - Window size: input={category_config.window['input_size']}, horizon={category_config.window['horizon']}")
    
    # Load data
    print(f"\n[2/5] Loading data...")
    data_config = category_config.data
    data_reader = DataReader(
        data_dir=data_config['data_dir'],
        file_pattern=data_config['file_pattern']
    )
    
    try:
        data = data_reader.load(years=data_config['years'])
    except FileNotFoundError:
        print("[WARNING] Combined year files not found, trying pattern-based loading...")
        file_prefix = data_config.get('file_prefix', 'Outboundreports')
        data = data_reader.load_by_file_pattern(
            years=data_config['years'],
            file_prefix=file_prefix
        )
    
    print(f"  - Loaded {len(data)} samples")
    
    # Get available brands for this category
    print(f"\n[3/5] Discovering brands in category '{args.category}'...")
    cat_col = data_config['cat_col']
    
    try:
        available_brands = get_available_brands(data, args.category, cat_col, args.brand_col)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1
    
    print(f"  - Found {len(available_brands)} brand(s) in {args.category}:")
    for i, brand in enumerate(available_brands, 1):
        # Count samples for this brand
        brand_data = data[(data[cat_col] == args.category) & (data[args.brand_col] == brand)]
        print(f"    {i}. {brand} ({len(brand_data):,} samples)")
    
    # Determine which brands to train
    if args.brands:
        # User specified specific brands
        brands_to_train = [b for b in args.brands if b in available_brands]
        missing_brands = [b for b in args.brands if b not in available_brands]
        
        if missing_brands:
            print(f"\n[WARNING] Requested brands not found in data: {missing_brands}")
        
        if not brands_to_train:
            print(f"[ERROR] None of the requested brands found in category '{args.category}'")
            print(f"        Available brands: {available_brands}")
            return 1
    else:
        # Train all available brands
        brands_to_train = available_brands
    
    print(f"\n[4/5] Will train {len(brands_to_train)} brand model(s):")
    for i, brand in enumerate(brands_to_train, 1):
        brand_output_name = f"{args.category}_{brand.replace(' ', '_').replace('/', '_')}"
        print(f"  {i}. {brand} -> outputs/spike-anchored/{brand_output_name}/")
    
    # Train each brand
    print(f"\n[5/5] Training brand models...")
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, brand in enumerate(brands_to_train, 1):
        print(f"\n{'=' * 80}")
        print(f"BRAND {idx}/{len(brands_to_train)}: {args.category} - {brand}")
        print(f"{'=' * 80}")
        
        # Check if model already exists
        if args.skip_existing:
            brand_output_name = f"{args.category}_{brand.replace(' ', '_').replace('/', '_')}"
            model_path = Path(category_config.output['output_dir']) / brand_output_name / 'models' / 'best_model.pth'
            if model_path.exists():
                print(f"[SKIP] Model already exists at {model_path}")
                skipped += 1
                continue
        
        try:
            result = train_brand_model(
                category=args.category,
                brand=brand,
                data=data,
                base_config=category_config,
                brand_col=args.brand_col
            )
            results.append({
                'category': args.category,
                'brand': brand,
                'status': 'success',
                'result': result
            })
            successful += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to train brand '{brand}': {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'category': args.category,
                'brand': brand,
                'status': 'failed',
                'error': str(e)
            })
            failed += 1
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Category: {args.category}")
    print(f"Total brands: {len(brands_to_train)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if successful > 0:
        print(f"\n✓ Successfully trained {successful} brand model(s):")
        for res in results:
            if res['status'] == 'success':
                brand_output_name = f"{res['category']}_{res['brand'].replace(' ', '_').replace('/', '_')}"
                print(f"  - {res['brand']} -> outputs/spike-anchored/{brand_output_name}/")
    
    if failed > 0:
        print(f"\n✗ Failed to train {failed} brand model(s):")
        for res in results:
            if res['status'] == 'failed':
                print(f"  - {res['brand']}: {res['error']}")
    
    print(f"\n{'=' * 80}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
