"""Brand-Level Prediction Script for Category Brands.

This script makes predictions for each BRAND within a category using
the brand-specific models trained by train_by_brand.py.

Each brand model is loaded from:
- outputs/DRY_BRANDNAME/models/best_model.pth

Predictions are saved to:
- outputs/DRY_BRANDNAME/predictions_YYYY.csv

Usage:
    # Use categories from config.data.major_categories and dates from config.inference
    python predict_by_brand.py

    # Predict only for DRY (override categories from config)
    python predict_by_brand.py --category DRY

    # Predict only for DRY with explicit date range
    python predict_by_brand.py --category DRY --start 2025-01-01 --end 2025-12-31
"""
import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
import argparse
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict

from config import load_config, load_holidays
from src.data import (
    DataReader,
    ForecastDataset,
    encode_categories,
    add_temporal_features,
    aggregate_daily,
    add_cbm_density_features,
    apply_scaling,
    inverse_transform_scaling
)
from src.data.preprocessing import (
    get_vietnam_holidays,
    add_day_of_week_cyclical_features,
    add_eom_features,
    add_weekday_volume_tier_features,
    apply_sunday_to_monday_carryover,
    add_operational_status_flags,
    add_seasonal_active_window_features,
)
from src.models import RNNWithCategory
from src.predict import (
    load_model_for_prediction,
    prepare_prediction_data,
    predict_direct_multistep,
    predict_direct_multistep_rolling,
    get_historical_window_data
)

# Import utility functions from train_by_brand.py
from train_by_brand import get_available_brands, filter_data_by_brand


def predict_brand_model(
    category: str,
    brand: str,
    data: pd.DataFrame,
    base_config,
    start_date: date,
    end_date: date,
    brand_col: str = "BRAND",
    save_predictions: bool = True
) -> pd.DataFrame:
    """
    Make predictions for a specific brand within a category.
    
    Args:
        category: Category name (e.g., "DRY")
        brand: Brand name (e.g., "AFC")
        data: Full dataset (will be filtered to category and brand)
        base_config: Configuration object for the category
        start_date: First date to predict
        end_date: Last date to predict
        brand_col: Name of brand column
        save_predictions: Whether to save predictions to CSV
    
    Returns:
        DataFrame with predictions
    """
    print(f"\n{'=' * 80}")
    print(f"Predicting Brand: {category} - {brand}")
    print(f"{'=' * 80}")
    
    # Filter data to this brand
    print(f"\n[1/6] Filtering data to category '{category}' and brand '{brand}'...")
    cat_col = base_config.data['cat_col']
    brand_data = filter_data_by_brand(data, category, brand, cat_col, brand_col)
    print(f"  - Filtered to {len(brand_data)} samples for {category} - {brand}")
    
    # Get brand-specific paths
    brand_output_name = f"{category}_{brand.replace(' ', '_').replace('/', '_')}"
    brand_output_dir = Path(base_config.output['output_dir']) / brand_output_name
    brand_model_dir = brand_output_dir / 'models'
    brand_model_path = brand_model_dir / 'best_model.pth'
    brand_scaler_path = brand_model_dir / 'scaler.pkl'  # Fixed: scaler is in models subdirectory
    
    print(f"  - Brand output directory: {brand_output_dir}")
    print(f"  - Model path: {brand_model_path}")
    
    # Check if model exists
    if not brand_model_path.exists():
        raise FileNotFoundError(
            f"Model not found for brand '{brand}' at {brand_model_path}\n"
            f"Please train the brand model first using: python train_by_brand.py --category {category} --brands {brand}"
        )
    
    # Load model and scaler
    print(f"\n[2/6] Loading model and scaler...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model state
    checkpoint = torch.load(brand_model_path, map_location=device)
    model_state = checkpoint.get('model_state_dict', checkpoint)
    
    # Load model config from checkpoint or use base config
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        print(f"  - Loaded model config from checkpoint")
    else:
        model_config = base_config
        print(f"  - Using base config for model")
    
    # Helper function to access config (handles both Config object and dict)
    def get_model_config(config_obj):
        """Get model config dict from either Config object or dict."""
        if isinstance(config_obj, dict):
            return config_obj.get('model', config_obj)
        else:
            return config_obj.model
    
    model_cfg = get_model_config(model_config)
    
    # Create model instance
    model = RNNWithCategory(
        input_dim=model_cfg['input_dim'],
        hidden_size=model_cfg['hidden_size'],
        n_layers=model_cfg['n_layers'],
        output_dim=model_cfg['output_dim'],
        num_categories=1,  # Brand models are category-specific (num_categories=1)
        cat_emb_dim=model_cfg.get('cat_emb_dim', 5),
        dropout_prob=model_cfg.get('dropout_prob', 0.2),
        use_layer_norm=model_cfg.get('use_layer_norm', True)
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    print(f"  - Model loaded: input_dim={model_cfg['input_dim']}, "
          f"hidden_size={model_cfg['hidden_size']}, "
          f"output_dim={model_cfg['output_dim']}")
    
    # Load scaler
    if brand_scaler_path.exists():
        with open(brand_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  - Scaler loaded from {brand_scaler_path}")
    else:
        print(f"  - WARNING: Scaler not found at {brand_scaler_path}, predictions may be incorrect")
        scaler = None
    
    # Prepare prediction data
    print(f"\n[3/6] Preparing prediction data...")
    time_col = base_config.data['time_col']
    target_col = base_config.data['target_col']
    feature_cols = base_config.data['feature_cols']
    
    # Add extra features that are used during training (same as in mvp_train.py)
    extra_features = [
        "lunar_month_sin", "lunar_month_cos",
        "lunar_day_sin", "lunar_day_cos",
        "days_to_tet", "days_to_mid_autumn",
        "is_active_season", "days_until_peak", "is_golden_window", "is_peak_loss_window",
        "cbm_per_qty", "cbm_per_qty_last_year"
    ]
    for feat in extra_features:
        if feat not in feature_cols:
            feature_cols.append(feat)
    
    # Remove target column from feature_cols (it's not an input feature)
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    base_config.set("data.feature_cols", feature_cols)
    
    # Ensure time column is datetime and sort
    if not pd.api.types.is_datetime64_any_dtype(brand_data[time_col]):
        brand_data[time_col] = pd.to_datetime(brand_data[time_col])
    brand_data = brand_data.sort_values(time_col).reset_index(drop=True)
    
    # Import feature engineering functions from mvp_predict.py and mvp_train.py
    from mvp_predict import (
        add_weekend_features,
        add_lunar_calendar_features,
        add_lunar_cyclical_features,
        add_days_to_tet_feature,
        add_rolling_and_momentum_features
    )
    from mvp_train import (
        add_days_to_mid_autumn_feature,
        add_days_since_holiday
    )
    
    # Add all required features (same as training)
    print("  - Adding temporal features...")
    brand_data = add_temporal_features(
        brand_data,
        time_col=time_col,
        month_sin_col="month_sin",
        month_cos_col="month_cos",
        dayofmonth_sin_col="dayofmonth_sin",
        dayofmonth_cos_col="dayofmonth_cos"
    )
    
    print("  - Adding weekend features...")
    brand_data = add_weekend_features(
        brand_data,
        time_col=time_col,
        is_weekend_col="is_weekend",
        day_of_week_col="day_of_week"
    )
    
    print("  - Adding cyclical day-of-week features...")
    brand_data = add_day_of_week_cyclical_features(
        brand_data,
        time_col=time_col,
        day_of_week_sin_col="day_of_week_sin",
        day_of_week_cos_col="day_of_week_cos"
    )
    
    print("  - Adding weekday volume tier features...")
    brand_data = add_weekday_volume_tier_features(
        brand_data,
        time_col=time_col,
        weekday_volume_tier_col="weekday_volume_tier",
        is_high_volume_weekday_col="is_high_volume_weekday"
    )
    
    print("  - Adding Is_Monday feature...")
    from src.data.preprocessing import add_is_monday_feature
    brand_data = add_is_monday_feature(
        brand_data,
        time_col=time_col,
        is_monday_col="Is_Monday"
    )
    
    print("  - Adding EOM features...")
    brand_data = add_eom_features(
        brand_data,
        time_col=time_col,
        is_eom_col="is_EOM",
        days_until_month_end_col="days_until_month_end",
        eom_window_days=3
    )
    
    print("  - Adding lunar calendar features...")
    brand_data = add_lunar_calendar_features(
        brand_data,
        time_col=time_col,
        lunar_month_col="lunar_month",
        lunar_day_col="lunar_day"
    )
    
    print("  - Adding lunar cyclical features...")
    brand_data = add_lunar_cyclical_features(
        brand_data,
        lunar_month_col="lunar_month",
        lunar_day_col="lunar_day",
        lunar_month_sin_col="lunar_month_sin",
        lunar_month_cos_col="lunar_month_cos",
        lunar_day_sin_col="lunar_day_sin",
        lunar_day_cos_col="lunar_day_cos"
    )
    
    print("  - Adding Vietnamese holiday features...")
    from mvp_train import add_holiday_features_vietnam
    brand_data = add_holiday_features_vietnam(
        brand_data,
        time_col=time_col,
        holiday_indicator_col="holiday_indicator",
        days_until_holiday_col="days_until_next_holiday",
        days_since_holiday_col="days_since_holiday"
    )
    
    print("  - Adding Tet countdown feature...")
    brand_data = add_days_to_tet_feature(
        brand_data,
        time_col=time_col,
        days_to_tet_col="days_to_tet"
    )
    
    print("  - Adding Mid-Autumn countdown feature...")
    brand_data = add_days_to_mid_autumn_feature(
        brand_data,
        time_col=time_col,
        days_to_mid_autumn_col="days_to_mid_autumn"
    )
    
    print("  - Adding rolling and momentum features...")
    brand_data = add_rolling_and_momentum_features(
        brand_data,
        time_col=time_col,
        target_col=target_col,
        cat_col=cat_col,
        rolling_7_col="rolling_mean_7d",
        rolling_30_col="rolling_mean_30d",
        momentum_col="momentum_3d_vs_14d"
    )
    
    # Add CBM density features (always add - they're part of training feature set)
    print("  - Adding CBM density features...")
    brand_data = add_cbm_density_features(
        brand_data,
        qty_col="Total QTY" if "Total QTY" in brand_data.columns else target_col,
        cbm_col=target_col,
        density_col="cbm_per_qty",
        density_last_year_col="cbm_per_qty_last_year",
        time_col=time_col,
        cat_col=cat_col
    )
    
    # Add seasonal features for category
    print(f"  - Adding seasonal features for category '{category}'...")
    brand_data = add_seasonal_active_window_features(
        brand_data,
        time_col=time_col,
        cat_col=cat_col,
        is_active_season_col="is_active_season",
        days_until_peak_col="days_until_peak",
        is_golden_window_col="is_golden_window"
    )
    
    # Ensure is_peak_loss_window exists (it's in the training feature set)
    if 'is_peak_loss_window' not in brand_data.columns:
        print("  - Creating is_peak_loss_window feature (default=0 for non-MOONCAKE)...")
        brand_data['is_peak_loss_window'] = 0
    
    # Apply scaling if scaler exists
    if scaler is not None:
        print("  - Applying scaling...")
        brand_data = apply_scaling(
            brand_data,
            scaler=scaler,
            target_col=target_col
        )
    
    # Get historical window for prediction
    print(f"\n[4/6] Extracting historical window...")
    window_size = base_config.window['input_size']
    
    # Get the last N days before start_date
    window_end_date = start_date - timedelta(days=1)
    window_data = get_historical_window_data(
        historical_data=brand_data,
        end_date=window_end_date,
        config=base_config,
        num_days=window_size
    )
    
    print(f"  - Window size: {len(window_data)} samples")
    print(f"  - Window date range: {window_data[time_col].min()} to {window_data[time_col].max()}")
    
    # Make predictions
    print(f"\n[5/6] Making predictions from {start_date} to {end_date}...")
    num_days = (end_date - start_date).days + 1
    horizon = base_config.window['horizon']
    
    if num_days > horizon:
        print(f"  - Prediction period ({num_days} days) exceeds horizon ({horizon} days)")
        print(f"  - Using rolling prediction mode...")
        predictions_df = predict_direct_multistep_rolling(
            model=model,
            device=device,
            initial_window_data=window_data,
            start_date=start_date,
            end_date=end_date,
            config=base_config,
            cat_id=0,  # Brand models use cat_id=0
            category=category,  # Pass category for seasonal logic
            historical_data=brand_data  # Pass full historical data for YoY features
        )
    else:
        print(f"  - Using direct multi-step prediction...")
        predictions_df = predict_direct_multistep(
            model=model,
            device=device,
            initial_window_data=window_data,
            start_date=start_date,
            end_date=end_date,
            config=base_config,
            cat_id=0,  # Brand models use cat_id=0
            category=category  # Pass category for seasonal logic
        )
    
    # Inverse transform predictions
    if scaler is not None:
        print("  - Inverse transforming predictions...")
        predictions_df['predicted'] = scaler.inverse_transform(
            predictions_df[['predicted']]
        ).flatten()
    
    # Add category and brand columns
    predictions_df['CATEGORY'] = category
    predictions_df['BRAND'] = brand
    
    # Save predictions
    if save_predictions:
        print(f"\n[6/6] Saving predictions...")
        year_start = start_date.year
        year_end = end_date.year
        if year_start == year_end:
            pred_filename = f"predictions_{year_start}.csv"
        else:
            pred_filename = f"predictions_{year_start}_{year_end}.csv"
        
        pred_output_path = brand_output_dir / pred_filename
        predictions_df.to_csv(pred_output_path, index=False)
        print(f"  - Saved to: {pred_output_path}")
    
    print(f"\n[SUCCESS] Brand prediction completed for {category} - {brand}")
    print(f"  - Predicted {len(predictions_df)} days")
    print(f"  - Mean prediction: {predictions_df['predicted'].mean():.2f}")
    print(f"  - Total volume: {predictions_df['predicted'].sum():.2f}")
    
    return predictions_df


def main():
    """
    Main execution function for Brand-Level Prediction Pipeline.
    
    Categories and dates:
    - If --category is provided: predict only that category
    - If --category is omitted: use config.data.major_categories from config.yaml
    - Prediction dates default to config.inference.prediction_start / prediction_end
      unless --start / --end overrides are provided.
    """
    parser = argparse.ArgumentParser(
        description="Make predictions for each BRAND within one or more categories"
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Category to predict brands for (e.g., DRY, FRESH, TET). '
             'If omitted, uses data.major_categories from config.yaml'
    )
    parser.add_argument(
        '--brands',
        type=str,
        nargs='*',
        help='Specific brands to predict (if not specified, predicts all brands with trained models)'
    )
    parser.add_argument(
        '--brand-col',
        type=str,
        default='BRAND',
        help='Name of BRAND column in data (default: BRAND)'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Start date for predictions (YYYY-MM-DD). If not specified, uses config.inference.prediction_start'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date for predictions (YYYY-MM-DD). If not specified, uses config.inference.prediction_end'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to prediction data file. If not specified, uses config.inference.prediction_data_path'
    )
    parser.add_argument(
        '--combine-output',
        action='store_true',
        help='Combine all brand predictions into a single file'
    )
    
    args = parser.parse_args()
    
    # Load base configuration to get major_categories and default inference settings
    base_config = load_config()
    data_cfg = base_config.data
    major_categories = data_cfg.get("major_categories", [])
    
    # Resolve list of categories to predict
    if args.category:
        categories_to_predict = [args.category]
    else:
        if not major_categories:
            print("[ERROR] No category specified and data.major_categories is empty in config.yaml")
            return 1
        categories_to_predict = major_categories
    
    print("=" * 80)
    print("MDLZ FORECASTING: Brand-Level Prediction Pipeline")
    print("=" * 80)
    print(f"Categories to predict: {categories_to_predict}")
    print("=" * 80)
    
    overall_success = True
    
    # Loop over each category
    for cat_idx, category in enumerate(categories_to_predict, 1):
        print(f"\n{'=' * 80}")
        print(f"CATEGORY {cat_idx}/{len(categories_to_predict)}: {category}")
        print(f"{'=' * 80}")
        
        # Load category-specific configuration (merges with base)
        print(f"\n[1/4] Loading configuration for category '{category}'...")
        try:
            category_config = load_config(category=category)
        except FileNotFoundError:
            print(f"[WARNING] No category-specific config found for '{category}'")
            print(f"         Expected: config/config_{category}.yaml")
            print(f"         Falling back to base config...")
            category_config = base_config
        
        # Get prediction dates (use overrides if provided, else config.inference)
        if args.start:
            start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        else:
            start_date = datetime.strptime(
                category_config.inference.get('prediction_start', '2025-01-01'),
                '%Y-%m-%d'
            ).date()
        
        if args.end:
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
        else:
            end_date = datetime.strptime(
                category_config.inference.get('prediction_end', '2025-12-31'),
                '%Y-%m-%d'
            ).date()
        
        print(f"  - Prediction period: {start_date} to {end_date}")
        print(f"  - Total days: {(end_date - start_date).days + 1}")
        
        # Load data
        print(f"\n[2/4] Loading data...")
        data_config = category_config.data
        
        # Determine data file
        if args.data_file:
            data_file = args.data_file
        else:
            data_file = category_config.inference.get('prediction_data_path')
            if not data_file:
                # Try to construct from years
                years = [start_date.year]
                if end_date.year != start_date.year:
                    years.append(end_date.year)
                data_config['years'] = years
        
        if data_file and os.path.exists(data_file):
            print(f"  - Loading from: {data_file}")
            data = pd.read_csv(data_file)
        else:
            # Load from data_dir using DataReader
            data_reader = DataReader(
                data_dir=data_config['data_dir'],
                file_pattern=data_config['file_pattern']
            )
            
            years = data_config['years']
            try:
                data = data_reader.load(years=years)
            except FileNotFoundError:
                print("[WARNING] Combined year files not found, trying pattern-based loading...")
                file_prefix = data_config.get('file_prefix', 'Outboundreports')
                data = data_reader.load_by_file_pattern(
                    years=years,
                    file_prefix=file_prefix
                )
        
        print(f"  - Loaded {len(data)} samples")
        
        # Ensure BRAND column exists; if not, fall back to historical data with BRAND (data_cat)
        brand_col_name = args.brand_col
        if brand_col_name not in data.columns:
            print(f"\n  [INFO] Column '{brand_col_name}' not found in prediction data.")
            print(f"        Falling back to historical data from data_dir/file_pattern for BRAND-level predictions...")
            data_reader = DataReader(
                data_dir=data_config['data_dir'],
                file_pattern=data_config['file_pattern']
            )
            years = data_config['years']
            try:
                data = data_reader.load(years=years)
                print(f"  - Reloaded {len(data)} samples from historical data with BRAND column")
            except FileNotFoundError:
                print(f"  [ERROR] Could not reload historical data with BRAND column.")
                overall_success = False
                continue
        
        # Get available brands for this category
        print(f"\n[3/4] Discovering brands with trained models...")
        cat_col = data_config['cat_col']
        
        try:
            available_brands = get_available_brands(data, category, cat_col, args.brand_col)
        except ValueError as e:
            print(f"[ERROR] {e}")
            overall_success = False
            continue
        
        # Check which brands have trained models
        base_output_dir = Path(category_config.output['output_dir'])
        brands_with_models = []
        
        for brand in available_brands:
            brand_output_name = f"{category}_{brand.replace(' ', '_').replace('/', '_')}"
            brand_model_path = base_output_dir / brand_output_name / 'models' / 'best_model.pth'
            if brand_model_path.exists():
                brands_with_models.append(brand)
        
        print(f"  - Found {len(brands_with_models)} brand(s) with trained models:")
        for i, brand in enumerate(brands_with_models, 1):
            brand_output_name = f"{category}_{brand.replace(' ', '_').replace('/', '_')}"
            print(f"    {i}. {brand} (outputs/{brand_output_name}/)")
        
        if not brands_with_models:
            print(f"\n[ERROR] No trained models found for brands in category '{category}'")
            print(f"        Please train brand models first using: python train_by_brand.py --category {category}")
            overall_success = False
            continue
        
        # Determine which brands to predict
        if args.brands and args.category:
            # Only filter brands when user explicitly targeted a single category
            brands_to_predict = [b for b in args.brands if b in brands_with_models]
            missing_brands = [b for b in args.brands if b not in brands_with_models]
            
            if missing_brands:
                print(f"\n[WARNING] Requested brands don't have trained models: {missing_brands}")
            
            if not brands_to_predict:
                print(f"[ERROR] None of the requested brands have trained models")
                print(f"        Brands with models: {brands_with_models}")
                overall_success = False
                continue
        else:
            # Predict all brands with trained models
            brands_to_predict = brands_with_models
        
        print(f"\n[4/4] Will predict {len(brands_to_predict)} brand(s) for category '{category}':")
        for i, brand in enumerate(brands_to_predict, 1):
            brand_output_name = f"{category}_{brand.replace(' ', '_').replace('/', '_')}"
            print(f"  {i}. {brand} -> outputs/{brand_output_name}/")
        
        # Predict each brand
        print(f"\n{'=' * 80}")
        print("Starting brand predictions...")
        print(f"{'=' * 80}")
        
        all_predictions = []
        successful = 0
        failed = 0
        
        for idx, brand in enumerate(brands_to_predict, 1):
            print(f"\n{'=' * 80}")
            print(f"BRAND {idx}/{len(brands_to_predict)}: {category} - {brand}")
            print(f"{'=' * 80}")
            
            try:
                predictions_df = predict_brand_model(
                    category=category,
                    brand=brand,
                    data=data,
                    base_config=category_config,
                    start_date=start_date,
                    end_date=end_date,
                    brand_col=args.brand_col,
                    save_predictions=True
                )
                all_predictions.append(predictions_df)
                successful += 1
            except Exception as e:
                print(f"\n[ERROR] Failed to predict brand '{brand}': {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                overall_success = False
        
        # Combine predictions if requested
        if args.combine_output and all_predictions:
            print(f"\n{'=' * 80}")
            print("Combining predictions...")
            print(f"{'=' * 80}")
            
            combined_df = pd.concat(all_predictions, ignore_index=True)
            combined_df = combined_df.sort_values(['date', 'BRAND']).reset_index(drop=True)
            
            year_start = start_date.year
            year_end = end_date.year
            if year_start == year_end:
                combined_filename = f"{category}_all_brands_predictions_{year_start}.csv"
            else:
                combined_filename = f"{category}_all_brands_predictions_{year_start}_{year_end}.csv"
            
            combined_output_path = base_output_dir / combined_filename
            combined_df.to_csv(combined_output_path, index=False)
            print(f"  - Combined predictions saved to: {combined_output_path}")
            print(f"  - Total rows: {len(combined_df)}")
            print(f"  - Brands: {combined_df['BRAND'].unique().tolist()}")
        
        # Print summary for this category
        print(f"\n{'=' * 80}")
        print("PREDICTION SUMMARY")
        print(f"{'=' * 80}")
        print(f"Category: {category}")
        print(f"Prediction period: {start_date} to {end_date}")
        print(f"Total brands: {len(brands_to_predict)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if all_predictions:
            print(f"\n[SUCCESS] Predicted {successful} brand(s):")
            for pred_df in all_predictions:
                brand = pred_df['BRAND'].iloc[0]
                total_volume = pred_df['predicted'].sum()
                print(f"  - {brand}: {len(pred_df)} days, total volume = {total_volume:.2f}")
        
        print(f"\n{'=' * 80}")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit(main())
