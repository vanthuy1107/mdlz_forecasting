"""Prediction script for HybridLSTM models.

This script makes predictions using the HybridLSTM model trained by train_hybrid_lstm.py.
It loads the category-level model and generates forecasts for the specified date range.

Usage:
    # Predict for DRY category for 2025
    python predict_hybrid_lstm.py --category DRY --start 2025-01-01 --end 2025-12-31
    
    # Predict with custom input data
    python predict_hybrid_lstm.py --category DRY --data-path dataset/test/data_2025.csv
    
    # Predict and save results
    python predict_hybrid_lstm.py --category DRY --start 2025-01-01 --end 2025-12-31 --save
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models import HybridLSTM
from src.data.preprocessing import (
    add_temporal_features,
    add_holiday_features,
    add_weekday_volume_tier_features,
    add_eom_features,
)
from src.data.expanding_features import (
    add_expanding_statistical_features,
    add_expanding_rolling_features,
    add_expanding_momentum_features,
    add_brand_tier_feature,
)
from src.utils.evaluation import calculate_forecast_metrics


def load_config(config_path: str = "config/config.yaml", category: Optional[str] = None):
    """Load configuration file."""
    from config import Config
    config = Config(config_path=config_path, category=category)
    return config


def load_data(
    data_dir: str,
    years: List[int],
    category: str,
    cat_col: str = "CATEGORY",
) -> pd.DataFrame:
    """Load and combine data files for specified years."""
    dfs = []
    for year in years:
        file_path = Path(data_dir) / f"data_{year}.csv"
        if file_path.exists():
            print(f"  Loading data_{year}.csv...")
            df = pd.read_csv(file_path)
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined data: {len(df_combined):,} rows")
    
    # Filter to category
    if cat_col in df_combined.columns:
        df_filtered = df_combined[df_combined[cat_col] == category].copy()
        print(f"  Filtered to {category}: {len(df_filtered):,} rows")
        return df_filtered
    else:
        return df_combined


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
) -> pd.DataFrame:
    """
    Prepare features for HybridLSTM prediction.
    Uses the same feature engineering as train_hybrid_lstm.py.
    """
    print("\n" + "=" * 80)
    print("PREPARING FEATURES")
    print("=" * 80)
    
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by time
    df = df.sort_values([cat_col, time_col] if cat_col in df.columns else [time_col]).reset_index(drop=True)
    
    print(f"\n1. Adding calendar-based features...")
    
    # Add temporal features
    df = add_temporal_features(df, time_col=time_col)
    print("   [+] Added: month_sin, month_cos, dayofmonth_sin, dayofmonth_cos")
    
    # Add holiday features
    df = add_holiday_features(df, time_col=time_col)
    print("   [+] Added: holiday_indicator, days_until_next_holiday")
    
    # Add days_since_holiday feature
    from src.data.preprocessing import get_us_holidays
    from datetime import timedelta
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    extended_min = min_date - timedelta(days=365)
    holidays = get_us_holidays(extended_min, max_date)
    holidays_sorted = sorted(holidays, reverse=True)
    
    def get_days_since_holiday(date):
        if pd.isna(date):
            return 365
        current_date = date.date() if isinstance(date, pd.Timestamp) else date
        for holiday in holidays_sorted:
            if holiday <= current_date:
                return (current_date - holiday).days
        return 365
    
    df["days_since_holiday"] = df[time_col].apply(get_days_since_holiday)
    print("   [+] Added: days_since_holiday")
    
    # Add lunar calendar features
    from src.utils.lunar_utils import solar_to_lunar_date
    
    def get_lunar_date(date):
        if pd.isna(date):
            return pd.Series([8, 15])
        current_date = date.date() if isinstance(date, pd.Timestamp) else date
        lunar_month, lunar_day = solar_to_lunar_date(current_date)
        return pd.Series([lunar_month, lunar_day])
    
    df[["lunar_month", "lunar_day"]] = df[time_col].apply(get_lunar_date)
    print("   [+] Added: lunar_month, lunar_day")
    
    # Add weekday features
    df = add_weekday_volume_tier_features(df, time_col=time_col)
    print("   [+] Added: weekday_volume_tier, is_high_volume_weekday")
    
    # Add EOM features
    df = add_eom_features(df, time_col=time_col)
    print("   [+] Added: is_EOM, days_until_month_end")
    
    # Add day of week cyclical features
    df["day_of_week"] = df[time_col].dt.dayofweek
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["Is_Monday"] = (df["day_of_week"] == 0).astype(int)
    print("   [+] Added: day_of_week_sin, day_of_week_cos, is_weekend, Is_Monday")
    
    print(f"\n2. Adding expanding window statistical features...")
    
    # Add expanding statistical features
    df = add_expanding_statistical_features(
        df,
        target_col=target_col,
        time_col=time_col,
        cat_col=cat_col,
        features_to_add=["weekday_avg", "category_avg"],
    )
    print("   [+] Added: weekday_avg_expanding, category_avg_expanding")
    
    # Add expanding rolling features
    df = add_expanding_rolling_features(
        df,
        target_col=target_col,
        time_col=time_col,
        cat_col=cat_col,
        windows=[7, 30],
    )
    print("   [+] Added: rolling_mean_7d_expanding, rolling_mean_30d_expanding")
    
    # Add expanding momentum features
    df = add_expanding_momentum_features(
        df,
        target_col=target_col,
        time_col=time_col,
        cat_col=cat_col,
        short_window=3,
        long_window=14,
    )
    print("   [+] Added: momentum_3d_vs_14d_expanding")
    
    # Add brand tier if BRAND column exists
    if "BRAND" in df.columns:
        df = add_brand_tier_feature(
            df,
            target_col=target_col,
            time_col=time_col,
            brand_col="BRAND",
            cat_col=cat_col,
        )
        print("   [+] Added: brand_tier_expanding")
    
    print(f"\n3. Feature preparation complete!")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Total rows: {len(df)}")
    
    return df


def create_prediction_windows(
    df: pd.DataFrame,
    temporal_features: List[str],
    static_features: List[str],
    input_size: int,
    time_col: str = "ACTUALSHIPDATE",
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Create sliding windows for prediction.
    
    Returns:
        X_temporal: Array of shape (n_windows, input_size, n_temporal_features)
        X_static: Array of shape (n_windows, n_static_features)
        prediction_dates: List of dates for each prediction window
    """
    X_temporal_list = []
    X_static_list = []
    prediction_dates = []
    
    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Create windows
    for i in range(len(df) - input_size):
        # Temporal window (last input_size rows)
        temporal_window = df.iloc[i:i+input_size][temporal_features].values
        X_temporal_list.append(temporal_window)
        
        # Static features (from the last row of the window)
        static_row = df.iloc[i+input_size-1][static_features].values
        X_static_list.append(static_row)
        
        # Prediction date (the date after the window)
        pred_date = df.iloc[i+input_size][time_col]
        prediction_dates.append(pred_date)
    
    X_temporal = np.array(X_temporal_list, dtype=np.float32)
    X_static = np.array(X_static_list, dtype=np.float32)
    
    return X_temporal, X_static, prediction_dates


def predict_with_hybrid_lstm(
    model: HybridLSTM,
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Make predictions using HybridLSTM model.
    
    Args:
        model: Trained HybridLSTM model
        X_temporal: Temporal features array (n_samples, input_size, n_temporal_features)
        X_static: Static features array (n_samples, n_static_features)
        device: Device to use for prediction
    
    Returns:
        predictions: Array of shape (n_samples, horizon)
    """
    model.eval()
    predictions = []
    
    # Convert to tensors
    X_temporal_tensor = torch.tensor(X_temporal, dtype=torch.float32).to(device)
    X_static_tensor = torch.tensor(X_static, dtype=torch.float32).to(device)
    
    # Batch prediction
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_temporal_tensor), batch_size):
            batch_temporal = X_temporal_tensor[i:i+batch_size]
            batch_static = X_static_tensor[i:i+batch_size]
            
            # Create dummy category tensor (not used in category-specific model)
            batch_cat = torch.zeros(len(batch_temporal), dtype=torch.long, device=device)
            
            # Predict
            batch_pred = model(batch_temporal, batch_static, batch_cat)
            predictions.append(batch_pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    return predictions


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="Make predictions using trained HybridLSTM model"
    )
    parser.add_argument("--category", type=str, default="DRY", help="Category to predict")
    parser.add_argument("--brand", type=str, help="Specific brand to predict (optional, if not set predicts all brands)")
    parser.add_argument("--brands", type=str, nargs="+", help="List of brands to predict (optional)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-path", type=str, help="Path to prediction data CSV")
    parser.add_argument("--model-path", type=str, help="Path to trained model (default: outputs/hybrid_lstm/{CATEGORY}/models/best_model.pth)")
    parser.add_argument("--save", action="store_true", help="Save predictions to CSV")
    parser.add_argument("--output-dir", type=str, default="outputs/hybrid_lstm", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--brand-col", type=str, default="BRAND", help="Name of brand column")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 80)
    print("HYBRID LSTM PREDICTION")
    print("=" * 80)
    print(f"\nCategory: {args.category}")
    if args.brand:
        print(f"Brand: {args.brand} (single brand)")
    elif args.brands:
        print(f"Brands: {', '.join(args.brands)} ({len(args.brands)} brands)")
    else:
        print(f"Brand: ALL (category-level)")
    print(f"Device: {device}")
    
    # Load configuration
    config = load_config(category=args.category)
    
    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(args.output_dir) / args.category / "models" / "best_model.pth"
    
    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print(f"Please train the model first using: python train_hybrid_lstm.py --category {args.category}")
        return 1
    
    print(f"Model path: {model_path}")
    
    # Load data
    print(f"\n[INFO] Loading data...")
    if args.data_path:
        df = pd.read_csv(args.data_path)
        print(f"  Loaded {len(df):,} rows from {args.data_path}")
        # Filter to category
        if config.data['cat_col'] in df.columns:
            df = df[df[config.data['cat_col']] == args.category].copy()
            print(f"  Filtered to {args.category}: {len(df):,} rows")
    else:
        # Load from config data directory
        df = load_data(
            config.data['data_dir'],
            config.data['years'],
            args.category,
            config.data['cat_col']
        )
    
    # Determine brands to predict
    brands_to_predict = []
    if args.brand:
        brands_to_predict = [args.brand]
    elif args.brands:
        brands_to_predict = args.brands
    elif args.brand_col in df.columns:
        # Get all unique brands in the data
        brands_to_predict = sorted(df[args.brand_col].unique().tolist())
        print(f"  Found {len(brands_to_predict)} brands in data: {', '.join(brands_to_predict[:5])}{'...' if len(brands_to_predict) > 5 else ''}")
    
    # Store original data for brand filtering
    df_all = df.copy()
    
    # Prepare features for all data first (before brand filtering)
    df_prepared_all = prepare_features(
        df_all,
        target_col=config.data['target_col'],
        time_col=config.data['time_col'],
        cat_col=config.data['cat_col'],
    )
    
    # Get feature lists from config
    if 'hybrid' in config.model:
        temporal_features = config.model['hybrid']['temporal_features']
        static_features = config.model['hybrid']['static_features']
    else:
        print("\n[ERROR] HybridLSTM configuration not found in config.yaml")
        return 1
    
    # Verify features exist
    missing_temporal = [f for f in temporal_features if f not in df_prepared_all.columns]
    missing_static = [f for f in static_features if f not in df_prepared_all.columns]
    
    if missing_temporal or missing_static:
        print(f"\n[ERROR] Missing features:")
        if missing_temporal:
            print(f"  Temporal: {missing_temporal}")
        if missing_static:
            print(f"  Static: {missing_static}")
        return 1
    
    print(f"\n[INFO] Feature groups:")
    print(f"  Temporal features ({len(temporal_features)}): {temporal_features[:3]}...")
    print(f"  Static features ({len(static_features)}): {static_features[:3]}...")
    
    # Load model
    print(f"\n[INFO] Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint contains model config
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        input_size = checkpoint.get('input_size', 30)
        horizon = checkpoint.get('horizon', 30)
        print(f"  Loaded model configuration from checkpoint")
    else:
        # Fallback to config file (for older models)
        print(f"  Using configuration from config file (older model format)")
        input_size = config.window.get('input_size', 30)
        horizon = config.window.get('horizon', 30)
        model_config = {
            'temporal_input_dim': len(temporal_features),
            'static_input_dim': len(static_features),
            'num_categories': 1,
            'cat_emb_dim': 5,
            'hidden_size': config.model.get('hidden_size', 128),
            'n_layers': config.model.get('n_layers', 2),
            'output_dim': horizon,
            'dropout_prob': config.model.get('dropout_prob', 0.2),
            'dense_hidden_sizes': config.model['hybrid'].get('dense_hidden_sizes', [64, 32]),
            'fusion_hidden_sizes': config.model['hybrid'].get('fusion_hidden_sizes', [64]),
        }
    
    # Create model with loaded configuration
    model = HybridLSTM(**model_config)
    
    # Load model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully")
    print(f"  Input size: {input_size} days")
    print(f"  Forecast horizon: {horizon} days")
    
    # ============================================================================
    # MAKE PREDICTIONS (PER BRAND IF SPECIFIED)
    # ============================================================================
    all_results = []
    
    if brands_to_predict and args.brand_col in df_prepared_all.columns:
        # Brand-level predictions
        print(f"\n[INFO] Making predictions for {len(brands_to_predict)} brand(s)...")
        
        for brand_idx, brand in enumerate(brands_to_predict, 1):
            print(f"\n  [{brand_idx}/{len(brands_to_predict)}] Processing brand: {brand}")
            
            # Filter data to this brand
            df_brand = df_prepared_all[df_prepared_all[args.brand_col] == brand].copy()
            
            if len(df_brand) < input_size:
                print(f"      [SKIP] Not enough data ({len(df_brand)} rows < {input_size} required)")
                continue
            
            # Create prediction windows for this brand
            X_temporal, X_static, prediction_dates = create_prediction_windows(
                df_brand,
                temporal_features,
                static_features,
                input_size,
                config.data['time_col'],
            )
            
            if len(X_temporal) == 0:
                print(f"      [SKIP] No prediction windows created")
                continue
            
            print(f"      Created {len(X_temporal):,} prediction windows")
            
            # Make predictions
            predictions = predict_with_hybrid_lstm(model, X_temporal, X_static, device)
            
            # Create results for this brand
            for i, pred_date in enumerate(prediction_dates):
                for day_ahead in range(horizon):
                    forecast_date = pred_date + pd.Timedelta(days=day_ahead)
                    
                    # Get actual value if it exists in the data
                    actual_mask = df_brand[config.data['time_col']] == forecast_date
                    actual_cbm = df_brand.loc[actual_mask, config.data['target_col']].values[0] if actual_mask.any() else None
                    
                    predicted_cbm = predictions[i, day_ahead]
                    
                    result = {
                        'brand': brand,
                        'prediction_date': pred_date,
                        'forecast_date': forecast_date,
                        'day_ahead': day_ahead + 1,
                        'predicted_cbm': predicted_cbm,
                        'actual_cbm': actual_cbm,
                    }
                    
                    # Calculate error metrics if actual value exists
                    if actual_cbm is not None:
                        result['error'] = predicted_cbm - actual_cbm
                        result['abs_error'] = abs(predicted_cbm - actual_cbm)
                        result['pct_error'] = ((predicted_cbm - actual_cbm) / actual_cbm * 100) if actual_cbm != 0 else None
                    else:
                        result['error'] = None
                        result['abs_error'] = None
                        result['pct_error'] = None
                    
                    all_results.append(result)
            
            print(f"      Generated {len(predictions) * horizon:,} predictions")
    
    else:
        # Category-level predictions (no brand filtering)
        print(f"\n[INFO] Making category-level predictions...")
        
        # Create prediction windows
        X_temporal, X_static, prediction_dates = create_prediction_windows(
            df_prepared_all,
            temporal_features,
            static_features,
            input_size,
            config.data['time_col'],
        )
        
        print(f"  Created {len(X_temporal):,} prediction windows")
        
        # Make predictions
        predictions = predict_with_hybrid_lstm(model, X_temporal, X_static, device)
        print(f"  Generated predictions: shape {predictions.shape}")
        
        # Create results
        for i, pred_date in enumerate(prediction_dates):
            for day_ahead in range(horizon):
                forecast_date = pred_date + pd.Timedelta(days=day_ahead)
                
                # Get actual value if it exists in the data
                actual_mask = df_prepared_all[config.data['time_col']] == forecast_date
                actual_cbm = df_prepared_all.loc[actual_mask, config.data['target_col']].values[0] if actual_mask.any() else None
                
                predicted_cbm = predictions[i, day_ahead]
                
                result = {
                    'prediction_date': pred_date,
                    'forecast_date': forecast_date,
                    'day_ahead': day_ahead + 1,
                    'predicted_cbm': predicted_cbm,
                    'actual_cbm': actual_cbm,
                }
                
                # Calculate error metrics if actual value exists
                if actual_cbm is not None:
                    result['error'] = predicted_cbm - actual_cbm
                    result['abs_error'] = abs(predicted_cbm - actual_cbm)
                    result['pct_error'] = ((predicted_cbm - actual_cbm) / actual_cbm * 100) if actual_cbm != 0 else None
                else:
                    result['error'] = None
                    result['abs_error'] = None
                    result['pct_error'] = None
                
                all_results.append(result)
    
    df_results = pd.DataFrame(all_results)
    
    # Filter by date range if specified
    if args.start:
        start_date = pd.to_datetime(args.start)
        df_results = df_results[df_results['forecast_date'] >= start_date]
    
    if args.end:
        end_date = pd.to_datetime(args.end)
        df_results = df_results[df_results['forecast_date'] <= end_date]
    
    print(f"\n[INFO] Prediction results:")
    print(f"  Total predictions: {len(df_results):,}")
    if 'brand' in df_results.columns:
        print(f"  Brands: {df_results['brand'].nunique()}")
    print(f"  Date range: {df_results['forecast_date'].min()} to {df_results['forecast_date'].max()}")
    
    # Calculate summary statistics for predictions with actuals
    df_with_actuals = df_results[df_results['actual_cbm'].notna()].copy()
    if len(df_with_actuals) > 0:
        mae = df_with_actuals['abs_error'].mean()
        rmse = np.sqrt((df_with_actuals['error'] ** 2).mean())
        mape = df_with_actuals['pct_error'].abs().mean()
        
        print(f"\n[INFO] Error Metrics (on {len(df_with_actuals):,} samples with actuals):")
        print(f"  MAE:  {mae:.2f} CBM")
        print(f"  RMSE: {rmse:.2f} CBM")
        print(f"  MAPE: {mape:.2f}%")
        
        # Brand-level metrics if applicable
        if 'brand' in df_with_actuals.columns:
            print(f"\n[INFO] Error Metrics by Brand:")
            brand_metrics = df_with_actuals.groupby('brand').agg({
                'abs_error': 'mean',
                'error': lambda x: np.sqrt((x ** 2).mean()),
                'pct_error': lambda x: x.abs().mean()
            }).round(2)
            brand_metrics.columns = ['MAE', 'RMSE', 'MAPE']
            print(brand_metrics.to_string())
    else:
        print(f"\n[INFO] No actual values found in data (future predictions only)")
    
    # Display sample
    print(f"\nSample predictions:")
    display_cols = [col for col in df_results.columns if col in ['brand', 'prediction_date', 'forecast_date', 'day_ahead', 'predicted_cbm', 'actual_cbm', 'error', 'abs_error']]
    print(df_results[display_cols].head(10).to_string(index=False))
    
    # Save if requested
    if args.save:
        output_dir = Path(args.output_dir) / args.category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'brand' in df_results.columns and len(brands_to_predict) == 1:
            # Single brand prediction
            brand_name = brands_to_predict[0].replace(' ', '_')
            output_file = output_dir / f"predictions_{brand_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        elif 'brand' in df_results.columns and len(brands_to_predict) > 1:
            # Multiple brands
            output_file = output_dir / f"predictions_all_brands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            # Category-level
            output_file = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df_results.to_csv(output_file, index=False)
        print(f"\n[+] Predictions saved to: {output_file}")
        
        # If multiple brands, also save summary by brand
        if 'brand' in df_results.columns and len(brands_to_predict) > 1:
            # Prediction summary
            summary_by_brand = df_results.groupby('brand')['predicted_cbm'].agg(['count', 'mean', 'sum']).reset_index()
            summary_by_brand.columns = ['brand', 'num_predictions', 'avg_predicted_cbm', 'total_predicted_cbm']
            
            # Add error metrics if actuals exist
            df_with_actuals = df_results[df_results['actual_cbm'].notna()]
            if len(df_with_actuals) > 0:
                error_summary = df_with_actuals.groupby('brand').agg({
                    'actual_cbm': 'mean',
                    'abs_error': 'mean',
                    'error': lambda x: np.sqrt((x ** 2).mean()),
                    'pct_error': lambda x: x.abs().mean()
                }).reset_index()
                error_summary.columns = ['brand', 'avg_actual_cbm', 'MAE', 'RMSE', 'MAPE']
                summary_by_brand = summary_by_brand.merge(error_summary, on='brand', how='left')
            
            summary_file = output_dir / f"summary_by_brand_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            summary_by_brand.to_csv(summary_file, index=False)
            print(f"[+] Brand summary saved to: {summary_file}")
            print(f"\nBrand Summary:")
            print(summary_by_brand.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
