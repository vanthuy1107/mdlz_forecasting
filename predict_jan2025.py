"""Prediction script for January 2025 using MVP test model.

This script loads the trained model from mvp_test.py and makes predictions
for January 2025 data, filtering to DRY category only.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from config import load_config
from src.data import (
    DataReader,
    ForecastDataset,
    slicing_window_category,
    encode_categories,
    add_holiday_features,
    add_temporal_features
)
from src.models import RNNWithCategory
from src.training import Trainer
from src.utils import plot_difference


def load_model_for_prediction(model_path: str, config):
    """Load trained model from checkpoint."""
    # Build model with same architecture
    model_config = config.model
    num_categories = model_config.get('num_categories')
    if num_categories is None:
        raise ValueError("num_categories must be set in config")
    
    model = RNNWithCategory(
        num_categories=num_categories,
        cat_emb_dim=model_config['cat_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        output_dim=model_config['output_dim']
    )
    
    # Load checkpoint
    device = torch.device(config.training['device'])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  - Model loaded from: {model_path}")
    print(f"  - Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    return model, device


def prepare_prediction_data(data, config, cat2id):
    """
    Prepare data for prediction using the same preprocessing as training.
    
    Args:
        data: DataFrame with raw data
        config: Configuration object
        cat2id: Category to ID mapping from training
    
    Returns:
        Prepared DataFrame ready for window creation
    """
    data_config = config.data
    time_col = data_config['time_col']
    cat_col = data_config['cat_col']
    cat_id_col = data_config['cat_id_col']
    
    # Ensure time column is datetime and sort
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)
    
    # Add temporal features
    print("  - Adding temporal features...")
    data = add_temporal_features(
        data,
        time_col=time_col,
        month_sin_col="month_sin",
        month_cos_col="month_cos",
        dayofmonth_sin_col="dayofmonth_sin",
        dayofmonth_cos_col="dayofmonth_cos"
    )
    
    # Add holiday features
    print("  - Adding holiday features...")
    data = add_holiday_features(
        data,
        time_col=time_col,
        holiday_indicator_col="holiday_indicator",
        days_until_holiday_col="days_until_next_holiday"
    )
    
    # Encode categories using the same mapping from training
    print("  - Encoding categories...")
    data = data.copy()
    data[cat_id_col] = data[cat_col].map(cat2id)
    
    # Check for unknown categories
    unknown_cats = data[data[cat_id_col].isna()][cat_col].unique()
    if len(unknown_cats) > 0:
        print(f"  [WARNING] Unknown categories found: {unknown_cats}")
        print(f"  [WARNING] These will be filtered out")
        data = data[data[cat_id_col].notna()].copy()
    
    return data


def create_prediction_windows(data, config):
    """
    Create sliding windows for prediction and extract corresponding dates.
    
    Returns:
        Tuple of (X_pred, y_pred, cat_pred, dates) where dates are the
        prediction dates corresponding to each window.
    """
    window_config = config.window
    data_config = config.data
    
    feature_cols = data_config['feature_cols']
    target_col = [data_config['target_col']]
    cat_id_col = data_config['cat_id_col']
    time_col_name = data_config['time_col']
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    
    print(f"  - Creating prediction windows (input_size={input_size}, horizon={horizon})...")
    
    # Use slicing_window_category but also extract dates
    X_pred, y_pred, cat_pred = slicing_window_category(
        data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=[cat_id_col],
        time_col=[time_col_name]
    )
    
    # Extract dates corresponding to each prediction window
    # The date for prediction at index i corresponds to data point at input_size + i
    # Normalize to date only (remove time component)
    dates = []
    cat_col_name = cat_id_col
    
    for cat, g in data.groupby(cat_col_name, sort=False):
        g = g.sort_values(time_col_name)
        # Extract dates for each window (date at prediction point)
        # Normalize to date only (remove time)
        for i in range(len(g) - input_size - horizon + 1):
            pred_datetime = g.iloc[i + input_size][time_col_name]
            # Convert to date only (normalize time to 00:00:00)
            if pd.api.types.is_datetime64_any_dtype(pd.Series([pred_datetime])):
                pred_date = pd.to_datetime(pred_datetime).normalize()  # Keeps as datetime but time = 00:00:00
            else:
                pred_date = pd.to_datetime(pred_datetime).normalize()
            dates.append(pred_date)
    
    dates = pd.to_datetime(dates).normalize()  # Ensure all dates are normalized
    
    print(f"  - Created {len(X_pred)} prediction windows")
    return X_pred, y_pred, cat_pred, dates


def main():
    """Main prediction function."""
    print("=" * 80)
    print("JANUARY 2025 PREDICTION USING MVP TEST MODEL")
    print("=" * 80)
    
    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config()
    
    # Override to match MVP test settings
    config.set('data.years', [2024])  # For loading category mapping reference
    data_config = config.data
    
    # Load 2024 data to get category mapping (same as MVP test)
    print("\n[2/6] Loading reference data to get category mapping...")
    data_reader = DataReader(
        data_dir=data_config['data_dir'],
        file_pattern=data_config['file_pattern']
    )
    
    try:
        ref_data = data_reader.load(years=[2024])
    except FileNotFoundError:
        print("[WARNING] Trying pattern-based loading...")
        ref_data = data_reader.load_by_file_pattern(
            years=[2024],
            file_prefix=data_config.get('file_prefix', 'Outboundreports')
        )
    
    # Filter to DRY and encode to get category mapping
    ref_data = ref_data[ref_data[data_config['cat_col']] == "DRY"].copy()
    _, cat2id, num_categories = encode_categories(ref_data, data_config['cat_col'])
    config.set('model.num_categories', num_categories)
    
    print(f"  - Category mapping: {cat2id}")
    print(f"  - Number of categories: {num_categories}")
    
    # Load 2025 data
    print("\n[3/6] Loading January 2025 data...")
    # Load from dataset/test/data_2025.csv
    data_2025_path = Path("dataset/test/data_2025.csv")
    
    if not data_2025_path.exists():
        raise FileNotFoundError(
            f"2025 data file not found at: {data_2025_path.absolute()}"
        )
    
    print(f"  - Loading from: {data_2025_path}")
    # Try different encodings if needed
    try:
        data_2025 = pd.read_csv(data_2025_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        try:
            data_2025 = pd.read_csv(data_2025_path, encoding='latin-1', low_memory=False)
        except Exception as e:
            data_2025 = pd.read_csv(data_2025_path, encoding='cp1252', low_memory=False)
    print(f"  - Loaded {len(data_2025)} samples")
    
    # Filter to DRY category and January 2025
    print("\n[4/6] Filtering data...")
    cat_col = data_config['cat_col']
    time_col = data_config['time_col']
    
    # Filter to DRY category
    data_2025 = data_2025[data_2025[cat_col] == "DRY"].copy()
    print(f"  - After DRY filter: {len(data_2025)} samples")
    
    # Filter to January 2025
    if not pd.api.types.is_datetime64_any_dtype(data_2025[time_col]):
        data_2025[time_col] = pd.to_datetime(data_2025[time_col])
    
    data_2025 = data_2025[
        (data_2025[time_col] >= '2025-01-01') & 
        (data_2025[time_col] < '2025-02-01')
    ].copy()
    print(f"  - After January 2025 filter: {len(data_2025)} samples")
    
    if len(data_2025) == 0:
        raise ValueError("No data found for January 2025 with DRY category")
    
    # Prepare data (add features, encode categories)
    print("\n[5/6] Preparing prediction data...")
    data_2025_prepared = prepare_prediction_data(data_2025, config, cat2id)
    
    # For prediction, we need historical context (last 30 days before Jan 2025)
    # If we have historical data, we should include it
    # For now, we'll use what we have in January 2025
    # Note: We need at least input_size samples to create windows
    
    # Create prediction windows
    X_pred, y_actual, cat_pred, pred_dates = create_prediction_windows(data_2025_prepared, config)
    
    if len(X_pred) == 0:
        raise ValueError(
            f"Not enough data to create prediction windows. "
            f"Need at least {config.window['input_size']} samples."
        )
    
    # Load trained model
    print("\n[6/6] Loading trained model...")
    model_path = Path("outputs/mvp_test/models/best_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please run mvp_test.py first to train the model."
        )
    
    model, device = load_model_for_prediction(str(model_path), config)
    
    # Create dataset and dataloader
    pred_dataset = ForecastDataset(X_pred, y_actual, cat_pred)
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=config.training['test_batch_size'],
        shuffle=False
    )
    
    # Make predictions
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS")
    print("=" * 80)
    
    # Create trainer for prediction
    trainer = Trainer(
        model=model,
        criterion=nn.MSELoss(),  # Not used for prediction, but required
        optimizer=torch.optim.Adam(model.parameters()),  # Not used, but required
        device=device
    )
    
    y_true, y_pred = trainer.predict(pred_loader)
    
    print(f"  - Predictions made: {len(y_pred)} samples")
    print(f"  - Prediction date range: {pred_dates.min()} to {pred_dates.max()}")
    
    # Calculate metrics
    mse = np.mean((y_true.flatten() - y_pred.flatten()) ** 2)
    mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
    rmse = np.sqrt(mse)
    
    print(f"\n  - MSE: {mse:.4f}")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    
    # Save predictions
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_dir = Path("outputs/mvp_test/predictions_jan2025")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions to CSV with dates (date only, no time)
    # Convert dates to date strings (MM/DD/YYYY format) for CSV output
    if isinstance(pred_dates, pd.DatetimeIndex):
        date_strings = pred_dates.strftime('%m/%d/%Y')
    else:
        date_strings = pd.to_datetime(pred_dates).dt.strftime('%m/%d/%Y')
    
    predictions_df = pd.DataFrame({
        'date': date_strings,
        'actual': y_true.flatten(),
        'predicted': y_pred.flatten(),
        'error': (y_true.flatten() - y_pred.flatten()),
        'abs_error': np.abs(y_true.flatten() - y_pred.flatten())
    })
    
    # Group by date and aggregate (sum for actual/predicted, sum for errors)
    predictions_df_grouped = predictions_df.groupby('date').agg({
        'actual': 'sum',
        'predicted': 'sum',
        'error': 'sum',
        'abs_error': 'sum'
    }).reset_index()
    
    # Recalculate error after aggregation (should match, but ensure consistency)
    predictions_df_grouped['error'] = predictions_df_grouped['actual'] - predictions_df_grouped['predicted']
    predictions_df_grouped['abs_error'] = np.abs(predictions_df_grouped['error'])
    
    # Sort by date to ensure chronological order
    # Convert to datetime for proper sorting, then back to string
    predictions_df_grouped['date_dt'] = pd.to_datetime(predictions_df_grouped['date'], format='%m/%d/%Y')
    predictions_df_grouped = predictions_df_grouped.sort_values('date_dt').reset_index(drop=True)
    predictions_df_grouped = predictions_df_grouped.drop(columns=['date_dt'])
    
    csv_path = output_dir / "predictions_jan2025.csv"
    predictions_df_grouped.to_csv(csv_path, index=False)
    print(f"  - Predictions saved to: {csv_path}")
    print(f"  - Grouped by date: {len(predictions_df_grouped)} unique dates")
    
    # Generate plot
    plot_path = output_dir / "predictions_jan2025.png"
    n_samples = min(100, len(y_true))
    plot_difference(
        y_true[:n_samples],
        y_pred[:n_samples],
        save_path=str(plot_path),
        show=False
    )
    print(f"  - Plot saved to: {plot_path}")
    
    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("January 2025 Prediction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: outputs/mvp_test/models/best_model.pth\n")
        f.write(f"Data: January 2025, DRY category only\n")
        f.write(f"Data source: dataset/test/data_2025.csv\n")
        f.write(f"Number of predictions: {len(y_pred)}\n")
        f.write(f"Date range: {pred_dates.min()} to {pred_dates.max()}\n\n")
        f.write("Metrics:\n")
        f.write(f"  MSE:  {mse:.4f}\n")
        f.write(f"  MAE:  {mae:.4f}\n")
        f.write(f"  RMSE: {rmse:.4f}\n")
    print(f"  - Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

