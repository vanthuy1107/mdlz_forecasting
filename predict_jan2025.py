"""Prediction script for full year 2025 using MVP test model.

This script loads the trained model from mvp_test.py and makes predictions
for full year 2025 data, filtering to DRY category only.

Two modes are supported:
1. Teacher Forcing (Test Evaluation): Uses actual ground truth QTY values from 2025 as features
2. Recursive (Production Forecast): Uses model's own predictions as inputs for future dates
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date

from config import load_config
from src.data import (
    DataReader,
    ForecastDataset,
    slicing_window_category,
    encode_categories,
    add_holiday_features,
    add_temporal_features
)
from src.data.preprocessing import get_us_holidays
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


def predict_recursive(
    model,
    device,
    initial_window_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    config,
    cat_id: int
):
    """
    Recursive prediction mode: uses model's own predictions as inputs.
    
    This function simulates a true production forecast where future QTY values
    are unknown. It uses the model's previous predictions to build the input
    window for subsequent predictions.
    
    Args:
        model: Trained PyTorch model (already on device and in eval mode)
        device: PyTorch device
        initial_window_data: DataFrame with last 30 days of historical data (Dec 2024)
                            Must have all feature columns and be sorted by time
        start_date: First date to predict (e.g., date(2025, 1, 1))
        end_date: Last date to predict (e.g., date(2025, 1, 31))
        config: Configuration object
        cat_id: Category ID (integer)
    
    Returns:
        DataFrame with columns: date, predicted, actual (if available)
    """
    window_config = config.window
    data_config = config.data
    
    input_size = window_config['input_size']
    feature_cols = data_config['feature_cols']
    target_col = data_config['target_col']
    time_col = data_config['time_col']
    
    # Validate initial window has enough data
    if len(initial_window_data) < input_size:
        raise ValueError(
            f"Initial window must have at least {input_size} samples, "
            f"got {len(initial_window_data)}"
        )
    
    # Get last input_size rows as the initial window
    window = initial_window_data.tail(input_size).copy()
    window = window.sort_values(time_col).reset_index(drop=True)
    
    predictions = []
    current_date = start_date
    
    print(f"  - Starting recursive prediction from {start_date} to {end_date}")
    print(f"  - Initial window: {window[time_col].min()} to {window[time_col].max()}")
    
    while current_date <= end_date:
        # Create feature vector for current_date using current window
        # Calendar features will be computed for current_date
        # QTY values come from the window (which includes previous predictions)
        
        # Get the feature values from the window (last input_size timesteps)
        # The window contains: [t-29, t-28, ..., t-1] where t is current_date
        window_features = window[feature_cols].values  # Shape: (input_size, n_features)
        
        # We need to update the last row's calendar features for current_date
        # But keep QTY from the window (which may be predicted)
        # For all features except QTY, compute for current_date
        
        # Create a row for current_date with correct calendar features
        current_datetime = pd.Timestamp(current_date)
        
        # Compute temporal features for current_date
        month = current_datetime.month
        dayofmonth = current_datetime.day
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        dayofmonth_sin = np.sin(2 * np.pi * (dayofmonth - 1) / 31)
        dayofmonth_cos = np.cos(2 * np.pi * (dayofmonth - 1) / 31)
        
        # Compute holiday features for current_date
        extended_end = end_date + timedelta(days=365)
        holidays = get_us_holidays(current_date, extended_end)
        holiday_set = set(holidays)
        
        holiday_indicator = 1 if current_date in holiday_set else 0
        
        # Calculate days until next holiday
        next_holiday = None
        for holiday in holidays:
            if holiday > current_date:
                next_holiday = holiday
                break
        
        if next_holiday:
            days_until_next_holiday = (next_holiday - current_date).days
        else:
            days_until_next_holiday = 365
        
        # Create input tensor from current window
        # The window contains features from dates [t-30, t-29, ..., t-1]
        # Calendar features in the window are correct for their respective dates
        # QTY values in the window include previous predictions for dates >= start_date
        X_window = torch.tensor(
            window_features,
            dtype=torch.float32
        ).unsqueeze(0).to(device)  # Shape: (1, input_size, n_features)
        
        cat_tensor = torch.tensor([cat_id], dtype=torch.long).to(device)
        
        # Make prediction
        with torch.no_grad():
            pred = model(X_window, cat_tensor).cpu().item()
        
        # Store prediction
        predictions.append({
            'date': current_date,
            'predicted': pred
        })
        
        # Update window: remove oldest row, add new row with prediction
        # Create new row with predicted QTY and calendar features for current_date
        new_row = window.iloc[-1:].copy()  # Copy last row as template
        new_row[time_col] = current_datetime
        new_row[target_col] = pred  # Use predicted QTY
        new_row['month_sin'] = month_sin
        new_row['month_cos'] = month_cos
        new_row['dayofmonth_sin'] = dayofmonth_sin
        new_row['dayofmonth_cos'] = dayofmonth_cos
        new_row['holiday_indicator'] = holiday_indicator
        new_row['days_until_next_holiday'] = days_until_next_holiday
        
        # Remove oldest row and append new row
        window = pd.concat([window.iloc[1:], new_row], ignore_index=True)
        window = window.sort_values(time_col).reset_index(drop=True)
        
        # Move to next date
        current_date += timedelta(days=1)
    
    predictions_df = pd.DataFrame(predictions)
    return predictions_df


def get_historical_window_data(
    historical_data: pd.DataFrame,
    end_date: date,
    config,
    num_days: int = 30
):
    """
    Extract the last N days of historical data to initialize recursive prediction window.
    
    Args:
        historical_data: DataFrame with historical data (e.g., 2024 data)
        end_date: Last date to include (e.g., date(2024, 12, 31))
        config: Configuration object
        num_days: Number of days to extract (default: 30, matching input_size)
    
    Returns:
        DataFrame with last num_days of data, sorted by time
    """
    time_col = config.data['time_col']
    
    # Filter to dates up to end_date
    historical_data = historical_data[
        pd.to_datetime(historical_data[time_col]).dt.date <= end_date
    ].copy()
    
    # Sort by time
    historical_data = historical_data.sort_values(time_col).reset_index(drop=True)
    
    # Get last num_days (group by date first, then take last num_days of unique dates)
    # Since there may be multiple rows per date (different categories), we need to handle this
    historical_data['date_only'] = pd.to_datetime(historical_data[time_col]).dt.date
    unique_dates = historical_data['date_only'].unique()
    unique_dates = sorted(unique_dates)
    
    if len(unique_dates) < num_days:
        print(f"  [WARNING] Only {len(unique_dates)} unique dates available, requested {num_days}")
        selected_dates = unique_dates
    else:
        selected_dates = unique_dates[-num_days:]
    
    window_data = historical_data[
        historical_data['date_only'].isin(selected_dates)
    ].copy()
    
    # Remove temporary column
    window_data = window_data.drop(columns=['date_only'])
    
    return window_data.sort_values(time_col).reset_index(drop=True)


def main():
    """Main prediction function with both Teacher Forcing and Recursive modes."""
    print("=" * 80)
    print("JANUARY 2025 PREDICTION - TEACHER FORCING vs RECURSIVE MODES")
    print("=" * 80)
    
    # Load configuration
    print("\n[1/7] Loading configuration...")
    config = load_config()
    
    # Override to match MVP test settings
    config.set('data.years', [2024])  # For loading category mapping reference
    data_config = config.data
    
    # Load 2024 data to get category mapping and historical window
    print("\n[2/7] Loading historical 2024 data...")
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
    
    cat_id = cat2id.get("DRY")
    if cat_id is None:
        raise ValueError("DRY category not found in training data")
    
    print(f"  - Category mapping: {cat2id}")
    print(f"  - Number of categories: {num_categories}")
    print(f"  - DRY category ID: {cat_id}")
    
    # Prepare historical 2024 data for recursive prediction initialization
    print("\n[3/7] Preparing historical 2024 data...")
    historical_data_prepared = prepare_prediction_data(ref_data.copy(), config, cat2id)
    
    # Get last 30 days of December 2024 for initial window
    historical_window = get_historical_window_data(
        historical_data_prepared,
        end_date=date(2024, 12, 31),
        config=config,
        num_days=config.window['input_size']
    )
    print(f"  - Historical window: {len(historical_window)} samples")
    print(f"  - Date range: {historical_window[data_config['time_col']].min()} to {historical_window[data_config['time_col']].max()}")
    
    # Load 2025 data
    print("\n[4/7] Loading 2025 data...")
    data_2025_path = Path("dataset/test/data_2025.csv")
    
    if not data_2025_path.exists():
        raise FileNotFoundError(
            f"2025 data file not found at: {data_2025_path.absolute()}"
        )
    
    print(f"  - Loading from: {data_2025_path}")
    try:
        data_2025 = pd.read_csv(data_2025_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        try:
            data_2025 = pd.read_csv(data_2025_path, encoding='latin-1', low_memory=False)
        except Exception as e:
            data_2025 = pd.read_csv(data_2025_path, encoding='cp1252', low_memory=False)
    print(f"  - Loaded {len(data_2025)} samples")
    
    # Filter to DRY category and January 2025 only
    print("\n[5/7] Filtering data...")
    cat_col = data_config['cat_col']
    time_col = data_config['time_col']
    
    # Filter to DRY category
    data_2025 = data_2025[data_2025[cat_col] == "DRY"].copy()
    print(f"  - After DRY filter: {len(data_2025)} samples")
    
    # Filter to January 2025 only
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
    data_2025_prepared = prepare_prediction_data(data_2025, config, cat2id)
    
    # Create prediction windows (for teacher forcing mode)
    X_pred, y_actual, cat_pred, pred_dates = create_prediction_windows(data_2025_prepared, config)
    
    if len(X_pred) == 0:
        raise ValueError(
            f"Not enough data to create prediction windows. "
            f"Need at least {config.window['input_size']} samples."
        )
    
    # Load trained model
    print("\n[6/7] Loading trained model...")
    model_path = Path("outputs/mvp_test/models/best_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please run mvp_test.py first to train the model."
        )
    
    model, device = load_model_for_prediction(str(model_path), config)
    
    # ========================================================================
    # MODE 1: TEACHER FORCING (Test Evaluation) - Uses actual 2025 values
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODE 1: TEACHER FORCING (Test Evaluation)")
    print("=" * 80)
    print("Using actual ground truth QTY values from 2025 as features.")
    print("This mode is suitable for model evaluation but not production forecasting.")
    
    # Create dataset and dataloader
    pred_dataset = ForecastDataset(X_pred, y_actual, cat_pred)
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=config.training['test_batch_size'],
        shuffle=False
    )
    
    # Create trainer for prediction
    trainer = Trainer(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        device=device
    )
    
    y_true_tf, y_pred_tf = trainer.predict(pred_loader)
    
    print(f"  - Predictions made: {len(y_pred_tf)} samples")
    print(f"  - Prediction date range: {pred_dates.min()} to {pred_dates.max()}")
    
    # Calculate metrics
    mse_tf = np.mean((y_true_tf.flatten() - y_pred_tf.flatten()) ** 2)
    mae_tf = np.mean(np.abs(y_true_tf.flatten() - y_pred_tf.flatten()))
    rmse_tf = np.sqrt(mse_tf)
    
    print(f"\n  - MSE:  {mse_tf:.4f}")
    print(f"  - MAE:  {mae_tf:.4f}")
    print(f"  - RMSE: {rmse_tf:.4f}")
    
    # ========================================================================
    # MODE 2: RECURSIVE (Production Forecast) - Uses model's own predictions
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODE 2: RECURSIVE (Production Forecast)")
    print("=" * 80)
    print("Using model's own predictions as inputs for future dates.")
    print("This mode simulates true production forecasting.")
    
    # Filter historical window to same category
    historical_window_filtered = historical_window[
        historical_window[data_config['cat_id_col']] == cat_id
    ].copy()
    
    if len(historical_window_filtered) < config.window['input_size']:
        # If not enough data for this category, use all data and duplicate
        print(f"  [WARNING] Only {len(historical_window_filtered)} samples for category, need {config.window['input_size']}")
        historical_window_filtered = historical_window[
            historical_window[data_config['cat_id_col']] == cat_id
        ].copy()
        # Take last available samples and repeat if needed
        if len(historical_window_filtered) > 0:
            last_row = historical_window_filtered.iloc[-1:].copy()
            while len(historical_window_filtered) < config.window['input_size']:
                historical_window_filtered = pd.concat([historical_window_filtered, last_row], ignore_index=True)
        else:
            raise ValueError(f"No historical data found for category ID {cat_id}")
    
    # Get last input_size samples
    historical_window_filtered = historical_window_filtered.tail(config.window['input_size']).copy()
    historical_window_filtered = historical_window_filtered.sort_values(time_col).reset_index(drop=True)
    
    # Run recursive prediction
    recursive_preds = predict_recursive(
        model=model,
        device=device,
        initial_window_data=historical_window_filtered,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        config=config,
        cat_id=cat_id
    )
    
    # Merge with actual values for comparison
    # Get actual values grouped by date
    actuals_by_date = data_2025_prepared.groupby(
        pd.to_datetime(data_2025_prepared[time_col]).dt.date
    )[data_config['target_col']].sum().reset_index()
    actuals_by_date.columns = ['date', 'actual']
    actuals_by_date['date'] = pd.to_datetime(actuals_by_date['date']).dt.date
    
    recursive_results = recursive_preds.merge(
        actuals_by_date,
        on='date',
        how='left'
    )
    
    # Calculate metrics (only for dates with actuals)
    recursive_results_with_actuals = recursive_results[recursive_results['actual'].notna()]
    
    if len(recursive_results_with_actuals) > 0:
        mse_rec = np.mean((recursive_results_with_actuals['actual'] - recursive_results_with_actuals['predicted']) ** 2)
        mae_rec = np.mean(np.abs(recursive_results_with_actuals['actual'] - recursive_results_with_actuals['predicted']))
        rmse_rec = np.sqrt(mse_rec)
        
        print(f"\n  - Predictions made: {len(recursive_preds)} samples")
        print(f"  - Samples with actuals: {len(recursive_results_with_actuals)}")
        print(f"  - MSE:  {mse_rec:.4f}")
        print(f"  - MAE:  {mae_rec:.4f}")
        print(f"  - RMSE: {rmse_rec:.4f}")
    else:
        print(f"\n  - Predictions made: {len(recursive_preds)} samples")
        print(f"  - No actual values available for comparison")
        mse_rec = mae_rec = rmse_rec = np.nan
    
    # ========================================================================
    # COMPARISON AND SAVING
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: TEACHER FORCING vs RECURSIVE")
    print("=" * 80)
    
    # Prepare teacher forcing results for comparison
    if isinstance(pred_dates, pd.DatetimeIndex):
        tf_dates = pred_dates.date
    else:
        tf_dates = pd.to_datetime(pred_dates).dt.date
    
    tf_results = pd.DataFrame({
        'date': tf_dates,
        'actual': y_true_tf.flatten(),
        'predicted': y_pred_tf.flatten()
    })
    tf_results = tf_results.groupby('date').agg({
        'actual': 'sum',
        'predicted': 'sum'
    }).reset_index()
    
    # Compare metrics
    print(f"\nMetrics Comparison:")
    print(f"{'Metric':<15} {'Teacher Forcing':<20} {'Recursive':<20} {'Difference':<20}")
    print("-" * 75)
    if not np.isnan(mae_rec):
        print(f"{'MAE':<15} {mae_tf:<20.4f} {mae_rec:<20.4f} {mae_rec - mae_tf:<20.4f}")
        print(f"{'RMSE':<15} {rmse_tf:<20.4f} {rmse_rec:<20.4f} {rmse_rec - rmse_tf:<20.4f}")
        print(f"{'MSE':<15} {mse_tf:<20.4f} {mse_rec:<20.4f} {mse_rec - mse_tf:<20.4f}")
        print(f"\nError increase: {(mae_rec / mae_tf - 1) * 100:.2f}% (MAE)")
        print(f"Error increase: {(rmse_rec / rmse_tf - 1) * 100:.2f}% (RMSE)")
    else:
        print(f"{'MAE':<15} {mae_tf:<20.4f} {'N/A':<20}")
        print(f"{'RMSE':<15} {rmse_tf:<20.4f} {'N/A':<20}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_dir = Path("outputs/mvp_test/predictions_jan2025")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Teacher Forcing results
    tf_output_path = output_dir / "predictions_teacher_forcing.csv"
    tf_results['date'] = pd.to_datetime(tf_results['date']).dt.strftime('%m/%d/%Y')
    tf_results['error'] = tf_results['actual'] - tf_results['predicted']
    tf_results['abs_error'] = np.abs(tf_results['error'])
    tf_results = tf_results.sort_values('date')
    tf_results.to_csv(tf_output_path, index=False)
    print(f"  - Teacher Forcing results: {tf_output_path}")
    
    # Save Recursive results
    rec_output_path = output_dir / "predictions_recursive.csv"
    recursive_results['date'] = pd.to_datetime(recursive_results['date']).dt.strftime('%m/%d/%Y')
    recursive_results['error'] = recursive_results['actual'] - recursive_results['predicted']
    recursive_results['abs_error'] = np.abs(recursive_results['error'])
    recursive_results = recursive_results.sort_values('date')
    recursive_results.to_csv(rec_output_path, index=False)
    print(f"  - Recursive results: {rec_output_path}")
    
    # Save comparison summary
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("January 2025 Prediction Comparison: Teacher Forcing vs Recursive\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: outputs/mvp_test/models/best_model.pth\n")
        f.write(f"Data: January 2025, DRY category only\n")
        f.write(f"Data source: dataset/test/data_2025.csv\n\n")
        
        f.write("Teacher Forcing Mode (Test Evaluation):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Description: Uses actual ground truth QTY values from 2025 as features\n")
        f.write(f"  Suitable for: Model evaluation on test set\n")
        f.write(f"  Number of predictions: {len(y_pred_tf)}\n")
        f.write(f"  Date range: {pred_dates.min()} to {pred_dates.max()}\n")
        f.write(f"  MSE:  {mse_tf:.4f}\n")
        f.write(f"  MAE:  {mae_tf:.4f}\n")
        f.write(f"  RMSE: {rmse_tf:.4f}\n\n")
        
        f.write("Recursive Mode (Production Forecast):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Description: Uses model's own predictions as inputs\n")
        f.write(f"  Suitable for: Production forecasting of unknown future dates\n")
        f.write(f"  Number of predictions: {len(recursive_preds)}\n")
        f.write(f"  Date range: 2025-01-01 to 2025-01-31\n")
        if not np.isnan(mae_rec):
            f.write(f"  MSE:  {mse_rec:.4f}\n")
            f.write(f"  MAE:  {mae_rec:.4f}\n")
            f.write(f"  RMSE: {rmse_rec:.4f}\n\n")
        else:
            f.write(f"  MSE:  N/A (no actual values available)\n")
            f.write(f"  MAE:  N/A (no actual values available)\n")
            f.write(f"  RMSE: N/A (no actual values available)\n\n")
        
        if not np.isnan(mae_rec):
            f.write("Comparison:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  MAE increase:  {(mae_rec / mae_tf - 1) * 100:.2f}%\n")
            f.write(f"  RMSE increase: {(rmse_rec / rmse_tf - 1) * 100:.2f}%\n")
            f.write(f"  MSE increase:  {(mse_rec / mse_tf - 1) * 100:.2f}%\n")
            f.write(f"\n  Note: Recursive mode shows higher error due to error accumulation.\n")
            f.write(f"  This is expected and represents true production forecast performance.\n")
    
    print(f"  - Comparison summary: {summary_path}")
    
    # Generate plots
    if len(y_true_tf) > 0:
        plot_path_tf = output_dir / "predictions_teacher_forcing.png"
        n_samples = min(100, len(y_true_tf))
        plot_difference(
            y_true_tf[:n_samples],
            y_pred_tf[:n_samples],
            save_path=str(plot_path_tf),
            show=False
        )
        print(f"  - Teacher Forcing plot: {plot_path_tf}")
    
    if len(recursive_results_with_actuals) > 0:
        plot_path_rec = output_dir / "predictions_recursive.png"
        plot_difference(
            recursive_results_with_actuals['actual'].values,
            recursive_results_with_actuals['predicted'].values,
            save_path=str(plot_path_rec),
            show=False
        )
        print(f"  - Recursive plot: {plot_path_rec}")
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

