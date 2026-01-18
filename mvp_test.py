"""MVP Test Script for MDLZ Warehouse Prediction System.

This script verifies the entire pipeline using a small subset of data:
- Loads 2023 and 2024 data
- Filters to DRY category only
- Trains for 20 epochs with spike_aware_mse loss
- Uses Vietnamese holidays (Tet, Mid-Autumn Festival, etc.)
- Generates test predictions and plots
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import date, timedelta
from typing import List

from config import load_config
from src.data import (
    DataReader,
    ForecastDataset,
    slicing_window_category,
    encode_categories,
    split_data,
    add_temporal_features
)
from src.models import RNNWithCategory
from src.training import Trainer
from src.utils import plot_difference, spike_aware_mse


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
    
    # Define Vietnamese holidays by year
    vietnam_holidays = {
        2023: {
            'tet': [date(2023, 1, 20), date(2023, 1, 21), date(2023, 1, 22), 
                   date(2023, 1, 23), date(2023, 1, 24), date(2023, 1, 25), date(2023, 1, 26)],
            'mid_autumn': [date(2023, 9, 29)],
            'independence': [date(2023, 9, 2)],
            'labor': [date(2023, 4, 30), date(2023, 5, 1)]
        },
        2024: {
            'tet': [date(2024, 2, 8), date(2024, 2, 9), date(2024, 2, 10),
                   date(2024, 2, 11), date(2024, 2, 12), date(2024, 2, 13), date(2024, 2, 14)],
            'mid_autumn': [date(2024, 9, 17)],
            'independence': [date(2024, 9, 2)],
            'labor': [date(2024, 4, 30), date(2024, 5, 1)]
        },
        2025: {
            'tet': [date(2025, 1, 27), date(2025, 1, 28), date(2025, 1, 29),
                   date(2025, 1, 30), date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2)],
            'mid_autumn': [date(2025, 10, 6)],
            'independence': [date(2025, 9, 2)],
            'labor': [date(2025, 4, 30), date(2025, 5, 1)]
        }
    }
    
    # Collect all holidays in the date range
    current = start_date
    while current <= end_date:
        year = current.year
        if year in vietnam_holidays:
            year_holidays = vietnam_holidays[year]
            holidays.extend(year_holidays['tet'])
            holidays.extend(year_holidays['mid_autumn'])
            holidays.extend(year_holidays['independence'])
            holidays.extend(year_holidays['labor'])
        current = date(year + 1, 1, 1)
    
    # Filter to date range and remove duplicates
    holidays = [h for h in holidays if start_date <= h <= end_date]
    holidays = sorted(list(set(holidays)))
    
    return holidays


def add_holiday_features_vietnam(
    df: pd.DataFrame,
    time_col: str = "ACTUALSHIPDATE",
    holiday_indicator_col: str = "holiday_indicator",
    days_until_holiday_col: str = "days_until_next_holiday"
) -> pd.DataFrame:
    """
    Add Vietnamese holiday-related features to DataFrame.
    
    Creates:
    - holiday_indicator: Binary (0 or 1) indicating if date is a Vietnamese holiday
    - days_until_next_holiday: Number of days until the next Vietnamese holiday
    
    Args:
        df: DataFrame with time column.
        time_col: Name of time column (should be datetime or date).
        holiday_indicator_col: Name for holiday indicator column.
        days_until_holiday_col: Name for days until next holiday column.
    
    Returns:
        DataFrame with added holiday features.
    """
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Get date range from data
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    
    # Extend range slightly to ensure we have next holidays for all dates
    extended_max = max_date + timedelta(days=365)
    
    # Get Vietnamese holidays
    holidays = get_vietnam_holidays(min_date, extended_max)
    holiday_set = set(holidays)
    
    # Initialize columns
    df[holiday_indicator_col] = 0
    df[days_until_holiday_col] = np.nan
    
    # Process each row
    for idx, row in df.iterrows():
        current_date = row[time_col].date()
        
        # Set holiday indicator
        if current_date in holiday_set:
            df.at[idx, holiday_indicator_col] = 1
        
        # Calculate days until next holiday
        next_holiday = None
        for holiday in holidays:
            if holiday > current_date:
                next_holiday = holiday
                break
        
        if next_holiday:
            days_until = (next_holiday - current_date).days
            df.at[idx, days_until_holiday_col] = days_until
        else:
            # If no holiday found (shouldn't happen with extended range), set to large value
            df.at[idx, days_until_holiday_col] = 365
    
    # Fill any remaining NaN values (shouldn't happen, but safety check)
    df[days_until_holiday_col] = df[days_until_holiday_col].fillna(365)
    
    return df


def main():
    """Main execution function for MVP test."""
    print("=" * 80)
    print("MVP TEST: MDLZ Warehouse Prediction System")
    print("=" * 80)
    
    # Load default configuration
    print("\n[1/8] Loading configuration...")
    config = load_config()
    
    # Override configuration for MVP test
    print("\n[2/8] Applying MVP test overrides...")
    config.set('data.years', [2023, 2024])
    config.set('training.epochs', 8)  # Increased to 20 epochs for better learning
    config.set('training.loss', 'spike_aware_mse')  # Force spike_aware_mse loss
    
    # Set dedicated output directory
    mvp_output_dir = "outputs/mvp_test"
    mvp_models_dir = os.path.join(mvp_output_dir, "models")
    os.makedirs(mvp_output_dir, exist_ok=True)
    os.makedirs(mvp_models_dir, exist_ok=True)
    config.set('output.output_dir', mvp_output_dir)
    config.set('output.model_dir', mvp_models_dir)
    config.set('output.save_model', True)  # Ensure model saving is enabled
    
    print(f"  - Data years: {config.data['years']}")
    print(f"  - Training epochs: {config.training['epochs']}")
    print(f"  - Loss function: spike_aware_mse (forced)")
    print(f"  - Output directory: {config.output['output_dir']}")
    
    # Load data
    print("\n[3/8] Loading data...")
    data_config = config.data
    data_reader = DataReader(
        data_dir=data_config['data_dir'],
        file_pattern=data_config['file_pattern']
    )
    
    # Load 2023 and 2024 data
    try:
        data = data_reader.load(years=data_config['years'])
    except FileNotFoundError:
        print("[WARNING] Combined year files not found, trying pattern-based loading...")
        file_prefix = data_config.get('file_prefix', 'Outboundreports')
        data = data_reader.load_by_file_pattern(
            years=data_config['years'],
            file_prefix=file_prefix
        )
    
    print(f"  - Loaded {len(data)} samples before filtering")
    
    # Fix DtypeWarning: Cast columns (0, 4) to string/category to resolve mixed types
    if len(data.columns) > 0:
        col_0 = data.columns[0]
        if col_0 in data.columns:
            data[col_0] = data[col_0].astype(str)
    if len(data.columns) > 4:
        col_4 = data.columns[4]
        if col_4 in data.columns:
            data[col_4] = data[col_4].astype(str)
    print("  - Fixed DtypeWarning by casting columns 0 and 4 to string")
    
    # Filter to DRY category only
    print("\n[4/8] Filtering data to DRY category...")
    cat_col = data_config['cat_col']
    samples_before = len(data)
    
    if cat_col not in data.columns:
        raise ValueError(f"Category column '{cat_col}' not found in data. Available columns: {list(data.columns)}")
    
    data = data[data[cat_col] == "DRY"].copy()
    samples_after = len(data)
    
    print(f"  - Samples before filtering: {samples_before}")
    print(f"  - Samples after filtering (CATEGORY == 'DRY'): {samples_after}")
    
    if samples_after == 0:
        raise ValueError("No samples found with CATEGORY == 'DRY'. Please check your data.")
    
    # Ensure time column is datetime and sort by time
    time_col = data_config['time_col']
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)
    
    # Feature engineering: Add temporal features
    print("\n[5/8] Adding temporal features...")
    data = add_temporal_features(
        data,
        time_col=time_col,
        month_sin_col="month_sin",
        month_cos_col="month_cos",
        dayofmonth_sin_col="dayofmonth_sin",
        dayofmonth_cos_col="dayofmonth_cos"
    )
    
    # Feature engineering: Add Vietnamese holiday features
    print("  - Adding Vietnamese holiday features...")
    data = add_holiday_features_vietnam(
        data,
        time_col=time_col,
        holiday_indicator_col="holiday_indicator",
        days_until_holiday_col="days_until_next_holiday"
    )
    
    # Encode categories
    print("  - Encoding categories...")
    data, cat2id, num_categories = encode_categories(data, cat_col)
    cat_id_col = data_config['cat_id_col']
    
    # Update config with num_categories for model
    config.set('model.num_categories', num_categories)
    print(f"  - Number of categories: {num_categories}")
    print(f"  - Category mapping: {cat2id}")
    
    # Split data
    print("\n[6/8] Splitting data...")
    train_data, val_data, test_data = split_data(
        data,
        train_size=data_config['train_size'],
        val_size=data_config['val_size'],
        test_size=data_config['test_size'],
        temporal=True
    )
    print(f"  - Train samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - Test samples: {len(test_data)}")
    
    # Create windows using slicing_window_category
    print("\n[7/8] Creating sliding windows...")
    window_config = config.window
    feature_cols = data_config['feature_cols']
    target_col = [data_config['target_col']]
    cat_col_list = [cat_id_col]
    time_col_list = [time_col]
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    
    print(f"  - Input size: {input_size}")
    print(f"  - Horizon: {horizon}")
    print(f"  - Feature columns: {feature_cols}")
    
    print("  - Creating training windows...")
    X_train, y_train, cat_train = slicing_window_category(
        train_data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=cat_col_list,
        time_col=time_col_list
    )
    
    print("  - Creating validation windows...")
    X_val, y_val, cat_val = slicing_window_category(
        val_data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=cat_col_list,
        time_col=time_col_list
    )
    
    print("  - Creating test windows...")
    X_test, y_test, cat_test = slicing_window_category(
        test_data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=cat_col_list,
        time_col=time_col_list
    )
    
    print(f"  - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"  - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Create datasets
    train_dataset = ForecastDataset(X_train, y_train, cat_train)
    val_dataset = ForecastDataset(X_val, y_val, cat_val)
    test_dataset = ForecastDataset(X_test, y_test, cat_test)
    
    # Create data loaders
    training_config = config.training
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['val_batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['test_batch_size'],
        shuffle=False
    )
    
    # Build model
    print("\n[8/8] Building model and trainer...")
    model_config = config.model
    model = RNNWithCategory(
        num_categories=num_categories,
        cat_emb_dim=model_config['cat_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        output_dim=model_config['output_dim']
    )
    print(f"  - Model: {model_config['name']}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build loss, optimizer, scheduler
    # Force spike_aware_mse loss function for Vietnamese market demand spikes
    print("  - Using spike_aware_mse loss (3x weight for top 20% QTY values)...")
    criterion = spike_aware_mse
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate']
    )
    
    scheduler_config = training_config.get('scheduler', {})
    scheduler_name = scheduler_config.get('name')
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 3),
            min_lr=scheduler_config.get('min_lr', 1e-5)
        )
    else:
        scheduler = None
    
    # Setup device
    device = torch.device(training_config['device'])
    print(f"  - Device: {device}")
    
    # Create trainer
    save_dir = config.output.get('model_dir')
    print(f"  - Model save directory: {save_dir}")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        log_interval=training_config['log_interval'],
        save_dir=save_dir
    )
    
    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    train_losses, val_losses = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config['epochs'],
        save_best=True,
        verbose=True
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    test_loss, y_true, y_pred = trainer.evaluate(
        test_loader,
        return_predictions=True
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test samples: {len(y_true)}")
    
    # Generate prediction plot
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    output_dir = config.output['output_dir']
    plot_path = os.path.join(output_dir, "test_predictions.png")
    
    # Use a reasonable number of samples for plotting
    n_samples = min(100, len(y_true))
    plot_difference(
        y_true[:n_samples],
        y_pred[:n_samples],
        save_path=plot_path,
        show=False
    )
    print(f"  - Prediction plot saved to: {plot_path}")
    
    print("\n" + "=" * 80)
    print("MVP TEST COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"  - Test predictions plot: {plot_path}")
    if save_dir:
        model_checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        print(f"  - Model checkpoint: {model_checkpoint_path}")
        if os.path.exists(model_checkpoint_path):
            print(f"    ✓ Model successfully saved")
        else:
            print(f"    ⚠ Warning: Model checkpoint not found at expected path")
    print(f"  - Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()

