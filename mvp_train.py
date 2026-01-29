"""
MVP Test Training Script for MDLZ Warehouse Prediction System
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import time
import json
import pickle

from config import load_config
from src.data import (
    DataReader,
    ForecastDataset,
    RollingGroupScaler,
    slicing_window,
    encode_brands
)
from src.data.preprocessing import add_features
from src.models import RNNForecastor
from src.training import Trainer


def train_single_model(data, config):
    """
    Train a single model with optional BRAND filtering.
    
    Args:
        data: Full DataFrame with all data
        config: Configuration object
    
    Returns:
        Dictionary with training results and metadata
    """
    print("TRAINING MODEL FOR ALL brands")
    print("=" * 80)
    
    data_config = config.data
    brand_col = data_config['brand_col']
    time_col = data_config['time_col']
    target_col = data_config['target_col']    
    
    # Update output directories with suffix
    mvp_output_dir = config.output.get('output_dir', 'outputs/mvp_test')
    mvp_models_dir = config.output.get('model_dir', os.path.join(mvp_output_dir, 'models'))

    os.makedirs(mvp_output_dir, exist_ok=True)
    os.makedirs(mvp_models_dir, exist_ok=True)
    config.set('output.output_dir', mvp_output_dir)
    config.set('output.model_dir', mvp_models_dir)
    
    # Ensure time column is datetime and sort by time
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)
    
    # Feature engineering: Add temporal features
    print("\n[5/8] Adding temporal features...")
    data = add_features(data, time_col, brand_col, target_col)
    

    # Encode brands
    print("  - Encoding brands...")
    data, brand2id, num_brands = encode_brands(data, brand_col)
    brand_id_col = data_config['brand_id_col']
    
    # Update config with num_brands for model
    config.set('model.num_brands', num_brands)
    print(f"  - Number of brands: {num_brands}")
    print(f"  - BRAND mapping: {brand2id}")
    
    
    # Fit scaler on training data and apply to all splits
    print(f"\n[6/8] Scaling target values in column '{target_col}'...")
    scaler = RollingGroupScaler(
        group_col=brand_id_col,
        time_col=time_col,
        feature_cols=target_col,
        lookback_months=6
    )
    # scaler = fit_scaler(train_data, target_col=target_col_for_model)
    test_end_date = pd.Timestamp("2025-01-01")
    data[time_col] = pd.to_datetime(data[time_col])
    scaler.fit(data, test_end_date)
    data_scaled = scaler.transform(data)
   
    
    # Create windows using slicing_window_BRAND
    print("\n[7/8] Creating sliding windows...")
    window_config = config.window
    feature_cols = data_config['feature_cols']
    # The model's supervised target column (may be residual or absolute)
    input_size = window_config['input_size']
    horizon = window_config['horizon']

    # Ensure model input_dim matches the final feature count (including new lunar encodings)
    num_features = len(feature_cols)
    config.set("model.input_dim", num_features)
    model_config = config.model

    print(f"  - Input size: {input_size}")
    print(f"  - Horizon: {horizon}")
    print(f"  - Feature columns: {feature_cols}")
    
    print("  - Creating training windows...")
    X_train, y_train, cat_train = slicing_window(
        data_scaled,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        brand_col=brand_id_col,
        time_col=time_col,
        label_start_date=None,
        label_end_date=None,
        return_dates=False
    )
    
    
    print(f"  - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Create datasets
    train_dataset = ForecastDataset(X_train, y_train, cat_train)
    
    # Create data loaders
    training_config = config.training
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True
    )
    
    # Build model
    print("\n[8/8] Building model and trainer...")
    model = RNNForecastor(
        num_brands=num_brands,
        brand_emb_dim=model_config['brand_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        output_dim=horizon
    )
    print(f"  - Model: {model_config['name']}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build loss, optimizer, scheduler
    # Force spike_aware_mse loss function for Vietnamese market demand spikes
    print("  - Using HuberLoss loss (delta=0.4)...")
    criterion = nn.HuberLoss(delta=0.4)
    
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
    
    # Convert config to dictionary for saving in checkpoint and metadata
    config_dict = config._config.copy()
    
    train_losses, val_losses = trainer.fit(
        train_loader=train_loader,
        val_loader=None,
        epochs=training_config['epochs'],
        save_best=True,
        verbose=True,
        config=config_dict
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save scaler for use in prediction
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  - Scaler saved to: {scaler_path}")
    
    # Save training metadata (including a compact text summary of this log)
    print("\n[9/9] Saving training metadata...")
    log_summary_lines = [
        f"Number of brands: {num_brands}",
        f"BRAND mapping: {brand2id}",
        f"Best validation loss: {trainer.best_val_loss:.4f}",
        f"Training time (seconds): {training_time:.2f}"
    ]
    metadata = {
        'training_config': {
            'epochs': training_config['epochs'],
            'batch_size': training_config['batch_size'],
            'learning_rate': training_config['learning_rate'],
            'loss_function': training_config['loss'],
            'device': training_config['device']
        },
        'model_config': dict(model_config),
        'data_config': {
            'years': data_config['years'],
            'brand_col': data_config['brand_col'],
            'feature_cols': data_config['feature_cols'],
            'target_col': data_config['target_col'],
            'daily_aggregation': True,  # Flag indicating daily aggregation was used
            'scaling': {
                'method': 'RobustScaler',
                'valid_group': list(scaler.valid_groups_)
            }
        },
        'window_config': dict(window_config),
        'training_results': {
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': trainer.best_val_loss,
            'training_time_seconds': training_time
        },
        'log_summary': "\n".join(log_summary_lines),
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  - Metadata saved to: {metadata_path}")
    
    # Generate prediction plot
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    output_dir = config.output['output_dir']
    
    
    result = {
        'output_dir': output_dir,
        'model_dir': save_dir,
        'training_time': training_time
    }
    
    return result


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
    # Use years from config.yaml instead of hardcoding
    config.set('data.years', config.data.get('years'))
    # Longer warm-up and lower LR for complex feature set
    # training.epochs and training.learning_rate now come from config.yaml
    config.set('training.learning_rate', 0.001)
    config.set('training.loss', 'spike_aware_mse')  # Force spike_aware_mse loss
    
    # Set base output directory
    mvp_output_dir = "outputs/mvp_test"
    mvp_models_dir = os.path.join(mvp_output_dir, "models")
    os.makedirs(mvp_output_dir, exist_ok=True)
    os.makedirs(mvp_models_dir, exist_ok=True)
    config.set('output.output_dir', mvp_output_dir)
    config.set('output.model_dir', mvp_models_dir)
    config.set('output.save_model', True)  # Ensure model saving is enabled
    
    # Get BRAND mode from config
    data_config = config.data
    
    print(f"  - Data years: {config.data['years']}")
    print(f"  - Training epochs: {config.training['epochs']}")
    print(f"  - Loss function: spike_aware_mse (forced)")
    print(f"  - Base output directory: {mvp_output_dir}")
    
    # Load data
    print("\n[3/8] Loading data...")
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
    
    # Fix DtypeWarning: Cast columns (0, 4) to string/BRAND to resolve mixed types
    if len(data.columns) > 0:
        col_0 = data.columns[0]
        if col_0 in data.columns:
            data[col_0] = data[col_0].astype(str)
    if len(data.columns) > 4:
        col_4 = data.columns[4]
        if col_4 in data.columns:
            data[col_4] = data[col_4].astype(str)
    print("  - Fixed DtypeWarning by casting columns 0 and 4 to string")
    
    # Get available brands
    brand_col = data_config['brand_col']
    if brand_col not in data.columns:
        raise ValueError(f"BRAND column '{brand_col}' not found in data. Available columns: {list(data.columns)}")
    
    available_brands = sorted(data[brand_col].unique().tolist())
    print(f"  - Available brands in data: {available_brands}")
    
    # Determine training tasks based on BRAND_mode
    
    # Execute training tasks
    results = []

    print(f"\n{'=' * 80}")
    print(f"TRAINING")
    print(f"{'=' * 80}")
    
    try:
        result = train_single_model(data, config)
        results.append(result)
        
        print(f"\n✓ Training completed successfully")
        print(f"  - Results saved to: {result['output_dir']}")
        print(f"  - Training time: {result['training_time']:.2f} seconds ({result['training_time']/60:.2f} minutes)")
    except Exception as e:
        print(f"\n Train failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # Print final summary
    print("\n" + "=" * 80)
    print("MVP TEST COMPLETE!")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"   - Output directory: {result['output_dir']}")
        print(f"   - Model checkpoint: {os.path.join(result['model_dir'], 'best_model.pth')}")
        print(f"   - Training time: {result['training_time']:.2f} seconds ({result['training_time']/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
