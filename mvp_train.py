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
from src.utils import spike_aware_huber, seed_everything, seed_worker, SEED

seed_everything(SEED)

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
    config.set('output.save_model', True) 
    
    # Ensure time column is datetime and sort by time
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)
    
    # Feature engineering: Add temporal features
    print("\n[5/8] Adding temporal features...")
    data = add_features(data, time_col, brand_col, target_col)
    target_col = "residual"

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
        lookback_months=3
    )
    y = data[target_col]
    print("\nBEFORE SCALE:")
    print("min:", y.min())
    print("max:", y.max())
    print("p50:", y.quantile(0.5))
    print("p90:", y.quantile(0.9))
    print("p99:", y.quantile(0.99))
    test_end_date = pd.Timestamp("2025-01-01")
    data[time_col] = pd.to_datetime(data[time_col])
    scaler.fit(data, test_end_date)
    data_scaled = scaler.transform(data)

    y = data_scaled[target_col]
    print("\nAFTER SCALE:")
    print("min:", y.min())
    print("max:", y.max())
    print("p50:", y.quantile(0.5))
    print("p90:", y.quantile(0.9))
    print("p99:", y.quantile(0.99))
   
    
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
    print(f"  - Embedding dims: {model_config['brand_emb_dim']}")
    print(f"  - Feature dims: {model_config['input_dim']}")
    print(f"  - Feature columns: {feature_cols}")
    
    print("  - Creating training windows...")
    X_train, y_train, baselines_train, cat_train = slicing_window(
        data_scaled,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        baseline_col='baseline',
        brand_col=brand_id_col,
        time_col=time_col,
        label_start_date=None,
        label_end_date=test_end_date,
        return_dates=False
    )
    
    
    print(f"  - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Create datasets
    train_dataset = ForecastDataset(X_train, y_train, cat_train)
    
    # Create data loaders
    training_config = config.training
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Build model
    print("\n[8/8] Building model and trainer...")
    model = RNNForecastor(
        num_brands=num_brands,
        brand_emb_dim=model_config['brand_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        dropout_prob=model_config['dropout_prob'],
        output_dim=model_config['output_dim']
    )
    print(f"  - Model: {model_config['name']}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build loss, optimizer, scheduler
    # w_loss = [1.0, 0.0]
    # criterion = lambda yhat, y: (
    #     w_loss[0] * nn.HuberLoss(delta=0.9)(yhat, y)
    #     + w_loss[1] * nn.MSELoss()(yhat, y)
    # )
    # criterion = lambda yhat, y: (
    #     spike_aware_huber(yhat, y, delta=0.9, alpha=2.0)
    # )
    criterion = nn.HuberLoss(delta=0.8)
    
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
    
    # Set base output directory
    mvp_output_dir = "outputs/mvp_test"
    mvp_models_dir = os.path.join(mvp_output_dir, "models")
    os.makedirs(mvp_output_dir, exist_ok=True)
    os.makedirs(mvp_models_dir, exist_ok=True)
    config.set('output.output_dir', mvp_output_dir)
    config.set('output.model_dir', mvp_models_dir)
    
    # Get BRAND mode from config
    data_config = config.data
    
    print(f"  - Data years: {config.data['years']}")
    print(f"  - Training epochs: {config.training['epochs']}")
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
        data = data[~data["BRAND"].isin(["KINH DO CAKE", "LU"])]
    except FileNotFoundError:
        print("[WARNING] Combined year files not found, trying pattern-based loading...")
        file_prefix = data_config.get('file_prefix', 'Outboundreports')
        data = data_reader.load_by_file_pattern(
            years=data_config['years'],
            file_prefix=file_prefix
        )
    
    print(f"  - Loaded {len(data)} samples")
    
    # Get available brands
    brand_col = data_config['brand_col']
    if brand_col not in data.columns:
        raise ValueError(f"BRAND column '{brand_col}' not found in data. Available columns: {list(data.columns)}")
    
    available_brands = sorted(data[brand_col].unique().tolist())
    print(f"  - Available brands in data: {available_brands}")

    print(f"\n{'=' * 80}")
    print(f"TRAINING")
    print(f"{'=' * 80}")
    
    try:
        result = train_single_model(data, config)
        
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
    print(f"   - Output directory: {result['output_dir']}")
    print(f"   - Model checkpoint: {os.path.join(result['model_dir'], 'best_model.pth')}")
    print(f"   - Training time: {result['training_time']:.2f} seconds ({result['training_time']/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
