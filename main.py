"""Main entry point for MDLZ Warehouse Quantity Prediction System."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
import pandas as pd

from config import load_config
from src.data import (
    DataReader,
    ForecastDataset,
    slicing_window_category,
    encode_categories,
    split_data,
    add_holiday_features,
    add_temporal_features
)
from src.models import RNNWithCategory, RNNForecastor
from src.training import Trainer
from src.utils import (
    spike_aware_mse,
    plot_difference,
    plot_learning_curve,
    save_pred_actual_txt
)


def build_model(config):
    """Build model based on configuration."""
    model_config = config.model
    model_name = model_config['name']
    
    if model_name == 'RNNWithCategory':
        # Get num_categories from config (should be set after data loading)
        num_categories = model_config.get('num_categories')
        if num_categories is None:
            raise ValueError("num_categories must be set in config after data loading")
        
        model = RNNWithCategory(
            num_categories=num_categories,
            cat_emb_dim=model_config['cat_emb_dim'],
            input_dim=model_config['input_dim'],
            hidden_size=model_config['hidden_size'],
            n_layers=model_config['n_layers'],
            output_dim=model_config['output_dim']
        )
    elif model_name == 'RNNForecastor':
        model = RNNForecastor(
            embedding_dim=model_config['input_dim'],
            hidden_size=model_config['hidden_size'],
            out_dim=model_config['output_dim'],
            n_layers=model_config['n_layers'],
            dropout_prob=model_config['dropout_prob']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def build_criterion(config):
    """Build loss function based on configuration."""
    loss_name = config.training['loss']
    
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'spike_aware_mse':
        return spike_aware_mse
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def build_optimizer(model, config):
    """Build optimizer based on configuration."""
    optimizer_name = config.training['optimizer']
    learning_rate = config.training['learning_rate']
    
    if optimizer_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer, config):
    """Build learning rate scheduler based on configuration."""
    scheduler_config = config.training.get('scheduler', {})
    scheduler_name = scheduler_config.get('name')
    
    if scheduler_name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 3),
            min_lr=scheduler_config.get('min_lr', 1e-5)
        )
    elif scheduler_name is None:
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def prepare_data_pipeline(config):
    """Prepare data pipeline: load, encode, split, and create windows."""
    data_config = config.data
    window_config = config.window
    
    # Load data from combined year files (created by combine_data.py)
    # Expected files: data_2022.csv, data_2023.csv, data_2024.csv, data_2025.csv
    data_reader = DataReader(
        data_dir=data_config['data_dir'],
        file_pattern=data_config['file_pattern']
    )
    
    # Load from combined year files (standard pattern)
    # Falls back to pattern-based loading if combined files don't exist
    try:
        data = data_reader.load(years=data_config['years'])
    except FileNotFoundError:
        print("[Main] Combined year files not found, trying pattern-based loading from source files...")
        print("[Main] Note: Run 'python combine_data.py' first to create combined year files.")
        file_prefix = data_config.get('file_prefix', 'Outboundreports')
        data = data_reader.load_by_file_pattern(
            years=data_config['years'],
            file_prefix=file_prefix
        )
    
    # Ensure time column is datetime and sort by time for temporal split
    time_col = data_config['time_col']
    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)
    
    # Add temporal features (month_sin, month_cos, dayofmonth_sin, dayofmonth_cos)
    print("Adding temporal features...")
    data = add_temporal_features(
        data,
        time_col=time_col,
        month_sin_col="month_sin",
        month_cos_col="month_cos",
        dayofmonth_sin_col="dayofmonth_sin",
        dayofmonth_cos_col="dayofmonth_cos"
    )
    
    # Add holiday features
    print("Adding holiday features...")
    data = add_holiday_features(
        data,
        time_col=time_col,
        holiday_indicator_col="holiday_indicator",
        days_until_holiday_col="days_until_next_holiday"
    )
    
    # Encode categories
    cat_col = data_config['cat_col']
    data, cat2id, num_categories = encode_categories(data, cat_col)
    cat_id_col = data_config['cat_id_col']
    
    # Update config with num_categories for model
    config.set('model.num_categories', num_categories)
    
    # Split data
    train_data, val_data, test_data = split_data(
        data,
        train_size=data_config['train_size'],
        val_size=data_config['val_size'],
        test_size=data_config['test_size'],
        temporal=True
    )
    
    # Create windows
    feature_cols = data_config['feature_cols']
    target_col = [data_config['target_col']]
    cat_col_list = [cat_id_col]
    time_col = [data_config['time_col']]
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    
    print("Creating training windows...")
    X_train, y_train, cat_train = slicing_window_category(
        train_data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=cat_col_list,
        time_col=time_col
    )
    
    print("Creating validation windows...")
    X_val, y_val, cat_val = slicing_window_category(
        val_data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=cat_col_list,
        time_col=time_col
    )
    
    print("Creating test windows...")
    X_test, y_test, cat_test = slicing_window_category(
        test_data,
        input_size,
        horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_col=cat_col_list,
        time_col=time_col
    )
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
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
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'X_test': X_test,
        'y_test': y_test,
        'cat_test': cat_test,
        'X_train': X_train,
        'y_train': y_train,
        'cat_train': cat_train
    }


def train(config_path: str = None):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Prepare data
    data_dict = prepare_data_pipeline(config)
    
    # Build model
    model = build_model(config)
    
    # Build loss, optimizer, scheduler
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # Setup device
    device = torch.device(config.training['device'])
    print(f"Using device: {device}")
    
    # Create trainer
    training_config = config.training
    output_config = config.output
    save_dir = output_config.get('model_dir') if output_config.get('save_model') else None
    
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
    print("\nStarting training...")
    train_losses, val_losses = trainer.fit(
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        epochs=training_config['epochs'],
        save_best=True,
        verbose=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, y_true, y_pred = trainer.evaluate(
        data_dict['test_loader'],
        return_predictions=True
    )
    print(f"Test loss: {test_loss:.4f}")
    
    # Visualizations
    output_dir = output_config['output_dir']
    visualize_config = output_config.get('visualize', {})
    save_plots = visualize_config.get('save_plots', True)
    show_plots = visualize_config.get('show_plots', False)
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Learning curve
        plot_learning_curve(
            train_losses,
            val_losses,
            save_path=os.path.join(output_dir, "learning_curve.png"),
            show=show_plots
        )
        
        # Test predictions
        inference_config = config.inference
        n_samples = inference_config.get('n_samples_for_plot', 100)
        
        plot_difference(
            y_true[:n_samples],
            y_pred[:n_samples],
            save_path=os.path.join(output_dir, "test_predictions.png"),
            show=show_plots
        )
        
        # Train predictions (for comparison)
        train_labels, train_preds = trainer.predict(data_dict['train_loader'])
        plot_difference(
            train_labels[:n_samples],
            train_preds[:n_samples],
            save_path=os.path.join(output_dir, "train_fit.png"),
            show=show_plots
        )
        
        # Save predictions to text file
        if inference_config.get('save_predictions', True):
            save_pred_actual_txt(
                os.path.join(output_dir, "predictions.txt"),
                y_true[:n_samples],
                y_pred[:n_samples],
                index=range(len(y_true[:n_samples]))
            )
    
    print(f"\nTraining complete! Results saved to {output_dir}")
    return trainer, config, data_dict


def predict(model_path: str, config_path: str = None, data_path: str = None):
    """Inference function for making predictions."""
    # Load configuration
    config = load_config(config_path)
    
    # Prepare data
    if data_path is None:
        data_dict = prepare_data_pipeline(config)
        test_loader = data_dict['test_loader']
    else:
        # Load custom data for prediction
        raise NotImplementedError("Custom data loading for inference not yet implemented")
    
    # Build model
    model = build_model(config)
    
    # Load checkpoint
    device = torch.device(config.training['device'])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Make predictions
    trainer = Trainer(
        model=model,
        criterion=build_criterion(config),
        optimizer=build_optimizer(model, config),
        device=device
    )
    
    y_true, y_pred = trainer.predict(test_loader)
    
    return y_true, y_pred


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MDLZ Warehouse Quantity Prediction System")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict'],
        default='train',
        help='Mode: train or predict'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model checkpoint for inference'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(config_path=args.config)
    elif args.mode == 'predict':
        if args.model_path is None:
            raise ValueError("--model-path is required for predict mode")
        predict(model_path=args.model_path, config_path=args.config)

