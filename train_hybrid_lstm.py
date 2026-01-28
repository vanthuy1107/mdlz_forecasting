"""Training script for Multi-Input Hybrid LSTM model.

This script demonstrates how to train the HybridLSTM model with the dual-branch
architecture (LSTM + Dense paths) using proper time-series cross-validation and
expanding window feature engineering to prevent data leakage.

Key Features:
1. Dual-branch architecture: Separate temporal and static feature processing
2. Leak-free feature engineering: Expanding windows with no look-ahead bias
3. Asymmetric loss function: Penalizes under-forecasting more than over-forecasting
4. Category-specific training: Independent models per category (DRY, TET, etc.)

Usage:
    # Train DRY category with HybridLSTM
    python train_hybrid_lstm.py --category DRY
    
    # Train with custom hyperparameters
    python train_hybrid_lstm.py --category DRY --epochs 50 --learning-rate 0.001
    
    # Use asymmetric loss (recommended for FMCG)
    python train_hybrid_lstm.py --category DRY --loss asymmetric_mse --under-penalty 3.0

Requirements:
    - Data files: dataset/data_cat/data_2023.csv, data_2024.csv
    - Masterdata: masterdata/masterdata category-brand.xlsx
    - Config: config/config.yaml (or config/config_{CATEGORY}.yaml)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from typing import List, Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models import HybridLSTM
from src.data.dataset import HybridForecastDataset
from src.data.preprocessing import (
    add_temporal_features,
    add_holiday_features,
    add_weekday_volume_tier_features,
    add_eom_features,
    split_data,
)
from src.data.expanding_features import (
    add_expanding_statistical_features,
    add_expanding_rolling_features,
    add_expanding_momentum_features,
    add_brand_tier_feature,
)
from src.utils.losses import asymmetric_mse_loss, AsymmetricMSELoss
from src.utils.evaluation import calculate_forecast_metrics


def load_config(config_path: str = "config/config.yaml", category: Optional[str] = None):
    """Load configuration file.
    
    Args:
        config_path: Path to base config file.
        category: Category name (if category-specific config exists, it overrides base).
    
    Returns:
        Config object with properties: .data, .model, .training, etc.
    """
    from config import Config
    
    # Create Config object (it handles category-specific loading internally)
    config = Config(config_path=config_path, category=category)
    
    return config


def prepare_data_with_expanding_features(
    df: pd.DataFrame,
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    config: dict = None,
) -> pd.DataFrame:
    """
    Prepare data with leak-free expanding window features.
    
    This function adds all necessary features for the HybridLSTM model:
    - Temporal features (cyclical encodings)
    - Holiday features
    - Weekday volume tiers
    - End-of-month features
    - Expanding statistical features (NO DATA LEAKAGE)
    - Expanding rolling features
    - Expanding momentum features
    
    Args:
        df: Raw data DataFrame.
        target_col: Name of target column.
        time_col: Name of time column.
        cat_col: Name of category column.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with all features added.
    """
    print("\n" + "=" * 80)
    print("PREPARING DATA WITH EXPANDING WINDOW FEATURES")
    print("=" * 80)
    
    df = df.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by category and time
    df = df.sort_values([cat_col, time_col]).reset_index(drop=True)
    
    print(f"\n1. Adding calendar-based features (no leakage)...")
    
    # Add temporal features (cyclical encodings)
    df = add_temporal_features(df, time_col=time_col)
    print("   [+] Added: month_sin, month_cos, dayofmonth_sin, dayofmonth_cos")
    
    # Add holiday features (calendar-based, no leakage)
    df = add_holiday_features(df, time_col=time_col)
    print("   [+] Added: holiday_indicator, days_until_next_holiday")
    
    # Add days_since_holiday feature
    from src.data.preprocessing import get_us_holidays
    from datetime import timedelta
    min_date = df[time_col].min().date()
    max_date = df[time_col].max().date()
    extended_min = min_date - timedelta(days=365)  # Look back for holidays
    holidays = get_us_holidays(extended_min, max_date)
    holidays_sorted = sorted(holidays, reverse=True)
    
    # Vectorized approach using apply
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
    
    # Add lunar calendar features (for seasonal categories)
    from src.utils.lunar_utils import solar_to_lunar_date
    
    # Vectorized approach using apply
    def get_lunar_date(date):
        if pd.isna(date):
            return pd.Series([8, 15])  # Default
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
    
    print(f"\n2. Adding expanding window statistical features (LEAK-FREE)...")
    
    # Add expanding statistical features (weekday avg, category avg)
    df = add_expanding_statistical_features(
        df,
        target_col=target_col,
        time_col=time_col,
        cat_col=cat_col,
        features_to_add=["weekday_avg", "category_avg"],
    )
    print("   [+] Added: weekday_avg_expanding, category_avg_expanding")
    
    # Add expanding rolling features (7-day, 30-day rolling means)
    df = add_expanding_rolling_features(
        df,
        target_col=target_col,
        time_col=time_col,
        cat_col=cat_col,
        windows=[7, 30],
    )
    print("   [+] Added: rolling_mean_7d_expanding, rolling_mean_30d_expanding")
    
    # Add expanding momentum features (3d vs 14d)
    df = add_expanding_momentum_features(
        df,
        target_col=target_col,
        time_col=time_col,
        cat_col=cat_col,
        short_window=3,
        long_window=14,
    )
    print("   [+] Added: momentum_3d_vs_14d_expanding")
    
    # Optional: Add brand tier features if brand column exists
    if "BRAND" in df.columns:
        df = add_brand_tier_feature(
            df,
            target_col=target_col,
            brand_col="BRAND",
            cat_col=cat_col,
            time_col=time_col,
        )
        print("   [+] Added: brand_tier_expanding")
    
    print(f"\n3. Feature preparation complete!")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Total rows: {len(df)}")
    
    return df


def create_hybrid_datasets(
    df: pd.DataFrame,
    temporal_features: List[str],
    static_features: List[str],
    target_col: str = "Total CBM",
    time_col: str = "ACTUALSHIPDATE",
    cat_col: str = "CATEGORY",
    input_size: int = 30,
    horizon: int = 30,
    train_size: float = 0.7,
    val_size: float = 0.1,
) -> Tuple[HybridForecastDataset, HybridForecastDataset, HybridForecastDataset]:
    """
    Create train/val/test datasets for HybridLSTM.
    
    This function:
    1. Splits data temporally (no shuffling)
    2. Creates sliding windows for each split
    3. Separates temporal and static features
    4. Returns HybridForecastDataset instances
    
    Args:
        df: DataFrame with all features.
        temporal_features: List of temporal feature names (for LSTM branch).
        static_features: List of static feature names (for Dense branch).
        target_col: Name of target column.
        time_col: Name of time column.
        cat_col: Name of category column.
        input_size: Input window size (e.g., 30 days).
        horizon: Forecast horizon (e.g., 30 days).
        train_size: Training set ratio.
        val_size: Validation set ratio.
    
    Returns:
        (train_dataset, val_dataset, test_dataset) tuple.
    """
    print("\n" + "=" * 80)
    print("CREATING HYBRID DATASETS")
    print("=" * 80)
    
    # Split data temporally
    print(f"\n1. Splitting data temporally...")
    train_df, val_df, test_df = split_data(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=1.0 - train_size - val_size,
        temporal=True,
    )
    
    print(f"   Train: {len(train_df)} rows ({train_df[time_col].min()} to {train_df[time_col].max()})")
    print(f"   Val:   {len(val_df)} rows ({val_df[time_col].min()} to {val_df[time_col].max()})")
    print(f"   Test:  {len(test_df)} rows ({test_df[time_col].min()} to {test_df[time_col].max()})")
    
    # Create sliding windows for each split
    print(f"\n2. Creating sliding windows (input_size={input_size}, horizon={horizon})...")
    
    def create_windows(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sliding windows for a single split."""
        X_temporal_list = []
        X_static_list = []
        y_list = []
        
        # Process per category
        for cat_name, cat_group in data.groupby(cat_col, sort=False):
            cat_group = cat_group.sort_values(time_col).reset_index(drop=True)
            
            # Extract features
            temporal_data = cat_group[temporal_features].values
            static_data = cat_group[static_features].values
            target_data = cat_group[target_col].values
            
            # Create sliding windows
            for i in range(len(cat_group) - input_size - horizon + 1):
                # Input window: [i, i+input_size)
                X_temporal = temporal_data[i:i+input_size]
                X_static = static_data[i+input_size-1]  # Take static features from last timestep
                
                # Target window: [i+input_size, i+input_size+horizon)
                y = target_data[i+input_size:i+input_size+horizon]
                
                X_temporal_list.append(X_temporal)
                X_static_list.append(X_static)
                y_list.append(y)
        
        X_temporal = np.array(X_temporal_list)
        X_static = np.array(X_static_list)
        y = np.array(y_list)
        
        return X_temporal, X_static, y
    
    # Create windows for each split
    X_train_temporal, X_train_static, y_train = create_windows(train_df)
    X_val_temporal, X_val_static, y_val = create_windows(val_df)
    X_test_temporal, X_test_static, y_test = create_windows(test_df)
    
    print(f"   Train: {len(X_train_temporal)} windows")
    print(f"   Val:   {len(X_val_temporal)} windows")
    print(f"   Test:  {len(X_test_temporal)} windows")
    
    # Create datasets
    print(f"\n3. Creating HybridForecastDataset instances...")
    
    train_dataset = HybridForecastDataset(X_train_temporal, X_train_static, y_train)
    val_dataset = HybridForecastDataset(X_val_temporal, X_val_static, y_val)
    test_dataset = HybridForecastDataset(X_test_temporal, X_test_static, y_test)
    
    print(f"   [+] Datasets created successfully")
    print(f"   Temporal features: {len(temporal_features)}")
    print(f"   Static features: {len(static_features)}")
    
    return train_dataset, val_dataset, test_dataset


def train_hybrid_model(
    model: HybridLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    learning_rate: float = 0.001,
    loss_fn = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "outputs/hybrid_lstm/best_model.pth",
    input_size: int = 30,
    horizon: int = 30,
    dense_hidden_sizes: List[int] = None,
    fusion_hidden_sizes: List[int] = None,
) -> Tuple[HybridLSTM, List[float], List[float]]:
    """
    Train HybridLSTM model.
    
    Args:
        model: HybridLSTM model instance.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        loss_fn: Loss function (if None, uses AsymmetricMSELoss).
        device: Device to train on ('cuda' or 'cpu').
        save_path: Path to save best model.
    
    Returns:
        (trained_model, train_losses, val_losses) tuple.
    """
    print("\n" + "=" * 80)
    print("TRAINING HYBRID LSTM MODEL")
    print("=" * 80)
    
    model = model.to(device)
    
    # Default loss function: Asymmetric MSE (penalizes under-forecasting)
    if loss_fn is None:
        loss_fn = AsymmetricMSELoss(under_penalty=3.0, over_penalty=1.0)
        print(f"\nUsing Asymmetric MSE Loss (under_penalty=3.0, over_penalty=1.0)")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: Adam")
    print(f"Scheduler: ReduceLROnPlateau")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            X_temporal, X_static, y = batch
            X_temporal = X_temporal.to(device)
            X_static = X_static.to(device)
            y = y.to(device)
            
            # Create dummy category IDs (single category per model)
            x_cat = torch.zeros(X_temporal.size(0), dtype=torch.long, device=device)
            
            # Forward pass
            y_pred = model(X_temporal, X_static, x_cat)
            
            # Compute loss
            loss = loss_fn(y_pred, y) if callable(loss_fn) else loss_fn.forward(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                X_temporal, X_static, y = batch
                X_temporal = X_temporal.to(device)
                X_static = X_static.to(device)
                y = y.to(device)
                
                x_cat = torch.zeros(X_temporal.size(0), dtype=torch.long, device=device)
                
                y_pred = model(X_temporal, X_static, x_cat)
                loss = loss_fn(y_pred, y) if callable(loss_fn) else loss_fn.forward(y_pred, y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model with configuration
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state and configuration
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'temporal_input_dim': model.temporal_input_dim,
                    'static_input_dim': model.static_input_dim,
                    'num_categories': model.num_categories,
                    'cat_emb_dim': model.cat_emb_dim,
                    'hidden_size': model.hidden_size,
                    'n_layers': model.n_layers,
                    'output_dim': model.output_dim,
                    'dropout_prob': model.dropout_prob,
                    'dense_hidden_sizes': dense_hidden_sizes or [64, 32],
                    'fusion_hidden_sizes': fusion_hidden_sizes or [64],
                },
                'input_size': input_size,
                'horizon': horizon,
            }
            torch.save(checkpoint, save_path)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f} [BEST]")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
    
    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint)
    print(f"\n[+] Training complete! Best model saved to: {save_path}")
    print(f"[+] Best validation loss: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Input Hybrid LSTM for MDLZ forecasting"
    )
    parser.add_argument("--category", type=str, default="DRY", help="Category to train (DRY, TET, MOONCAKE, etc.)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--input-size", type=int, default=30, help="Input window size (days)")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon (days)")
    parser.add_argument("--loss", type=str, default="asymmetric_mse", choices=["mse", "asymmetric_mse"], help="Loss function")
    parser.add_argument("--under-penalty", type=float, default=3.0, help="Under-forecast penalty for asymmetric loss")
    parser.add_argument("--over-penalty", type=float, default=1.0, help="Over-forecast penalty for asymmetric loss")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--output-dir", type=str, default="outputs/hybrid_lstm", help="Output directory")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 80)
    print("MDLZ HYBRID LSTM TRAINING")
    print("=" * 80)
    print(f"\nCategory: {args.category}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Input Size: {args.input_size} days")
    print(f"Forecast Horizon: {args.horizon} days")
    print(f"Loss Function: {args.loss}")
    if args.loss == "asymmetric_mse":
        print(f"  Under-penalty: {args.under_penalty}")
        print(f"  Over-penalty: {args.over_penalty}")
    print(f"Device: {device}")
    print(f"Output Directory: {args.output_dir}")
    
    # Load configuration
    config = load_config(category=args.category)
    
    # ============================================================================
    # DATA LOADING
    # ============================================================================
    print(f"\n[INFO] Loading data for category: {args.category}")
    
    # Load data files
    data_dir = Path(config.data['data_dir'])
    years = config.data['years']
    file_pattern = config.data['file_pattern']
    
    # Load all year files
    dfs = []
    for year in years:
        file_path = data_dir / file_pattern.format(year=year)
        if file_path.exists():
            print(f"  Loading {file_path.name}...")
            df_year = pd.read_csv(file_path)
            dfs.append(df_year)
        else:
            print(f"  [WARNING] File not found: {file_path}")
    
    if not dfs:
        print(f"[ERROR] No data files found for years {years} in {data_dir}")
        print(f"[ERROR] Expected files: {[file_pattern.format(year=y) for y in years]}")
        return 1
    
    # Combine all years
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined data: {len(df):,} rows")
    
    # Filter to category
    if 'CATEGORY' in df.columns:
        df = df[df['CATEGORY'] == args.category].copy()
        print(f"  Filtered to {args.category}: {len(df):,} rows")
    else:
        print(f"[ERROR] CATEGORY column not found in data")
        return 1
    
    if len(df) == 0:
        print(f"[ERROR] No data found for category: {args.category}")
        return 1
    
    # ============================================================================
    # FEATURE ENGINEERING (LEAK-FREE)
    # ============================================================================
    df_prepared = prepare_data_with_expanding_features(
        df,
        target_col=config.data['target_col'],
        time_col=config.data['time_col'],
        cat_col=config.data['cat_col'],
        config=config,
    )
    
    # ============================================================================
    # DEFINE FEATURE GROUPS
    # ============================================================================
    print(f"\n[INFO] Defining feature groups...")
    
    # Get feature lists from config
    if 'hybrid' in config.model:
        temporal_features = config.model['hybrid']['temporal_features']
        static_features = config.model['hybrid']['static_features']
    else:
        # Default feature split if not in config
        temporal_features = [
            'Total CBM',
            'rolling_mean_7d_expanding',
            'rolling_mean_30d_expanding',
            'momentum_3d_vs_14d_expanding',
        ]
        
        static_features = [
            'month_sin', 'month_cos',
            'dayofmonth_sin', 'dayofmonth_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'holiday_indicator', 'days_until_next_holiday',
            'is_weekend', 'Is_Monday',
            'is_EOM', 'days_until_month_end',
            'weekday_volume_tier', 'is_high_volume_weekday',
            'weekday_avg_expanding', 'category_avg_expanding',
        ]
    
    # Verify features exist
    missing_temporal = [f for f in temporal_features if f not in df_prepared.columns]
    missing_static = [f for f in static_features if f not in df_prepared.columns]
    
    if missing_temporal:
        print(f"[ERROR] Missing temporal features: {missing_temporal}")
        return 1
    if missing_static:
        print(f"[ERROR] Missing static features: {missing_static}")
        return 1
    
    print(f"  Temporal features ({len(temporal_features)}): {temporal_features[:3]}...")
    print(f"  Static features ({len(static_features)}): {static_features[:3]}...")
    
    # ============================================================================
    # CREATE DATASETS
    # ============================================================================
    train_dataset, val_dataset, test_dataset = create_hybrid_datasets(
        df_prepared,
        temporal_features=temporal_features,
        static_features=static_features,
        target_col=config.data['target_col'],
        time_col=config.data['time_col'],
        cat_col=config.data['cat_col'],
        input_size=args.input_size,
        horizon=args.horizon,
        train_size=config.data['train_size'],
        val_size=config.data['val_size'],
    )
    
    # ============================================================================
    # INITIALIZE MODEL
    # ============================================================================
    print(f"\n[INFO] Initializing HybridLSTM model...")
    
    model = HybridLSTM(
        num_categories=1,  # Single category per model
        cat_emb_dim=config.model['cat_emb_dim'],
        temporal_input_dim=len(temporal_features),
        static_input_dim=len(static_features),
        hidden_size=config.model['hidden_size'],
        n_layers=config.model['n_layers'],
        dense_hidden_sizes=config.model['hybrid']['dense_hidden_sizes'],
        fusion_hidden_sizes=config.model['hybrid']['fusion_hidden_sizes'],
        output_dim=args.horizon,
        dropout_prob=config.model['dropout_prob'],
        use_layer_norm=config.model['use_layer_norm'],
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============================================================================
    # CREATE DATA LOADERS
    # ============================================================================
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # ============================================================================
    # SETUP LOSS FUNCTION
    # ============================================================================
    if args.loss == "asymmetric_mse":
        loss_fn = AsymmetricMSELoss(
            under_penalty=args.under_penalty,
            over_penalty=args.over_penalty,
        )
    else:
        loss_fn = nn.MSELoss()
    
    # ============================================================================
    # TRAIN MODEL
    # ============================================================================
    output_dir = Path(args.output_dir) / args.category
    save_path = output_dir / "models" / "best_model.pth"
    
    model, train_losses, val_losses = train_hybrid_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        loss_fn=loss_fn,
        device=device,
        save_path=str(save_path),
        input_size=args.input_size,
        horizon=args.horizon,
        dense_hidden_sizes=config.model['hybrid']['dense_hidden_sizes'],
        fusion_hidden_sizes=config.model['hybrid']['fusion_hidden_sizes'],
    )
    
    # ============================================================================
    # EVALUATE ON TEST SET
    # ============================================================================
    print(f"\n[INFO] Evaluating on test set...")
    
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            X_temporal, X_static, y = batch
            X_temporal = X_temporal.to(device)
            X_static = X_static.to(device)
            y = y.to(device)
            
            # Dummy category IDs
            x_cat = torch.zeros(X_temporal.size(0), dtype=torch.long, device=device)
            
            # Forward pass
            y_pred = model(X_temporal, X_static, x_cat)
            
            # Compute loss
            loss = loss_fn(y_pred, y) if callable(loss_fn) else loss_fn.forward(y_pred, y)
            test_loss += loss.item()
            
            all_predictions.append(y_pred.cpu().numpy())
            all_actuals.append(y.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # Calculate metrics
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    metrics = calculate_forecast_metrics(actuals, predictions)
    
    print(f"\n{'=' * 80}")
    print(f"TEST SET RESULTS")
    print(f"{'=' * 80}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MSE:  {metrics['mse']:.2f}")
    print(f"{'=' * 80}")
    
    # Save metrics
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'mse': float(metrics['mse']),
            'category': args.category,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'loss_function': args.loss,
            'under_penalty': args.under_penalty,
            'over_penalty': args.over_penalty,
        }, f, indent=2)
    
    print(f"\n[+] Metrics saved to: {metrics_path}")
    print(f"[+] Model saved to: {save_path}")
    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'=' * 80}")
    
    return 0


if __name__ == "__main__":
    exit(main())
