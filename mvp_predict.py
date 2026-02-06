"""
Module for performing model inference and walk-forward testing.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
import json
import pickle
import shutil
import re
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List

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
from src.utils import plot_difference, upload_to_google_sheets, GSPREAD_AVAILABLE, save_monthly_forecast
from src.utils.visualization import plot_monthly_forecast
from src.utils.metrics import generate_accuracy_report
from src.utils.google_sheets import upload_history_prediction
from src.utils.date import load_holidays
from src.utils import spike_aware_huber, seed_everything, seed_worker, SEED

seed_everything(SEED)


def fill_missing_dates(
    df: pd.DataFrame,
    time_col: str,
    brand_col: str,
    target_col: str,
    feature_cols: List[str] = None,
    fill_target: float = 0.0,
):
    """Ensure each brand has continuous daily dates between its min and max.

    - Adds rows for missing dates per brand.
    - Sets `target_col` to `fill_target` for filled rows.
    - If `feature_cols` provided, fills them with 0 for new rows.

    Returns a new DataFrame with added rows (sorted by time_col).
    """
    if df is None or len(df) == 0:
        return df

    df = df.copy()
    # Normalize datetime column
    df[time_col] = pd.to_datetime(df[time_col])

    rows_to_add = []
    brands = df[brand_col].dropna().unique()
    for brand in brands:
        brand_mask = df[brand_col] == brand
        brand_dates = pd.to_datetime(df.loc[brand_mask, time_col]).dt.normalize()
        if brand_dates.empty:
            continue
        start = brand_dates.min()
        end = brand_dates.max()
        full_idx = pd.date_range(start=start, end=end, freq='D')
        existing = pd.DatetimeIndex(brand_dates.values)
        missing = full_idx.difference(existing)
        for d in missing:
            new_row = {time_col: d, brand_col: brand, target_col: fill_target}
            if feature_cols:
                for f in feature_cols:
                    new_row[f] = 0
            rows_to_add.append(new_row)

    if len(rows_to_add) > 0:
        fill_df = pd.DataFrame(rows_to_add)
        # Preserve column order where possible
        df = pd.concat([df, fill_df], axis=0, ignore_index=True, sort=False)
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(by=[brand_col, time_col]).reset_index(drop=True)

    return df

def load_model_for_test(model_path: str, config):
    """Load trained model from checkpoint and scaler."""
    # Load metadata first to get the training-time model/data config
    model_dir = Path(model_path).parent
    metadata_path = model_dir / "metadata.json"
    
    # ------------------------------------------------------------------
    # 1) Recover num_brands and full model architecture from metadata
    # ------------------------------------------------------------------
    num_brands = None
    model_config = None
    feature_cols = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Get full model_config (includes num_brands, input_dim, etc.)
        model_config = metadata.get('model_config', {})
        if 'num_brands' in model_config:
            num_brands = model_config['num_brands']

        # Also recover the exact feature column list used during training
        data_config = metadata.get("data_config", {})
        feature_cols = data_config.get("feature_cols")
    
    # Fallback to config if metadata doesn't have it
    if num_brands is None:
        model_config = config.model
        num_brands = model_config.get('num_brands')
    
        if num_brands is None:
            raise ValueError("num_brands must be found in model metadata or config")

    # If we have the training-time feature list, push it into the live config
    # so that window creation uses the exact same ordering and dimensionality.
    if feature_cols is not None:
        config.set("data.feature_cols", list(feature_cols))
    
    print(f"  - Loading model with num_brands={num_brands} (from trained model)")
    
    # ------------------------------------------------------------------
    # 2) Build model with the *exact* architecture used during training
    #    (input_dim, hidden_size, n_layers, etc. come from metadata)
    # -----------------------------------------------------------------
    model = RNNForecastor(
        num_brands=num_brands,
        brand_emb_dim=model_config['brand_emb_dim'],
        input_dim=model_config['input_dim'],
        hidden_size=model_config['hidden_size'],
        n_layers=model_config['n_layers'],
        output_dim=model_config['output_dim']
    )
    
    # Load checkpoint
    device = torch.device(config.training['device'])
    ckpt = torch.load(model_path, map_location=device)

    cfg = ckpt.get("config", {})
    model_cfg = cfg["model"]
    assert model_cfg["input_dim"] == model.input_dim
    assert model_cfg["brand_emb_dim"] == model.brand_emb_dim
    assert model_cfg["num_brands"] == model.num_brands

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  - Model loaded from: {model_path}")
    print(f"  - Best validation loss: {ckpt.get('best_val_loss', 'N/A'):.4f}")
    
    # Load scaler from same directory as model (model_dir already defined above)
    scaler_path = model_dir / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  - Scaler loaded from: {scaler_path}")
    else:
        print(f"  [WARNING] Scaler not found at {scaler_path}, test will be in scaled space")
    
    return model, device, scaler

def walk_forward_test(
    model : nn.Module,
    scaler : RollingGroupScaler,
    data : pd.DataFrame,
    time_col : str,
    feature_cols : List[str],
    target_col : str,
    brand_id_col : str,
    input_size : int,
    horizon : int,
    test_start_date : pd.Timestamp,
    test_end_date : pd.Timestamp,
    update_freq : pd.DateOffset,
    device
):
    """
    Perform walk-forward testing by chunking time periods and updating scaler.
    
    For each chunk:
    - Fit scaler on all data BEFORE chunk_start
    - Call slicing_window_brand with [chunk_start, chunk_end) range
    - slicing_window_brand automatically creates windows for each day in range
    
    Args:
        model: Trained PyTorch model for prediction.
        scaler: RollingGroupScaler fitted on training data.
        data: Full DataFrame containing historical + test data.
        time_col: Name of the time column in data.
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        brand_id_col: Name of the brand ID column.
        input_size: Number of past days used as input (28).
        horizon: Number of future days to predict (28).
        test_start_date: Start date for testing (inclusive).
        test_end_date: End date for testing (inclusive).
        update_freq: Frequency to update scaler (e.g., pd.DateOffset(months=2)).
        device: Torch device (CPU or GPU).

    Returns:
        DataFrame with columns: ['predicted', 'actual', 'brand', 'date', 'window_start_date'].
    """
    # Convert dates to pd.Timestamp
    if isinstance(test_start_date, date) and not isinstance(test_start_date, datetime):
        test_start_date = pd.Timestamp(test_start_date)
    else:
        test_start_date = pd.Timestamp(test_start_date)
    
    if isinstance(test_end_date, date) and not isinstance(test_end_date, datetime):
        test_end_date = pd.Timestamp(test_end_date)
    else:
        test_end_date = pd.Timestamp(test_end_date)
    
    model.eval()

    all_y_pred = []
    all_y_true = []
    all_baselines = []
    all_brand = []
    all_dates = []

    chunk_start = test_start_date
    chunk_idx = 0
    while chunk_start < test_end_date:
        chunk_idx += 1
        chunk_end = min(chunk_start + update_freq, test_end_date)

        print(f"\n[WALK-FORWARD] Chunk {chunk_idx}: {chunk_start.date()} to {chunk_end.date()}")
        print(f"       Scaler will be fit on data before {chunk_start.date()}")
        print(f"       Slicing windows for range [{chunk_start.date()}, {chunk_end.date()})")
        
        # Fit scaler on all data BEFORE chunk_start
        print(f"  [1/4] Fitting scaler on data before {chunk_start.date()}")
        scaler.fit(data, chunk_start)

        # Scale all data up to chunk_end
        print(f"  [2/4] Scaling data up to {chunk_end.date()}")
        df_chunk = data[
            data[time_col] < (chunk_end + pd.Timedelta(days=horizon))
        ].copy()

        df_chunk_scaled = scaler.transform(df_chunk)
        print(f"       Scaled {len(df_chunk_scaled)} rows")

        # Slice windows for entire chunk range
        # slicing_window_brand will create windows for each day in [chunk_start, chunk_end)
        print(f"  [3/4] Creating sliding windows for each day in range")
        X_test, y_test, baselines_test, brand_test, test_dates, off_flags = slicing_window(
            df=df_chunk_scaled,
            input_size=input_size,
            horizon=horizon,
            feature_cols=feature_cols,
            target_col=target_col,
            baseline_col='baseline',
            brand_col=brand_id_col,
            time_col=time_col,
            off_holiday_col="is_off_holiday",
            label_start_date=chunk_start,
            label_end_date=chunk_end,
            return_dates=True,
            return_off_holiday=True,
        )

        if len(X_test) == 0:
            print(f"  ⚠️  No samples in this chunk, skipping...")
            chunk_start = chunk_end
            continue

        print(f"       Created {len(X_test)} windows")
        if len(test_dates) > 0:
            print(
                f"       Window date range: "
                f"{pd.to_datetime(test_dates).min().date()} "
                f"to {pd.to_datetime(test_dates).max().date()}"
            )

        # Predict
        print(f"  [4/4] Running predictions on {len(X_test)} windows")
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
            brands = torch.tensor(brand_test, dtype=torch.long).view(-1).to(device)
            outputs = model(inputs, brands).cpu().numpy()

        print(f"       Output shape: {outputs.shape}")

        # Inverse transform
        y_pred = (
            scaler.inverse_transform_y(outputs[:, [0]], brand_test)
            .reshape(-1, 1)
            + baselines_test[:, [0]]
        )
        y_pred[off_flags == 1] = 0

        y_true = (
            scaler.inverse_transform_y(y_test[:, [0]], brand_test)
            .reshape(-1, 1)
            + baselines_test[:, [0]]
        )


        # Collect
        all_y_pred.append(y_pred)
        all_y_true.append(y_true)
        all_baselines.append(baselines_test)
        all_brand.append(brand_test)
        all_dates.append(test_dates)

        chunk_start = chunk_end

    # Concatenate all chunks
    if len(all_y_pred) == 0:
        raise ValueError("No windows were created during walk-forward test!")
    
    # Concatenate all chunks
    y_pred_all = np.concatenate(all_y_pred, axis=0)    # (N, 1)
    y_true_all = np.concatenate(all_y_true, axis=0)  # (N, 1)
    brand_all = np.concatenate(all_brand, axis=0)    # (N,)
    dates_all = np.concatenate(all_dates, axis=0)    # (N,)
    baseline_all = np.concatenate(
        [b[:, 0] for b in all_baselines], axis=0
    )   # shape (N,)

    print(f"\n[WALK-FORWARD] Complete!")
    print(f"  - Total windows: {len(y_pred_all)}")
    print(f"  - Forecast type: one-step-ahead (t+1)")
    print(f"  - Date range: {dates_all.min()} to {dates_all.max()}")

    # Remove horizon dimension
    preds_flat = y_pred_all.squeeze()
    actuals_flat = y_true_all.squeeze()

    # Direct mapping (no expansion)
    results_df = pd.DataFrame({
        'predicted': preds_flat,
        'actual': actuals_flat,
        'baseline': baseline_all,
        'brand': brand_all,
        'date': pd.to_datetime(dates_all),
        'window_start_date': pd.to_datetime(dates_all)
    })

    final = results_df.sort_values(by=['brand', 'date']).reset_index(drop=True)
    return final



def main():
    print("=" * 80)
    print("TESTING PHASE")
    print("=" * 80)
    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config()
    # Get column names from config first
    data_config = config.data
    brand_col = data_config['brand_col']
    time_col = data_config['time_col']
    target_col = data_config['target_col']

    inference_config = config.inference or {}
    test_data_path = inference_config.get(
        "test_data_path",
        "dataset/test/data_prediction.csv",
    )
    test_data_path = Path(test_data_path)
    test_start = pd.to_datetime(
        inference_config.get("test_start", "2025-01-01")
    )
    # End date + 1
    test_end = pd.to_datetime(
        inference_config.get("test_end", "2025-12-31")
    ) + pd.Timedelta(days=1)

    if test_end < test_start:
        raise ValueError(
            f"inference.test_end ({test_end}) "
            f"must be on or after inference.test_start ({test_start})"
        )
    
    historical_year = test_start.year - 1
    print("=" * 80)
    print(f"\n[2/6] Loading historical {historical_year} data...")
    data_reader = DataReader(
        data_dir=data_config['data_dir'],
        file_pattern=data_config['file_pattern']
    )

    # Load history data
    print(f"  - Loading historical data for years: {historical_year}")
    ref_data = data_reader.load(years=[historical_year])
    ref_data = ref_data[~ref_data["BRAND"].isin(["KINH DO CAKE", "LU"])]

    # DIAGNOSTIC: Check historical data coverage    
    if len(ref_data) > 0 and time_col in ref_data.columns:
        ref_data_dates = pd.to_datetime(ref_data[time_col])
        print(f"  - Historical data loaded: {len(ref_data)} rows")
        print(f"  - Date range: {ref_data_dates.min().date()} to {ref_data_dates.max().date()}")

    # Encode all brands from reference data (don't filter yet - need full mapping)
    # Note: We encode here to get brand2id mapping, but num_brands for model will come from trained model metadata
    _, trained_brand2id, num_brands = encode_brands(ref_data, data_config['brand_col'])
    # Don't overwrite config.model.num_brands here - it will be loaded from trained model metadata
    
    print(f"  - brand mapping: {trained_brand2id}")
    print(f"  - Number of brands in data: {num_brands}")
    
    # Determine which brands to predict
    # Filter out NaN values and convert to string to handle mixed types
    unique_ref_brands = ref_data[brand_col].dropna().astype(str).unique().tolist()
    available_brands = sorted([brand for brand in unique_ref_brands if brand.lower() != 'nan'])
    print(f"  - Available brands in training data: {available_brands}")

    # Load test data
    print("\n[3/6] Loading test data...")
    
    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test data file not found at: {test_data_path.absolute()}"
        )
    print(f"  - Loading from: {test_data_path}")
    test_data = pd.read_csv(test_data_path, encoding='utf-8', low_memory=False)
    test_data = test_data[~test_data["BRAND"].isin(["KINH DO CAKE", "LU"])]
    print(f"  - Loaded {len(test_data)} samples")
    if len(test_data) == 0:
        raise ValueError(
                    f"Test data file is empty or contains no valid data."
                )

    # Filter to desired test window (keep all brands for now - will filter per brand later)
    print("\n[4/6] Cliping date range...")
    
    # Filter to configured test window first
    if not pd.api.types.is_datetime64_any_dtype(test_data[time_col]):
        # The test CSV can contain dates like "13/01/YYYY" (dd/mm/YYYY).
        # Use a robust parser that supports mixed formats and day-first dates.
        test_data[time_col] = pd.to_datetime(
            test_data[time_col],
            format="mixed",
            dayfirst=True,
        )
    
    # Store original data to check date range if filtering returns empty
    temp_data = test_data.copy()
    temp_data = temp_data[
        (temp_data[time_col] >= test_start)
        & (temp_data[time_col] < test_end)
    ]
    print(
        f"  - After date filter [{test_start} .. {test_end}): "
        f"{len(temp_data)} samples"
    )

    # Check if no valid data in test_date range
    if len(temp_data) == 0:
        min_date = temp_data[time_col].min()
        max_date = temp_data[time_col].max()
        raise ValueError(
            f"No data found in test file for the specified date range "
            f"[{test_start} .. {test_end}).\n"
            f"Available date range in file: {min_date.date()} to {max_date.date()}\n"
            f"Please update config.inference.test_start and test_end to "
            f"match the available date range, or use a different test data file."
        )
    
    # Check which brands are available in test data
    # Filter out NaN values and convert to string to handle mixed types
    unique_brands = test_data[brand_col].dropna().astype(str).unique().tolist()
    available_test_brands = sorted([brand for brand in unique_brands if brand.lower() != 'nan'])
    test_year = test_start.year
    print(f"  - Available brands in {test_year} data: {available_test_brands}")
    
    # Filter brands_to_test to only those available in both reference and test data
    brands_to_test = [brand for brand in available_brands if brand in available_test_brands]
    
    if len(brands_to_test) == 0:
        raise ValueError(f"No matching brands found between reference data and {test_year} data. "
                        f"Reference: {available_brands}, {test_year}: {available_test_brands}")
    print(f"  - Final brands to predict: {brands_to_test}")

    # =====================================================================
    # [REFINEMENT] Prepare data before calling walk_forward_test()
    # =====================================================================
    print("\n[5/6] Preparing data for walk-forward test...")
    
    # 1. Extract key parameters from config
    print("  [5.1] Extracting config parameters...")
    feature_cols = data_config.get('feature_cols', [])
    target_col = data_config.get('target_col', 'Total CBM')
    input_size = config.window.get('input_size', 7)
    horizon = config.window.get('horizon', 2)
    
    # 2. Prepare historical data (ref_data) with features
    print("  [5.2] Preparing historical data with features...")
    data_prepared = add_features(
        ref_data.copy(),
        time_col=data_config['time_col'],
        brand_col=data_config['brand_col'],
        target_col=target_col
    )
    print(f"    - Added all features to historical data")
    
    
    # 3. Prepare test data with same features
    print("  [5.3] Preparing test data with features...")
    test_data_prepared = add_features(
        test_data.copy(),
        time_col=data_config['time_col'],
        brand_col=data_config['brand_col'],
        target_col=target_col
    )
    print(f"    - Added all features to test data")

    target_col = "residual"
    print(f"    - Feature columns: {len(feature_cols)} columns")
    print(f"    - Target: {target_col}")
    print(f"    - Input size: {input_size} days")
    print(f"    - Horizon: {horizon} days")
    
    # 4. Combine data: use last 6 months of historical + all test data
    # (This ensures enough input history for rolling windows and scaler updates)
    print("  [5.4] Combining last 6 months of historical with test data...")
    
    # Get last 6 months of historical data
    data_prepared[time_col] = pd.to_datetime(data_prepared[time_col])
    last_date = data_prepared[time_col].max()
    six_months_ago = last_date - pd.DateOffset(months=6)
    data_prepared_last6m = data_prepared[data_prepared[time_col] >= six_months_ago].copy()
    
    print(f"    - Historical data (last 6 months): {len(data_prepared_last6m)} rows")
    print(f"      Date range: {data_prepared_last6m[time_col].min().date()} to {data_prepared_last6m[time_col].max().date()}")
    
    # Combine with test data
    data = pd.concat([data_prepared_last6m, test_data_prepared], axis=0, ignore_index=True)
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)
    print(f"    - Combined data: {len(data)} rows")
    print(f"    - Date range: {data[time_col].min().date()} to {data[time_col].max().date()}")
    
    # 4.5 Encode brands using TRAINING-TIME mapping to match model embedding
    print("  [5.4.5] Encoding brands using training-time mapping...")
    if trained_brand2id is not None:
        # Use training-time mapping to ensure brand IDs match what model expects
        brand_id_col = data_config['brand_id_col']
        data[brand_id_col] = data[brand_col].map(trained_brand2id)
        
        # Check for unmapped brands (NaN in brand_id_col) BEFORE converting to int
        unmapped_mask = data[brand_id_col].isna()
        unmapped_count = unmapped_mask.sum()
        if unmapped_count > 0:
            print(f"    [WARNING] {unmapped_count} rows have brands not in training set:")
            unmapped_brands = data[unmapped_mask][brand_col].unique()
            print(f"    [WARNING] Unmapped brands: {unmapped_brands}")
            print(f"    [WARNING] Dropping {unmapped_count} rows with unmapped brands")
            data = data[~unmapped_mask].reset_index(drop=True)
        
        # Now safe to convert to int
        data[brand_id_col] = data[brand_id_col].astype(int)
        
        # Verify brand IDs are in valid range
        max_brand_id = data[brand_id_col].max()
        min_brand_id = data[brand_id_col].min()
        print(f"    - Created {brand_id_col} column: IDs range [{min_brand_id}, {max_brand_id}]")
        print(f"    - Model expects brand IDs in range [0, {num_brands - 1}]")
        
        if max_brand_id >= num_brands:
            raise ValueError(
                f"brand ID {max_brand_id} exceeds model's num_brands ({num_brands}). "
                f"Training and test data may have different brands."
            )
    else:
        # Fallback: create mapping on the fly (not recommended, may cause mismatches)
        print("    [WARNING] No training-time mapping found, creating new mapping from test data")
        data, _, _ = encode_brands(data, brand_col)
        brand_id_col = data_config['brand_id_col']
    
    # 5. Ensure proper date types for walk_forward_test
    print("  [5.5] Formatting date parameters...")
    test_start_date = pd.to_datetime(test_start).date()
    test_end_date = pd.to_datetime(test_end).date()  # Already exclusive from config, don't subtract 1
    print(f"    - Test window: {test_start_date} to {test_end_date} (end is exclusive)")

    # =====================================================================
    # Load pretrained model
    # =====================================================================
    print("\n[6/6] Loading pretrained models...")
    
    # Load model from new directory structure
    model_dir_path = Path(f"outputs/mvp_test/models")
    model_path = model_dir_path / "best_model.pth"
        
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please run mvp_train.py first to train brand-specific models."
        )
        
    print(f"  - Using model from: {model_path}")
    model, device, scaler = load_model_for_test(str(model_path), config)
    # =====================================================================
    # [PAUSE HERE] Walk-forward test (ready to call)
    # =====================================================================
    print("\n[READY] Prepared data for walk-forward test:")
    print(f"  - Data shape: {data.shape}")
    print(f"  - Feature columns: {feature_cols}")
    print(f"  - Test dates: {test_start_date} to {test_end_date}")
    
    # ------------------------------------------------------------------
    # Walk-forward test (t+1 only)
    # ------------------------------------------------------------------
    test_results = walk_forward_test(
        model=model,
        scaler=scaler,
        data=data,
        time_col=time_col,
        feature_cols=feature_cols,
        target_col=target_col,
        brand_id_col=brand_id_col,
        input_size=input_size,
        horizon=1,   # explicit, even if ignored internally
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        update_freq=pd.DateOffset(months=3),
        device=device
    )

    # Keep only 2025
    test_results = test_results[
        test_results['date'].dt.year == 2025
    ].copy()

    # Business constraint: no negative demand
    # Demand zero on OFF HOLIDAYS
    test_results['predicted'] = test_results['predicted'].clip(lower=0)
    holidays = load_holidays()
    off_holidays = {
        d
        for year_data in holidays.values()
        for d in year_data.get("off", [])
    }

    test_results.loc[
        test_results["date"].dt.date.isin(off_holidays),
        "predicted"
    ] = 0

    # Keep only essential columns
    test_results = test_results[
        ['predicted', 'actual', 'baseline', 'brand', 'date']
    ]

    print(f"[EVAL] One-step forecasts: {len(test_results)} rows")
    print(
        f"[EVAL] Date range: "
        f"{test_results['date'].min().date()} → "
        f"{test_results['date'].max().date()}"
    )

    if len(test_results) > 0:
        print(f"[EVAL] Brands: {sorted(test_results['brand'].unique())}")
        save_monthly_forecast(test_results)
    else:
        print("[EVAL] No forecasts generated")


    # Generate accuracy report
    print("\n[REPORTING] Generating accuracy report...")

    if len(test_results) > 0:
        id_to_brand_name = (
            {v: k for k, v in trained_brand2id.items()}
            if trained_brand2id else {}
        )

        report = generate_accuracy_report(
            test_results,
            actual_col='actual',
            forecast_col='predicted',
            baseline_col='baseline',
            brand_col='brand',
            date_col='date',
            brand_name_map=id_to_brand_name,
            output_path='outputs/accuracy_report.txt'
        )

        # print(report)
    else:
        print("  - No data available for accuracy report")

    
    # Generate plots for each brand/month combination
    if config.output['visualize'].get('save_plots', False):
        print("\n[PLOTTING] Generating monthly forecast plots...")

        if len(test_results) > 0:
            test_results['month'] = (
                test_results['date']
                .dt.to_period('M')
                .astype(str)
            )

            id_to_brand_name = (
                {v: k for k, v in trained_brand2id.items()}
                if trained_brand2id else {}
            )

            for brand in sorted(test_results['brand'].unique()):
                brand_name = id_to_brand_name.get(brand, str(brand))
                brand_data = test_results[test_results['brand'] == brand]

                print(f"  - brand {brand} ({brand_name})")

                for month in sorted(brand_data['month'].unique()):
                    plot_monthly_forecast(
                        test_results,
                        brand=brand,
                        brand_name=brand_name,
                        month_str=month,
                        output_dir="outputs/plots",
                        show=False,
                        plot_baseline=True,   # 👈 control
                    )


            print("  - All plots saved to: outputs/plots/")
        else:
            print("  - No data available for plotting")


if __name__ == "__main__":
    main()