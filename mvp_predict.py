"""Prediction script using MVP test model.

This script loads the trained model from mvp_test.py and makes predictions
for the configured prediction period.

Two modes are supported:
1. Teacher Forcing (Test Evaluation): Uses actual ground truth QTY values from prediction data as features
2. Recursive (Production Forecast): Uses model's own predictions as inputs for future dates
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
    slicing_window_category,
    encode_categories,
    aggregate_daily
)
from src.data.preprocessing import add_feature
from src.models import RNNForecastor
from src.training import Trainer
from src.utils import plot_difference, upload_to_google_sheets, GSPREAD_AVAILABLE, save_monthly_forecast
from src.utils.visualization import plot_monthly_forecast
from src.utils.metrics import generate_accuracy_report
from src.utils.google_sheets import upload_history_prediction


def fill_missing_dates(
    df: pd.DataFrame,
    time_col: str,
    cat_col: str,
    target_col: str,
    feature_cols: List[str] = None,
    fill_target: float = 0.0,
):
    """Ensure each category has continuous daily dates between its min and max.

    - Adds rows for missing dates per category.
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
    cats = df[cat_col].dropna().unique()
    for cat in cats:
        cat_mask = df[cat_col] == cat
        cat_dates = pd.to_datetime(df.loc[cat_mask, time_col]).dt.normalize()
        if cat_dates.empty:
            continue
        start = cat_dates.min()
        end = cat_dates.max()
        full_idx = pd.date_range(start=start, end=end, freq='D')
        existing = pd.DatetimeIndex(cat_dates.values)
        missing = full_idx.difference(existing)
        for d in missing:
            new_row = {time_col: d, cat_col: cat, target_col: fill_target}
            if feature_cols:
                for f in feature_cols:
                    new_row[f] = 0
            rows_to_add.append(new_row)

    if len(rows_to_add) > 0:
        fill_df = pd.DataFrame(rows_to_add)
        # Preserve column order where possible
        df = pd.concat([df, fill_df], axis=0, ignore_index=True, sort=False)
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(by=[cat_col, time_col]).reset_index(drop=True)

    return df

def load_model_for_test(model_path: str, config):
    """Load trained model from checkpoint and scaler."""
    # Load metadata first to get the training-time model/data config
    model_dir = Path(model_path).parent
    metadata_path = model_dir / "metadata.json"
    
    # ------------------------------------------------------------------
    # 1) Recover num_categories and full model architecture from metadata
    # ------------------------------------------------------------------
    num_categories = None
    model_config = None
    feature_cols = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Get full model_config (includes num_categories, input_dim, etc.)
        model_config = metadata.get('model_config', {})
        if 'num_categories' in model_config:
            num_categories = model_config['num_categories']

        # Also recover the exact feature column list used during training
        data_config = metadata.get("data_config", {})
        feature_cols = data_config.get("feature_cols")
    
    # Fallback to config if metadata doesn't have it
    if num_categories is None:
        model_config = config.model
        num_categories = model_config.get('num_categories')
    
        if num_categories is None:
            raise ValueError("num_categories must be found in model metadata or config")
    
    cat2id = None  # Training-time category mapping
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Try to extract category mapping from log_summary
        log_summary = metadata.get('log_summary', '')
        # Look for "Category mapping: {...}" in log_summary
        match = re.search(r"Category mapping: ({[^}]+})", log_summary)
        if match:
            try:
                # Parse the dictionary string from log_summary
                cat2id_str = match.group(1)
                # Convert single quotes to double quotes for JSON parsing
                cat2id_str = cat2id_str.replace("'", '"')
                cat2id = json.loads(cat2id_str)
            except:
                pass

    # If we have the training-time feature list, push it into the live config
    # so that window creation uses the exact same ordering and dimensionality.
    if feature_cols is not None:
        config.set("data.feature_cols", list(feature_cols))
    
    print(f"  - Loading model with num_categories={num_categories} (from trained model)")

    if cat2id:
        print(f"  - Training-time category mapping: {cat2id}")
    
    # ------------------------------------------------------------------
    # 2) Build model with the *exact* architecture used during training
    #    (input_dim, hidden_size, n_layers, etc. come from metadata)
    # -----------------------------------------------------------------
    model = RNNForecastor(
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
    
    # Load scaler from same directory as model (model_dir already defined above)
    scaler_path = model_dir / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  - Scaler loaded from: {scaler_path}")
    else:
        print(f"  [WARNING] Scaler not found at {scaler_path}, test will be in scaled space")
    
    return model, device, scaler, cat2id

def walk_forward_test(
    model : nn.Module,
    scaler : RollingGroupScaler,
    data : pd.DataFrame,
    time_col : str,
    feature_cols : List[str],
    target_col : str,
    cat_id_col : str,
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
    - Call slicing_window_category with [chunk_start, chunk_end) range
    - slicing_window_category automatically creates windows for each day in range
    
    Args:
        model: Trained PyTorch model for prediction.
        scaler: RollingGroupScaler fitted on training data.
        data: Full DataFrame containing historical + test data.
        time_col: Name of the time column in data.
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        cat_id_col: Name of the category ID column.
        input_size: Number of past days used as input (28).
        horizon: Number of future days to predict (28).
        test_start_date: Start date for testing (inclusive).
        test_end_date: End date for testing (inclusive).
        update_freq: Frequency to update scaler (e.g., pd.DateOffset(months=2)).
        device: Torch device (CPU or GPU).

    Returns:
        DataFrame with columns: ['predicted', 'actual', 'category', 'date', 'window_start_date'].
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

    all_preds = []
    all_y_true = []
    all_cat = []
    all_dates = []

    chunk_start = test_start_date - pd.DateOffset(months=1)
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
        # slicing_window_category will create windows for each day in [chunk_start, chunk_end)
        print(f"  [3/4] Creating sliding windows for each day in range")
        X_test, y_test, cat_test, test_dates = slicing_window_category(
            df=df_chunk_scaled,
            input_size=input_size,
            horizon=horizon,
            feature_cols=feature_cols,
            target_col=target_col,
            cat_col=cat_id_col,
            time_col=time_col,
            label_start_date=chunk_start,
            label_end_date=chunk_end,
            return_dates=True
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
            cats = torch.tensor(cat_test, dtype=torch.long).view(-1).to(device)
            outputs = model(inputs, cats).cpu().numpy()

        print(f"       Output shape: {outputs.shape}")

        # Inverse transform
        preds = scaler.inverse_transform_y(outputs, cat_test)
        y_true = scaler.inverse_transform_y(y_test, cat_test)

        # Collect
        all_preds.append(preds)
        all_y_true.append(y_true)
        all_cat.append(cat_test)
        all_dates.append(test_dates)

        chunk_start = chunk_end

    # Concatenate all chunks
    if len(all_preds) == 0:
        raise ValueError("No windows were created during walk-forward test!")
    
    preds_all = np.concatenate(all_preds, axis=0)  # shape: (N, horizon)
    y_true_all = np.concatenate(all_y_true, axis=0)  # shape: (N, horizon)
    cat_all = np.concatenate(all_cat, axis=0)  # shape: (N,)
    dates_all = np.concatenate(all_dates, axis=0)  # shape: (N,)

    print(f"\n[WALK-FORWARD] Complete!")
    print(f"  - Total windows: {len(preds_all)}")
    print(f"  - Horizon: {preds_all.shape[1]} days")
    print(f"  - Total predictions: {len(preds_all) * preds_all.shape[1]}")
    print(f"  - Date range: {dates_all.min()} to {dates_all.max()}")

    # Flatten predictions and actuals: (N, horizon) -> (N*horizon,)
    preds_flat = preds_all.flatten()
    actuals_flat = y_true_all.flatten()
    
    # Expand categories and dates
    cat_expanded = np.repeat(cat_all, preds_all.shape[1])
    
    # For each window date, create horizon consecutive dates
    dates_expanded = []
    window_start_dates = []
    for window_date in dates_all:
        window_date = pd.Timestamp(window_date)
        for day_offset in range(preds_all.shape[1]):
            dates_expanded.append(window_date + timedelta(days=day_offset))
            window_start_dates.append(window_date)
    dates_expanded = np.array(dates_expanded)
    window_start_dates = np.array(window_start_dates)

    # Convert to DataFrame
    results_df = pd.DataFrame({
        'predicted': preds_flat,
        'actual': actuals_flat,
        'category': cat_expanded,
        'date': dates_expanded,
        'window_start_date': window_start_dates
    })

    final = results_df.sort_values(by=['category', 'date']).reset_index(drop=True)

    return final


def main():
    print("=" * 80)
    print("TESTING PHASE")
    print("=" * 80)
    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config()
    data_config = config.data

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

    # Fill missing dates per category in historical data (target set to 0)
    try:
        # Note: data_config keys are not yet extracted here, so infer minimal defaults
        # We'll use the config values later when available; this conservative fill
        # fills only the `time` and category columns that exist in the loaded DF.
        print("  - Filling missing dates for historical data (per category)...")
        # Infer column names if available
        inf_time_col = data_config.get('time_col') if 'data_config' in locals() else None
        inf_cat_col = data_config.get('cat_col') if 'data_config' in locals() else None
        inf_target_col = data_config.get('target_col') if 'data_config' in locals() else None
        if inf_time_col and inf_cat_col and inf_target_col:
            before_rows = len(ref_data)
            ref_data = fill_missing_dates(ref_data, inf_time_col, inf_cat_col, inf_target_col)
            after_rows = len(ref_data)
            print(f"    - Historical: added {after_rows - before_rows} rows (now {after_rows})")
        else:
            # Delay fill until we have explicit names later in the pipeline
            print("    - Skipping automatic fill now; will ensure fill after config extraction")
    except Exception as e:
        print(f"    [WARNING] Failed to fill historical dates: {e}")

    # DIAGNOSTIC: Check historical data coverage
    # Get column names from config first
    cat_col = data_config['cat_col']
    time_col = data_config['time_col']
    target_col = data_config['target_col']
    
    if len(ref_data) > 0 and time_col in ref_data.columns:
        ref_data_dates = pd.to_datetime(ref_data[time_col])
        print(f"  - Historical data loaded: {len(ref_data)} rows")
        print(f"  - Date range: {ref_data_dates.min().date()} to {ref_data_dates.max().date()}")

    # Encode all categories from reference data (don't filter yet - need full mapping)
    # Note: We encode here to get cat2id mapping, but num_categories for model will come from trained model metadata
    _, trained_cat2id, num_categories = encode_categories(ref_data, data_config['cat_col'])
    # Don't overwrite config.model.num_categories here - it will be loaded from trained model metadata
    
    print(f"  - Category mapping: {trained_cat2id}")
    print(f"  - Number of categories in data: {num_categories}")
    
    # Determine which categories to predict
    # Filter out NaN values and convert to string to handle mixed types
    unique_ref_cats = ref_data[cat_col].dropna().astype(str).unique().tolist()
    available_categories = sorted([cat for cat in unique_ref_cats if cat.lower() != 'nan'])
    print(f"  - Available categories in training data: {available_categories}")

    # Load test data
    print("\n[3/6] Loading test data...")
    
    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test data file not found at: {test_data_path.absolute()}"
        )
    print(f"  - Loading from: {test_data_path}")
    test_data = pd.read_csv(test_data_path, encoding='utf-8', low_memory=False)

    # Fill missing dates per category in test data (target set to 0)
    try:
        print("  - Filling missing dates for test data (per category)...")
        # Use time/cat/target names from data_config (available above)
        td_time_col = data_config.get('time_col')
        td_cat_col = data_config.get('cat_col')
        td_target_col = data_config.get('target_col')
        before_rows = len(test_data)
        test_data = fill_missing_dates(test_data, td_time_col, td_cat_col, td_target_col)
        after_rows = len(test_data)
        print(f"    - Test data: added {after_rows - before_rows} rows (now {after_rows})")
    except Exception as e:
        print(f"    [WARNING] Failed to fill test data dates: {e}")

    print(f"  - Loaded {len(test_data)} samples")
    if len(test_data) == 0:
        raise ValueError(
                    f"Test data file is empty or contains no valid data."
                )

    # Filter to desired test window (keep all categories for now - will filter per category later)
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
    
    # Check which categories are available in test data
    # Filter out NaN values and convert to string to handle mixed types
    unique_cats = test_data[cat_col].dropna().astype(str).unique().tolist()
    available_test_categories = sorted([cat for cat in unique_cats if cat.lower() != 'nan'])
    test_year = test_start.year
    print(f"  - Available categories in {test_year} data: {available_test_categories}")
    
    # Filter categories_to_test to only those available in both reference and test data
    categories_to_test = [cat for cat in available_categories if cat in available_test_categories]
    
    if len(categories_to_test) == 0:
        raise ValueError(f"No matching categories found between reference data and {test_year} data. "
                        f"Reference: {available_categories}, {test_year}: {available_test_categories}")
    print(f"  - Final categories to predict: {categories_to_test}")

    # =====================================================================
    # [REFINEMENT] Prepare data before calling walk_forward_test()
    # =====================================================================
    print("\n[5/6] Preparing data for walk-forward test...")
    
    # 1. Extract key parameters from config
    print("  [5.1] Extracting config parameters...")
    feature_cols = data_config.get('feature_cols', [])
    target_col = data_config.get('target_col', 'Total CBM')
    input_size = config.window.get('input_size', 28)
    horizon = config.window.get('horizon', 28)
    
    print(f"    - Feature columns: {len(feature_cols)} columns")
    print(f"    - Target: {target_col}")
    print(f"    - Input size: {input_size} days")
    print(f"    - Horizon: {horizon} days")
    
    # 2. Prepare historical data (ref_data) with features
    print("  [5.2] Preparing historical data with features...")
    data_prepared = add_feature(
        ref_data.copy(),
        time_col=data_config['time_col'],
        cat_col=data_config['cat_col'],
        target_col=target_col
    )
    print(f"    - Added all features to historical data")
    
    # Aggregate to daily level
    data_prepared = aggregate_daily(data_prepared, data_config['time_col'], target_col, cat_col, feature_cols)
    print(f"    - Aggregated to daily: {len(data_prepared)} rows")
    
    # 3. Prepare test data with same features
    print("  [5.3] Preparing test data with features...")
    test_data_prepared = add_feature(
        test_data.copy(),
        time_col=data_config['time_col'],
        cat_col=data_config['cat_col'],
        target_col=target_col
    )
    print(f"    - Added all features to test data")
    
    # Aggregate test data to daily level
    test_data_prepared = aggregate_daily(test_data_prepared, data_config['time_col'], target_col, cat_col, feature_cols)
    print(f"    - Aggregated test to daily: {len(test_data_prepared)} rows")
    
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
    
    # 4.5 Encode categories using TRAINING-TIME mapping to match model embedding
    print("  [5.4.5] Encoding categories using training-time mapping...")
    if trained_cat2id is not None:
        # Use training-time mapping to ensure category IDs match what model expects
        cat_id_col = data_config['cat_id_col']
        data = data.copy()
        data[cat_id_col] = data[cat_col].map(trained_cat2id)
        
        # Check for unmapped categories (NaN in cat_id_col) BEFORE converting to int
        unmapped_mask = data[cat_id_col].isna()
        unmapped_count = unmapped_mask.sum()
        if unmapped_count > 0:
            print(f"    [WARNING] {unmapped_count} rows have categories not in training set:")
            unmapped_cats = data[unmapped_mask][cat_col].unique()
            print(f"    [WARNING] Unmapped categories: {unmapped_cats}")
            print(f"    [WARNING] Dropping {unmapped_count} rows with unmapped categories")
            data = data[~unmapped_mask].reset_index(drop=True)
        
        # Now safe to convert to int
        data[cat_id_col] = data[cat_id_col].astype(int)
        
        # Verify category IDs are in valid range
        max_cat_id = data[cat_id_col].max()
        min_cat_id = data[cat_id_col].min()
        print(f"    - Created {cat_id_col} column: IDs range [{min_cat_id}, {max_cat_id}]")
        print(f"    - Model expects category IDs in range [0, {num_categories - 1}]")
        
        if max_cat_id >= num_categories:
            raise ValueError(
                f"Category ID {max_cat_id} exceeds model's num_categories ({num_categories}). "
                f"Training and test data may have different categories."
            )
    else:
        # Fallback: create mapping on the fly (not recommended, may cause mismatches)
        print("    [WARNING] No training-time mapping found, creating new mapping from test data")
        data, _, _ = encode_categories(data, cat_col)
        cat_id_col = data_config['cat_id_col']
    
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
            f"Please run mvp_train.py first to train category-specific models."
        )
        
    print(f"  - Using model from: {model_path}")
    model, device, scaler, trained_cat2id = load_model_for_test(str(model_path), config)

    # =====================================================================
    # [PAUSE HERE] Walk-forward test (ready to call)
    # =====================================================================
    print("\n[READY] Prepared data for walk-forward test:")
    print(f"  - Data shape: {data.shape}")
    print(f"  - Feature columns: {feature_cols}")
    print(f"  - Test dates: {test_start_date} to {test_end_date}")
    
    # TODO: Uncomment below to run walk-forward test
    for cat in data[cat_col].unique():
        cat_data = data[data[cat_col] == cat]
        all_dates = pd.date_range(cat_data[time_col].min(), cat_data[time_col].max())
        missing = set(all_dates) - set(cat_data[time_col])
        print(f"Category {cat}: missing {len(missing)} days")

    test_results = walk_forward_test(
        model=model,
        scaler=scaler,
        data=data,
        time_col=time_col,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_id_col=cat_id_col,
        input_size=input_size,
        horizon=horizon,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        update_freq=pd.DateOffset(months=2), 
        device=device
    )
    
    test_results = test_results[test_results['date'].dt.year == 2025]
    # results_df contains columns: ['predicted', 'actual', 'category', 'date']
    print(f"\nWalk-forward test completed with {len(test_results)} samples")
    
    # Filter for 28-day horizon starting from day 1 of each month
    print("\n[FILTERING] Creating 28-day monthly window dataset...")
    test_results['date'] = pd.to_datetime(test_results['date'])
    test_results['window_start_date'] = pd.to_datetime(test_results['window_start_date'])
    
    # Filter for 28-day windows starting from day 1 of each month
    # Logic: Keep only predictions from windows where window_start_date.day == 1
    print(f"  - Total rows before filtering: {len(test_results)}")
    print(f"  - Unique window_start_dates: {len(test_results['window_start_date'].unique())}")
    print(f"  - Window start dates range: {test_results['window_start_date'].min().date()} to {test_results['window_start_date'].max().date()}")
    print(f"  - Window start months: {sorted(test_results['window_start_date'].dt.month.unique())}")
    
    # Add helper column to check if window starts on day 1
    test_results['window_start_day'] = test_results['window_start_date'].dt.day
    test_results['days_from_window_start'] = (test_results['date'] - test_results['window_start_date']).dt.days
    
    # Filter: Keep only windows starting on day 1, and only first 28 days of each window (0-29)
    test_results_monthly_df = test_results[
        (test_results['window_start_day'] == 1) & 
        (test_results['days_from_window_start'] >= 0) &
        (test_results['days_from_window_start'] <= 27)
    ].copy()
    
    # Clean up temporary columns, keep only the 4 main columns
    cols_to_keep = ['predicted', 'actual', 'category', 'date']
    test_results_monthly_df = test_results_monthly_df[[c for c in cols_to_keep if c in test_results_monthly_df.columns]]
    
    print(f"  - Output rows (day 1 windows, 28 days each): {len(test_results_monthly_df)}")
    
    if len(test_results_monthly_df) > 0:
        print(f"  - Categories: {test_results_monthly_df['category'].unique()}")
        print(f"  - Date range: {test_results_monthly_df['date'].min().date()} to {test_results_monthly_df['date'].max().date()}")
        save_monthly_forecast(test_results_monthly_df)
        print(test_results_monthly_df.head(10))
    else:
        print("  - No monthly windows found (day 1 windows)")
    
    # Generate plots for each category/month combination
    print("\n[PLOTTING] Generating monthly forecast plots...")
    if len(test_results_monthly_df) > 0:
        test_results_monthly_df['date'] = pd.to_datetime(test_results_monthly_df['date'])
        test_results_monthly_df['month'] = test_results_monthly_df['date'].dt.to_period('M').astype(str)
        
        # Create reverse mapping: ID -> Name for plot filenames
        id_to_cat_name = {v: k for k, v in trained_cat2id.items()} if trained_cat2id else {}
        print(f"  - Category ID to Name mapping: {id_to_cat_name}")
        
        # Group by category and month
        for category in sorted(test_results_monthly_df['category'].unique()):
            # Get category name for filename
            cat_name = id_to_cat_name.get(category, str(category))
            print(f"  - Category {category} ({cat_name}):")
            cat_data = test_results_monthly_df[test_results_monthly_df['category'] == category]
            
            for month in sorted(cat_data['month'].unique()):
                plot_monthly_forecast(
                    test_results_monthly_df,
                    category=category,
                    category_name=cat_name,
                    month_str=month,
                    output_dir="outputs/plots",
                    show=False
                )
        print(f"  - All plots saved to: outputs/plots/")
    else:
        print("  - No data available for plotting")
    
    # Generate accuracy report
    print("\n[REPORTING] Generating accuracy report...")
    if len(test_results_monthly_df) > 0:
        # Prepare category name mapping for report
        id_to_cat_name = {v: k for k, v in trained_cat2id.items()} if trained_cat2id else {}
        
        # Generate and save report
        report = generate_accuracy_report(
            test_results_monthly_df,
            actual_col='actual',
            forecast_col='predicted',
            category_col='category',
            date_col='date',
            category_name_map=id_to_cat_name,
            output_path='outputs/accuracy_report.txt'
        )
        print(report)
    else:
        print("  - No data available for accuracy report")

if __name__ == "__main__":
    main()