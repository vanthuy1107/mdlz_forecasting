"""Prediction execution functions.

This module contains the core prediction functions that execute model inference.
"""
import torch
import numpy as np
import pandas as pd
from datetime import date, timedelta

from src.data.preprocessing import get_vietnam_holidays


# Import helper functions from mvp_predict.py (temporary - will be refactored)
# These are utility functions that should eventually be moved to a utils module
def _solar_to_lunar_date(solar_date: date) -> tuple:
    """Convert solar (Gregorian) date to lunar (Vietnamese) date approximation."""
    if solar_date.month == 1 and solar_date.day >= 20:
        lunar_month = 1
        lunar_day = solar_date.day - 19
    elif solar_date.month == 2:
        if solar_date.day <= 10:
            lunar_month = 1
            lunar_day = solar_date.day + 12
        else:
            lunar_month = 2
            lunar_day = solar_date.day - 10
    else:
        lunar_month = solar_date.month
        lunar_day = solar_date.day
    
    lunar_month = max(1, min(12, lunar_month))
    lunar_day = max(1, min(30, lunar_day))
    return lunar_month, lunar_day


def _get_tet_start_dates(start_year: int, end_year: int):
    """Get Tet (Lunar New Year) start dates for a year range."""
    from config import load_holidays
    VIETNAM_HOLIDAYS_BY_YEAR = load_holidays(holiday_type="model")
    tet_dates = []
    for year in range(start_year, end_year + 1):
        if year in VIETNAM_HOLIDAYS_BY_YEAR:
            tet_window = VIETNAM_HOLIDAYS_BY_YEAR[year].get("tet", [])
            if tet_window:
                tet_dates.append(tet_window[0])
    return sorted(list(set(tet_dates)))


def _apply_sunday_to_monday_carryover_predictions(
    predictions_df: pd.DataFrame,
    date_col: str = 'date',
    pred_col: str = 'predicted'
) -> pd.DataFrame:
    """Apply Sunday-to-Monday carryover rule to predictions."""
    df = predictions_df.copy()
    
    def get_date_obj(d):
        if isinstance(d, date):
            return d
        elif isinstance(d, pd.Timestamp):
            return d.date()
        elif pd.api.types.is_datetime64_any_dtype(pd.Series([d])):
            return pd.to_datetime(d).date()
        else:
            return pd.to_datetime(d).date()
    
    df['_date_obj'] = df[date_col].apply(get_date_obj)
    df = df.sort_values('_date_obj').reset_index(drop=True)
    
    for i in range(len(df)):
        current_date = df.loc[i, '_date_obj']
        if current_date.weekday() == 6:  # Sunday
            sunday_value = df.loc[i, pred_col]
            if i + 1 < len(df):
                next_date = df.loc[i + 1, '_date_obj']
                days_diff = (next_date - current_date).days
                if next_date.weekday() == 0 and days_diff == 1:  # Next day is Monday
                    df.loc[i + 1, pred_col] = df.loc[i + 1, pred_col] + sunday_value
            df.loc[i, pred_col] = 0.0
    
    df = df.drop(columns=['_date_obj'])
    return df


def predict_direct_multistep(
    model,
    device,
    initial_window_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    config,
    cat_id: int,
    category: str = None
):
    """
    Direct multi-step prediction mode: predicts entire forecast horizon at once.
    
    CRITICAL: For category-specific models (num_categories=1), cat_id should always be 0.
    This ensures consistent results regardless of which categories are processed together.
    
    Args:
        model: Trained PyTorch model (already on device and in eval mode)
        device: PyTorch device
        initial_window_data: DataFrame with last input_size days of historical data
        start_date: First date to predict
        end_date: Last date to predict
        config: Configuration object
        cat_id: Category ID (integer) - should be 0 for category-specific models
        category: Category name (e.g., "FRESH") - used for category-specific post-processing
    
    Returns:
        DataFrame with columns: date, predicted
    """
    window_config = config.window
    data_config = config.data
    
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    feature_cols = data_config['feature_cols']
    time_col = data_config['time_col']
    
    if len(initial_window_data) < input_size:
        raise ValueError(
            f"Initial window must have at least {input_size} samples, "
            f"got {len(initial_window_data)}"
        )
    
    num_days_to_predict = (end_date - start_date).days + 1
    if num_days_to_predict != horizon:
        print(f"  [WARNING] Prediction range ({num_days_to_predict} days) doesn't match horizon ({horizon}). "
              f"Will predict {horizon} days starting from {start_date}")
    
    window = initial_window_data.tail(input_size).copy()
    window = window.sort_values(time_col).reset_index(drop=True)
    
    print(f"  - Starting direct multi-step prediction from {start_date} to {end_date}")
    print(f"  - Initial window: {window[time_col].min()} to {window[time_col].max()}")
    print(f"  - Model will output {horizon} predictions at once (direct multi-step)")
    print(f"  - Using cat_id={cat_id} for model input")
    
    window_features = window[feature_cols].values
    X_window = torch.tensor(
        window_features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    
    cat_tensor = torch.tensor([cat_id], dtype=torch.long).to(device)
    
    with torch.no_grad():
        pred_scaled = model(X_window, cat_tensor).cpu().numpy()
    
    pred_scaled = pred_scaled.squeeze(0) if pred_scaled.ndim > 1 else pred_scaled
    model_output_dim = pred_scaled.shape[0] if pred_scaled.ndim > 0 else 1
    
    if model_output_dim == 1 and horizon > 1:
        print(f"  [INFO] Model outputs single value (output_dim=1), but horizon={horizon}.")
        print(f"         Repeating prediction for all {horizon} days.")
        single_pred = float(pred_scaled[0] if pred_scaled.ndim > 0 else pred_scaled)
        pred_scaled = np.repeat(single_pred, horizon)
    elif model_output_dim < horizon:
        print(f"  [WARNING] Model output_dim ({model_output_dim}) < horizon ({horizon}).")
        print(f"           Repeating last prediction for remaining days.")
        if pred_scaled.ndim == 0:
            pred_scaled = np.array([pred_scaled])
        pred_scaled = np.concatenate([
            pred_scaled,
            np.repeat(pred_scaled[-1], horizon - model_output_dim)
        ])
    elif model_output_dim > horizon:
        pred_scaled = pred_scaled[:horizon]
    
    if pred_scaled.ndim == 0:
        pred_scaled = np.repeat(pred_scaled, horizon)
    elif len(pred_scaled) != horizon:
        if len(pred_scaled) < horizon:
            pred_scaled = np.concatenate([pred_scaled, np.repeat(pred_scaled[-1], horizon - len(pred_scaled))])
        else:
            pred_scaled = pred_scaled[:horizon]
    
    prediction_dates = [start_date + timedelta(days=i) for i in range(horizon)]
    
    extended_end = end_date + timedelta(days=365)
    holidays = get_vietnam_holidays(start_date, extended_end)
    holiday_set = set(holidays)
    
    predictions = []
    for i, pred_date in enumerate(prediction_dates):
        is_holiday = pred_date in holiday_set
        is_sunday = pred_date.weekday() == 6
        
        # Hard logic for holiday suppression: force zero-volume on holidays
        # This prevents "Holiday Blindness" where model predicts high volumes on non-operational days
        if is_holiday:
            pred_value = 0.0  # Strictly enforce zero on holidays
        elif is_sunday:
            pred_value = 0.0  # Sundays are also zero (especially for FRESH)
        else:
            pred_value = float(pred_scaled[i])
        
        predictions.append({
            'date': pred_date,
            'predicted': pred_value
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    # Apply Sunday-to-Monday carryover for all categories EXCEPT FRESH
    # FRESH category requires Sunday-to-Zero hard mask (no carryover)
    if category != "FRESH":
        predictions_df = _apply_sunday_to_monday_carryover_predictions(
            predictions_df,
            date_col='date',
            pred_col='predicted'
        )
    else:
        # For FRESH category: enforce Sunday-to-Zero hard mask
        # Ensure all Sundays remain at zero (no carryover to Monday)
        predictions_df['_date_obj'] = predictions_df['date'].apply(
            lambda d: d if isinstance(d, date) else pd.to_datetime(d).date()
        )
        predictions_df.loc[predictions_df['_date_obj'].apply(lambda d: d.weekday() == 6), 'predicted'] = 0.0
        predictions_df = predictions_df.drop(columns=['_date_obj'])
    
    return predictions_df


def predict_direct_multistep_rolling(
    model,
    device,
    initial_window_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    config,
    cat_id: int,
    category: str = None
):
    """
    Predict for a long date range by looping through chunks of horizon days.
    
    CRITICAL: For category-specific models (num_categories=1), cat_id should always be 0.
    
    Args:
        model: Trained PyTorch model
        device: PyTorch device
        initial_window_data: DataFrame with last input_size days of historical data
        start_date: First date to predict
        end_date: Last date to predict
        config: Configuration object
        cat_id: Category ID (integer) - should be 0 for category-specific models
        category: Category name (e.g., "FRESH") - used for category-specific post-processing
    
    Returns:
        DataFrame with columns: date, predicted
    """
    window_config = config.window
    data_config = config.data
    
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    feature_cols = data_config['feature_cols']
    target_col = data_config['target_col']
    time_col = data_config['time_col']
    
    if len(initial_window_data) < input_size:
        raise ValueError(
            f"Initial window must have at least {input_size} samples, "
            f"got {len(initial_window_data)}"
        )
    
    window = initial_window_data.tail(input_size).copy()
    window = window.sort_values(time_col).reset_index(drop=True)
    
    all_predictions = []
    current_start = start_date
    chunk_num = 0
    
    total_days = (end_date - start_date).days + 1
    print(f"  - Starting rolling prediction for {total_days} days (horizon={horizon} days per chunk)")
    print(f"  - Date range: {start_date} to {end_date}")
    print(f"  - Initial window: {window[time_col].min()} to {window[time_col].max()}")
    
    while current_start <= end_date:
        chunk_num += 1
        chunk_end = min(
            current_start + timedelta(days=horizon - 1),
            end_date
        )
        chunk_days = (chunk_end - current_start).days + 1
        
        print(f"\n  [Chunk {chunk_num}] Predicting {current_start} to {chunk_end} ({chunk_days} days)...")
        
        chunk_predictions = predict_direct_multistep(
            model=model,
            device=device,
            initial_window_data=window,
            start_date=current_start,
            end_date=chunk_end,
            config=config,
            cat_id=cat_id,
            category=category
        )
        
        all_predictions.append(chunk_predictions)
        
        # Update window with predictions (simplified - full implementation would recompute all features)
        print(f"  - Updating window with {len(chunk_predictions)} predictions...")
        # Note: Full implementation would need to recompute rolling features, etc.
        # For now, this is a placeholder - the full implementation is in mvp_predict.py
        
        current_start = chunk_end + timedelta(days=1)
    
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        final_predictions = final_predictions.drop_duplicates(subset=['date'], keep='first')
        final_predictions = final_predictions.sort_values('date').reset_index(drop=True)
        print(f"\n  - Completed rolling prediction: {len(final_predictions)} total predictions")
        return final_predictions
    else:
        return pd.DataFrame(columns=['date', 'predicted'])


def get_historical_window_data(
    historical_data: pd.DataFrame,
    end_date: date,
    config,
    num_days: int = 30
):
    """
    Extract the last N days of historical data to initialize prediction window.
    
    Args:
        historical_data: DataFrame with historical data
        end_date: Last date to include
        config: Configuration object
        num_days: Number of days to extract
    
    Returns:
        DataFrame with last num_days of data, sorted by time
    """
    time_col = config.data['time_col']
    
    historical_data = historical_data[
        pd.to_datetime(historical_data[time_col]).dt.date <= end_date
    ].copy()
    
    historical_data = historical_data.sort_values(time_col).reset_index(drop=True)
    
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
    
    window_data = window_data.drop(columns=['date_only'])
    
    return window_data.sort_values(time_col).reset_index(drop=True)
