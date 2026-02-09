"""Prediction execution functions.

This module contains the core prediction functions that execute model inference.
"""
import torch
import numpy as np
import pandas as pd
from datetime import date, timedelta

from src.data.preprocessing import get_vietnam_holidays, add_days_until_lunar_08_01_feature


# Import helper functions from mvp_predict.py (temporary - will be refactored)
# These are utility functions that should eventually be moved to a utils module
def _solar_to_lunar_date(solar_date: date) -> tuple:
    """
    Convert solar (Gregorian) date to lunar (Vietnamese) date using anchor points.
    
    Uses known Mid-Autumn Festival dates (Lunar Month 8, Day 15) as anchor points:
    - 2023: Sep 29 = Lunar 08-15
    - 2024: Sep 17 = Lunar 08-15
    - 2025: Oct 6 = Lunar 08-15
    - 2026: Sep 25 = Lunar 08-15
    """
    # Anchor points: Mid-Autumn Festival dates (Lunar Month 8, Day 15)
    mid_autumn_anchors = {
        2023: date(2023, 9, 29),   # Lunar 08-15
        2024: date(2024, 9, 17),   # Lunar 08-15
        2025: date(2025, 10, 6),   # Lunar 08-15
        2026: date(2026, 9, 25),   # Lunar 08-15
    }
    
    # Find the closest Mid-Autumn anchor
    year = solar_date.year
    if year not in mid_autumn_anchors:
        if year < 2023:
            anchor_year = 2023
        elif year > 2026:
            anchor_year = 2026
        else:
            anchor_year = year
    else:
        anchor_year = year
    
    anchor_date = mid_autumn_anchors[anchor_year]
    days_diff = (solar_date - anchor_date).days
    
    # Start from Lunar Month 8, Day 15
    lunar_month = 8
    lunar_day = 15
    
    # Adjust by days difference (approximate: 29.5 days per lunar month)
    if days_diff > 0:
        lunar_day += days_diff
        while lunar_day > 30:
            lunar_day -= 30
            lunar_month += 1
            if lunar_month > 12:
                lunar_month = 1
    else:
        lunar_day += days_diff
        while lunar_day < 1:
            lunar_day += 30
            lunar_month -= 1
            if lunar_month < 1:
                lunar_month = 12
    
    lunar_month = max(1, min(12, lunar_month))
    lunar_day = max(1, min(30, lunar_day))
    return lunar_month, lunar_day


def _get_is_active_season_mooncake(pred_date: date) -> int:
    """
    Calculate is_active_season for MOONCAKE category.
    
    MOONCAKE is active between Lunar Months 7-9 AND Gregorian months 7-9 (July-September).
    
    CRITICAL FIX: Narrowed from Lunar Months 6-9 to 7-9 to prevent early June predictions.
    Added Gregorian month constraints (7-9) to ensure predictions only in July-September.
    
    For 2025:
    - June 2025: Lunar Month 6 (OFF - too early, prevented by Gregorian constraint)
    - July 2025: Lunar Month 6-7 (ON if Lunar Month >= 7, otherwise OFF)
    - August 2025: Lunar Month 7-8 (ON - peak season)
    - September 2025: Lunar Month 8-9 (ON - peak season)
    - October 2025: Lunar Month 9 (OFF - too late, prevented by Gregorian constraint)
    
    Args:
        pred_date: Prediction date (Gregorian)
    
    Returns:
        1 if in active season (Lunar Months 7-9 AND Gregorian months 7-9), 0 otherwise
    """
    # Get lunar month and Gregorian month
    lunar_month, lunar_day = _solar_to_lunar_date(pred_date)
    gregorian_month = pred_date.month
    
    # Active season: Lunar Months 7-9 AND Gregorian months 7-9 (July-September only)
    # This prevents early predictions in June and late predictions in October
    is_active = (lunar_month >= 7) and (lunar_month <= 9) and (gregorian_month >= 7) and (gregorian_month <= 9)
    
    return 1 if is_active else 0


def _get_is_golden_window_mooncake(pred_date: date) -> int:
    """
    Calculate is_golden_window for MOONCAKE category.
    
    Golden Window: Lunar Months 6.15 to 8.01 (peak buildup period)
    
    Args:
        pred_date: Prediction date (Gregorian)
    
    Returns:
        1 if in Golden Window, 0 otherwise
    """
    lunar_month, lunar_day = _solar_to_lunar_date(pred_date)
    is_golden = (
        ((lunar_month == 6) and (lunar_day >= 15)) or
        (lunar_month == 7) or
        ((lunar_month == 8) and (lunar_day <= 1))
    )
    return 1 if is_golden else 0


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


def _apply_dow_anchored_hybrid_baseline(
    predictions_df: pd.DataFrame,
    history_df: pd.DataFrame,
    time_col: str,
    target_col: str,
    alpha: float = 0.5,
    hist_weeks: int = 8,
) -> pd.DataFrame:
    """
    Apply DOW-Anchored Hybrid Baseline smoothing to a sequence of predictions.

    This implements the following decomposition for each forecast date T:
        B_pred(T)   = 7-day rolling mean of model predictions
        B_hist(T)   = historical mean for the same DOW over the last N weeks
        B_hybrid(T) = alpha * B_pred(T) + (1 - alpha) * B_hist(T)

        RNN_residual(T)   = Raw_Pred(T) - B_pred(T)
        Final_Forecast(T) = B_hybrid(T) + RNN_residual(T)

    which is equivalent to:
        Final_Forecast(T) = Raw_Pred(T)
                            + (1 - alpha) * (B_hist(T) - B_pred(T))

    All computations are performed in the model's working scale
    (typically StandardScaler space for the target column). The caller
    is responsible for inverse-transforming back to absolute units.
    """
    if predictions_df is None or len(predictions_df) == 0:
        return predictions_df
    if history_df is None or len(history_df) == 0:
        return predictions_df

    preds = predictions_df.copy()

    # Ensure we have a proper datetime column for history
    if time_col not in history_df.columns:
        return preds

    hist = history_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(hist[time_col]):
        hist[time_col] = pd.to_datetime(hist[time_col])

    # Restrict history to the most recent hist_weeks window when possible
    try:
        max_hist_date = hist[time_col].dt.date.max()
        lookback_days = max(hist_weeks * 7, 7)
        min_hist_date = max_hist_date - timedelta(days=lookback_days - 1)
        mask_recent = hist[time_col].dt.date >= min_hist_date
        hist_recent = hist.loc[mask_recent].copy()
        if len(hist_recent) == 0:
            hist_recent = hist
    except Exception:
        hist_recent = hist

    if target_col not in hist_recent.columns:
        return preds

    # Coerce target to numeric to avoid dtype issues
    hist_recent[target_col] = pd.to_numeric(hist_recent[target_col], errors="coerce")
    hist_recent = hist_recent.dropna(subset=[target_col])
    if len(hist_recent) == 0:
        return preds

    # Compute DOW -> historical mean mapping (B_hist)
    hist_recent["_dow"] = hist_recent[time_col].dt.dayofweek
    dow_means = hist_recent.groupby("_dow")[target_col].mean()
    global_mean = hist_recent[target_col].mean()

    # Sort predictions by date to enforce temporal order
    preds = preds.sort_values("date").reset_index(drop=True)

    def _get_weekday(d):
        if isinstance(d, date):
            return d.weekday()
        if isinstance(d, pd.Timestamp):
            return d.dayofweek
        # Fallback for string / numpy.datetime64
        return pd.to_datetime(d).dayofweek

    dows = preds["date"].apply(_get_weekday)
    b_hist_values = dows.map(lambda d: dow_means.get(d, global_mean)).to_numpy()

    # Dynamic component: 7-day rolling mean of model predictions (B_pred)
    y = preds["predicted"].astype(float).to_numpy()
    if len(y) == 0:
        return preds

    b_pred_values = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        start_idx = max(0, i - 6)  # Up to 7 days including current
        b_pred_values[i] = np.mean(y[start_idx : i + 1])

    # Residual relative to dynamic baseline
    residual_values = y - b_pred_values

    # Hybrid baseline and final adjustment
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    b_hybrid_values = alpha_clamped * b_pred_values + (1.0 - alpha_clamped) * b_hist_values
    final_values = b_hybrid_values + residual_values

    preds["predicted"] = final_values
    return preds


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
    target_col = data_config['target_col']
    
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
    
    # Ensure all required features exist in the window (create missing ones with proper or default values)
    missing_features = [f for f in feature_cols if f not in window.columns]
    if missing_features:
        print(f"  - WARNING: Missing features in window: {missing_features}")
        print(f"  - Creating missing features (computed where possible, else default)...")
        for feat in missing_features:
            if feat == "is_peak_loss_window":
                # For is_peak_loss_window, calculate based on lunar dates if available
                if "lunar_month" in window.columns and "lunar_day" in window.columns:
                    lunar_month = window["lunar_month"]
                    lunar_day = window["lunar_day"]
                    is_peak_loss_window = (
                        ((lunar_month == 7) & (lunar_day >= 15)) |
                        ((lunar_month == 8) & (lunar_day <= 15))
                    )
                    window["is_peak_loss_window"] = is_peak_loss_window.astype(int)
                else:
                    window["is_peak_loss_window"] = 0
            elif feat == "is_august":
                # Binary: 1 if Gregorian month == 8, else 0 (critical for MOONCAKE)
                if time_col in window.columns:
                    t = pd.to_datetime(window[time_col])
                    window["is_august"] = (t.dt.month == 8).astype(int)
                else:
                    window["is_august"] = 0
            elif feat == "days_until_lunar_08_01":
                # Countdown to Lunar 08-01; requires lunar_month/lunar_day
                if "lunar_month" in window.columns and "lunar_day" in window.columns:
                    window = add_days_until_lunar_08_01_feature(
                        window,
                        time_col=time_col,
                        lunar_month_col="lunar_month",
                        lunar_day_col="lunar_day",
                        days_until_lunar_08_01_col="days_until_lunar_08_01",
                    )
                else:
                    window["days_until_lunar_08_01"] = 365  # fallback when lunar cols missing
            else:
                # For other missing features, use 0 as default
                window[feat] = 0
    
    print(f"  - Starting direct multi-step prediction from {start_date} to {end_date}")
    print(f"  - Initial window: {window[time_col].min()} to {window[time_col].max()}")
    print(f"  - Model will output {horizon} predictions at once (direct multi-step)")
    print(f"  - Using cat_id={cat_id} for model input")
    
    # ENHANCED: Input Injection (Warm-up) for MOONCAKE
    # If the input window is all zeros (off-season), inject lunar-aligned historical patterns
    # This prevents "Zero-locking" where the RNN fails to anticipate the upcoming peak
    window_features = window[feature_cols].values
    
    if category == "MOONCAKE":
        # Check if window volume is near-zero
        target_col_idx = feature_cols.index(data_config['target_col'])
        window_volume = window_features[:, target_col_idx]
        
        if np.sum(np.abs(window_volume)) <= 1.0:  # Near-zero window
            print(f"  [WARM-UP] Zero input window detected for MOONCAKE")
            # Inject lunar-aligned historical patterns
            from src.utils.lunar_utils import inject_lunar_aligned_warmup
            
            # Get window dates
            window_dates = pd.to_datetime(window[time_col]).dt.date.tolist()
            
            # Use initial_window_data as historical data source
            # (it should contain the full historical dataset if passed correctly)
            window_features = inject_lunar_aligned_warmup(
                input_window=window_features,
                window_dates=window_dates,
                historical_data=initial_window_data if len(initial_window_data) > input_size else window,
                target_col=data_config['target_col'],
                time_col=time_col,
                cat_col=data_config.get('cat_col', 'CATEGORY'),
                category=category,
                target_col_idx=target_col_idx
            )
    
    X_window = torch.tensor(
        window_features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    
    # NOTE: Input scaling check for off-season features
    # Only the target column (Total CBM) is scaled using StandardScaler, NOT the input features.
    # Input features (is_active_season, lunar_month, etc.) remain in their natural ranges:
    # - Binary features (is_active_season, is_golden_window): 0 or 1
    # - Cyclical features (sin/cos): -1 to 1
    # - Countdown features: raw day counts
    # This ensures off-season features (is_active_season=0) are not scaled to "near-zero"
    # values that the LSTM might interpret as small signals. The binary 0/1 encoding
    # provides a clear, unambiguous signal to the model.
    
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
    
    # DEBUG: Print holidays for January 2025
    jan_2025_holidays = [h for h in holidays if h.year == 2025 and h.month == 1]
    if len(jan_2025_holidays) > 0:
        print(f"\n  [DEBUG] January 2025 holidays loaded: {sorted(jan_2025_holidays)}")
        print(f"  [DEBUG] Total holidays in range: {len(holidays)}")
    
    predictions = []
    for i, pred_date in enumerate(prediction_dates):
        is_holiday = pred_date in holiday_set
        is_sunday = pred_date.weekday() == 6
        
        # DEBUG: Print for specific dates
        if pred_date.year == 2025 and pred_date.month == 1 and 25 <= pred_date.day <= 31:
            print(f"  [DEBUG] Date {pred_date}: is_holiday={is_holiday}, is_sunday={is_sunday}, raw_pred={float(pred_scaled[i]):.2f}")
        
        # Hard logic for holiday suppression: force zero-volume on holidays
        # This prevents "Holiday Blindness" where model predicts high volumes on non-operational days
        if is_holiday:
            pred_value = 0.0  # Strictly enforce zero on holidays
        elif is_sunday:
            pred_value = 0.0  # Sundays are also zero (especially for FRESH)
        else:
            pred_value = float(pred_scaled[i])
        
        # CRITICAL FIX: Hard-Masking Logic for MOONCAKE
        # Re-enforce strict off-season masking: Final_Pred = Raw_Pred * is_active_season
        # This prevents predictions from leaking into off-season (e.g., January)
        if category == "MOONCAKE":
            is_active_season = _get_is_active_season_mooncake(pred_date)
            pred_value = pred_value * is_active_season
            # REMOVED: Golden Window hard masking
            # Golden Window should be used for loss weighting during training, NOT for hard masking during prediction
            # Hard masking was causing valid active-season predictions to be zeroed out
            # The model should learn to predict lower values outside Golden Window through training, not forced zeros
        
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
    
    # ------------------------------------------------------------------
    # DOW-Anchored Hybrid Baseline (optional, controlled via config)
    # ------------------------------------------------------------------
    # When enabled, re-center the prediction trajectory around a hybrid
    # baseline that combines:
    #   - B_pred(T): 7-day rolling mean of the model's own predictions
    #   - B_hist(T): DOW-specific historical mean from recent weeks
    # This is applied in the model's working scale; inverse-scaling is
    # handled later by the calling pipeline.
    inference_cfg = getattr(config, "inference", None)
    if inference_cfg is not None:
        dow_cfg = inference_cfg.get("dow_hybrid_baseline", {})
        if dow_cfg.get("enabled", False):
            alpha = float(dow_cfg.get("alpha", 0.5))
            hist_weeks = int(dow_cfg.get("hist_weeks", 8))
            predictions_df = _apply_dow_anchored_hybrid_baseline(
                predictions_df=predictions_df,
                history_df=initial_window_data,
                time_col=time_col,
                target_col=target_col,
                alpha=alpha,
                hist_weeks=hist_weeks,
            )
    
    return predictions_df


def predict_direct_multistep_rolling(
    model,
    device,
    initial_window_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    config,
    cat_id: int,
    category: str = None,
    historical_data: pd.DataFrame = None
):
    """
    Predict for a long date range by looping through CALENDAR MONTH chunks.
    
    ENHANCED: Predictions are now aligned with calendar month boundaries.
    Each chunk predicts one complete month (from 1st to last day of month).
    
    CRITICAL: For category-specific models (num_categories=1), cat_id should always be 0.
    
    Args:
        model: Trained PyTorch model
        device: PyTorch device
        initial_window_data: DataFrame with last input_size days of historical data
        start_date: First date to predict (will be adjusted to 1st of month if not already)
        end_date: Last date to predict
        config: Configuration object
        cat_id: Category ID (integer) - should be 0 for category-specific models
        category: Category name (e.g., "FRESH") - used for category-specific post-processing
        historical_data: Optional full historical dataset for recomputing YoY features (critical for MOONCAKE)
    
    Returns:
        DataFrame with columns: date, predicted
    """
    from src.data.preprocessing import add_year_over_year_volume_features
    
    window_config = config.window
    data_config = config.data
    
    input_size = window_config['input_size']
    horizon = window_config['horizon']
    feature_cols = data_config['feature_cols']
    target_col = data_config['target_col']
    time_col = data_config['time_col']
    cat_col = data_config.get('cat_col', 'CATEGORY')
    
    # ENHANCED: Support full-month windows (don't truncate to input_size if using full months)
    # Full-month windows capture monthly trends better than arbitrary N-day windows
    use_full_month_window = len(initial_window_data) >= input_size
    
    if len(initial_window_data) < input_size:
        print(f"  [WARNING] Initial window has {len(initial_window_data)} samples, less than input_size={input_size}")
        print(f"            Using all available data...")
        window = initial_window_data.copy()
    elif use_full_month_window and len(initial_window_data) > input_size:
        # Use the full window if it's larger than input_size (e.g., full month data)
        print(f"  [FULL-MONTH MODE] Using full window ({len(initial_window_data)} days) instead of truncating to input_size={input_size}")
        window = initial_window_data.copy()
    else:
        # Standard mode: use last input_size days
        window = initial_window_data.tail(input_size).copy()
    
    window = window.sort_values(time_col).reset_index(drop=True)
    
    # Build combined historical dataset for YoY feature lookup
    # This includes both the initial window and any historical data provided
    combined_historical = window.copy()
    if historical_data is not None and len(historical_data) > 0:
        # Ensure historical data has same columns and is sorted
        hist_cols = [col for col in window.columns if col in historical_data.columns]
        if len(hist_cols) > 0:
            hist_subset = historical_data[hist_cols].copy()
            hist_subset = hist_subset.sort_values(time_col).reset_index(drop=True)
            # Combine: historical data + current window (avoid duplicates)
            combined_historical = pd.concat([hist_subset, window], ignore_index=True)
            combined_historical = combined_historical.drop_duplicates(subset=[time_col], keep='last')
            combined_historical = combined_historical.sort_values(time_col).reset_index(drop=True)
    
    all_predictions = []
    current_start = start_date
    chunk_num = 0
    
    total_days = (end_date - start_date).days + 1
    print(f"  - Starting rolling prediction for {total_days} days")
    print(f"  - Date range: {start_date} to {end_date}")
    print(f"  - Prediction mode: CALENDAR MONTH chunks (each chunk = one complete month)")
    print(f"  - Initial window: {window[time_col].min()} to {window[time_col].max()}")
    if historical_data is not None:
        print(f"  - Historical data available: {len(historical_data)} rows for YoY feature lookup")
    
    while current_start <= end_date:
        chunk_num += 1
        
        # ENHANCED: Use calendar month boundaries instead of fixed horizon
        # Each chunk predicts ONE COMPLETE MONTH from the 1st to the last day of that month
        # This aligns predictions with natural business cycles
        
        # Get the month boundaries
        chunk_year = current_start.year
        chunk_month = current_start.month
        
        # If current_start is NOT the 1st of the month, adjust it
        if current_start.day != 1:
            print(f"  [WARNING] Current start date {current_start} is not the 1st of month. Adjusting to month start.")
            current_start = date(chunk_year, chunk_month, 1)
        
        # Calculate the last day of this month
        if chunk_month == 12:
            next_month_year = chunk_year + 1
            next_month = 1
        else:
            next_month_year = chunk_year
            next_month = chunk_month + 1
        
        # Last day of current month = day before 1st of next month
        from datetime import date as dt_date
        first_of_next_month = dt_date(next_month_year, next_month, 1)
        chunk_end = first_of_next_month - timedelta(days=1)
        
        # Don't exceed the requested end_date
        chunk_end = min(chunk_end, end_date)
        
        chunk_days = (chunk_end - current_start).days + 1
        
        print(f"\n  [Chunk {chunk_num}] Predicting {current_start} to {chunk_end} ({chunk_days} days) - Month {chunk_month:02d}/{chunk_year}...")
        
        # Check if chunk_days exceeds model's horizon
        if chunk_days > horizon:
            print(f"  [WARNING] Month has {chunk_days} days but model horizon is {horizon}. Will predict {horizon} days.")
            chunk_end = current_start + timedelta(days=horizon - 1)
        
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
        
        # Update window with predictions and recompute features
        print(f"  - Updating window with {len(chunk_predictions)} predictions...")
        
        # Get the last row from window as a template
        last_row_template = window.iloc[-1:].copy()
        
        # For each predicted date, create a feature row
        new_rows = []
        for _, pred_row in chunk_predictions.iterrows():
            pred_date = pred_row['date']
            pred_value = pred_row['predicted']  # Already in scaled space
            
            # Create new row based on template
            new_row = last_row_template.copy()
            new_row[time_col] = pd.Timestamp(pred_date)
            new_row[target_col] = pred_value  # Use predicted value
            
            # Compute temporal features for this date
            pred_datetime = pd.Timestamp(pred_date)
            month = pred_datetime.month
            dayofmonth = pred_datetime.day
            new_row['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
            new_row['dayofmonth_sin'] = np.sin(2 * np.pi * (dayofmonth - 1) / 31)
            new_row['dayofmonth_cos'] = np.cos(2 * np.pi * (dayofmonth - 1) / 31)
            
            # Compute weekend features
            day_of_week = pred_datetime.dayofweek
            new_row['is_weekend'] = 1 if day_of_week >= 5 else 0
            new_row['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            new_row['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # CRITICAL FIX: Root Cause #3 - LSTM State Reset via Input Window Adjustment
            # When predicting the 1st day of a new month, we need to "signal" to the LSTM that
            # there's a significant context change (month boundary crossing).
            # Since we can't directly reset LSTM hidden states during inference, we use a 
            # "Month-Boundary Signal Amplification" strategy:
            # - Inject strong early-month penalty signals into the most recent days of the input window
            # - This helps counteract the "momentum" from EOM high-volume data
            # - Only applies when we're about to predict Day 1 of a new month
            
            if dayofmonth == 1:
                # This is Day 1 prediction - apply LSTM state reset strategy
                # Modify the last 3-5 days in the window to inject month-boundary signals
                print(f"  [LSTM STATE RESET] Month boundary detected at {pred_date}. Amplifying early-month signals in input window...")
                
                # Get the last 3 rows of the window (the most recent context)
                window_length = len(window)
                amplification_zone_size = min(3, window_length)
                
                # For each row in the amplification zone, boost early-month penalty features
                for amp_idx in range(amplification_zone_size):
                    window_idx = window_length - amplification_zone_size + amp_idx
                    
                    # Amplify early-month signals (if features exist)
                    if 'is_first_5_days' in window.columns:
                        window.loc[window_idx, 'is_first_5_days'] = 1  # Signal upcoming month boundary
                    if 'post_peak_signal' in window.columns:
                        window.loc[window_idx, 'post_peak_signal'] = 1.0  # Maximum decay signal
                    if 'is_high_vol_weekday_AND_early_month' in window.columns:
                        # If this day in the window is a high-volume weekday, suppress it
                        window_day_of_week = int(window.loc[window_idx, 'day_of_week_sin'] * 7 / (2 * np.pi))  # Approximate
                        if window.loc[window_idx, 'is_high_volume_weekday'] == 1 if 'is_high_volume_weekday' in window.columns else False:
                            window.loc[window_idx, 'is_high_vol_weekday_AND_early_month'] = -2
            
            # Compute lunar calendar features
            lunar_month, lunar_day = _solar_to_lunar_date(pred_date)
            new_row['lunar_month'] = lunar_month
            new_row['lunar_day'] = lunar_day
            
            # Compute lunar cyclical features
            new_row['lunar_month_sin'] = np.sin(2 * np.pi * (lunar_month - 1) / 12.0)
            new_row['lunar_month_cos'] = np.cos(2 * np.pi * (lunar_month - 1) / 12.0)
            new_row['lunar_day_sin'] = np.sin(2 * np.pi * (lunar_day - 1) / 30.0)
            new_row['lunar_day_cos'] = np.cos(2 * np.pi * (lunar_day - 1) / 30.0)
            
            # Compute seasonal features for MOONCAKE
            if category == "MOONCAKE":
                # Active season: Lunar Months 6-9
                is_active = (lunar_month >= 6) and (lunar_month <= 9)
                new_row['is_active_season'] = 1 if is_active else 0
                
                # Golden Window: Lunar Months 6.15 to 8.01
                is_golden = (
                    ((lunar_month == 6) and (lunar_day >= 15)) or
                    (lunar_month == 7) or
                    ((lunar_month == 8) and (lunar_day <= 1))
                )
                new_row['is_golden_window'] = 1 if is_golden else 0
                
                # Peak Loss Window: Lunar Months 7.15 to 8.15
                is_peak_loss = (
                    ((lunar_month == 7) and (lunar_day >= 15)) or
                    ((lunar_month == 8) and (lunar_day <= 15))
                )
                new_row['is_peak_loss_window'] = 1 if is_peak_loss else 0
            
            # ENHANCED: Gregorian-anchored features using lunar_utils
            if 'is_august' in new_row.columns:
                new_row['is_august'] = 1 if month == 8 else 0
            if 'days_until_lunar_08_01' in new_row.columns:
                from src.utils.lunar_utils import compute_days_until_lunar_08_01
                new_row['days_until_lunar_08_01'] = compute_days_until_lunar_08_01(pred_date)
            
            # CRITICAL FIX: Root Cause #1 - Update Early Month Penalty Features Dynamically
            # These features were previously inherited from the EOM template, causing Day 1 to miss the early-month signal
            # Now we recompute them for each prediction date to ensure accurate penalty signals
            
            # 1. early_month_low_tier: Tiered penalty signal
            if 'early_month_low_tier' in new_row.columns:
                if dayofmonth <= 5:
                    new_row['early_month_low_tier'] = -10  # EXTREME low volume (days 1-5)
                elif dayofmonth <= 10:
                    new_row['early_month_low_tier'] = 1    # Transitioning low volume (days 6-10)
                else:
                    new_row['early_month_low_tier'] = 2    # Normal days
            
            # 2. is_early_month_low: Binary flag for days 1-10
            if 'is_early_month_low' in new_row.columns:
                new_row['is_early_month_low'] = 1 if dayofmonth <= 10 else 0
            
            # 3. is_first_5_days: Binary flag for severe drop period (days 1-5)
            if 'is_first_5_days' in new_row.columns:
                new_row['is_first_5_days'] = 1 if dayofmonth <= 5 else 0
            
            # 4. is_first_3_days: Binary flag for maximum penalty period (days 1-3)
            if 'is_first_3_days' in new_row.columns:
                new_row['is_first_3_days'] = 1 if dayofmonth <= 3 else 0
            
            # 5. days_from_month_start: Gradient signal (0-based: 0 on 1st, 1 on 2nd, etc.)
            if 'days_from_month_start' in new_row.columns:
                new_row['days_from_month_start'] = dayofmonth - 1
            
            # 6. post_peak_signal: Exponential decay to break EOM momentum
            if 'post_peak_signal' in new_row.columns:
                lambda_decay = 0.15  # Same as preprocessing.py
                new_row['post_peak_signal'] = np.exp(-lambda_decay * (dayofmonth - 1))
            
            # 7. is_high_vol_weekday_AND_early_month: Explicit interaction feature
            # Suppresses weekday boost during early month (prevents "Logic Collision")
            if 'is_high_vol_weekday_AND_early_month' in new_row.columns:
                # First check if this is a high-volume weekday (Mon/Wed/Fri)
                is_high_vol_weekday = day_of_week in [0, 2, 4]  # Monday=0, Wednesday=2, Friday=4
                
                if is_high_vol_weekday and dayofmonth <= 5:
                    new_row['is_high_vol_weekday_AND_early_month'] = -2  # STRONG suppression (days 1-5)
                elif is_high_vol_weekday and dayofmonth <= 10:
                    new_row['is_high_vol_weekday_AND_early_month'] = -1  # Moderate suppression (days 6-10)
                else:
                    new_row['is_high_vol_weekday_AND_early_month'] = 0   # No suppression
            
            # 8. Update is_high_volume_weekday if present
            if 'is_high_volume_weekday' in new_row.columns:
                new_row['is_high_volume_weekday'] = 1 if day_of_week in [0, 2, 4] else 0
            
            # 9. Update Is_Monday if present
            if 'Is_Monday' in new_row.columns:
                new_row['Is_Monday'] = 1 if day_of_week == 0 else 0
            
            new_rows.append(new_row)
        
        # Append new rows to window
        if new_rows:
            new_rows_df = pd.concat(new_rows, ignore_index=True)
            window = pd.concat([window, new_rows_df], ignore_index=True)
            window = window.sort_values(time_col).reset_index(drop=True)
            
            # Update combined historical dataset with new predictions
            combined_historical = pd.concat([combined_historical, new_rows_df], ignore_index=True)
            combined_historical = combined_historical.drop_duplicates(subset=[time_col], keep='last')
            combined_historical = combined_historical.sort_values(time_col).reset_index(drop=True)
            
            # Recompute rolling features
            qty_values = window[target_col].values
            for i in range(len(window)):
                # Rolling mean 7d
                start_idx_7d = max(0, i - 6)
                rolling_mean_7d = np.mean(qty_values[start_idx_7d:i+1])
                
                # Rolling mean 30d
                start_idx_30d = max(0, i - 29)
                rolling_mean_30d = np.mean(qty_values[start_idx_30d:i+1])
                
                # Momentum: 3d vs 14d
                start_idx_3d = max(0, i - 2)
                start_idx_14d = max(0, i - 13)
                rolling_mean_3d = np.mean(qty_values[start_idx_3d:i+1]) if i >= 2 else np.mean(qty_values[:i+1])
                rolling_mean_14d = np.mean(qty_values[start_idx_14d:i+1]) if i >= 13 else np.mean(qty_values[:i+1])
                momentum_3d_vs_14d = rolling_mean_3d - rolling_mean_14d
                
                window.loc[i, 'rolling_mean_7d'] = rolling_mean_7d
                window.loc[i, 'rolling_mean_30d'] = rolling_mean_30d
                window.loc[i, 'momentum_3d_vs_14d'] = momentum_3d_vs_14d
            
            # ENHANCED: Recompute YoY features for MOONCAKE using Lunar-Aligned lookup
            # This is essential because YoY features are the primary signal for seasonal products
            if category == "MOONCAKE" and 'cbm_last_year' in feature_cols:
                print(f"  - Recomputing YoY features for MOONCAKE using ENHANCED Lunar-Aligned lookup...")
                # Use enhanced lunar_utils for precise Lunar-to-Lunar date mapping
                from src.utils.lunar_utils import get_lunar_aligned_yoy_lookup
                
                if len(combined_historical) > 0:
                    # For each row in window, compute lunar-aligned YoY features
                    window_dates = pd.to_datetime(window[time_col]).dt.date if not pd.api.types.is_datetime64_any_dtype(window[time_col]) else window[time_col].dt.date
                    
                    yoy_updated_count = 0
                    yoy_sample_values = []
                    for idx, win_date in enumerate(window_dates):
                        if isinstance(win_date, pd.Timestamp):
                            win_date = win_date.date()
                        
                        # Use enhanced lunar-aligned YoY lookup
                        cbm_last_year_val, cbm_2_years_ago_val = get_lunar_aligned_yoy_lookup(
                            current_date=win_date,
                            historical_data=combined_historical,
                            target_col=target_col,
                            time_col=time_col,
                            cat_col=cat_col,
                            category=category
                        )
                        
                        # Update window with YoY values
                        window.loc[idx, 'cbm_last_year'] = cbm_last_year_val
                        window.loc[idx, 'cbm_2_years_ago'] = cbm_2_years_ago_val
                        yoy_updated_count += 1
                        
                        # Collect sample values for debugging (first 5 non-zero values)
                        if cbm_last_year_val > 0 and len(yoy_sample_values) < 5:
                            # Also get the SOURCE dates that were matched
                            from src.utils.lunar_utils import find_lunar_aligned_date_from_previous_year, solar_to_lunar_date
                            source_date_1y = find_lunar_aligned_date_from_previous_year(
                                win_date, combined_historical, time_col, years_back=1
                            )
                            source_date_2y = find_lunar_aligned_date_from_previous_year(
                                win_date, combined_historical, time_col, years_back=2
                            )
                            lunar_current = solar_to_lunar_date(win_date)
                            yoy_sample_values.append((
                                win_date, lunar_current, 
                                source_date_1y, cbm_last_year_val,
                                source_date_2y, cbm_2_years_ago_val
                            ))
                    
                    print(f"  - Updated YoY features for {yoy_updated_count}/{len(window)} window rows")
                    if yoy_sample_values:
                        print(f"  - Sample YoY Lunar-Aligned Lookups (showing SOURCE dates matched):")
                        for sample_date, lunar_cur, src_1y, cbm_ly, src_2y, cbm_2y in yoy_sample_values[:3]:
                            lunar_str = f"Lunar {lunar_cur[0]:02d}-{lunar_cur[1]:02d}"
                            print(f"      {sample_date} ({lunar_str}) -> {src_1y}: {cbm_ly:.2f} CBM (1y ago)")
                            if src_2y:
                                print(f"                                          -> {src_2y}: {cbm_2y:.2f} CBM (2y ago)")
                    else:
                        print(f"  - WARNING: All YoY values are zero! This may cause underprediction.")
                else:
                    print(f"  - WARNING: No historical data available for YoY feature lookup")
            
            # ENHANCED: Maintain full-month window for next iteration
            # Instead of truncating to input_size, keep data from start of previous month to current date
            # This ensures consistent monthly context throughout the rolling prediction
            if use_full_month_window and len(window) > input_size:
                # Get the last date in the window
                last_window_date = pd.to_datetime(window[time_col]).dt.date.max()
                
                # Calculate start of previous month
                last_year = last_window_date.year
                last_month = last_window_date.month
                if last_month == 1:
                    prev_month_year = last_year - 1
                    prev_month = 12
                else:
                    prev_month_year = last_year
                    prev_month = last_month - 1
                
                from datetime import date as dt_date
                full_month_start = dt_date(prev_month_year, prev_month, 1)
                
                # Keep data from start of previous month onwards
                window['_date_only'] = pd.to_datetime(window[time_col]).dt.date
                window = window[window['_date_only'] >= full_month_start].copy()
                window = window.drop(columns=['_date_only'])
                window = window.sort_values(time_col).reset_index(drop=True)
                print(f"  - Maintaining full-month window: {len(window)} rows from {window[time_col].min()} to {window[time_col].max()}")
            else:
                # Standard mode: keep only the last input_size rows for next iteration
                window = window.tail(input_size).copy()
                window = window.sort_values(time_col).reset_index(drop=True)
        
        # ENHANCED: Move to the 1st day of the NEXT MONTH (not just +1 day)
        # This ensures each chunk aligns with calendar month boundaries
        chunk_month = current_start.month
        chunk_year = current_start.year
        
        if chunk_month == 12:
            next_month_year = chunk_year + 1
            next_month = 1
        else:
            next_month_year = chunk_year
            next_month = chunk_month + 1
        
        from datetime import date as dt_date
        current_start = dt_date(next_month_year, next_month, 1)
    
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
    num_days: int = 30,
    use_full_months: bool = True
):
    """
    Extract the last N days of historical data to initialize prediction window.
    
    ENHANCED: Supports full-month alignment mode for better trend capture.
    When use_full_months=True, returns data from complete months (last month + current month).
    
    Args:
        historical_data: DataFrame with historical data
        end_date: Last date to include
        config: Configuration object
        num_days: Number of days to extract (used when use_full_months=False)
        use_full_months: If True, use full month boundaries instead of fixed num_days
    
    Returns:
        DataFrame with historical data, sorted by time
    """
    time_col = config.data['time_col']
    
    historical_data = historical_data[
        pd.to_datetime(historical_data[time_col]).dt.date <= end_date
    ].copy()
    
    historical_data = historical_data.sort_values(time_col).reset_index(drop=True)
    
    historical_data['date_only'] = pd.to_datetime(historical_data[time_col]).dt.date
    
    if use_full_months:
        # Use complete months: last month + current month (up to end_date)
        # This captures monthly trends better than arbitrary N-day windows
        
        # Get the month boundaries
        end_year = end_date.year
        end_month = end_date.month
        
        # Calculate last month (previous month)
        if end_month == 1:
            last_month_year = end_year - 1
            last_month = 12
        else:
            last_month_year = end_year
            last_month = end_month - 1
        
        # Start from the first day of last month
        from datetime import date as dt_date
        start_date = dt_date(last_month_year, last_month, 1)
        
        print(f"  [FULL-MONTH MODE] Window: {start_date} to {end_date} (last month + current month)")
        
        # Select data from start_date to end_date
        window_data = historical_data[
            (historical_data['date_only'] >= start_date) &
            (historical_data['date_only'] <= end_date)
        ].copy()
    else:
        # Original behavior: use last N days
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
