"""Window creation utilities for prediction.

This module handles creating prediction windows from prepared data.
"""
import pandas as pd
import numpy as np

from src.data import slicing_window_category


def create_prediction_windows(data: pd.DataFrame, config):
    """
    Create sliding windows for prediction and extract corresponding dates.
    
    Args:
        data: Prepared DataFrame with all features and category IDs
        config: Configuration object
    
    Returns:
        Tuple of (X_pred, y_pred, cat_pred, dates) where:
        - X_pred: Input sequences (n_samples, input_size, n_features)
        - y_pred: Target values (n_samples, horizon) - may be NaN for future dates
        - cat_pred: Category IDs for each window (n_samples,)
        - dates: List of prediction dates corresponding to each window
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
