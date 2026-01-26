"""Prediction module for MDLZ forecasting system.

This module provides functions for loading models, preparing data, and making predictions.
All functions are designed to ensure consistent results regardless of which categories
are being processed together.
"""

from .loader import load_model_for_prediction
from .prepare import prepare_prediction_data
from .windows import create_prediction_windows
from .predictor import (
    predict_direct_multistep,
    predict_direct_multistep_rolling,
    get_historical_window_data
)

__all__ = [
    'load_model_for_prediction',
    'prepare_prediction_data',
    'create_prediction_windows',
    'predict_direct_multistep',
    'predict_direct_multistep_rolling',
    'get_historical_window_data',
]
