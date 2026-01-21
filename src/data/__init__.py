"""Data loading, preprocessing, and dataset utilities."""
from .loader import DataReader
from .dataset import ForecastDataset
from .preprocessing import (
    slicing_window,
    slicing_window_multivariate,
    slicing_window_category,
    encode_categories,
    split_data,
    prepare_data,
    add_holiday_features,
    add_temporal_features,
    aggregate_daily,
    add_cbm_density_features,
    fit_scaler,
    apply_scaling,
    inverse_transform_scaling
)

__all__ = [
    'DataReader',
    'ForecastDataset',
    'slicing_window',
    'slicing_window_multivariate',
    'slicing_window_category',
    'encode_categories',
    'split_data',
    'prepare_data',
    'add_holiday_features',
    'add_temporal_features',
    'aggregate_daily',
    'add_cbm_density_features',
    'fit_scaler',
    'apply_scaling',
    'inverse_transform_scaling'
]

