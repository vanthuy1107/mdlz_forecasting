"""Data loading, preprocessing, and dataset utilities."""
from .loader import DataReader
from .dataset import ForecastDataset
from .scaler import RollingGroupScaler
from .utils import slicing_window
from .feature_engineer import FeatureEngineer, add_history_features, add_calendar_features 

__all__ = [
    'DataReader',
    'ForecastDataset',
    'FeatureEngineer',
    'slicing_window',
    'build_single_window',
    'add_baseline',
    'add_history_features',
    'add_calendar_features',
    'encode_brands',
    'add_features'
]

