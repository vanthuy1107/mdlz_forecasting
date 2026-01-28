"""PyTorch Dataset classes."""
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Union
import numpy as np


class ForecastDataset(Dataset):
    """Dataset for time series forecasting with optional category information."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cat: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize ForecastDataset.
        
        Args:
            X: Input features of shape (N_samples, T, D_features).
            y: Target values of shape (N_samples,) or (N_samples, horizon).
            cat: Optional category IDs of shape (N_samples,).
            transform: Optional transform function to apply to X.
        """
        self.X = X
        self.y = y
        self.cat = cat
        self.transform = transform
        
        # Validate shapes
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")
        
        if cat is not None and len(cat) != len(X):
            raise ValueError(f"cat must have same length as X, got {len(cat)} and {len(X)}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Union[tuple, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index.
        
        Returns:
            If cat is provided: (X, cat, y) tuple of tensors.
            Otherwise: (X, y) tuple of tensors.
        """
        X = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            X = self.transform(X)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Handle y shape: if 2D with single dimension, squeeze it
        if y.ndim > 0 and y.shape[0] == 1:
            y = y.squeeze(-1)
        
        if self.cat is not None:
            cat = torch.tensor(self.cat[idx], dtype=torch.long)
            # Ensure cat is scalar or 1D
            if cat.ndim > 0:
                cat = cat.squeeze()
            return X, cat, y
        else:
            return X, y


class HybridForecastDataset(Dataset):
    """Dataset for Hybrid LSTM with separate temporal and static features.
    
    This dataset class supports the dual-branch HybridLSTM architecture by
    separating features into:
    - Temporal features: Sequential historical data (goes to LSTM branch)
    - Static features: Metadata and domain knowledge (goes to Dense branch)
    
    Example:
        Temporal features (LSTM branch):
        - Historical CBM values (sequence)
        - Rolling averages
        - Momentum indicators
        
        Static features (Dense branch):
        - Brand ID (embedded)
        - Day-of-week statistics (e.g., "average Monday volume")
        - Holiday proximity (days until Tet)
        - Growth momentum (MoM/YoY)
        - Cyclical encodings (sin/cos of day, month)
    """
    
    def __init__(
        self,
        X_temporal: np.ndarray,
        X_static: np.ndarray,
        y: np.ndarray,
        cat: Optional[np.ndarray] = None,
        transform_temporal: Optional[Callable] = None,
        transform_static: Optional[Callable] = None,
    ):
        """
        Initialize HybridForecastDataset.
        
        Args:
            X_temporal: Temporal sequence features of shape (N, T, D_temporal).
                       These are fed to the LSTM branch.
            X_static: Static/meta features of shape (N, D_static).
                     These are fed to the Dense branch.
            y: Target values of shape (N,) or (N, horizon).
            cat: Optional category IDs of shape (N,).
            transform_temporal: Optional transform for temporal features.
            transform_static: Optional transform for static features.
        """
        self.X_temporal = X_temporal
        self.X_static = X_static
        self.y = y
        self.cat = cat
        self.transform_temporal = transform_temporal
        self.transform_static = transform_static
        
        # Validate shapes
        if len(X_temporal) != len(X_static):
            raise ValueError(
                f"X_temporal and X_static must have same length, "
                f"got {len(X_temporal)} and {len(X_static)}"
            )
        
        if len(X_temporal) != len(y):
            raise ValueError(
                f"X_temporal and y must have same length, "
                f"got {len(X_temporal)} and {len(y)}"
            )
        
        if cat is not None and len(cat) != len(X_temporal):
            raise ValueError(
                f"cat must have same length as X_temporal, "
                f"got {len(cat)} and {len(X_temporal)}"
            )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.X_temporal)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample.
        
        Args:
            idx: Sample index.
        
        Returns:
            (X_temporal, X_static, cat, y) tuple of tensors if cat is provided.
            (X_temporal, X_static, y) tuple of tensors if cat is not provided.
        """
        X_temporal = self.X_temporal[idx]
        X_static = self.X_static[idx]
        y = self.y[idx]
        
        # Apply transforms if provided
        if self.transform_temporal:
            X_temporal = self.transform_temporal(X_temporal)
        
        if self.transform_static:
            X_static = self.transform_static(X_static)
        
        # Convert to tensors
        X_temporal = torch.tensor(X_temporal, dtype=torch.float32)
        X_static = torch.tensor(X_static, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Handle y shape: if 2D with single dimension, squeeze it
        if y.ndim > 0 and y.shape[0] == 1:
            y = y.squeeze(-1)
        
        if self.cat is not None:
            cat = torch.tensor(self.cat[idx], dtype=torch.long)
            # Ensure cat is scalar or 1D
            if cat.ndim > 0:
                cat = cat.squeeze()
            return X_temporal, X_static, cat, y
        else:
            return X_temporal, X_static, y

