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

