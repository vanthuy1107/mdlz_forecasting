"""Custom loss functions."""
import torch


def spike_aware_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Spike-aware MSE loss that assigns higher weight to high values (spikes).
    
    This loss function assigns 3x weight to top 20% values (spikes) to better
    handle sudden demand surges.
    
    Args:
        y_pred: Predicted values tensor.
        y_true: True values tensor.
    
    Returns:
        Weighted MSE loss.
    """
    threshold = torch.quantile(y_true, 0.8)  # top 20%
    weight = torch.where(
        y_true > threshold,
        torch.tensor(3.0, device=y_true.device),
        torch.tensor(1.0, device=y_true.device)
    )
    return torch.mean(weight * (y_pred - y_true)**2)

