"""Custom loss functions."""
import torch
from typing import Optional, Dict


def spike_aware_mse(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    category_ids: Optional[torch.Tensor] = None,
    category_loss_weights: Optional[Dict[int, float]] = None,
    is_monday: Optional[torch.Tensor] = None,
    monday_loss_weight: float = 3.0
) -> torch.Tensor:
    """
    Spike-aware MSE loss that assigns higher weight to high values (spikes).
    
    This loss function assigns 3x weight to top 20% values (spikes) to better
    handle sudden demand surges. Optionally supports category-specific loss weights
    and Monday-specific weighting for FRESH category.
    
    Args:
        y_pred: Predicted values tensor.
        y_true: True values tensor.
        category_ids: Optional tensor of category IDs (same batch size as y_pred/y_true).
        category_loss_weights: Optional dict mapping category_id -> loss_weight multiplier.
        is_monday: Optional tensor indicating Monday samples (1 for Monday, 0 otherwise).
                  Shape should match y_true (can be 1D for single-step or 2D for multi-step).
        monday_loss_weight: Weight multiplier for Monday samples (default: 3.0).
    
    Returns:
        Weighted MSE loss.
    """
    threshold = torch.quantile(y_true, 0.8)  # top 20%
    base_weight = torch.where(
        y_true > threshold,
        torch.tensor(3.0, device=y_true.device),
        torch.tensor(1.0, device=y_true.device)
    )
    
    # Apply Monday-specific weighting if provided
    if is_monday is not None:
        # Ensure is_monday matches y_true shape
        if is_monday.ndim == 1 and y_true.ndim == 2:
            # Expand 1D is_monday to 2D to match (batch, horizon) shape
            is_monday = is_monday.unsqueeze(-1).expand_as(y_true)
        elif is_monday.ndim == 2 and y_true.ndim == 1:
            # If y_true is 1D but is_monday is 2D, take first column or mean
            is_monday = is_monday[:, 0] if is_monday.shape[1] > 0 else is_monday.mean(dim=1)
        
        # Apply Monday weight: 3.0x for Monday samples (Is_Monday == 1)
        monday_weight = torch.where(
            is_monday > 0.5,  # Is_Monday == 1
            torch.tensor(monday_loss_weight, device=y_true.device),
            torch.tensor(1.0, device=y_true.device)
        )
        base_weight = base_weight * monday_weight
    
    # Apply category-specific loss weights if provided
    if category_ids is not None and category_loss_weights is not None:
        # Ensure category_ids is 1D
        if category_ids.dim() > 1:
            category_ids = category_ids.squeeze(-1)
        category_ids = category_ids.long()
        
        # Create weight tensor matching y_true shape
        category_weight = torch.ones_like(y_true, device=y_true.device)
        
        # Apply category-specific weights
        for cat_id, loss_weight in category_loss_weights.items():
            cat_mask = category_ids == cat_id
            if cat_mask.any():
                # For multi-step forecasting, y_true/y_pred may be 2D (batch, horizon)
                # We need to expand the mask to match
                if y_true.ndim == 2:
                    cat_mask_expanded = cat_mask.unsqueeze(-1).expand_as(y_true)
                else:
                    cat_mask_expanded = cat_mask
                category_weight[cat_mask_expanded] = loss_weight
        
        # Combine base spike weight with category weight
        weight = base_weight * category_weight
    else:
        weight = base_weight
    
    return torch.mean(weight * (y_pred - y_true)**2)

