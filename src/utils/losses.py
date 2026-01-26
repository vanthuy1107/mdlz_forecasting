"""Custom loss functions."""
import torch
import torch.nn as nn
from typing import Optional, Dict


def spike_aware_mse(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    category_ids: Optional[torch.Tensor] = None,
    category_loss_weights: Optional[Dict[int, float]] = None,
    is_monday: Optional[torch.Tensor] = None,
    monday_loss_weight: float = 3.0,
    is_golden_window: Optional[torch.Tensor] = None,
    golden_window_weight: float = 1.0,
    use_smooth_l1: bool = False,
    smooth_l1_beta: float = 1.0
) -> torch.Tensor:
    """
    Spike-aware MSE loss that assigns higher weight to high values (spikes).
    
    This loss function assigns 3x weight to top 20% values (spikes) to better
    handle sudden demand surges. Optionally supports category-specific loss weights,
    Monday-specific weighting for FRESH category, and Golden Window weighting for
    MOONCAKE category (Lunar Months 6.15 to 8.01).
    
    Args:
        y_pred: Predicted values tensor.
        y_true: True values tensor.
        category_ids: Optional tensor of category IDs (same batch size as y_pred/y_true).
        category_loss_weights: Optional dict mapping category_id -> loss_weight multiplier.
        is_monday: Optional tensor indicating Monday samples (1 for Monday, 0 otherwise).
                  Shape should match y_true (can be 1D for single-step or 2D for multi-step).
        monday_loss_weight: Weight multiplier for Monday samples (default: 3.0).
        is_golden_window: Optional tensor indicating Golden Window samples (1 for Golden Window, 0 otherwise).
                         For MOONCAKE: Lunar Months 6.15 to 8.01. Shape should match y_true.
        golden_window_weight: Weight multiplier for Golden Window samples (default: 1.0, typically 3.0 for MOONCAKE).
        use_smooth_l1: If True, use SmoothL1Loss instead of MSE (default: False).
        smooth_l1_beta: Beta parameter for SmoothL1Loss (default: 1.0).
    
    Returns:
        Weighted loss (MSE or SmoothL1).
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
    
    # Apply Golden Window weighting if provided (for MOONCAKE category)
    # Peak-Phase Loss Amplification: Apply 10x weight when is_golden_window == 1
    # This helps the model be less "scared" to predict 400+ CBM peaks
    if is_golden_window is not None:
        # Ensure is_golden_window matches y_true shape
        if is_golden_window.ndim == 1 and y_true.ndim == 2:
            is_golden_window = is_golden_window.unsqueeze(-1).expand_as(y_true)
        elif is_golden_window.ndim == 2 and y_true.ndim == 1:
            is_golden_window = is_golden_window[:, 0] if is_golden_window.shape[1] > 0 else is_golden_window.mean(dim=1)
        
        # Apply Golden Window weight: multiplier for Golden Window samples
        # Apply 10x amplification for peak-phase samples (is_golden_window == 1)
        peak_phase_multiplier = 10.0  # 10x weight for peak-phase samples
        golden_weight = torch.where(
            is_golden_window > 0.5,  # In Golden Window
            torch.tensor(golden_window_weight * peak_phase_multiplier, device=y_true.device),
            torch.tensor(1.0, device=y_true.device)
        )
        base_weight = base_weight * golden_weight
    
    # Apply category-specific loss weights if provided
    # CRITICAL: For MOONCAKE, loss_weight should only apply to active days (is_active_season == 1)
    # Goal: "High Precision during the season, Zero outside the season"
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
                # Apply loss_weight only where category matches
                # Note: If is_active_season is available in inputs, it should be used to
                # further restrict loss_weight to active days only. This is handled
                # by the training pipeline which extracts is_active_season from inputs.
                category_weight[cat_mask_expanded] = loss_weight
        
        # Combine base spike weight with category weight
        weight = base_weight * category_weight
    else:
        weight = base_weight
    
    # Choose loss function: SmoothL1Loss or MSE
    if use_smooth_l1:
        # Use SmoothL1Loss (Huber loss variant)
        loss_fn = nn.SmoothL1Loss(reduction='none', beta=smooth_l1_beta)
        loss = loss_fn(y_pred, y_true)
    else:
        # Use MSE
        loss = (y_pred - y_true)**2
    
    return torch.mean(weight * loss)


def focal_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Focal Loss implementation for handling class imbalance and hard examples.
    
    Focal Loss is designed to address class imbalance by down-weighting easy examples
    and focusing training on hard examples. It's particularly useful for scenarios
    with many zero values and few high-value spikes (like MOONCAKE category).
    
    Formula: FL = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the probability of the true class.
    
    For regression, we adapt it as:
    FL = alpha * |error|^gamma * loss_base
    where loss_base is MSE or SmoothL1.
    
    Args:
        y_pred: Predicted values tensor.
        y_true: True values tensor.
        alpha: Weighting factor for the loss (default: 1.0).
        gamma: Focusing parameter. Higher gamma down-weights easy examples more (default: 2.0).
        reduction: Reduction method: 'mean', 'sum', or 'none' (default: 'mean').
    
    Returns:
        Focal loss value.
    """
    # Compute base loss (MSE)
    mse = (y_pred - y_true) ** 2
    
    # Compute error magnitude (normalized)
    error_magnitude = torch.abs(y_pred - y_true)
    # Normalize by max error in batch to get relative difficulty
    max_error = error_magnitude.max()
    if max_error > 0:
        normalized_error = error_magnitude / (max_error + 1e-8)
    else:
        normalized_error = error_magnitude
    
    # Focal weight: (1 - normalized_error)^gamma
    # Higher error -> lower weight (harder example gets more focus)
    # Actually, for regression, we want to focus on high-error examples
    # So we use: (normalized_error)^gamma to up-weight high errors
    focal_weight = (normalized_error + 1e-8) ** gamma
    
    # Apply alpha weighting
    focal_loss_value = alpha * focal_weight * mse
    
    if reduction == 'mean':
        return torch.mean(focal_loss_value)
    elif reduction == 'sum':
        return torch.sum(focal_loss_value)
    else:
        return focal_loss_value


def huber_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    delta: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Huber Loss (Smooth L1 Loss) implementation.
    
    Huber Loss is less sensitive to outliers than MSE and provides a smooth
    transition between L1 and L2 loss. It's more robust for handling spikes
    and outliers in time series data.
    
    Formula:
    L_delta = 0.5 * (y_pred - y_true)^2  if |error| <= delta
    L_delta = delta * |error| - 0.5 * delta^2  if |error| > delta
    
    Args:
        y_pred: Predicted values tensor.
        y_true: True values tensor.
        delta: Threshold parameter. Errors larger than delta use L1 loss (default: 1.0).
        reduction: Reduction method: 'mean', 'sum', or 'none' (default: 'mean').
    
    Returns:
        Huber loss value.
    """
    error = y_pred - y_true
    abs_error = torch.abs(error)
    
    # Quadratic term for small errors
    quadratic = torch.clamp(abs_error, max=delta)
    # Linear term for large errors
    linear = abs_error - quadratic
    
    loss = 0.5 * quadratic ** 2 + delta * linear
    
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss

