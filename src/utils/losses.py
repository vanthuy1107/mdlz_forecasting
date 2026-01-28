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
    is_peak_loss_window: Optional[torch.Tensor] = None,
    peak_loss_window_weight: float = 1.0,
    is_august: Optional[torch.Tensor] = None,
    august_boost_weight: float = 1.0,
    use_smooth_l1: bool = False,
    smooth_l1_beta: float = 1.0
) -> torch.Tensor:
    """
    Spike-aware MSE loss that assigns higher weight to high values (spikes).
    
    This loss function assigns 3x weight to top 20% values (spikes) to better
    handle sudden demand surges. Optionally supports category-specific loss weights,
    Monday-specific weighting for FRESH category, and Golden Window weighting for
    MOONCAKE category (Gregorian August 1-31).
    
    Args:
        y_pred: Predicted values tensor.
        y_true: True values tensor.
        category_ids: Optional tensor of category IDs (same batch size as y_pred/y_true).
        category_loss_weights: Optional dict mapping category_id -> loss_weight multiplier.
        is_monday: Optional tensor indicating Monday samples (1 for Monday, 0 otherwise).
                  Shape should match y_true (can be 1D for single-step or 2D for multi-step).
        monday_loss_weight: Weight multiplier for Monday samples (default: 3.0).
        is_golden_window: Optional tensor indicating Golden Window samples (1 for Golden Window, 0 otherwise).
                         For MOONCAKE: Gregorian August 1-31. Shape should match y_true.
        golden_window_weight: Weight multiplier for Golden Window samples (default: 1.0, typically 12.0 for MOONCAKE).
        is_peak_loss_window: Optional tensor indicating Peak Loss Window samples (1 for Peak Loss Window, 0 otherwise).
                           For MOONCAKE: Lunar Months 7.15 to 8.15 (critical peak period). Shape should match y_true.
        peak_loss_window_weight: Weight multiplier for Peak Loss Window samples (default: 1.0, typically 20.0 for MOONCAKE).
        is_august: Optional tensor indicating August samples (1 for Gregorian month == 8, 0 otherwise).
                  For MOONCAKE: Reinforces August as strong signal anchor. Shape should match y_true.
        august_boost_weight: Weight multiplier for August samples (default: 1.0, typically 30.0 for MOONCAKE).
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
    
    # Apply Peak Loss Window weighting if provided (for MOONCAKE category - Lunar Months 7.15 to 8.15)
    # This is the critical peak period where errors must be heavily penalized
    if is_peak_loss_window is not None and peak_loss_window_weight > 1.0:
        # Ensure is_peak_loss_window matches y_true shape
        if is_peak_loss_window.ndim == 1 and y_true.ndim == 2:
            is_peak_loss_window = is_peak_loss_window.unsqueeze(-1).expand_as(y_true)
        elif is_peak_loss_window.ndim == 2 and y_true.ndim == 1:
            is_peak_loss_window = is_peak_loss_window[:, 0] if is_peak_loss_window.shape[1] > 0 else is_peak_loss_window.mean(dim=1)
        
        # Apply Peak Loss Window weight: multiplier for Peak Loss Window samples
        peak_weight = torch.where(
            is_peak_loss_window > 0.5,  # In Peak Loss Window (Lunar Months 7.15 to 8.15)
            torch.tensor(peak_loss_window_weight, device=y_true.device),
            torch.tensor(1.0, device=y_true.device)
        )
        base_weight = base_weight * peak_weight
    
    # Apply August boost weighting if provided (for MOONCAKE category - Gregorian August)
    # This reinforces August (Gregorian month 8) as a strong signal anchor
    if is_august is not None and august_boost_weight > 1.0:
        # Ensure is_august matches y_true shape
        if is_august.ndim == 1 and y_true.ndim == 2:
            is_august = is_august.unsqueeze(-1).expand_as(y_true)
        elif is_august.ndim == 2 and y_true.ndim == 1:
            is_august = is_august[:, 0] if is_august.shape[1] > 0 else is_august.mean(dim=1)
        
        # Apply August boost weight: multiplier for August samples (Gregorian month == 8)
        august_weight = torch.where(
            is_august > 0.5,  # In August (Gregorian month == 8)
            torch.tensor(august_boost_weight, device=y_true.device),
            torch.tensor(1.0, device=y_true.device)
        )
        base_weight = base_weight * august_weight
    
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


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for asymmetric regression with active season weighting.
    
    This loss function is designed to predict a specific quantile (e.g., 90th percentile)
    rather than the mean. It applies asymmetric penalties:
    - Higher penalty when actual value exceeds prediction (under-forecasting)
    - Lower penalty when actual value is below prediction (over-forecasting)
    
    For P90 (τ=0.9), the penalty ratio is 9:1 (0.9/0.1), meaning under-forecasting is
    penalized 9x more severely than over-forecasting.
    
    This is particularly useful for MOONCAKE category where the cost of under-forecasting
    (stock-outs) is much higher than the cost of slight over-forecasting (safety stock).
    
    Mathematical Formula:
    L_τ(y, ŷ) = {
        τ * (y - ŷ)    if y >= ŷ  (under-forecasting: higher penalty)
        (1-τ) * (ŷ - y)  if y < ŷ  (over-forecasting: lower penalty)
    }
    
    Where τ is the target quantile (e.g., 0.9 for 90th percentile).
    
    Active Season Weighting:
    When is_active_season is provided, the loss is multiplied by active_season_weight
    for samples in the active season. This ensures the model focuses on the entire
    active season (70+ days) rather than just the peak day.
    
    Peak Loss Window Weighting:
    When is_peak_loss_window is provided, the loss is multiplied by peak_loss_window_weight
    for samples in the peak loss window (Lunar Months 7.15 to 8.15). This ensures the model
    is heavily penalized for errors in the critical peak period.
    
    August Boost Weighting:
    When is_august is provided, the loss is multiplied by august_boost_weight for samples
    in August (Gregorian month == 8). This reinforces August as a strong signal anchor
    for MOONCAKE category, preventing phase shift errors.
    
    Args:
        quantile: Target quantile (default: 0.9 for P90). Must be between 0 and 1.
        reduction: Reduction method: 'mean', 'sum', or 'none' (default: 'mean').
        active_season_weight: Weight multiplier for active season samples (default: 1.0, no weighting).
        peak_loss_window_weight: Weight multiplier for peak loss window samples (default: 1.0, no weighting).
        august_boost_weight: Weight multiplier for August samples (default: 1.0, no weighting).
    """
    
    def __init__(self, quantile: float = 0.9, reduction: str = 'mean', active_season_weight: float = 1.0, peak_loss_window_weight: float = 1.0, august_boost_weight: float = 1.0):
        """
        Initialize Quantile Loss.
        
        Args:
            quantile: Target quantile (default: 0.9 for P90). Must be between 0 and 1.
            reduction: Reduction method: 'mean', 'sum', or 'none' (default: 'mean').
            active_season_weight: Weight multiplier for active season samples (default: 1.0).
            peak_loss_window_weight: Weight multiplier for peak loss window samples (default: 1.0).
        """
        super(QuantileLoss, self).__init__()
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")
        self.quantile = quantile
        self.reduction = reduction
        self.active_season_weight = active_season_weight
        self.peak_loss_window_weight = peak_loss_window_weight
        self.august_boost_weight = august_boost_weight
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        is_active_season: Optional[torch.Tensor] = None,
        is_peak_loss_window: Optional[torch.Tensor] = None,
        is_august: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute quantile loss with optional active season, peak loss window, and August boost weighting.
        
        Args:
            y_pred: Predicted values tensor (can be 1D or 2D).
            y_true: True values tensor (same shape as y_pred).
            category_ids: Optional category IDs (not used in quantile loss, kept for API compatibility).
            inputs: Optional input features (not used in quantile loss, kept for API compatibility).
            is_active_season: Optional tensor indicating active season samples (1 for active season, 0 otherwise).
                            Shape should match y_true (can be 1D for single-step or 2D for multi-step).
                            When provided, loss is multiplied by active_season_weight for active season samples.
            is_peak_loss_window: Optional tensor indicating peak loss window samples (1 for peak loss window, 0 otherwise).
                                Shape should match y_true (can be 1D for single-step or 2D for multi-step).
                                When provided, loss is multiplied by peak_loss_window_weight for peak loss window samples.
            is_august: Optional tensor indicating August samples (1 for Gregorian month == 8, 0 otherwise).
                      Shape should match y_true (can be 1D for single-step or 2D for multi-step).
                      When provided, loss is multiplied by august_boost_weight for August samples.
        
        Returns:
            Quantile loss value (scalar if reduction='mean' or 'sum', tensor otherwise).
        """
        # Ensure y_pred and y_true have the same shape
        if y_pred.shape != y_true.shape:
            # Try to broadcast if possible
            if y_pred.ndim == 1 and y_true.ndim == 2:
                y_pred = y_pred.unsqueeze(-1).expand_as(y_true)
            elif y_pred.ndim == 2 and y_true.ndim == 1:
                y_true = y_true.unsqueeze(-1).expand_as(y_pred)
            else:
                raise ValueError(
                    f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
                )
        
        # Compute error: positive when y_true > y_pred (under-forecasting)
        error = y_true - y_pred
        
        # Apply asymmetric penalty based on quantile
        # When error >= 0 (under-forecasting): penalty = quantile * error
        # When error < 0 (over-forecasting): penalty = (1 - quantile) * |error|
        # For P90 (τ=0.9): under-forecasting penalty = 0.9 * error, over-forecasting = 0.1 * |error|
        # Ratio: 0.9/0.1 = 9x more severe penalty for under-forecasting
        loss = torch.where(
            error >= 0,
            self.quantile * error,           # Under-forecasting: higher penalty (0.9 * error for P90)
            (1 - self.quantile) * (-error)   # Over-forecasting: lower penalty (0.1 * error for P90)
        )
        
        # Apply active season weighting if provided
        if is_active_season is not None and self.active_season_weight > 1.0:
            # Ensure is_active_season matches y_true shape
            if is_active_season.ndim == 1 and y_true.ndim == 2:
                is_active_season = is_active_season.unsqueeze(-1).expand_as(y_true)
            elif is_active_season.ndim == 2 and y_true.ndim == 1:
                is_active_season = is_active_season[:, 0] if is_active_season.shape[1] > 0 else is_active_season.mean(dim=1)
            
            # Create weight tensor: active_season_weight for active season, 1.0 otherwise
            season_weight = torch.where(
                is_active_season > 0.5,  # Active season indicator
                torch.tensor(self.active_season_weight, device=y_true.device),
                torch.tensor(1.0, device=y_true.device)
            )
            loss = loss * season_weight
        
        # Apply peak loss window weighting if provided (for MOONCAKE category - Lunar Months 7.15 to 8.15)
        # This is the critical peak period where errors must be heavily penalized
        if is_peak_loss_window is not None and self.peak_loss_window_weight > 1.0:
            # Ensure is_peak_loss_window matches y_true shape
            if is_peak_loss_window.ndim == 1 and y_true.ndim == 2:
                is_peak_loss_window = is_peak_loss_window.unsqueeze(-1).expand_as(y_true)
            elif is_peak_loss_window.ndim == 2 and y_true.ndim == 1:
                is_peak_loss_window = is_peak_loss_window[:, 0] if is_peak_loss_window.shape[1] > 0 else is_peak_loss_window.mean(dim=1)
            
            # Create weight tensor: peak_loss_window_weight for peak loss window, 1.0 otherwise
            peak_weight = torch.where(
                is_peak_loss_window > 0.5,  # In Peak Loss Window (Lunar Months 7.15 to 8.15)
                torch.tensor(self.peak_loss_window_weight, device=y_true.device),
                torch.tensor(1.0, device=y_true.device)
            )
            loss = loss * peak_weight
        
        # Apply August boost weighting if provided (for MOONCAKE category - Gregorian August)
        # This reinforces August (Gregorian month 8) as a strong signal anchor
        if is_august is not None and self.august_boost_weight > 1.0:
            # Ensure is_august matches y_true shape
            if is_august.ndim == 1 and y_true.ndim == 2:
                is_august = is_august.unsqueeze(-1).expand_as(y_true)
            elif is_august.ndim == 2 and y_true.ndim == 1:
                is_august = is_august[:, 0] if is_august.shape[1] > 0 else is_august.mean(dim=1)
            
            # Create weight tensor: august_boost_weight for August, 1.0 otherwise
            august_weight = torch.where(
                is_august > 0.5,  # In August (Gregorian month == 8)
                torch.tensor(self.august_boost_weight, device=y_true.device),
                torch.tensor(1.0, device=y_true.device)
            )
            loss = loss * august_weight
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}. Must be 'mean', 'sum', or 'none'.")


def quantile_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantile: float = 0.9,
    reduction: str = 'mean',
    category_ids: Optional[torch.Tensor] = None,
    inputs: Optional[torch.Tensor] = None,
    is_active_season: Optional[torch.Tensor] = None,
    active_season_weight: float = 1.0,
    is_peak_loss_window: Optional[torch.Tensor] = None,
    peak_loss_window_weight: float = 1.0,
    is_august: Optional[torch.Tensor] = None,
    august_boost_weight: float = 1.0
) -> torch.Tensor:
    """
    Quantile Loss (Pinball Loss) function for asymmetric regression.
    
    Convenience function that creates and calls QuantileLoss.
    This function signature matches the pattern used by other loss functions
    in this module for consistency with the training pipeline.
    
    Args:
        y_pred: Predicted values tensor.
        y_true: True values tensor.
        quantile: Target quantile (default: 0.9 for P90). Must be between 0 and 1.
        reduction: Reduction method: 'mean', 'sum', or 'none' (default: 'mean').
        category_ids: Optional category IDs (not used, kept for API compatibility).
        inputs: Optional input features (not used, kept for API compatibility).
        is_active_season: Optional tensor indicating active season samples (1 for active season, 0 otherwise).
        active_season_weight: Weight multiplier for active season samples (default: 1.0).
        is_peak_loss_window: Optional tensor indicating peak loss window samples (1 for peak loss window, 0 otherwise).
        peak_loss_window_weight: Weight multiplier for peak loss window samples (default: 1.0).
        is_august: Optional tensor indicating August samples (1 for Gregorian month == 8, 0 otherwise).
        august_boost_weight: Weight multiplier for August samples (default: 1.0).
    
    Returns:
        Quantile loss value.
    """
    loss_fn = QuantileLoss(quantile=quantile, reduction=reduction, active_season_weight=active_season_weight, peak_loss_window_weight=peak_loss_window_weight, august_boost_weight=august_boost_weight)
    return loss_fn(y_pred, y_true, category_ids=category_ids, inputs=inputs, is_active_season=is_active_season, is_peak_loss_window=is_peak_loss_window, is_august=is_august)


class AsymmetricMSELoss(nn.Module):
    """
    Asymmetric MSE Loss for FMCG Supply Chain Forecasting.
    
    In supply chain management, the cost of under-forecasting (stock-outs) is typically
    higher than the cost of over-forecasting (excess inventory). This loss function
    applies asymmetric penalties:
    
    - Under-forecast (y_true > y_pred): Higher penalty (α * MSE)
    - Over-forecast (y_true < y_pred): Lower penalty (β * MSE)
    
    Where α > β, typically α/β ≈ 2-5 for FMCG operations.
    
    Mathematical Formula:
    L(y, ŷ) = {
        α * (y - ŷ)²    if y > ŷ  (under-forecast: penalize heavily)
        β * (ŷ - y)²    if y ≤ ŷ  (over-forecast: penalize lightly)
    }
    
    This encourages the model to be slightly conservative (predict a bit higher)
    to avoid costly stock-outs during peak demand periods.
    
    Peak Season Boosting:
    During high-volatility months (TET/Mid-Autumn), we apply additional weight
    multipliers to both penalties to ensure accurate forecasts during critical periods.
    
    Args:
        under_penalty: Penalty weight for under-forecasting (α, default: 3.0).
        over_penalty: Penalty weight for over-forecasting (β, default: 1.0).
        reduction: Reduction method: 'mean', 'sum', or 'none' (default: 'mean').
        peak_season_boost: Additional multiplier for peak season periods (default: 1.0).
    
    Example:
        >>> loss_fn = AsymmetricMSELoss(under_penalty=3.0, over_penalty=1.0)
        >>> # If actual = 100, predicted = 80 (under-forecast by 20)
        >>> # Loss = 3.0 * (100 - 80)² = 3.0 * 400 = 1200
        >>> 
        >>> # If actual = 100, predicted = 120 (over-forecast by 20)
        >>> # Loss = 1.0 * (120 - 100)² = 1.0 * 400 = 400
        >>> # Ratio: 1200/400 = 3:1 (under-forecast penalty is 3x higher)
    """
    
    def __init__(
        self,
        under_penalty: float = 3.0,
        over_penalty: float = 1.0,
        reduction: str = 'mean',
        peak_season_boost: float = 1.0,
    ):
        """
        Initialize Asymmetric MSE Loss.
        
        Args:
            under_penalty: Weight for under-forecasting (default: 3.0).
            over_penalty: Weight for over-forecasting (default: 1.0).
            reduction: Reduction method: 'mean', 'sum', or 'none'.
            peak_season_boost: Multiplier for peak season periods.
        """
        super(AsymmetricMSELoss, self).__init__()
        self.under_penalty = under_penalty
        self.over_penalty = over_penalty
        self.reduction = reduction
        self.peak_season_boost = peak_season_boost
        
        if under_penalty < over_penalty:
            import warnings
            warnings.warn(
                f"under_penalty ({under_penalty}) < over_penalty ({over_penalty}). "
                f"For FMCG, under-forecasting should be penalized more heavily. "
                f"Consider setting under_penalty > over_penalty."
            )
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        is_peak_season: Optional[torch.Tensor] = None,
        category_ids: Optional[torch.Tensor] = None,
        category_loss_weights: Optional[Dict[int, float]] = None,
    ) -> torch.Tensor:
        """
        Compute asymmetric MSE loss with optional peak season boosting.
        
        Args:
            y_pred: Predicted values (batch, horizon) or (batch,).
            y_true: True values (same shape as y_pred).
            is_peak_season: Optional binary tensor indicating peak season (1) or not (0).
                           Shape should match y_true.
            category_ids: Optional category IDs for per-category loss weighting.
            category_loss_weights: Optional dict mapping category_id -> loss_weight.
        
        Returns:
            Asymmetric MSE loss value.
        """
        # Ensure same shape
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        
        # Calculate error
        error = y_true - y_pred
        squared_error = error ** 2
        
        # Apply asymmetric penalties
        # Under-forecast: error > 0 (actual > predicted) → higher penalty
        # Over-forecast: error <= 0 (actual <= predicted) → lower penalty
        asymmetric_loss = torch.where(
            error > 0,
            self.under_penalty * squared_error,  # Under-forecast penalty (α * MSE)
            self.over_penalty * squared_error,   # Over-forecast penalty (β * MSE)
        )
        
        # Apply peak season boosting if provided
        if is_peak_season is not None:
            # Ensure is_peak_season matches y_true shape
            if is_peak_season.ndim == 1 and y_true.ndim == 2:
                is_peak_season = is_peak_season.unsqueeze(-1).expand_as(y_true)
            elif is_peak_season.ndim == 2 and y_true.ndim == 1:
                is_peak_season = is_peak_season[:, 0] if is_peak_season.shape[1] > 0 else is_peak_season.mean(dim=1)
            
            # Apply peak season boost: multiply loss by peak_season_boost during peak periods
            peak_weight = torch.where(
                is_peak_season > 0.5,  # Peak season indicator
                torch.tensor(self.peak_season_boost, device=y_true.device),
                torch.tensor(1.0, device=y_true.device)
            )
            asymmetric_loss = asymmetric_loss * peak_weight
        
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
                    # For multi-step forecasting, expand mask to match y_true shape
                    if y_true.ndim == 2:
                        cat_mask_expanded = cat_mask.unsqueeze(-1).expand_as(y_true)
                    else:
                        cat_mask_expanded = cat_mask
                    category_weight[cat_mask_expanded] = loss_weight
            
            asymmetric_loss = asymmetric_loss * category_weight
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(asymmetric_loss)
        elif self.reduction == 'sum':
            return torch.sum(asymmetric_loss)
        elif self.reduction == 'none':
            return asymmetric_loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}. Must be 'mean', 'sum', or 'none'.")


def asymmetric_mse_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    under_penalty: float = 3.0,
    over_penalty: float = 1.0,
    reduction: str = 'mean',
    is_peak_season: Optional[torch.Tensor] = None,
    peak_season_boost: float = 1.0,
    category_ids: Optional[torch.Tensor] = None,
    category_loss_weights: Optional[Dict[int, float]] = None,
) -> torch.Tensor:
    """
    Asymmetric MSE Loss function for FMCG Supply Chain Forecasting.
    
    Convenience function that creates and calls AsymmetricMSELoss.
    
    In supply chain, the cost of under-forecasting (stock-outs) is typically
    higher than over-forecasting (excess inventory). This loss applies asymmetric
    penalties to encourage conservative forecasts.
    
    Args:
        y_pred: Predicted values.
        y_true: True values.
        under_penalty: Weight for under-forecasting (default: 3.0).
        over_penalty: Weight for over-forecasting (default: 1.0).
        reduction: Reduction method: 'mean', 'sum', or 'none'.
        is_peak_season: Optional binary tensor for peak season periods.
        peak_season_boost: Multiplier for peak season (default: 1.0).
        category_ids: Optional category IDs.
        category_loss_weights: Optional dict mapping category_id -> loss_weight.
    
    Returns:
        Asymmetric MSE loss value.
    
    Example:
        >>> # Standard usage (3:1 penalty ratio)
        >>> loss = asymmetric_mse_loss(pred, actual, under_penalty=3.0, over_penalty=1.0)
        >>> 
        >>> # High penalty for critical categories (5:1 ratio)
        >>> loss = asymmetric_mse_loss(pred, actual, under_penalty=5.0, over_penalty=1.0)
        >>> 
        >>> # With peak season boosting (TET/Mid-Autumn: 2x weight)
        >>> loss = asymmetric_mse_loss(
        ...     pred, actual, 
        ...     under_penalty=3.0, 
        ...     is_peak_season=peak_indicator,
        ...     peak_season_boost=2.0
        ... )
    """
    loss_fn = AsymmetricMSELoss(
        under_penalty=under_penalty,
        over_penalty=over_penalty,
        reduction=reduction,
        peak_season_boost=peak_season_boost,
    )
    return loss_fn(
        y_pred,
        y_true,
        is_peak_season=is_peak_season,
        category_ids=category_ids,
        category_loss_weights=category_loss_weights,
    )

