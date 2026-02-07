"""Custom loss functions."""
import torch
import torch.nn as nn
from typing import Optional, Dict


def get_dynamic_early_month_weight(day_of_month: torch.Tensor, base_weight: float = 100.0) -> torch.Tensor:
    """
    Compute dynamic early month loss weight based on day of month.
    
    SOLUTION 2: Dynamic Loss Weighting - HARD RESET STRATEGY V2 (CONFIGURABLE)
    This creates an EXTREME scheduled penalty with EXTENDED hard zone and EXPONENTIAL decay
    to fix residual over-prediction in Days 2-4 post-holiday period.
    
    Weight Schedule (configurable base_weight, default 100x):
    - Days 1-5: MAXIMUM PENALTY (base_weight) - Extended zero-tolerance zone for entire first week
    - Days 6-10: EXPONENTIAL decay from base_weight down to 1x
    - Days 11+: Standard weight (1x)
    
    Mathematical Formula:
    - If day <= 5: weight = base_weight (default 100.0, configurable to 50.0 for balance with weekday patterns)
    - If 6 <= day <= 10: weight = base_weight * exp(-lambda * (day - 5))
      * Lambda calculated dynamically: ln(base_weight) / 5 so that Day 10 reaches 1.0x
      * Day 6: base_weight * exp(-lambda * 1) (maintains high penalty)
      * Day 10: base_weight * exp(-lambda * 5) = 1.0x (smooth landing)
    - If day > 10: weight = 1.0
    
    Args:
        day_of_month: Tensor of day of month values (1-31).
                     Shape can be (batch,) or (batch, horizon).
        base_weight: Base weight for Days 1-5 (default: 100.0).
                    Can be reduced (e.g., 50.0) to allow weekday patterns to show through.
    
    Returns:
        Weight tensor with same shape as input, containing dynamic weights.
    """
    # Initialize all weights to 1.0 (default)
    weights = torch.ones_like(day_of_month, dtype=torch.float32, device=day_of_month.device)
    
    # Days 1-5: MAXIMUM PENALTY (configurable base_weight)
    # This keeps the model in "severe low volume" mode for the entire first 5 days
    mask_days_1_5 = (day_of_month >= 1) & (day_of_month <= 5)
    weights = torch.where(mask_days_1_5, torch.tensor(base_weight, device=day_of_month.device), weights)
    
    # Days 6-10: EXPONENTIAL decay from base_weight to 1
    # Formula: weight = base_weight * exp(-lambda * (day - 5))
    # Lambda dynamically calculated: ln(base_weight) / 5 so exp(-lambda * 5) = 1/base_weight
    # This gives: base_weight * (1/base_weight) = 1.0 at Day 10
    mask_days_6_10 = (day_of_month >= 6) & (day_of_month <= 10)
    lambda_decay = torch.log(torch.tensor(base_weight, device=day_of_month.device)) / 5.0
    exponential_decay = base_weight * torch.exp(-lambda_decay * (day_of_month - 5))
    weights = torch.where(mask_days_6_10, exponential_decay, weights)
    
    # Days 11+: Standard weight (1x) - already initialized to 1.0
    
    return weights


def spike_aware_mse(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    category_ids: Optional[torch.Tensor] = None,
    category_loss_weights: Optional[Dict[int, float]] = None,
    is_monday: Optional[torch.Tensor] = None,
    monday_loss_weight: float = 3.0,
    is_wednesday: Optional[torch.Tensor] = None,
    wednesday_loss_weight: float = 1.0,
    is_friday: Optional[torch.Tensor] = None,
    friday_loss_weight: float = 1.0,
    is_early_month: Optional[torch.Tensor] = None,
    early_month_loss_weight: float = 1.0,
    day_of_month: Optional[torch.Tensor] = None,
    use_dynamic_early_month_weight: bool = False,
    dynamic_early_month_base_weight: float = 100.0,
    is_golden_window: Optional[torch.Tensor] = None,
    golden_window_weight: float = 1.0,
    is_peak_loss_window: Optional[torch.Tensor] = None,
    peak_loss_window_weight: float = 1.0,
    is_august: Optional[torch.Tensor] = None,
    august_boost_weight: float = 1.0,
    use_smooth_l1: bool = False,
    smooth_l1_beta: float = 1.0,
    use_asymmetric_penalty: bool = False,
    over_pred_penalty: float = 2.0,
    under_pred_penalty: float = 1.0,
    apply_mean_error_constraint: bool = False,
    mean_error_weight: float = 0.1
) -> torch.Tensor:
    """
    Spike-aware MSE loss with Balanced Distribution focus and optional asymmetric penalties.
    
    This loss function can operate in two modes:
    
    1. PEAK-DEFENSE MODE (default, use_asymmetric_penalty=False):
       - Assigns 3x weight to top 20% values (spikes) to avoid under-prediction
       - Works well for preventing stock-outs but can cause upward bias
    
    2. BALANCED DISTRIBUTION MODE (use_asymmetric_penalty=True):
       - Asymmetric penalties during non-peak periods to eliminate upward bias
       - Over-predictions penalized more heavily than under-predictions in early month
       - Mean error constraint ensures monthly totals align with historical averages
       - Spike protection still active but balanced with downward correction
    
    Args:
        y_pred: Predicted values tensor.
        y_true: True values tensor.
        category_ids: Optional tensor of category IDs (same batch size as y_pred/y_true).
        category_loss_weights: Optional dict mapping category_id -> loss_weight multiplier.
        is_monday: Optional tensor indicating Monday samples (1 for Monday, 0 otherwise).
                  Shape should match y_true (can be 1D for single-step or 2D for multi-step).
        monday_loss_weight: Weight multiplier for Monday samples (default: 3.0).
        is_wednesday: Optional tensor indicating Wednesday samples (1 for Wednesday, 0 otherwise).
                     Shape should match y_true (can be 1D for single-step or 2D for multi-step).
        wednesday_loss_weight: Weight multiplier for Wednesday samples (default: 1.0).
        is_friday: Optional tensor indicating Friday samples (1 for Friday, 0 otherwise).
                  Shape should match y_true (can be 1D for single-step or 2D for multi-step).
        friday_loss_weight: Weight multiplier for Friday samples (default: 1.0).
        is_early_month: Optional tensor indicating early month samples (1 for days 1-10, 0 otherwise).
                       Shape should match y_true (can be 1D for single-step or 2D for multi-step).
                       Used only when use_dynamic_early_month_weight=False (static weighting).
        early_month_loss_weight: Weight multiplier for early month samples (default: 1.0, typically 5.0-15.0 for DRY).
                                Used only when use_dynamic_early_month_weight=False (static weighting).
        day_of_month: Optional tensor of day of month values (1-31) for dynamic early month weighting.
                     Shape should match y_true (can be 1D for single-step or 2D for multi-step).
                     Used only when use_dynamic_early_month_weight=True.
        use_dynamic_early_month_weight: If True, use dynamic weight schedule (Days 1-5: configurable, Days 6-10: exponential decay).
                                       If False, use static early_month_loss_weight for all days 1-10 (default: False).
        dynamic_early_month_base_weight: Base weight for Days 1-5 when use_dynamic_early_month_weight=True (default: 100.0).
                                        Can be reduced (e.g., 50.0) to allow weekday patterns (Wed/Fri) to show through.
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
        use_asymmetric_penalty: If True, apply asymmetric penalties (over-predictions penalized more in non-peak periods).
                               This eliminates upward bias while maintaining spike protection (default: False).
        over_pred_penalty: Penalty multiplier for over-predictions when use_asymmetric_penalty=True (default: 2.0).
                          Higher values more aggressively penalize over-prediction. Typical range: 1.5-3.0.
        under_pred_penalty: Penalty multiplier for under-predictions when use_asymmetric_penalty=True (default: 1.0).
                           Usually kept at 1.0 for balanced approach.
        apply_mean_error_constraint: If True, add mean error constraint to force monthly predictions to align
                                    with historical averages (default: False). This 'pulls down' inflated predictions.
        mean_error_weight: Weight for mean error constraint loss component (default: 0.1).
                          Higher values enforce stricter alignment with historical monthly averages.
    
    Returns:
        Weighted loss (MSE or SmoothL1) with optional asymmetric penalties and mean error constraint.
    """
    threshold = torch.quantile(y_true, 0.8)  # top 20%
    base_weight = torch.where(
        y_true > threshold,
        torch.tensor(3.0, device=y_true.device),
        torch.tensor(1.0, device=y_true.device)
    )
    
    # Apply Monday-specific weighting if provided (for FRESH category)
    if is_monday is not None and monday_loss_weight > 1.0:
        # Ensure is_monday matches y_true shape
        if is_monday.ndim == 1 and y_true.ndim == 2:
            # Expand 1D is_monday to 2D to match (batch, horizon) shape
            is_monday = is_monday.unsqueeze(-1).expand_as(y_true)
        elif is_monday.ndim == 2 and y_true.ndim == 1:
            # If y_true is 1D but is_monday is 2D, take first column or mean
            is_monday = is_monday[:, 0] if is_monday.shape[1] > 0 else is_monday.mean(dim=1)
        
        # Apply Monday weight for Monday samples (Is_Monday == 1)
        monday_weight = torch.where(
            is_monday > 0.5,  # Is_Monday == 1
            torch.tensor(monday_loss_weight, device=y_true.device),
            torch.tensor(1.0, device=y_true.device)
        )
        base_weight = base_weight * monday_weight
    
    # Apply Wednesday-specific weighting if provided (for FRESH category)
    if is_wednesday is not None and wednesday_loss_weight > 1.0:
        # Ensure is_wednesday matches y_true shape
        if is_wednesday.ndim == 1 and y_true.ndim == 2:
            is_wednesday = is_wednesday.unsqueeze(-1).expand_as(y_true)
        elif is_wednesday.ndim == 2 and y_true.ndim == 1:
            is_wednesday = is_wednesday[:, 0] if is_wednesday.shape[1] > 0 else is_wednesday.mean(dim=1)
        
        # Apply Wednesday weight for Wednesday samples
        wednesday_weight = torch.where(
            is_wednesday > 0.5,
            torch.tensor(wednesday_loss_weight, device=y_true.device),
            torch.tensor(1.0, device=y_true.device)
        )
        base_weight = base_weight * wednesday_weight
    
    # Apply Friday-specific weighting if provided (for FRESH category)
    if is_friday is not None and friday_loss_weight > 1.0:
        # Ensure is_friday matches y_true shape
        if is_friday.ndim == 1 and y_true.ndim == 2:
            is_friday = is_friday.unsqueeze(-1).expand_as(y_true)
        elif is_friday.ndim == 2 and y_true.ndim == 1:
            is_friday = is_friday[:, 0] if is_friday.shape[1] > 0 else is_friday.mean(dim=1)
        
        # Apply Friday weight for Friday samples
        friday_weight = torch.where(
            is_friday > 0.5,
            torch.tensor(friday_loss_weight, device=y_true.device),
            torch.tensor(1.0, device=y_true.device)
        )
        base_weight = base_weight * friday_weight
    
    # Apply Early Month-specific weighting if provided (for DRY category)
    # This helps fix over-prediction in early month (days 1-10)
    # Two modes: Static (legacy) or Dynamic (SOLUTION 2: HARD RESET STRATEGY V2 - Exponential Decay)
    if use_dynamic_early_month_weight and day_of_month is not None:
        # SOLUTION 2: Dynamic Loss Weighting with HARD RESET STRATEGY V2
        # Days 1-5: 50x, Days 6-10: exponential decay from 50x to 1x, Days 11+: 1x
        # This extended hard zone + exponential decay fixes the "post-holiday bump" issue
        # Ensure day_of_month matches y_true shape
        if day_of_month.ndim == 1 and y_true.ndim == 2:
            day_of_month = day_of_month.unsqueeze(-1).expand_as(y_true)
        elif day_of_month.ndim == 2 and y_true.ndim == 1:
            day_of_month = day_of_month[:, 0] if day_of_month.shape[1] > 0 else day_of_month.mean(dim=1)
        
        # Compute dynamic weight schedule
        dynamic_weight = get_dynamic_early_month_weight(day_of_month, base_weight=dynamic_early_month_base_weight)
        base_weight = base_weight * dynamic_weight
    elif is_early_month is not None and early_month_loss_weight > 1.0:
        # Legacy mode: Static weighting for all days 1-10
        # Ensure is_early_month matches y_true shape
        if is_early_month.ndim == 1 and y_true.ndim == 2:
            is_early_month = is_early_month.unsqueeze(-1).expand_as(y_true)
        elif is_early_month.ndim == 2 and y_true.ndim == 1:
            is_early_month = is_early_month[:, 0] if is_early_month.shape[1] > 0 else is_early_month.mean(dim=1)
        
        # Apply Early Month weight for early month samples (days 1-10)
        early_month_weight = torch.where(
            is_early_month > 0.5,
            torch.tensor(early_month_loss_weight, device=y_true.device),
            torch.tensor(1.0, device=y_true.device)
        )
        base_weight = base_weight * early_month_weight
    
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
    
    # Apply Asymmetric Penalty for Balanced Distribution (optional)
    # This addresses upward bias by penalizing over-predictions more heavily in non-peak periods
    # The asymmetric weight is incorporated into the main weight tensor
    if use_asymmetric_penalty:
        # Compute error: positive when over-predicting (y_pred > y_true)
        error = y_pred - y_true
        
        # Determine if we're in a peak period (where we want to maintain spike protection)
        # Peak periods: high volume spikes (top 20%), Golden Window, Peak Loss Window, August
        is_peak_period = (y_true > threshold)  # Spike detection (same as base_weight)
        
        # Add Golden Window to peak period mask if available
        if is_golden_window is not None:
            if is_golden_window.ndim == 1 and y_true.ndim == 2:
                is_golden_window_expanded = is_golden_window.unsqueeze(-1).expand_as(y_true)
            else:
                is_golden_window_expanded = is_golden_window
            is_peak_period = is_peak_period | (is_golden_window_expanded > 0.5)
        
        # Add Peak Loss Window to peak period mask if available
        if is_peak_loss_window is not None:
            if is_peak_loss_window.ndim == 1 and y_true.ndim == 2:
                is_peak_loss_window_expanded = is_peak_loss_window.unsqueeze(-1).expand_as(y_true)
            else:
                is_peak_loss_window_expanded = is_peak_loss_window
            is_peak_period = is_peak_period | (is_peak_loss_window_expanded > 0.5)
        
        # Add August to peak period mask if available (for MOONCAKE)
        if is_august is not None:
            if is_august.ndim == 1 and y_true.ndim == 2:
                is_august_expanded = is_august.unsqueeze(-1).expand_as(y_true)
            else:
                is_august_expanded = is_august
            is_peak_period = is_peak_period | (is_august_expanded > 0.5)
        
        # Apply asymmetric penalties ONLY in non-peak periods
        # In peak periods, we maintain the original spike-aware behavior (weight=1.0)
        asymmetric_weight = torch.ones_like(error, device=error.device)
        
        # Non-peak periods: Apply asymmetric penalty
        non_peak_mask = ~is_peak_period
        over_pred_mask = (error > 0) & non_peak_mask  # Over-predicting in non-peak period
        under_pred_mask = (error <= 0) & non_peak_mask  # Under-predicting in non-peak period
        
        # Over-predictions get higher penalty (default 2.0x) to eliminate upward bias
        asymmetric_weight = torch.where(
            over_pred_mask,
            torch.tensor(over_pred_penalty, device=error.device),
            asymmetric_weight
        )
        
        # Under-predictions get standard penalty (default 1.0x)
        asymmetric_weight = torch.where(
            under_pred_mask,
            torch.tensor(under_pred_penalty, device=error.device),
            asymmetric_weight
        )
        
        # Incorporate asymmetric weight into main weight tensor
        # This ensures asymmetric penalty works correctly with spike weights and other temporal weights
        weight = weight * asymmetric_weight
    
    # Apply existing category/temporal weights
    weighted_loss = weight * loss
    
    # Apply Mean Error Constraint (optional)
    # This forces the model to ensure predicted monthly totals align with historical averages
    # Effectively 'pulls down' inflated predictions across the entire month
    if apply_mean_error_constraint and mean_error_weight > 0:
        # Compute mean error (bias) across the batch
        # Positive mean error = systematic over-prediction (upward bias)
        # Negative mean error = systematic under-prediction
        mean_error = torch.mean(y_pred - y_true)
        
        # Penalize positive mean error (over-prediction bias) more heavily
        # This constraint ensures the model doesn't systematically over-predict
        # We use squared mean error to penalize large biases more heavily
        mean_error_loss = mean_error ** 2
        
        # Weight the mean error constraint
        # Typical values: 0.05-0.2 (higher = stricter alignment with historical averages)
        mean_error_loss = mean_error_weight * mean_error_loss
        
        # Combine main loss with mean error constraint
        # Main loss ensures per-sample accuracy
        # Mean error constraint ensures no systematic bias
        return torch.mean(weighted_loss) + mean_error_loss
    
    return torch.mean(weighted_loss)


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

