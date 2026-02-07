# Early Month Over-Prediction Solution

## Problem Statement

The DRY category model suffers from **"Early Month Over-prediction"**:

- **Root Cause**: End-of-month (EOM) volume spikes leave the LSTM's hidden state with "memory" of high volumes
- **Symptom**: Model predicts high values for days 1-10 of new month, but actual volume drops significantly (post-peak slump)
- **Pattern**: EOM spike → Model remembers → Early month over-prediction

## Implemented Solutions

This document describes two complementary solutions that address the issue from different angles:

1. **Solution 1: Post-Peak Decay Feature** (Structural/Input Fix)
2. **Solution 2: Dynamic Loss Weighting** (Training Signal Fix)

---

## Solution 1: Post-Peak Decay Feature

### Concept

Provide an **explicit signal** to the model that indicates how far we've moved away from the last EOM peak. This "decay" feature helps the LSTM Forget Gate learn to suppress high-volume hidden states carried over from the previous month.

### Mathematical Formula

```
V = exp(-λ × t)
```

Where:
- `V` = post-peak signal value
- `λ` = decay rate (0.3)
- `t` = day of month (0-indexed: 0 on 1st, 1 on 2nd, etc.)

### Feature Values

| Day of Month | Value | Interpretation |
|--------------|-------|----------------|
| Day 1 | 1.0 | Maximum risk of over-prediction |
| Day 2 | 0.74 | High risk |
| Day 5 | 0.30 | Medium risk |
| Day 10 | 0.06 | Low risk |
| Day 20 | 0.002 | Negligible |

### Why It Works

- **Non-linear decay**: Unlike linear "day of month" features, exponential decay captures the rapid drop-off in EOM influence
- **LSTM gate signal**: The Forget Gate can learn to use this feature to actively suppress previous month's high-volume hidden states
- **Structural fix**: Addresses the root cause by giving the model awareness of temporal distance from the peak event

### Implementation

**File**: `src/data/preprocessing.py`

Added to `add_early_month_low_volume_features()`:

```python
# SOLUTION 1: Post-Peak Decay Feature
lambda_decay = 0.3  # Decay rate
df['post_peak_signal'] = np.exp(-lambda_decay * df['days_from_month_start'])
```

**File**: `config/config_DRY.yaml`

Added to `feature_cols`:

```yaml
- "post_peak_signal"  # SOLUTION 1: Post-Peak Decay - breaks EOM momentum (exp decay)
```

---

## Solution 2: Dynamic Loss Weighting

### Concept

Replace the static loss multiplier (15x for all days 1-10) with a **gradient-based penalty schedule** that:

- Applies **maximum penalty** (20x) on days 1-3
- **Linear decay** from 20x to 1x for days 4-10
- Standard weight (1x) for days 11+

This eliminates the "cliff effect" and forces the optimizer to prioritize fixing early month errors.

### Weight Schedule

| Day Range | Weight Formula | Example Values |
|-----------|----------------|----------------|
| Days 1-3 | 20.0 (fixed) | Day 1: 20x, Day 2: 20x, Day 3: 20x |
| Days 4-10 | 20.0 - (day - 1) × (19.0/9.0) | Day 4: 13.67x, Day 7: 7.22x, Day 10: 1.0x |
| Days 11+ | 1.0 (standard) | Day 15: 1x, Day 25: 1x |

### Why It Works

1. **Eliminates cliff effect**: Smooth transition prevents abrupt penalty changes
2. **Backpropagation priority**: Errors on days 1-3 are treated as "catastrophic failures" (20x amplification)
3. **Gradient descent optimization**: Optimizer naturally prioritizes fixing the highest-weighted errors first
4. **Smooth convergence**: Linear decay ensures stable training dynamics

### Mathematical Formula

```python
def get_dynamic_early_month_weight(day_of_month):
    if day <= 3:
        return 20.0
    elif 4 <= day <= 10:
        return 20.0 - (day - 1) * (19.0 / 9.0)
    else:
        return 1.0
```

### Implementation

**File**: `src/utils/losses.py`

Added new function:

```python
def get_dynamic_early_month_weight(day_of_month: torch.Tensor) -> torch.Tensor:
    """
    Compute dynamic early month loss weight based on day of month.
    
    Weight Schedule:
    - Days 1-3: Maximum penalty (20x)
    - Days 4-10: Linear decay from 20x to 1x
    - Days 11+: Standard weight (1x)
    """
    weights = torch.ones_like(day_of_month, dtype=torch.float32)
    
    # Days 1-3: Maximum penalty
    mask_days_1_3 = (day_of_month >= 1) & (day_of_month <= 3)
    weights = torch.where(mask_days_1_3, torch.tensor(20.0), weights)
    
    # Days 4-10: Linear decay
    mask_days_4_10 = (day_of_month >= 4) & (day_of_month <= 10)
    linear_decay = 20.0 - (day_of_month - 1) * (19.0 / 9.0)
    weights = torch.where(mask_days_4_10, linear_decay, weights)
    
    return weights
```

Updated `spike_aware_mse()` signature:

```python
def spike_aware_mse(
    ...
    day_of_month: Optional[torch.Tensor] = None,
    use_dynamic_early_month_weight: bool = False,
    ...
):
```

Updated early month weighting logic:

```python
if use_dynamic_early_month_weight and day_of_month is not None:
    # SOLUTION 2: Dynamic Loss Weighting
    dynamic_weight = get_dynamic_early_month_weight(day_of_month)
    base_weight = base_weight * dynamic_weight
elif is_early_month is not None and early_month_loss_weight > 1.0:
    # Legacy mode: Static weighting
    ...
```

**File**: `mvp_train.py`

Added config extraction:

```python
use_dynamic_early_month_weight = False  # Default
if 'use_dynamic_early_month_weight' in cat_params:
    use_dynamic_early_month_weight = bool(cat_params['use_dynamic_early_month_weight'])
    if use_dynamic_early_month_weight:
        print(f"  - SOLUTION 2: Dynamic Early Month Weighting ENABLED")
```

Added feature extraction:

```python
# Extract day_of_month for dynamic early month weighting
if use_dyn_early_month and inputs is not None:
    if days_from_month_start_idx is not None:
        # Convert 0-indexed days_from_month_start to 1-indexed day_of_month
        last_day_from_start = inputs[:, -1, days_from_month_start_idx]
        day_of_month_last = last_day_from_start + 1
        # Project forward for multi-step horizon
        day_of_month_horizon = day_of_month_last + torch.arange(1, horizon + 1)
```

**File**: `config/config_DRY.yaml`

Updated config:

```yaml
category_specific_params:
  DRY:
    # SOLUTION 2: Dynamic Early Month Loss Weighting
    use_dynamic_early_month_weight: true  # Enable gradient-based penalty schedule
    # Legacy static mode (commented out):
    # early_month_loss_weight: 15.0
```

**File**: `src/data/preprocessing.py`

Added `is_first_3_days` feature for maximum penalty period:

```python
# Create binary indicator for FIRST 3 DAYS (1st-3rd) - MAXIMUM PENALTY period
df['is_first_3_days'] = (day_of_month <= 3).astype(int)
```

Added to config:

```yaml
- "is_first_3_days"  # CRITICAL: Days 1-3 maximum penalty period
```

---

## How Both Solutions Work Together

### Complementary Mechanisms

1. **Solution 1 (Post-Peak Decay)**: Provides the model with **structural awareness** of temporal distance from EOM
   - Input-level fix: Model sees the decay signal in every training example
   - Enables LSTM gates to learn proper forgetting behavior
   
2. **Solution 2 (Dynamic Loss Weighting)**: Forces the model to **prioritize early month accuracy** during training
   - Training-level fix: Optimizer treats early month errors as critical
   - Ensures the model learns to use features like post_peak_signal effectively

### Synergy

- **Feature + Penalty**: The model learns that when `post_peak_signal` is high (early month), predictions must be accurate because the loss penalty is severe
- **Gradient flow**: High loss weights on days 1-3 → Large gradients → Model learns to attend to `post_peak_signal` → Better early month predictions
- **Robust learning**: Even if one solution is partially effective, the other provides reinforcement

---

## Configuration

### Enable Both Solutions

**File**: `config/config_DRY.yaml`

```yaml
# Feature columns (SOLUTION 1: Post-Peak Decay)
data:
  feature_cols:
    - "post_peak_signal"           # SOLUTION 1: Exponential decay from EOM
    - "is_first_3_days"            # For dynamic weighting
    - "days_from_month_start"      # For dynamic weighting
    # ... other features ...

# Loss weighting (SOLUTION 2: Dynamic Loss Weighting)
category_specific_params:
  DRY:
    use_dynamic_early_month_weight: true  # SOLUTION 2: Gradient-based penalty
```

### Fallback to Static Weighting (Legacy)

If you want to use the old static weighting approach:

```yaml
category_specific_params:
  DRY:
    use_dynamic_early_month_weight: false  # Disable dynamic weighting
    early_month_loss_weight: 15.0         # Use static 15x weight for days 1-10
```

---

## Expected Impact

### Before Implementation

- Days 1-10: Consistent over-prediction (actual volume = 50 CBM, predicted = 150 CBM)
- Model "remembers" EOM spike (e.g., Day 31: 300 CBM → Day 1: still expects high volume)
- Static 15x penalty on all days 1-10 (cliff effect on day 11)

### After Implementation

- Days 1-3: Aggressive correction with 20x penalty + decay signal → Model learns to suppress EOM memory
- Days 4-10: Gradual penalty reduction + decay signal → Smooth transition to normal volume
- Days 11+: Standard weighting → No unintended side effects
- Overall: Early month predictions align with actual post-peak slump

### Key Metrics to Monitor

1. **Early Month MAPE**: Should decrease significantly (target: <20% for days 1-5)
2. **Day 1-3 predictions**: Should drop to match actual low volumes
3. **Training loss**: Higher early in training (due to aggressive penalties), but converges to better final loss
4. **Overall accuracy**: Should remain stable or improve (days 11+ unaffected)

---

## Testing & Validation

### Train the Model

```bash
# Train with DRY config (both solutions enabled)
python mvp_train.py --config config/config_DRY.yaml --category DRY
```

### Verify Features

Check that training logs show:

```
- days_from_month_start feature found at index X (for dynamic early month weighting)
- SOLUTION 2: Dynamic Early Month Weighting ENABLED (Days 1-3: 20x, Days 4-10: linear decay, for DRY)
```

### Evaluate Predictions

After training, check predictions for early month periods:

```python
# Example: Evaluate early month predictions
df_pred = pd.read_csv("predictions_DRY.csv")
df_early = df_pred[df_pred['day_of_month'] <= 10]

# Compute MAPE for early month
mape_early = ((df_early['predicted'] - df_early['actual']).abs() / df_early['actual']).mean() * 100
print(f"Early Month MAPE: {mape_early:.2f}%")
```

---

## Troubleshooting

### Issue: Model still over-predicts on days 1-3

**Diagnosis**: Dynamic penalty may not be strong enough

**Solution**: Increase maximum penalty in `get_dynamic_early_month_weight()`:

```python
# Change from 20.0 to 30.0 or 40.0
mask_days_1_3 = (day_of_month >= 1) & (day_of_month <= 3)
weights = torch.where(mask_days_1_3, torch.tensor(30.0), weights)  # Increased to 30x
```

### Issue: Model under-predicts mid-to-late month

**Diagnosis**: Penalty decay may be too aggressive

**Solution**: Adjust decay schedule to be less steep:

```python
# Extend penalty period to day 15
elif 4 <= day <= 15:
    return 20.0 - (day - 1) * (19.0 / 14.0)  # Slower decay over 15 days
```

### Issue: Training loss is unstable

**Diagnosis**: High penalties causing gradient explosions

**Solution**: 
1. Reduce learning rate in config:
   ```yaml
   learning_rate: 0.0003  # Reduced from 0.0005
   ```
2. Enable gradient clipping in training code

---

## Files Modified

| File | Changes |
|------|---------|
| `src/data/preprocessing.py` | Added `post_peak_signal` feature, added `is_first_3_days` feature |
| `src/utils/losses.py` | Added `get_dynamic_early_month_weight()`, updated `spike_aware_mse()` |
| `mvp_train.py` | Added dynamic weighting logic, feature extraction for `day_of_month` |
| `config/config_DRY.yaml` | Added `post_peak_signal` to features, enabled `use_dynamic_early_month_weight` |

---

## References

### Solution 1: Post-Peak Decay

- **Concept**: Exponential decay feature to signal temporal distance from EOM peak
- **Formula**: `V = exp(-0.3 × t)`
- **Mechanism**: Allows LSTM Forget Gate to learn to suppress previous month's hidden states

### Solution 2: Dynamic Loss Weighting

- **Concept**: Gradient-based penalty schedule with smooth decay
- **Schedule**: Days 1-3: 20x → Days 4-10: linear decay to 1x → Days 11+: 1x
- **Mechanism**: Forces optimizer to prioritize early month accuracy during training

---

## Next Steps

1. **Train the model** with both solutions enabled
2. **Evaluate early month predictions** (days 1-10)
3. **Compare MAPE** before and after implementation
4. **Fine-tune parameters** if needed (decay rate λ, maximum penalty weight)
5. **Monitor training stability** (loss convergence, gradient norms)

---

## Contact

For questions or issues with this implementation, refer to:

- Implementation files in `src/` directory
- Configuration in `config/config_DRY.yaml`
- This documentation: `EARLY_MONTH_OVERPREDICTION_SOLUTION.md`
