# DRY Early Month Over-Prediction Fix

## Problem Statement

The DRY category model was **over-predicting by 2-3K CBM in early month** (days 1-5):
- **Predictions**: Started at ~7.1K CBM on day 1
- **Actuals**: Only ~3.6-4.2K CBM on days 1-5
- **Error**: ~2-3K CBM over-prediction (~60-70% error rate)

This occurred because:
1. The model was trained on historical data where early month patterns existed but weren't emphasized
2. The existing `early_month_low_tier` and `is_early_month_low` features were too simple (binary 0/1)
3. No loss weighting was applied to early month days, so the model didn't prioritize getting them right

## Solutions Implemented

### 1. **Added Early Month Loss Weighting** ✅

**Config Change** (`config/config_DRY.yaml`):
```yaml
category_specific_params:
  DRY:
    early_month_loss_weight: 5.0  # 5x weight on early month days (days 1-10)
```

This forces the model to pay **5x more attention** to early month predictions during training, similar to how FRESH uses Monday/Wednesday/Friday weighting (8x/6x/6x).

**Files Modified:**
- `config/config_DRY.yaml`: Added `early_month_loss_weight: 5.0`
- `src/utils/losses.py`: Added `is_early_month` and `early_month_loss_weight` parameters to `spike_aware_mse()`
- `mvp_train.py`: 
  - Added early_month_loss_weight extraction from config
  - Added is_early_month_low feature index detection
  - Added early month horizon detection logic
  - Passed early_month parameters to loss function

### 2. **Enhanced Feature Engineering** ✅

**Added `days_from_month_start` Feature**:
```yaml
feature_cols:
  - "early_month_low_tier"       # CRITICAL: Days 1-10 low volume signal
  - "is_early_month_low"         # CRITICAL: Binary early month indicator
  - "days_from_month_start"      # CRITICAL: Gradient signal (0, 1, 2, 3... 30)
```

The `days_from_month_start` feature provides a **continuous gradient** (0 on day 1, 1 on day 2, ..., 30 on day 31) that helps the model learn the progressive increase in volume as the month progresses.

**Improved `early_month_low_tier` Granularity**:

Changed from 2 tiers to 3 tiers in `src/data/preprocessing.py`:

**Before:**
- Tier 0: Days 1-10 (low volume)
- Tier 1: Days 11-31 (normal)

**After:**
- Tier 0: Days 1-5 (very low volume - critical pattern)
- Tier 1: Days 6-10 (transitioning low volume)
- Tier 2: Days 11-31 (normal volume)

This provides more granularity to capture the **severe drop in days 1-5** vs the recovery in days 6-10.

### 3. **Loss Function Enhancement** ✅

Updated `spike_aware_mse()` in `src/utils/losses.py`:

```python
def spike_aware_mse(
    ...
    is_early_month: Optional[torch.Tensor] = None,
    early_month_loss_weight: float = 1.0,
    ...
) -> torch.Tensor:
```

The function now applies early month weighting similar to weekday weighting:

```python
# Apply Early Month-specific weighting if provided (for DRY category)
if is_early_month is not None and early_month_loss_weight > 1.0:
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
```

## How It Works

### Training Time
1. During each training batch, the model extracts `is_early_month_low` feature from inputs
2. For each horizon step in the forecast, it identifies if that day falls in the early month period
3. The loss function applies **5x weight multiplier** to early month predictions:
   - If prediction is for days 1-10 → **5.0x loss weight**
   - If prediction is for days 11-31 → **1.0x loss weight** (baseline)

### Feature Interaction
The model now has three complementary signals:
1. **`early_month_low_tier`** (0/1/2): Categorical indicator of volume tier
2. **`is_early_month_low`** (0/1): Binary flag for early month
3. **`days_from_month_start`** (0-30): Continuous gradient showing position in month

These work together to give the model a clear understanding of:
- **Where** we are in the month (categorical + binary)
- **How far** we are from month start (continuous)
- **How much to care** about getting it right (loss weighting)

## Expected Impact

After retraining with these changes:

- **Days 1-5 predictions** should decrease from ~7.1K → closer to ~4.0K actual
- **Days 6-10 predictions** should show gradual recovery toward normal levels
- **Days 11-31 predictions** should remain stable (already accurate)

The 5x loss weight ensures the model will **prioritize accuracy on early month days** during training, forcing it to learn the distinctive low-volume pattern.

## Files Modified

1. **`config/config_DRY.yaml`**: 
   - Added `early_month_loss_weight: 5.0`
   - Added `days_from_month_start` to feature_cols
   - Added comments marking early month features as CRITICAL

2. **`src/data/preprocessing.py`**:
   - Enhanced `add_early_month_low_volume_features()` to use 3 tiers (0/1/2) instead of 2 tiers (0/1)

3. **`src/utils/losses.py`**:
   - Added `is_early_month` and `early_month_loss_weight` parameters
   - Added early month weighting logic after Friday weighting

4. **`mvp_train.py`**:
   - Added early_month_loss_weight config extraction (~line 1697)
   - Added is_early_month_low feature index detection (~line 1747)
   - Added early_month_loss_weight to create_criterion signature (~line 1765)
   - Added early month horizon extraction logic (~line 1884)
   - Added is_early_month parameter to spike_aware_mse call (~line 1963)
   - Updated create_criterion call to include early_month parameters (~line 1976)

## Next Steps

1. **Retrain the DRY model** with these changes:
   ```bash
   python mvp_train.py
   ```

2. **Verify the fix** by checking:
   - Training logs should show: "Early Month loss weight: 5.0x (days 1-10, for DRY)"
   - Predictions on days 1-5 should be closer to actuals (3.6-4.2K range)
   - Overall MAPE/MAE should improve for DRY category

3. **Fine-tune if needed**:
   - If still over-predicting: Increase `early_month_loss_weight` to 8.0 or 10.0
   - If under-predicting: Decrease `early_month_loss_weight` to 3.0 or 4.0
   - If days 1-5 vs days 6-10 need different treatment: Can add separate weights

## Why This Works Better Than Alternatives

### Alternative 1: Remove Early Month Features
❌ **Not recommended**: Removing the features would leave the model blind to early month patterns entirely. The model needs to know when it's early month.

### Alternative 2: Add More Training Data
❌ **Insufficient**: More data alone won't fix the issue if the model doesn't prioritize early month accuracy.

### Alternative 3: Adjust Feature Scaling
❌ **Indirect**: Feature scaling changes affect all predictions, not just early month. Loss weighting is more targeted.

### Our Approach: Loss Weighting + Enhanced Features ✅
- **Targeted**: Only affects early month predictions
- **Flexible**: Easy to tune the weight (5x, 8x, 10x, etc.)
- **Proven**: Same approach successfully fixed FRESH Monday/Wednesday/Friday under-prediction
- **Interpretable**: Clear signal to the model: "care more about these days"

---

**Date**: 2026-02-07
**Category**: DRY
**Issue**: Early month over-prediction (days 1-5: predicted 7.1K vs actual 3.6-4.2K)
**Solution**: 5x loss weighting + 3-tier feature + days_from_month_start gradient
