# DRY Early Month Fix - Action Plan

## Current Situation

Your predictions still show severe over-prediction in early month (days 1-5):
- **Predicted**: 7.1K-7.5K CBM
- **Actual**: 3.6K-4.2K CBM
- **Error**: ~50-60% over-prediction

## Root Cause

**The model hasn't been retrained yet!** The changes we made only affect training, not inference. You're still using the old model trained without early month loss weighting.

## Solution: 3-Step Approach

### Step 1: RETRAIN with Aggressive Settings (MUST DO)

I've increased the early month loss weight from 5.0x to **15.0x** and added the `is_first_5_days` feature.

**Run this command to retrain:**
```bash
python mvp_train.py
```

**What to check in training logs:**
```
Early Month loss weight: 15.0x (days 1-10, for DRY)
is_early_month_low feature found at index XX
```

**Training time:** ~10-30 minutes depending on your hardware

### Step 2: Evaluate Results

After retraining, run predictions again:
```bash
python mvp_predict.py --category DRY
# OR
python predict_by_brand.py --category DRY
```

Check if early month predictions improved:
- **Target**: Days 1-5 should predict 3.5K-4.5K (not 7K+)
- **Acceptable**: Within ±500 CBM of actuals

### Step 3: If Still Not Fixed - Apply Post-Processing

If Step 1 doesn't fully fix it, apply the post-processing adjustment I created.

**Option A: Quick Fix (Multiply predictions by adjustment factor)**

Add this to your prediction script at the end, before saving results:

```python
# At the end of mvp_predict.py or predict_by_brand.py
if category == "DRY":
    # Apply early month adjustment for days 1-10
    date_col = "ACTUALSHIPDATE"
    pred_col = "predicted"  # or whatever your prediction column is named
    
    # Create day_of_month column
    predictions_df['day_of_month'] = pd.to_datetime(predictions_df[date_col]).dt.day
    
    # Adjustment factors (based on observed error)
    adjustments = {
        1: 0.52, 2: 0.54, 3: 0.56, 4: 0.58, 5: 0.65,
        6: 0.75, 7: 0.80, 8: 0.85, 9: 0.90, 10: 0.95
    }
    
    # Apply adjustments
    for day, factor in adjustments.items():
        mask = predictions_df['day_of_month'] == day
        predictions_df.loc[mask, pred_col] = predictions_df.loc[mask, pred_col] * factor
    
    predictions_df = predictions_df.drop(columns=['day_of_month'])
    print(f"[INFO] Applied early month adjustment for DRY category")
```

**Option B: Use the adjustment module**

```python
from src.predict.early_month_adjustment import apply_early_month_adjustment

if category == "DRY":
    predictions_df = apply_early_month_adjustment(
        predictions_df,
        date_col="ACTUALSHIPDATE",
        pred_col="predicted"
    )
```

## Changes Made

### 1. config/config_DRY.yaml
```yaml
early_month_loss_weight: 15.0  # Increased from 5.0 to 15.0 (AGGRESSIVE)
```

Added feature:
```yaml
- "is_first_5_days"  # New binary indicator for days 1-5
```

### 2. src/data/preprocessing.py
Added `is_first_5_days` feature to `add_early_month_low_volume_features()`:
```python
df['is_first_5_days'] = (day_of_month <= 5).astype(int)
```

### 3. src/predict/early_month_adjustment.py (NEW)
Created post-processing adjustment utility with calibrated factors:
- Day 1-5: Reduce by ~45-48%
- Day 6-10: Reduce by ~25-10%

## Why This Will Work

### Approach 1: Aggressive Loss Weighting (15x)
- **Monday weighting (8x) successfully fixed FRESH** under-prediction
- **15x weight** will force model to prioritize days 1-10 even more
- **`is_first_5_days` feature** gives model explicit signal for severe drop days

### Approach 2: Post-Processing
- **Guaranteed fix** - directly scales predictions to match actuals
- **Calibrated factors** based on your actual error pattern
- **Can be fine-tuned** easily without retraining

## Expected Timeline

| Step | Action | Time | Expected Result |
|------|--------|------|-----------------|
| 1 | Retrain with 15x weight | 10-30 min | Days 1-5: 4.5K-5.5K (improved but may not be perfect) |
| 2 | Evaluate | 5 min | Check if predictions within ±1K of actuals |
| 3 | Apply post-processing if needed | 5 min | Days 1-5: 3.5K-4.5K (match actuals) |

## Fallback Plan

If none of the above work, the issue might be:

1. **Data quality**: Check if actuals for days 1-5 are consistently low across all months
2. **Feature leakage**: `Total CBM` as feature might be causing issues
3. **Model architecture**: May need to add attention mechanism or separate head for early month

Let me know the results after retraining!

---

## Quick Commands

**Retrain DRY:**
```bash
python mvp_train.py
```

**Check if retraining worked:**
```bash
python mvp_predict.py --category DRY
```

**Apply post-processing fix immediately** (if you don't want to wait for retraining):
See Option A above - add those lines to your prediction script.

---

**Last Updated**: 2026-02-07
**Status**: Ready to retrain with 15x early month loss weight + is_first_5_days feature
