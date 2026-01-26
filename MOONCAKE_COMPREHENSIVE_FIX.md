# MOONCAKE Comprehensive Fix Plan

## Current Status (After YoY Fix)

Results are still poor:
- **July**: 0.00% accuracy (zeroed by masking)
- **August**: 7.63% accuracy (predicted 459.06 vs actual 6014.93)
- **September**: 8.96% accuracy (predicted 493.06 vs actual 4575.75)
- **October**: 0.00% accuracy (zeroed by masking)

## Remaining Issues

### Issue 1: Seasonal Masking Too Aggressive ⚠️⚠️

**Problem**: Only August-September (solar months 8-9) are allowed, causing July and October to be zeroed.

**Fix Required**: Update `get_is_active_season_mooncake()` to allow Lunar Months 6-9, which should correspond to Solar months 6-9 (June-September).

**Files to Fix**:
- `mvp_predict.py` lines 2638-2674
- `src/predict/predictor.py` lines 36-65

### Issue 2: Model Under-Prediction (Even with YoY Features) ⚠️⚠️⚠️

**Problem**: Even in allowed months (August-September), predictions are only ~8% of actual values.

**Possible Causes**:
1. **Model not trained properly** - May need retraining with correct features
2. **Scaling issues** - Target scaling may be compressing peak values
3. **Feature importance** - Model may not be using YoY features effectively
4. **Loss function** - May need stronger weighting for peak periods

**Investigation Needed**:
- Check if YoY features are actually non-zero during prediction (add logging)
- Verify model was trained with YoY features included
- Check model weights to see if it's using YoY features
- Review training logs for loss convergence

### Issue 3: Golden Window Hard Masking ⚠️

**Problem**: Additional masking zeros predictions outside Golden Window (Lunar Months 6.15-8.01).

**Fix Required**: Remove hard masking, use Golden Window only for loss weighting during training.

**File to Fix**: `src/predict/predictor.py` lines 271-274

## Action Plan

### Step 1: Verify YoY Features Are Working

Add diagnostic logging to verify YoY features are populated:

```python
# In src/predict/prepare.py, after YoY feature calculation
if current_category == "MOONCAKE":
    # Log sample values
    sample_dates = data.head(5)
    for _, row in sample_dates.iterrows():
        print(f"  Date: {row[time_col]}, cbm_last_year: {row.get('cbm_last_year', 0):.2f}, "
              f"cbm_2_years_ago: {row.get('cbm_2_years_ago', 0):.2f}")
```

### Step 2: Fix Seasonal Masking

Update to allow Lunar Months 6-9 (Solar months 6-9):

```python
# In mvp_predict.py and src/predict/predictor.py
def get_is_active_season_mooncake(row_date):
    # Allow Solar months 6-9 (June-September) instead of just 8-9
    solar_month = row_date.month
    is_active = (solar_month >= 6) and (solar_month <= 9)
    return 1 if is_active else 0
```

### Step 3: Remove Golden Window Hard Masking

Remove the hard zero logic in `src/predict/predictor.py`:

```python
# REMOVE THIS:
is_golden_window = _get_is_golden_window_mooncake(pred_date)
if is_golden_window == 0:
    pred_value = 0.0  # ❌ Remove this
```

### Step 4: Check Model Training

Verify the model was trained with:
- YoY features included in feature_cols
- Proper loss weights (loss_weight=20.0, golden_window_weight=10.0)
- Sufficient epochs (currently 20-40)

### Step 5: Retrain Model (If Needed)

If YoY features weren't included during training, or if model architecture needs adjustment:
1. Ensure YoY features are in feature_cols during training
2. Consider increasing model capacity (hidden_size, n_layers)
3. Increase training epochs
4. Verify loss weights are being applied

## Expected Improvements After Fixes

1. **July and October**: Should predict small but non-zero values (not zeroed)
2. **August-September**: Should predict much closer to actual (50-80% of actual, not 8%)
3. **Overall Accuracy**: Should improve from 8.88% to 40-60%+

## Priority Order

1. **HIGH**: Fix seasonal masking (allows July/October predictions)
2. **HIGH**: Remove Golden Window hard masking
3. **MEDIUM**: Verify YoY features are working (add logging)
4. **MEDIUM**: Check if model needs retraining
5. **LOW**: Fine-tune model architecture/hyperparameters

## Testing

After each fix:
1. Re-run predictions
2. Check monthly accuracy metrics
3. Verify predictions are non-zero for July/October
4. Check if August/September predictions improve
