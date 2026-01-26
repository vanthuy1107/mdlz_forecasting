# MOONCAKE Prediction Results Analysis

## Summary of Results

Based on the terminal output (lines 885-929), here are the MOONCAKE prediction results for 2025:

### Overall Metrics
- **Total Predictions**: 390 samples
- **MSE**: 5777.41
- **MAE**: 25.27
- **RMSE**: 76.01
- **Accuracy**: 8.88% (very poor)

### Monthly Breakdown

| Month | Actual Total | Predicted Total | Accuracy (Σ|err|) | Status |
|-------|--------------|-----------------|-------------------|--------|
| **2025-07** | 22.85 | **0.00** | 0.00% | ❌ **ZEROED** |
| **2025-08** | 6014.93 | 459.06 | 7.63% | ⚠️ **SEVERELY UNDER-PREDICTED** |
| **2025-09** | 4575.75 | 493.06 | 8.96% | ⚠️ **SEVERELY UNDER-PREDICTED** |
| **2025-10** | 112.19 | **0.00** | 0.00% | ❌ **ZEROED** |

## Critical Issues Identified

### Issue 1: Overly Aggressive Seasonal Masking ⚠️

**Location**: `mvp_predict.py` lines 2625-2690, `src/predict/predictor.py` lines 36-65

**Problem**: 
- The `get_is_active_season_mooncake()` function only allows **Solar months 8 and 9** (August and September) as active season
- This causes **July and October to be automatically zeroed out**, even though:
  - July 2025 had actual volume of 22.85 CBM
  - October 2025 had actual volume of 112.19 CBM

**Code Reference**:
```python
# mvp_predict.py:2668
solar_month = row_date.month
is_active = (solar_month == 8) or (solar_month == 9)  # Only Aug-Sep allowed
```

**Root Cause**: 
- The config specifies `seasonal_active_window: "lunar_months_6_9"` (Lunar Months 6-9)
- But the implementation uses a conservative solar month approximation that's too restrictive
- Lunar Months 6-9 should roughly correspond to Solar months 6-9 (June-September), not just 8-9

### Issue 2: Severe Under-Prediction During Peak Season ⚠️⚠️

**Problem**:
- Even during the allowed months (August-September), predictions are **only ~8% of actual values**
- August: Predicted 459.06 vs Actual 6014.93 (7.63% of actual)
- September: Predicted 493.06 vs Actual 4575.75 (10.78% of actual)

**Possible Root Causes**:
1. **Model Training Issues**:
   - Model may not have learned the peak patterns effectively
   - Loss weights may need adjustment (currently `loss_weight: 20.0`, `golden_window_weight: 10.0`)
   - Model architecture may be too small (`hidden_size: 128`, `n_layers: 2`)

2. **Feature Engineering Issues**:
   - Lunar calendar conversion may be inaccurate
   - Golden Window features may not be properly encoded
   - `days_to_mid_autumn` feature may not be working correctly

3. **Scaling Issues**:
   - Target scaling may be compressing peak values
   - Inverse scaling may not be recovering the full magnitude

### Issue 3: Golden Window Over-Masking ⚠️

**Location**: `src/predict/predictor.py` lines 271-274

**Problem**:
- Additional masking logic zeros out predictions outside the Golden Window (Lunar Months 6.15-8.01)
- This may be zeroing valid predictions that are in active season but outside Golden Window
- The Golden Window should be a **weighting factor**, not a hard mask

**Code Reference**:
```python
# predictor.py:271-274
is_golden_window = _get_is_golden_window_mooncake(pred_date)
if is_golden_window == 0:
    pred_value = 0.0  # ❌ This zeros valid active-season predictions
```

### Issue 4: Teacher Forcing Results Empty ⚠️

**Problem**:
- Warning: "One or more arrays are empty. Creating empty DataFrame for teacher forcing results."
- This suggests teacher forcing mode is not working, making it impossible to compare recursive vs teacher forcing performance

## Recommendations

### Immediate Fixes (High Priority)

1. **Fix Seasonal Masking Logic**:
   - Update `get_is_active_season_mooncake()` to use proper lunar calendar conversion
   - Allow Lunar Months 6-9, which should roughly correspond to Solar months 6-9 (June-September)
   - Consider using actual lunar calendar library instead of approximation

2. **Remove Golden Window Hard Masking**:
   - Golden Window should be used for **loss weighting during training**, not for hard masking during prediction
   - Remove the hard zero logic in `predictor.py` lines 271-274
   - Allow predictions in active season (Lunar Months 6-9) even if outside Golden Window

3. **Investigate Model Under-Prediction**:
   - Check if model was trained with sufficient data for peak periods
   - Verify loss weights are being applied correctly during training
   - Consider increasing model capacity or training epochs
   - Review feature engineering, especially lunar calendar features

### Medium Priority

4. **Fix Teacher Forcing Mode**:
   - Debug why teacher forcing arrays are empty
   - Ensure proper comparison between recursive and teacher forcing modes

5. **Improve Lunar Calendar Accuracy**:
   - Replace simplified `solar_to_lunar_date()` with proper lunar calendar library
   - Use accurate lunar-to-solar date conversions for 2025

6. **Add Diagnostic Logging**:
   - Log when predictions are zeroed by masking
   - Log model output values before and after scaling
   - Log feature values for peak prediction days

### Long-Term Improvements

7. **Model Retraining**:
   - Retrain with corrected seasonal masking
   - Increase loss weights for peak periods if needed
   - Consider ensemble methods or transfer learning from TET model

8. **Validation**:
   - Validate predictions against historical MOONCAKE patterns
   - Check if 2025 data aligns with expected seasonal patterns

## Code Locations to Review

1. **Seasonal Masking**: 
   - `mvp_predict.py:2638-2674` (get_is_active_season_mooncake)
   - `src/predict/predictor.py:36-65` (_get_is_active_season_mooncake)

2. **Golden Window Masking**:
   - `src/predict/predictor.py:271-274` (hard zero logic)

3. **Model Training**:
   - `config/config_MOONCAKE.yaml` (loss weights, model architecture)
   - `mvp_train.py` (training loop, loss calculation)

4. **Feature Engineering**:
   - `mvp_predict.py:1517-1565` (lunar features, days_to_mid_autumn)
   - `src/data/preprocessing.py` (lunar calendar conversion)

## Expected vs Actual Behavior

**Expected**:
- July: Low volume (22.85 CBM) - should predict small but non-zero
- August: Peak season (6014.93 CBM) - should predict close to actual
- September: Peak season (4575.75 CBM) - should predict close to actual  
- October: Tail end (112.19 CBM) - should predict small but non-zero

**Actual**:
- July: Zeroed by masking ❌
- August: Only 7.63% of actual ❌
- September: Only 8.96% of actual ❌
- October: Zeroed by masking ❌

## Next Steps

1. ✅ Review this analysis
2. Fix seasonal masking to allow Lunar Months 6-9 (Solar months 6-9)
3. Remove Golden Window hard masking
4. Investigate model under-prediction (check training logs, model weights)
5. Retrain model if necessary
6. Re-run predictions and compare results
