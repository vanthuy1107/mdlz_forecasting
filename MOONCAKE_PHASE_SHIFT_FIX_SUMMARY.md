# MOONCAKE Phase-Shift and Magnitude Collapse Fix Summary

**Date**: 2025-01-27  
**Issue**: Model predicting peaks 30 days too early (July instead of August/September 2025)  
**Root Cause**: Gregorian Bias - Model learned from 2023/2024 peaks (earlier in year) and applied to 2025

## Critical Fixes Implemented

### 1. ✅ Fixed Lunar Date Conversion
**Problem**: Simplified `solar_to_lunar_date()` function was inaccurate, causing incorrect lunar date mapping.

**Solution**: 
- Updated all `solar_to_lunar_date()` functions in:
  - `mvp_train.py`
  - `mvp_predict.py`
  - `src/predict/predictor.py`
  - `src/predict/prepare.py`
- Uses Mid-Autumn Festival dates as anchor points:
  - 2023: Sep 29 = Lunar 08-15
  - 2024: Sep 17 = Lunar 08-15
  - 2025: Oct 6 = Lunar 08-15 ✓
  - 2026: Sep 25 = Lunar 08-15

**Verification**:
```
2025-08-15: Lunar (6, 23) ✓
2025-09-15: Lunar (7, 24) ✓
2025-10-06: Lunar (8, 15) ✓ (Mid-Autumn Festival)
```

### 2. ✅ Fixed YoY Features to Use Lunar Date Matching
**Problem**: `cbm_last_year` was matching by calendar date, causing phase-shift. For example:
- 2025-08-15 (Lunar 07-15) was matching 2024-08-15 (different lunar date)
- Should match 2024-08-XX that is also Lunar 07-15

**Solution**:
- Updated `add_year_over_year_volume_features()` in `src/data/preprocessing.py`
- Added `use_lunar_matching=True` parameter for MOONCAKE
- Matches by (category, lunar_month, lunar_day) instead of (category, calendar_date)
- Finds dates approximately 1 year earlier (300-400 days) with same lunar date

**Impact**: Model will now see correct year-over-year patterns aligned with lunar calendar.

### 3. ✅ Added Peak Loss Window (Lunar Months 7.15 to 8.15)
**Problem**: Loss function wasn't penalizing errors in the critical peak period.

**Solution**:
- Added `is_peak_loss_window` feature (Lunar Months 7.15 to 8.15)
- Updated `spike_aware_mse()` loss function to support `peak_loss_window_weight`
- Added `peak_loss_window_weight: 20.0` to `config_MOONCAKE.yaml`
- Loss function now applies 20x weight to errors in peak period

**Impact**: Model will be heavily penalized for missing the peak, forcing it to learn the correct timing.

### 4. ✅ Increased YoY Feature Influence
**Problem**: Model relied too heavily on `rolling_mean_7d`, causing magnitude collapse in recursive forecasting.

**Solution**:
- YoY features now use lunar date matching (ensures correct alignment)
- Added trend deviation features (`trend_vs_yoy_ratio`, `trend_vs_yoy_diff`)
- Model can now adjust YoY baseline based on recent 21-day trend

**Impact**: Model will prioritize YoY patterns over recent rolling means during golden window.

### 5. ✅ Updated Golden Window Calculation
**Problem**: Golden window was 6.15 to 8.01, but peak loss window needed to be 7.15 to 8.15.

**Solution**:
- Kept golden window as 6.15 to 8.01 (peak buildup period)
- Added separate peak loss window 7.15 to 8.15 (critical peak period for loss weighting)
- Both features are now created and used appropriately

## ⚠️ CRITICAL: Model Must Be Retrained

**The current predictions are using an OLD model trained with incorrect code.**

### Required Actions:
1. **Retrain the MOONCAKE model** using `mvp_train.py` with the fixed code
2. **Verify lunar dates** are correct in training data (should show August/September 2025 as Lunar Months 6-7)
3. **Verify YoY features** are populated correctly (should have non-zero values for prediction dates)
4. **Re-run predictions** with the newly trained model

### Expected Improvements After Retraining:
- **July 2025**: Should predict ~22 CBM (not 859 CBM) - off-season
- **August 2025**: Should predict ~6,000 CBM (not 714 CBM) - peak period
- **September 2025**: Should predict ~4,500 CBM (not 778 CBM) - peak period
- **October 2025**: Should predict ~112 CBM (not 0 CBM) - post-peak

## Verification Checklist

Before retraining, verify:
- [ ] Lunar date conversion: 2025-10-06 = Lunar (8, 15) ✓
- [ ] Golden window: August 15, 2025 should be in golden window (Lunar 6.23)
- [ ] Peak loss window: September 15, 2025 should be in peak loss window (Lunar 7.24)
- [ ] YoY features: `cbm_last_year` should have non-zero values for August/September 2025 dates
- [ ] Feature columns: `is_peak_loss_window` is in `feature_cols` for MOONCAKE

## Configuration Updates

Updated `config_MOONCAKE.yaml`:
- Added `peak_loss_window_weight: 20.0` for Lunar Months 7.15 to 8.15
- Loss function will apply 20x weight to errors in this critical period

## Next Steps

1. **Retrain model**: Run `python mvp_train.py` to train MOONCAKE with fixed code
2. **Verify training**: Check that lunar dates are correct in training logs
3. **Re-predict**: Run `python mvp_predict.py` with newly trained model
4. **Evaluate**: Compare new predictions to actuals - should see peak in August/September, not July
