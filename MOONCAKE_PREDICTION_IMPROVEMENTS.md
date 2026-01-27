# MOONCAKE Prediction Improvements

**Date**: 2025-01-27  
**Issue**: Poor prediction accuracy for MOONCAKE category
- July 2025: Predicted 1655.80 CBM, Actual 22.85 CBM (72x overprediction)
- August 2025: Predicted 1386.54 CBM, Actual 6014.93 CBM (77% underprediction)
- September 2025: Predicted 1444.34 CBM, Actual 4575.75 CBM (68% underprediction)
- October 2025: Predicted 0.00 CBM, Actual 112.19 CBM (no prediction)

## Root Causes Identified

1. **Missing YoY Feature Recomputation**: During rolling prediction, year-over-year features (`cbm_last_year`, `cbm_2_years_ago`) were not being recomputed when the window was updated with new predictions. This is critical for MOONCAKE because YoY features are the primary signal for seasonal products.

2. **Incorrect Active Season Logic**: The active season check was using solar months (6-9) instead of lunar months, causing July 2025 (Lunar Month 5-6, off-season) to be incorrectly identified as active season.

3. **Insufficient Historical Data Access**: The rolling prediction function didn't have access to the full historical dataset needed to compute YoY features using lunar date matching.

4. **Missing Peak Loss Window Feature**: The `is_peak_loss_window` feature was not being computed during rolling prediction window updates.

## Fixes Implemented

### 1. ✅ Fixed YoY Feature Recomputation in Rolling Prediction

**File**: `src/predict/predictor.py` - `predict_direct_multistep_rolling()`

**Changes**:
- Added `historical_data` parameter to the function to enable YoY feature lookup
- Implemented full feature recomputation when updating the window with new predictions
- Added logic to recompute `cbm_last_year` and `cbm_2_years_ago` using lunar date matching for MOONCAKE
- Uses combined historical dataset (initial window + historical data + new predictions) for accurate YoY lookup

**Impact**: Model will now have correct YoY features during recursive prediction, preventing magnitude collapse.

### 2. ✅ Fixed Active Season Logic to Use Lunar Months

**File**: `src/predict/predictor.py` - `_get_is_active_season_mooncake()`

**Changes**:
- Changed from solar month check (months 6-9) to lunar month check (Lunar Months 6-9)
- Added additional check: If at start of Lunar Month 6, only activate from day 15 onwards
- This ensures July 2025 (Lunar Month 5-6, mostly off-season) is correctly zeroed out

**Impact**: Off-season periods (like July 2025) will now be properly masked to zero.

### 3. ✅ Added Historical Data Parameter to Rolling Prediction

**File**: `mvp_predict.py` - Recursive prediction section

**Changes**:
- Updated call to `predict_direct_multistep_rolling()` to pass `historical_data_prepared_cat` for MOONCAKE
- Historical data is now available for YoY feature lookup during rolling prediction

**Impact**: YoY features can now be accurately computed using the full historical dataset.

### 4. ✅ Added Peak Loss Window Feature Computation

**File**: `src/predict/predictor.py` - `predict_direct_multistep_rolling()`

**Changes**:
- Added computation of `is_peak_loss_window` feature during window updates
- Peak Loss Window: Lunar Months 7.15 to 8.15 (critical peak period)

**Impact**: Model will have correct peak loss window features during recursive prediction.

## Expected Improvements

After retraining the model with these fixes:

1. **July 2025**: Should predict ~0-22 CBM (not 1655.80) - correctly identified as off-season
2. **August 2025**: Should predict ~6,000 CBM (not 1386.54) - YoY features will provide correct magnitude
3. **September 2025**: Should predict ~4,500 CBM (not 1444.34) - YoY features will provide correct magnitude
4. **October 2025**: Should predict ~112 CBM (not 0.00) - post-peak but still in active season

## Next Steps

### 1. Retrain the Model

**CRITICAL**: The current predictions are using an OLD model trained without these fixes.

```bash
python mvp_train.py
```

This will train MOONCAKE with:
- Correct lunar date conversion
- Lunar date matching for YoY features
- Peak loss window weighting
- All other fixes from `MOONCAKE_PHASE_SHIFT_FIX_SUMMARY.md`

### 2. Verify Training

Check training logs to ensure:
- Lunar dates are correct (August/September 2025 should be Lunar Months 6-7)
- YoY features are populated correctly (non-zero values for prediction dates)
- Peak loss window features are present

### 3. Re-run Predictions

```bash
python mvp_predict.py
```

The new predictions should show:
- Correct timing (peak in August/September, not July)
- Correct magnitude (matching actuals within 20-30%)
- Proper off-season masking (July should be near zero)

## Technical Details

### YoY Feature Recomputation Logic

The rolling prediction now:
1. Maintains a combined historical dataset (initial window + historical data + new predictions)
2. For each chunk of predictions, recomputes YoY features using `add_year_over_year_volume_features()` with `use_lunar_matching=True`
3. Updates the window with correct YoY features before the next chunk prediction
4. Ensures YoY features reference the correct historical dates using lunar date matching

### Active Season Logic

The active season check now:
1. Converts solar date to lunar date using `_solar_to_lunar_date()`
2. Checks if lunar month is between 6 and 9 (inclusive)
3. Additional check: If lunar month is 6 and day < 15, treat as off-season
4. Returns 0 for off-season (forces prediction to zero via hard masking)

## Files Modified

1. `src/predict/predictor.py`:
   - `predict_direct_multistep_rolling()`: Added historical_data parameter, YoY feature recomputation, peak loss window computation
   - `_get_is_active_season_mooncake()`: Fixed to use lunar months instead of solar months

2. `mvp_predict.py`:
   - Updated call to `predict_direct_multistep_rolling()` to pass historical data for MOONCAKE

## Verification Checklist

Before running predictions, verify:
- [ ] Model has been retrained with all fixes from `MOONCAKE_PHASE_SHIFT_FIX_SUMMARY.md`
- [ ] Lunar date conversion: 2025-10-06 = Lunar (8, 15) ✓
- [ ] Active season: July 2025 should be off-season (Lunar Month 5-6)
- [ ] Active season: August 2025 should be active (Lunar Month 6-7)
- [ ] YoY features: `cbm_last_year` should have non-zero values for August/September 2025 dates
- [ ] Feature columns: `is_peak_loss_window` is in `feature_cols` for MOONCAKE

## Related Documents

- `MOONCAKE_PHASE_SHIFT_FIX_SUMMARY.md`: Previous fixes for phase-shift and magnitude collapse
- `config/config_MOONCAKE.yaml`: MOONCAKE-specific configuration
- `src/data/preprocessing.py`: YoY feature computation logic
