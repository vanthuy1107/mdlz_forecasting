# Year-Over-Year Features Fix for MOONCAKE

## Problem Identified

The MOONCAKE model was not learning from same period last year and 2 years ago because:

1. **Year-over-year features (`cbm_last_year`, `cbm_2_years_ago`) were always 0.0 during prediction**
2. The `add_year_over_year_volume_features` function requires historical data in the same DataFrame to calculate YoY features
3. During prediction, only prediction data (2025) was passed to the function, so it couldn't find historical values (2023, 2024)

## Root Cause

The `add_year_over_year_volume_features` function works by:
1. Taking current data (e.g., 2025 dates)
2. Shifting dates forward by 1 year (2025 → 2026) and 2 years (2025 → 2027)
3. Merging back to find matching dates

**Problem**: When predicting 2025 dates, the DataFrame only contains 2025 data. When the function shifts 2025-08-15 forward to 2026-08-15, it can't find that date in the DataFrame (because 2026 data doesn't exist). Similarly, it can't find 2024-08-15 (which would be needed for 2025-08-15's `cbm_last_year`).

**Solution**: Combine historical data (2023, 2024) with prediction data (2025) BEFORE calling `add_year_over_year_volume_features`. This way:
- For 2025-08-15, the function can find 2024-08-15 (shifted forward from 2024-08-15)
- For 2025-08-15, the function can find 2023-08-15 (shifted forward from 2023-08-15)

## Fix Implementation

### 1. Modified `src/predict/prepare.py`

- Added `historical_data` parameter to `prepare_prediction_data()` function
- When processing MOONCAKE category:
  - If historical data is provided, combine it with prediction data
  - Calculate YoY features on the combined dataset
  - Extract only prediction data rows (with YoY features now populated)
- Added warning if historical data is not provided

### 2. Modified `mvp_predict.py`

- Updated call to `prepare_prediction_data()` for MOONCAKE category
- Passes prepared historical data (`historical_data_prepared_cat`) to enable YoY feature calculation

## Expected Impact

After this fix:
- `cbm_last_year` will contain actual values from 2024 (same calendar date)
- `cbm_2_years_ago` will contain actual values from 2023 (same calendar date)
- Model will be able to learn from historical patterns
- Predictions should improve significantly, especially during peak season (August-September)

## Testing

Run the diagnostic script to verify:
```bash
python check_yoy_features.py
```

Expected output:
- ✓ Year-over-year features CAN be calculated for 2025 when combined with historical data
- Non-zero `cbm_last_year` values for prediction dates

## Next Steps

1. Re-run predictions with the fix
2. Verify that `cbm_last_year` and `cbm_2_years_ago` are non-zero for prediction dates
3. Check if model predictions improve (should be closer to actual values during peak season)
4. If predictions still under-perform, investigate:
   - Model training (loss weights, architecture)
   - Feature scaling
   - Other feature engineering issues
