# Complete Month-Based Prediction System - Summary

## Overview
This document summarizes the complete enhancement to align predictions with calendar months at every level.

## Three-Level Enhancement

### Level 1: Initial Window - Full Month Context
**File**: `get_historical_window_data()` in `predictor.py`

**Change**: Use complete months for initial context window
- Before: Last N days (e.g., 28 days from Mar 4 - Mar 31)
- After: Full previous month + current month (e.g., Feb 1 - Mar 31)

### Level 2: Window Maintenance - Keep Full Months
**File**: `predict_direct_multistep_rolling()` in `predictor.py`

**Change**: Maintain full-month context after each chunk
- Before: Truncate to last 28 days (e.g., Aug 31 - Sep 27)
- After: Keep from start of previous month (e.g., Aug 1 - Sep 30)

### Level 3: Chunk Boundaries - Predict by Month
**File**: `predict_direct_multistep_rolling()` in `predictor.py`

**Change**: Align chunks with calendar month boundaries
- Before: Fixed 30-day chunks (May 31 - Jun 29)
- After: Complete calendar months (Jun 1 - Jun 30)

## Complete Prediction Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ INITIAL SETUP                                                   │
├─────────────────────────────────────────────────────────────────┤
│ Historical Data: Jan 2023 - Mar 2025                           │
│ Predict: Apr 2025 - Dec 2025                                   │
│                                                                 │
│ Initial Window (Full Month Context):                           │
│   └─> Feb 1, 2025 - Mar 31, 2025 (59 days = Feb + Mar)       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 1: APRIL 2025                                            │
├─────────────────────────────────────────────────────────────────┤
│ Input Window: Feb 1 - Mar 31 (Full Feb + Full Mar)            │
│ Predict: Apr 1 - Apr 30 (30 days)                             │
│ Updated Window: Mar 1 - Apr 30 (Full Mar + Full Apr)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 2: MAY 2025                                              │
├─────────────────────────────────────────────────────────────────┤
│ Input Window: Mar 1 - Apr 30 (Full Mar + Full Apr)            │
│ Predict: May 1 - May 31 (31 days) ← Handles 31-day month      │
│ Updated Window: Apr 1 - May 31 (Full Apr + Full May)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CHUNK 3: JUNE 2025                                             │
├─────────────────────────────────────────────────────────────────┤
│ Input Window: Apr 1 - May 31 (Full Apr + Full May)            │
│ Predict: Jun 1 - Jun 30 (30 days)                             │
│ Updated Window: May 1 - Jun 30 (Full May + Full Jun)          │
└─────────────────────────────────────────────────────────────────┘

... continues for each month ...
```

## Key Principles

### 1. Always Start on 1st
Every chunk starts on the **1st day of a month**, never mid-month.

### 2. Always End on Last Day
Every chunk ends on the **last day of a month** (28/29/30/31 depending on month).

### 3. Two-Month Context
Input window always includes **two complete calendar months**:
- Previous month (in full)
- Current month being predicted (in full after predictions)

### 4. Month-to-Month Transitions
When moving to next chunk:
```python
current_month = 5  # May
next_month = 6     # June
current_start = date(2025, 6, 1)  # Always 1st of next month
```

## Benefits by Use Case

### For DRY Category (Early-Month Overprediction)
- Model sees full March when predicting April 1st
- Clear month boundary signals
- Consistent early-month context

### For All Categories
- Natural business cycle alignment
- Easy monthly reporting
- Handles variable month lengths (28-31 days)
- Better start/mid/end of month pattern learning

### For Model Training
- Consider training with monthly sequences
- Loss function can weight by day-of-month
- Better generalization to month boundaries

## Configuration Requirements

### Recommended Config
```yaml
window:
  input_size: 60      # ~2 months of historical context
  horizon: 31         # Must handle longest month (31 days)
  stride: 1
```

### Minimum Requirements
- `input_size`: At least 60 (to fit 2 full months)
- `horizon`: At least 31 (to predict complete months with 31 days)

## Files Modified

1. **src/predict/predictor.py**
   - `get_historical_window_data()`: Full month initial window
   - `predict_direct_multistep_rolling()`: Calendar month chunks + window maintenance
   - `predict_direct_multistep()`: No changes (handles variable horizon)

2. **mvp_predict.py**
   - Updated 3 calls to `get_historical_window_data()` with `use_full_months=True`

3. **predict_by_brand.py**
   - Updated 1 call to `get_historical_window_data()` with `use_full_months=True`

## Expected Console Output

```
[FULL-MONTH MODE] Window: 2025-02-01 to 2025-03-31 (last month + current month)
[FULL-MONTH MODE] Using full window (59 days) instead of truncating to input_size=28

- Starting rolling prediction for 274 days
- Date range: 2025-04-01 to 2025-12-31
- Prediction mode: CALENDAR MONTH chunks (each chunk = one complete month)
- Initial window: 2025-02-01 00:00:00 to 2025-03-31 00:00:00

[Chunk 1] Predicting 2025-04-01 to 2025-04-30 (30 days) - Month 04/2025...
- Maintaining full-month window: 60 rows from 2025-03-01 to 2025-04-30

[Chunk 2] Predicting 2025-05-01 to 2025-05-31 (31 days) - Month 05/2025...
- Maintaining full-month window: 61 rows from 2025-04-01 to 2025-05-31

[Chunk 3] Predicting 2025-06-01 to 2025-06-30 (30 days) - Month 06/2025...
- Maintaining full-month window: 61 rows from 2025-05-01 to 2025-06-30

... continues for each month ...
```

## Validation Steps

1. **Check chunk boundaries**: Each chunk should be a complete month
2. **Check window dates**: Windows should start on 1st of previous month
3. **Check continuity**: No gaps or overlaps between chunks
4. **Check month lengths**: April (30), May (31), June (30), etc. all correct

## Next Steps

1. **Test with current model**: Run predictions and verify output format
2. **Consider retraining**: Model may benefit from learning with month-aligned windows
3. **Monitor metrics**: Check if early-month predictions improve
4. **Evaluate by month**: Compare predicted vs actual monthly totals

## Related Documents

- `FULL_MONTH_WINDOW_ENHANCEMENT.md`: Details on Level 1 & 2 (window context)
- `CALENDAR_MONTH_ALIGNMENT.md`: Details on Level 3 (chunk boundaries)
- `EARLY_MONTH_STRENGTHENED_RULES.md`: Original early-month issue analysis
