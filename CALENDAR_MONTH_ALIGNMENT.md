# Calendar Month-Aligned Prediction Enhancement

## Problem
The rolling prediction was using **arbitrary 30-day chunks** that didn't align with calendar months:
- Chunk 6: 2025-05-31 to 2025-06-29 (partial May + partial June)
- Chunk 10: 2025-09-28 to 2025-10-27 (partial Sep + partial Oct)

This causes several issues:
1. **Split months**: A single month's predictions are split across 2 chunks
2. **Partial context**: Window shows 2025-08-31 to 2025-09-27 (doesn't start from Aug 1st)
3. **Poor month boundaries**: Model doesn't understand "we're starting a new month"
4. **31-day months**: Months with 31 days should be in one chunk, not split

## Solution
Enhanced the rolling prediction to use **CALENDAR MONTH boundaries**:

### Key Changes

#### 1. Month-Aligned Chunks
**Before:**
```python
chunk_end = current_start + timedelta(days=horizon - 1)  # Fixed 30-day chunks
current_start = chunk_end + timedelta(days=1)            # Arbitrary advancement
```

**After:**
```python
# Each chunk = ONE COMPLETE MONTH from 1st to last day
chunk_start = date(year, month, 1)  # Always start on 1st of month
chunk_end = last_day_of_month       # End on 28/29/30/31 depending on month
current_start = date(next_year, next_month, 1)  # Move to 1st of next month
```

#### 2. Full-Month Input Windows
**Before:**
```python
window = last 28 days  # e.g., 2025-08-31 to 2025-09-27 (partial months)
```

**After:**
```python
window = previous_month_start to current_date  # e.g., 2025-08-01 to 2025-09-30 (full Aug + full Sep)
```

## Changes Made

### 1. Enhanced `predict_direct_multistep_rolling()` (predictor.py)

**Chunk Calculation:**
- Calculates first day of month: `date(year, month, 1)`
- Calculates last day of month: `date(next_year, next_month, 1) - 1 day`
- Adjusts `start_date` to month start if not already on the 1st
- Moves to next month's 1st day after each chunk

**Window Maintenance:**
- Keeps data from "start of previous month" onwards
- Ensures window always includes 2 complete months
- Example: When predicting October, window includes full August + full September

### 2. Updated Documentation
- Updated docstring to clarify calendar month chunking behavior
- Added note that `start_date` will be adjusted to 1st of month if needed

## Benefits

1. **Natural Business Cycles**: Predictions align with how businesses think (by month)
2. **Complete Month Context**: Each month is predicted as a whole unit
3. **Better Start-of-Month Predictions**: Model sees full previous month when predicting new month
4. **Handles Variable Month Lengths**: Automatically handles 28/29/30/31-day months
5. **Consistent Evaluation**: Easy to compare predicted vs actual by complete months

## Example Output

### Before (Arbitrary 30-day chunks)
```
[Chunk 5] Predicting 2025-05-01 to 2025-05-30 (30 days)...
- Initial window: 2025-04-03 00:00:00 to 2025-04-30 00:00:00

[Chunk 6] Predicting 2025-05-31 to 2025-06-29 (30 days)...  ← Split May/June
- Initial window: 2025-05-03 00:00:00 to 2025-05-30 00:00:00

[Chunk 10] Predicting 2025-09-28 to 2025-10-27 (30 days)... ← Split Sep/Oct
- Initial window: 2025-08-31 00:00:00 to 2025-09-27 00:00:00 ← Partial months
```

### After (Calendar month chunks)
```
[Chunk 5] Predicting 2025-05-01 to 2025-05-31 (31 days) - Month 05/2025...
- Initial window: 2025-04-01 00:00:00 to 2025-04-30 00:00:00  ← Full April

[Chunk 6] Predicting 2025-06-01 to 2025-06-30 (30 days) - Month 06/2025...
- Initial window: 2025-05-01 00:00:00 to 2025-05-31 00:00:00  ← Full May
- Maintaining full-month window: 61 rows from 2025-05-01 to 2025-06-30

[Chunk 10] Predicting 2025-10-01 to 2025-10-31 (31 days) - Month 10/2025...
- Initial window: 2025-09-01 00:00:00 to 2025-09-30 00:00:00  ← Full September
- Maintaining full-month window: 61 rows from 2025-09-01 to 2025-10-31
```

## Prediction Flow Example

```
Initial: Window = Feb 1 - Mar 31 (full Feb + full Mar)

Chunk 1: Predict Apr 1 - Apr 30 (30 days)
         Window = Mar 1 - Apr 30 (full Mar + full Apr)

Chunk 2: Predict May 1 - May 31 (31 days)  ← Handles 31-day month
         Window = Apr 1 - May 31 (full Apr + full May)

Chunk 3: Predict Jun 1 - Jun 30 (30 days)
         Window = May 1 - Jun 30 (full May + full Jun)

Chunk 4: Predict Jul 1 - Jul 31 (31 days)
         Window = Jun 1 - Jul 31 (full Jun + full Jul)
```

Each chunk:
- Predicts **exactly one complete calendar month**
- Input window includes **two complete calendar months** (previous + current)
- Next chunk starts on the **1st day of the next month**

## Impact on DRY Early-Month Predictions

This enhancement directly addresses the early-month overprediction issue:

1. **Clean Month Boundaries**: Model clearly sees "Apr 1st is start of new month"
2. **Full Previous Month Context**: When predicting April, window includes all of February + all of March
3. **No Split Predictions**: April is predicted as one complete unit, not split across chunks
4. **Better Trend Learning**: Model learns monthly patterns (early/mid/late month behavior)

## Model Horizon Considerations

If a month has more days than the model's horizon (e.g., 31 days but horizon=30):
```python
if chunk_days > horizon:
    print(f"[WARNING] Month has {chunk_days} days but model horizon is {horizon}")
    chunk_end = current_start + timedelta(days=horizon - 1)
```

**Recommendation**: Set `horizon: 31` in config to handle all months.

## Configuration Update

Update your config file to handle the longest month:

```yaml
window:
  input_size: 60      # ~2 months of input context
  horizon: 31         # Maximum month length (January, March, May, July, August, October, December)
  stride: 1
```

## Testing

Run prediction and verify output:
```bash
python mvp_predict.py --config config/config_DRY.yaml
```

Look for:
- `Prediction mode: CALENDAR MONTH chunks` at the start
- Chunk messages: `[Chunk N] Predicting YYYY-MM-01 to YYYY-MM-31 (31 days) - Month MM/YYYY`
- Window maintenance: `Maintaining full-month window: N rows from YYYY-MM-01 to YYYY-MM-DD`
- Each chunk should start on the 1st and end on the last day (28/29/30/31) of each month

## Validation

Check that predictions now have clean month boundaries:
```python
import pandas as pd
df = pd.read_csv('predictions.csv')
df['month'] = pd.to_datetime(df['date']).dt.month
print(df.groupby('month')['predicted'].sum())  # Each month's total
```

Each month should be complete (not split across chunks).
