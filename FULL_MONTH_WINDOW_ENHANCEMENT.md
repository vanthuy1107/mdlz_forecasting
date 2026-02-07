# Full-Month Window Enhancement

## Problem
The prediction window was using the last N days (e.g., 28 days) from historical data, which often resulted in **partial month coverage**. For example:
- Initial window: 2025-03-04 to 2025-03-31 (only 28 days of March, missing February)
- Rolling chunks: 2025-08-31 to 2025-09-27 (partial months)

This approach fails to capture **monthly trends** properly, as business patterns often follow monthly cycles (e.g., start-of-month vs end-of-month behavior).

**Additional Problem**: Even with the initial full-month window, the rolling prediction was **truncating back to input_size (28 days)** after each chunk, losing the monthly context.

## Solution
Enhanced the prediction system to support **full-month alignment throughout**:

### 1. Enhanced `get_historical_window_data()` - Initial Window
```python
def get_historical_window_data(
    historical_data: pd.DataFrame,
    end_date: date,
    config,
    num_days: int = 30,
    use_full_months: bool = True  # NEW PARAMETER
):
```

**Behavior when `use_full_months=True` (default):**
- **Includes complete months**: last month + current month (up to `end_date`)
- Example for `end_date=2025-03-31`:
  - Start date: 2025-02-01 (first day of previous month)
  - End date: 2025-03-31 (given end date)
  - Total: February (28 days) + March (31 days) = 59 days

### 2. Enhanced `predict_direct_multistep_rolling()` - Maintain Full Months
**Previous behavior:**
```python
# Truncated to input_size after each chunk - LOST monthly context
window = window.tail(input_size).copy()  # Only keeps last 28 days
```

**New behavior:**
```python
# Maintains full-month window throughout rolling prediction
if use_full_month_window and len(window) > input_size:
    # Keep data from start of previous month onwards
    full_month_start = date(prev_month_year, prev_month, 1)
    window = window[window['date'] >= full_month_start].copy()
else:
    # Standard mode: truncate to input_size
    window = window.tail(input_size).copy()
```

## Changes Made

### 1. Enhanced `get_historical_window_data()` (predictor.py)
- Added `use_full_months` parameter with full-month logic
- Calculates start date as first day of previous month
- Prints `[FULL-MONTH MODE]` message for visibility

### 2. Updated `predict_direct_multistep_rolling()` (predictor.py)
- No longer truncates to `input_size` when full-month window is provided
- **CRITICAL FIX**: Maintains full-month window after each chunk
- Dynamically calculates "start of previous month" based on current window end date
- Prints window maintenance messages for each chunk

### 3. Updated All Callers
Updated function calls in:
- `mvp_predict.py` (3 locations)
- `predict_by_brand.py` (1 location)

All now use `use_full_months=True` by default.

## Benefits

1. **Better Monthly Trend Capture**: RNN sees complete monthly patterns throughout prediction
2. **Start-of-Month Awareness**: Captures early-month vs late-month behavior differences
3. **More Context**: Provides 2 full months of data consistently (vs partial month)
4. **Seasonal Alignment**: Better for products with monthly sales cycles
5. **Rolling Consistency**: Maintains monthly context across all chunks, not just initial

## Example Output

### Initial Window (First Chunk)
```
[FULL-MONTH MODE] Window: 2025-02-01 to 2025-03-31 (last month + current month)
[FULL-MONTH MODE] Using full window (59 days) instead of truncating to input_size=28
- Initial window: 2025-02-01 00:00:00 to 2025-03-31 00:00:00

[Chunk 1] Predicting 2025-04-01 to 2025-04-30 (30 days)...
```

### Rolling Chunks (Maintaining Full Months)
```
[Chunk 10] Predicting 2025-09-28 to 2025-10-27 (30 days)...
- Starting direct multi-step prediction from 2025-09-28 to 2025-10-27
- Initial window: 2025-08-01 00:00:00 to 2025-09-27 00:00:00  ← FULL August + partial September
- Maintaining full-month window: 68 rows from 2025-08-01 to 2025-10-27  ← After predictions
```

**Before Fix:**
- Window: 2025-08-31 to 2025-09-27 (28 days, partial months)

**After Fix:**
- Window: 2025-08-01 to 2025-09-27 (full August + partial September)

## Impact on DRY Category
For the DRY early-month prediction issue:
- Model now sees complete monthly patterns throughout all predictions
- Better understanding of month-to-month transitions at EVERY step
- Should reduce early-month overprediction by providing consistent representative context
- Each chunk predicts with full awareness of "we're starting a new month"

## Technical Details

### Window Maintenance Logic
For each rolling chunk after adding predictions:
1. Get the last date in the window (e.g., 2025-10-27)
2. Calculate previous month start (e.g., 2025-09-01 if current is October)
3. Keep all rows from previous month start onwards
4. Result: Always maintains ~60-90 days of data spanning 2 full months

### Example Flow
```
Chunk 1: Window = Feb 1 - Mar 31 → Predict Apr → Window = Mar 1 - Apr 30
Chunk 2: Window = Mar 1 - Apr 30 → Predict May → Window = Apr 1 - May 30
Chunk 3: Window = Apr 1 - May 30 → Predict Jun → Window = May 1 - Jun 30
...
```

Each window always includes the full previous month + current month data.

## Testing
Run prediction with the updated code:
```bash
python mvp_predict.py --config config/config_DRY.yaml
```

Look for:
- `[FULL-MONTH MODE]` messages at the start
- `Maintaining full-month window: N rows from YYYY-MM-01` messages after each chunk
- Initial window should start on day 1 of previous month (e.g., 2025-02-01, not 2025-03-04)

