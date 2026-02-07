# Data Leakage Analysis Report
## MDLZ Forecast Pipeline - Causal Forecasting Verification

**Date**: 2025  
**Script Analyzed**: `mvp_test.py` and related pipeline files  
**Purpose**: Verify that the forecasting pipeline strictly follows causal forecasting principles with no data leakage

---

## Executive Summary

This report analyzes the data pipeline implementation in `mvp_test.py` and related files to verify that the forecasting system adheres to causal forecasting principles. The analysis covers four critical areas:

1. **Data Splitting Logic** - Temporal (chronological) split verification
2. **Window Slicing Logic** - Off-by-one error and temporal causality verification
3. **Feature Engineering** - Future information leakage check
4. **Inference Logic** - Recursive prediction mode verification

**Overall Assessment**: The training pipeline appears to follow causal principles, but **CRITICAL ISSUES** were identified in the inference logic for January 2025 predictions.

---

## 1. Data Splitting Verification ✅

### Location
- `src/data/preprocessing.py`: `split_data()` function (lines 361-407)
- `mvp_test.py`: Data splitting call (lines 282-291)

### Implementation Analysis

**Code Location**: `src/data/preprocessing.py:361-407`

```python
def split_data(data: pd.DataFrame, ..., temporal: bool = True):
    N = len(data)
    if temporal:
        train_end = int(train_size * N)
        val_end = train_end + int(val_size * N)
        train_data = data[:train_end].copy()
        val_data = data[train_end:val_end].copy()
        test_data = data[val_end:].copy()
```

**Verification in `mvp_test.py`**:
- Line 248: `data = data.sort_values(time_col).reset_index(drop=True)` ✅
- Line 282-287: `split_data(..., temporal=True)` ✅

### Findings

✅ **PASS**: Data splitting is strictly temporal (chronological)

- Data is sorted by `time_col` before splitting (line 248 in `mvp_test.py`)
- `temporal=True` parameter is explicitly set in all split calls
- Sequential split logic: `data[:train_end]`, `data[train_end:val_end]`, `data[val_end:]`
- No random shuffling occurs before the split
- The 70/10/20 ratio is applied chronologically: first 70% → train, next 10% → val, last 20% → test

**Risk Level**: ✅ **NONE** - No leakage detected

---

## 2. Window Slicing Verification ✅

### Location
- `src/data/preprocessing.py`: `slicing_window_category()` function (lines 96-141)

### Implementation Analysis

**Code Location**: `src/data/preprocessing.py:130-139`

```python
for cat, g in df.groupby(cat_col_name, sort=False):
    g = g.sort_values(time_col_name)  # ✅ Sort within each category
    X_data = g[feature_cols].values
    y_data = g[target_col_name].values
    
    for i in range(len(g) - input_size - horizon + 1):
        X.append(X_data[i:i+input_size])           # Positions [i, i+input_size)
        y.append(y_data[i+input_size:i+input_size+horizon])  # Positions [i+input_size, i+input_size+horizon)
```

**With `input_size=30` and `horizon=1`**:
- For index `i`: 
  - `X` = `X_data[i:i+30]` = positions `[i, i+1, ..., i+29]` (30 timesteps)
  - `y` = `y_data[i+30:i+31]` = position `[i+30]` (1 timestep)
- **Gap**: X ends at position `i+29`, y starts at position `i+30` ✅

### Findings

✅ **PASS**: Window slicing correctly maintains temporal causality

- Input window `X` contains data from `t-30` to `t-1` (positions `i` to `i+29`)
- Target `y` is at time `t` (position `i+30`)
- **No overlap** between X and y - there is exactly a 1-timestep gap
- Within each category group, data is sorted by time column before window creation
- The range `len(g) - input_size - horizon + 1` correctly prevents indexing beyond array bounds

**Example Timeline** (for `i=0`, `input_size=30`, `horizon=1`):
```
Time:  t-30, t-29, ..., t-2, t-1, [t], t+1, ...
X:     [─────── positions 0 to 29 ───────]
y:                                    [position 30]
```

**Risk Level**: ✅ **NONE** - No off-by-one error detected

---

## 3. Feature Engineering Verification ✅

### Location
- Holiday features: `src/data/preprocessing.py`: `add_holiday_features()` (lines 218-286)
- Temporal features: `src/data/preprocessing.py`: `add_temporal_features()` (lines 289-337)
- Holiday feature implementation in `mvp_test.py`: `add_holiday_features_vietnam()` (lines 97-165)

### Implementation Analysis

**Holiday Features** (`add_holiday_features_vietnam` in `mvp_test.py:141-160`):

```python
for idx, row in df.iterrows():
    current_date = row[time_col].date()
    
    # Set holiday indicator - uses ONLY current_date
    if current_date in holiday_set:
        df.at[idx, holiday_indicator_col] = 1
    
    # Calculate days until next holiday - uses ONLY current_date and calendar
    next_holiday = None
    for holiday in holidays:
        if holiday > current_date:  # ✅ Only future holidays in calendar
            next_holiday = holiday
            break
    
    if next_holiday:
        days_until = (next_holiday - current_date).days  # ✅ Calendar calculation
```

**Temporal Features** (`add_temporal_features` in `preprocessing.py:324-335`):

```python
month = df[time_col].dt.month          # ✅ From date only
dayofmonth = df[time_col].dt.day       # ✅ From date only
df[month_sin_col] = np.sin(2 * np.pi * (month - 1) / 12)
df[month_cos_col] = np.cos(2 * np.pi * (month - 1) / 12)
```

**QTY Feature**:
- `QTY` is included in `feature_cols` (config/config.yaml:17)
- QTY values are from **previous timesteps only** (part of X window, not y)
- This is correct - historical QTY values are legitimate features

### Findings

✅ **PASS**: Feature engineering uses only calendar/date information and historical data

- `holiday_indicator`: Derived from calendar (holiday list) - no future shipment data ✅
- `days_until_next_holiday`: Calculated from current date and holiday calendar - no future shipment data ✅
- `month_sin/cos`, `dayofmonth_sin/cos`: Derived from date column only ✅
- `QTY`: Historical values from `t-30` to `t-1` only (not from future) ✅

**Risk Level**: ✅ **NONE** - No leakage from future information

---

## 4. Inference Logic Verification ⚠️ CRITICAL ISSUE

### Location
- `predict_jan2025.py`: Main prediction logic (lines 172-394)

### Implementation Analysis

**Code Location**: `predict_jan2025.py:263`

```python
# Create prediction windows
X_pred, y_actual, cat_pred, pred_dates = create_prediction_windows(data_2025_prepared, config)
```

**The Problem**: `create_prediction_windows()` calls `slicing_window_category()` which uses **actual 2025 data**:

```python
# predict_jan2025.py:137-145
X_pred, y_pred, cat_pred = slicing_window_category(
    data,  # ⚠️ This is data_2025_prepared containing actual 2025 QTY values
    input_size,
    horizon,
    feature_cols=feature_cols,  # ⚠️ Includes 'QTY' as a feature
    ...
)
```

**What This Means**:
- When predicting January 2025, the input window `X_pred` includes actual QTY values from January 2025
- For example, to predict `2025-01-31`, the model uses QTY from `2024-12-02` to `2025-01-30` (actual values)
- **However**, the model also uses QTY from `2025-01-01` to `2025-01-30` as features
- These are **actual ground truth values from the test period**, not predictions

### Critical Issue

🚨 **DATA LEAKAGE DETECTED**: The current implementation uses **actual ground truth QTY values from 2025** as features in the input window when predicting 2025.

**Example Scenario** (predicting `2025-01-31`):
```
Input window X should contain:
- QTY from 2024-12-02 to 2024-12-31 (historical - OK)
- QTY from 2025-01-01 to 2025-01-30 (actual 2025 values - ⚠️ LEAKAGE)

Target y:
- QTY for 2025-01-31 (actual - OK, needed for evaluation)
```

**Why This Is A Problem**:
- In a true recursive forecasting scenario, the model should use **its own previous predictions** for January 2025, not actual values
- Using actual values makes the model appear more accurate than it would be in production
- This violates causal forecasting principles for true out-of-sample prediction

### Correct Implementation Should Be

For true recursive prediction in January 2025:
1. Use historical data up to `2024-12-31` for the initial window
2. For each day in January 2025:
   - Use the model's **prediction** from the previous day (not actual value)
   - Update the input window by shifting: remove oldest, add latest prediction
   - This is true autoregressive/recursive forecasting

**Current Behavior** (Non-Recursive/Teacher Forcing):
- Uses actual QTY values from 2025 as features
- Only suitable for **evaluation** of model quality on test set
- **Not suitable** for true forecasting of future unknown dates

### Findings

⚠️ **FAIL**: Inference logic does NOT use recursive mode

- `predict_jan2025.py` uses actual 2025 QTY values in input features
- No recursive/autoregressive prediction loop that uses previous predictions
- This is acceptable for **test set evaluation**, but not for true forecasting
- For production forecasting of unknown future dates, recursive mode is needed

**Risk Level**: ⚠️ **HIGH** - For true forecasting scenarios
**Risk Level**: ✅ **NONE** - For test set evaluation (if ground truth is available)

---

## Summary of Findings

| Aspect | Status | Risk Level | Notes |
|--------|--------|------------|-------|
| **1. Data Splitting** | ✅ PASS | None | Strictly temporal, no shuffling |
| **2. Window Slicing** | ✅ PASS | None | No off-by-one error, correct temporal causality |
| **3. Feature Engineering** | ✅ PASS | None | Uses only calendar and historical data |
| **4. Inference Logic** | ⚠️ **ISSUE** | High (for forecasting) | Uses actual ground truth, not recursive predictions |

---

## Recommendations

### 1. Immediate Actions

**For Test Set Evaluation** (Current Use Case):
- ✅ Current implementation is **acceptable** if the goal is to evaluate model performance on a held-out test set
- The model uses actual 2025 QTY values as features, which is fine for evaluation
- However, metrics may be **overly optimistic** compared to true production performance

**For Production Forecasting** (Future Unknown Dates):
- ❌ Current implementation is **NOT suitable** for forecasting unknown future dates
- Must implement **recursive prediction mode** that:
  1. Uses historical data up to the last known date
  2. For each future date, uses the model's previous predictions (not actual values)
  3. Updates the input window by shifting: removes oldest timestep, appends latest prediction

### 2. Implementation Suggestion

Create a new function `predict_recursive()` in `predict_jan2025.py`:

```python
def predict_recursive(model, initial_data, start_date, end_date, config):
    """
    Recursive prediction mode: uses model's own predictions as inputs.
    
    Args:
        initial_data: Historical data up to start_date (e.g., data up to 2024-12-31)
        start_date: First date to predict (e.g., 2025-01-01)
        end_date: Last date to predict (e.g., 2025-01-31)
        config: Configuration object
    
    Returns:
        DataFrame with predictions for each date
    """
    # Start with historical window
    window = initial_data[-config.window['input_size']:].copy()
    predictions = []
    
    current_date = start_date
    while current_date <= end_date:
        # Create input window from current window
        X = create_window_from_data(window, config)
        
        # Get prediction
        pred = model.predict(X)  # Predict for current_date
        
        # Update window: remove oldest, add prediction
        new_row = create_row_from_prediction(current_date, pred, config)
        window = window.iloc[1:].append(new_row)  # Shift window
        
        predictions.append({'date': current_date, 'predicted': pred})
        current_date += timedelta(days=1)
    
    return pd.DataFrame(predictions)
```

### 3. Documentation Update

Update `predict_jan2025.py` documentation to clarify:
- Current implementation is for **test set evaluation** (uses ground truth)
- For production forecasting, use recursive mode (not yet implemented)
- Clearly label output files: `predictions_test_evaluation.csv` vs `predictions_recursive_forecast.csv`

---

## Conclusion

The training pipeline (`mvp_test.py`) **strictly follows causal forecasting principles** with no data leakage in:
- ✅ Data splitting (temporal, no shuffling)
- ✅ Window slicing (no off-by-one error)
- ✅ Feature engineering (calendar-based only)

However, the inference pipeline (`predict_jan2025.py`) has a **critical limitation**:
- ⚠️ It uses actual ground truth values from 2025 as features
- ⚠️ This is acceptable for test set evaluation but not for true production forecasting
- ⚠️ Recursive prediction mode is not implemented

**For the current use case (evaluating model on test set)**: The implementation is acceptable, but metrics may be optimistic.

**For production forecasting of unknown future dates**: Recursive prediction mode must be implemented to avoid using unavailable future information.

---

**Report Generated**: 2025  
**Files Analyzed**: 
- `mvp_test.py`
- `src/data/preprocessing.py`
- `predict_jan2025.py`
- `config/config.yaml`
