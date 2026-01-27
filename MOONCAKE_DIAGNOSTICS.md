# MOONCAKE Prediction Diagnostics

**Date**: 2025-01-27  
**Current Results**:
- August 2025: Predicted 652.29 CBM, Actual 6014.93 CBM (89% underprediction)
- September 2025: Predicted 762.55 CBM, Actual 4575.75 CBM (83% underprediction)

## Critical Issues to Check

### 1. ⚠️ Model Retraining Status

**CRITICAL**: The model MUST be retrained with all fixes before predictions will improve.

**Check**:
- [ ] Has the model been retrained after implementing the fixes?
- [ ] Check model timestamp: `outputs/MOONCAKE/models/best_model.pth`
- [ ] Verify training logs show YoY features being used with lunar matching

**Action**: If model hasn't been retrained, run:
```bash
python mvp_train.py
```

### 2. ⚠️ Historical Data Availability

**Issue**: YoY features need historical data from 2024 (and 2023) to provide correct values for August/September 2025.

**Check**:
- [ ] Does `historical_data_prepared_cat` contain data from 2024?
- [ ] Are there actual values for MOONCAKE in August/September 2024?
- [ ] Check terminal output: "Passing historical data (X rows) for YoY feature lookup"

**Expected**: Historical data should have:
- August 2024: ~6,000 CBM (for August 2025 YoY lookup)
- September 2024: ~4,500 CBM (for September 2025 YoY lookup)

### 3. ⚠️ YoY Feature Values

**Issue**: Even if YoY features are computed, they might be zero or incorrect.

**Check Terminal Output**:
Look for lines like:
```
- Recomputing YoY features for MOONCAKE using historical data...
- Updated YoY features for X/Y window rows
- Sample YoY values (date, cbm_last_year, cbm_2_years_ago):
    2025-08-15: 150.23, 120.45
```

**Expected**: 
- `cbm_last_year` should be ~6,000 for August 2025 dates (matching August 2024)
- `cbm_last_year` should be ~4,500 for September 2025 dates (matching September 2024)
- If all values are zero, YoY lookup is failing

### 4. ⚠️ Feature Scaling

**Issue**: YoY features might be in original scale while model expects scaled values.

**Check**:
- [ ] Are YoY features being scaled before being passed to the model?
- [ ] The model was trained with scaled features, so inputs must also be scaled

**Note**: The scaler should be applied to the entire feature vector, including YoY features.

### 5. ⚠️ Lunar Date Matching

**Issue**: YoY lookup uses lunar date matching. If lunar dates are incorrect, lookup will fail.

**Check**:
- [ ] Verify lunar dates: 2025-08-15 should be Lunar Month 6-7 (not 5-6)
- [ ] Check that 2024-08-XX (with same lunar date) has data
- [ ] Lunar date matching window: 300-400 days ago for 1 year, 600-800 days for 2 years

### 6. ⚠️ Model Architecture

**Issue**: The model might not be using YoY features effectively even if they're present.

**Check**:
- [ ] Are `cbm_last_year` and `cbm_2_years_ago` in the feature columns?
- [ ] Check model training logs to see if YoY features had non-zero values during training
- [ ] Model might need more training epochs to learn YoY patterns

## Diagnostic Steps

### Quick Diagnostic Script

**EASIEST WAY**: Run the automated diagnostic script:

```bash
python diagnose_mooncake.py
```

This script will automatically check all the items below and provide a summary report.

### Manual Diagnostic Steps

If you prefer to run diagnostics manually:

#### Step 1: Verify Model Retraining

```bash
# Check model file timestamp
ls -la outputs/MOONCAKE/models/best_model.pth

# Check if training was done after fixes
# Model should be newer than MOONCAKE_PREDICTION_IMPROVEMENTS.md
```

#### Step 2: Check Historical Data

```python
# In Python, check historical data
import pandas as pd
from datetime import date

# Load historical data
ref_data = pd.read_csv("dataset/train/data_2024.csv")
mooncake_2024 = ref_data[ref_data['CATEGORY'] == 'MOONCAKE'].copy()
mooncake_2024['date'] = pd.to_datetime(mooncake_2024['ACTUALSHIPDATE']).dt.date

# Check August 2024
aug_2024 = mooncake_2024[
    (mooncake_2024['date'] >= date(2024, 8, 1)) & 
    (mooncake_2024['date'] <= date(2024, 8, 31))
]
print(f"August 2024 total: {aug_2024['Total CBM'].sum():.2f}")

# Check September 2024
sep_2024 = mooncake_2024[
    (mooncake_2024['date'] >= date(2024, 9, 1)) & 
    (mooncake_2024['date'] <= date(2024, 9, 30))
]
print(f"September 2024 total: {sep_2024['Total CBM'].sum():.2f}")
```

#### Step 3: Check YoY Feature Computation

The diagnostic script will test YoY feature computation. You can also check terminal output during prediction:

```python
# In src/predict/predictor.py, the code now prints sample YoY values
# Check terminal output for:
# - "Sample YoY values (date, cbm_last_year, cbm_2_years_ago):"
```

#### Step 4: Verify Feature Columns

```python
# Check if YoY features are in feature columns
import yaml
config = yaml.safe_load(open("config/config_MOONCAKE.yaml"))
feature_cols = config['data']['feature_cols']
print("cbm_last_year" in feature_cols)
print("cbm_2_years_ago" in feature_cols)
```

## Most Likely Causes

Based on the 89% underprediction, the most likely issues are:

1. **Model Not Retrained** (90% probability)
   - Old model doesn't know how to use YoY features
   - Solution: Retrain the model

2. **YoY Features Are Zero** (70% probability)
   - Historical data not available or lunar matching failing
   - Solution: Check historical data and lunar date matching

3. **Feature Scaling Issue** (30% probability)
   - YoY features not scaled correctly
   - Solution: Verify scaler is applied to all features

## Immediate Actions

1. **Retrain the model**:
   ```bash
   python mvp_train.py
   ```

2. **Check terminal output** during prediction for:
   - "Passing historical data (X rows) for YoY feature lookup"
   - "Sample YoY values" with non-zero values
   - Any warnings about missing historical data

3. **Verify historical data** contains 2024 MOONCAKE data with actual volumes

4. **Re-run predictions** after retraining:
   ```bash
   python mvp_predict.py
   ```

## Expected After Fixes

After retraining and fixing YoY features:
- August 2025: Should predict ~5,000-6,500 CBM (within 20% of actual)
- September 2025: Should predict ~4,000-5,000 CBM (within 20% of actual)

If predictions are still low after retraining, check:
1. Historical data availability
2. Lunar date matching accuracy
3. Feature scaling
4. Model training logs for YoY feature usage
