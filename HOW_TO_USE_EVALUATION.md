# How to Use evaluate_early_month.py

## Purpose

This script analyzes whether the **Early Month Over-prediction** fix is working by comparing predictions for:
- Days 1-3 (maximum 20x penalty)
- Days 4-10 (linear decay penalty)
- Days 11-31 (normal weighting)

## Steps

### Step 1: Re-train the Model

The training script now saves predictions to CSV automatically:

```powershell
python mvp_train.py --config config/config_DRY.yaml --category DRY
```

This will create: `outputs/DRY/test_predictions.csv`

### Step 2: Run the Evaluation

```powershell
python evaluate_early_month.py
```

Or specify a custom path:

```powershell
python evaluate_early_month.py outputs/DRY/test_predictions.csv
```

### Step 3: Interpret the Results

The script will show:

```
[DAYS 1-3 (20x Penalty)]
  MAE: XXX CBM
  Avg error (pred - actual): +XX CBM  ← Should be close to 0!
  Over-prediction rate: XX%  ← Should be < 70%

[DAYS 4-10 (Linear Decay Penalty)]
  MAE: XXX CBM
  ...

[EARLY MONTH (Days 1-10)]
  MAPE: XX%  ← Should be < 30%
  ...

[REST OF MONTH (Days 11-31)]
  MAPE: XX%
  ...

SUCCESS CRITERIA:
  ✓ or ✗ Early month MAPE < 30%
  ✓ or ✗ Over-prediction rate < 70%
  ✓ or ✗ Avg error < 100 CBM
  ✓ or ✗ Days 1-3 MAE < 300 CBM
```

## What to Look For

### ✅ Success Indicators:
- Early month MAPE similar to or better than rest of month
- Over-prediction rate around 50-60% (not 80-90%)
- Average error close to 0 (not +100 or +200 CBM)
- Days 1-3 have low MAE (the 20x penalty is working)

### ⚠️ Warning Signs:
- Early month MAPE >> rest of month MAPE
- Over-prediction rate > 80%
- Average error > +100 CBM (systematic over-prediction)
- Days 1-3 MAE very high (penalty not effective)

## If Results Are Not Good

### Option 1: Train Longer
Edit `config/config_DRY.yaml`:
```yaml
training:
  epochs: 50  # Increase from 20 to 50 or 100
```

### Option 2: Adjust Penalty
Edit `src/utils/losses.py`, line ~40:
```python
# Reduce maximum penalty from 20.0 to 15.0 or 12.0
mask_days_1_3 = (day_of_month >= 1) & (day_of_month <= 3)
weights = torch.where(mask_days_1_3, torch.tensor(15.0), weights)  # Reduced
```

### Option 3: Check Feature Usage
Verify in training log:
```
- days_from_month_start feature found at index 19 (for dynamic early month weighting)
- post_peak_signal in feature list
```

## Example Output

```
================================================================================
EARLY MONTH OVER-PREDICTION EVALUATION
================================================================================
Data file: outputs/DRY/test_predictions.csv
Total predictions: 2000

[DAYS 1-3 (20x Penalty)]
  Samples: 300
  MAE: 180.45 CBM
  Over-prediction rate: 55.0%
  Avg error (pred - actual): +25.30 CBM ✓ OK

[EARLY MONTH (Days 1-10)]
  Samples: 1000
  MAPE: 22.50%
  Over-prediction rate: 58.0%
  Avg error (pred - actual): +35.20 CBM ✓ OK

[REST OF MONTH (Days 11-31)]
  Samples: 1000
  MAPE: 20.15%
  Over-prediction rate: 52.0%

================================================================================
SUCCESS CRITERIA
================================================================================
  ✓ Early month MAPE < 30%: 22.50% (PASS)
  ✓ Over-prediction rate < 70%: 58.0% (PASS)
  ✓ Avg error < 100 CBM: +35.20 CBM (PASS)
  ✓ Days 1-3 MAE < 300 CBM: 180.45 CBM (PASS)

  🎉 SUCCESS: Early month over-prediction issue appears to be FIXED!
```

## Quick Start

```powershell
# 1. Train model (this now saves CSV automatically)
python mvp_train.py --config config/config_DRY.yaml --category DRY

# 2. Evaluate early month performance
python evaluate_early_month.py

# 3. Check if all criteria pass
```

That's it! The script will tell you if the fix is working or if you need to adjust parameters.
