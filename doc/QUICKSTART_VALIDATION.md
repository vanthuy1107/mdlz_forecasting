# Quick Start: Validating Early Month Hard Reset Fixes

## Overview
This guide helps you quickly validate and test the three root cause fixes for early-month over-prediction.

---

## Step 1: Run Validation Script

```bash
python validate_early_month_fixes.py
```

**Expected Output:**
```
================================================================================
VALIDATION REPORT: Early Month Hard Reset Fixes
================================================================================

✅ FIX #1 PASSED: All penalty features are correctly computed for early month days
✅ FIX #2 PASSED: Penalty features are preserved in feature list, separate from scaled target
✅ FIX #3 PASSED: Penalty signals are strong enough to counteract EOM momentum
✅ CODE VALIDATION PASSED: Fixes are implemented in predictor.py

🎉 ALL VALIDATIONS PASSED! The fixes are ready for testing.
```

**If Validation Fails:**
- Check that all files are up to date (especially `src/predict/predictor.py`)
- Review error messages for specific issues
- See `EARLY_MONTH_HARD_RESET_FIXES.md` for detailed fix descriptions

---

## Step 2: Test Predictions (Optional - Requires Trained Model)

### Option A: Use Existing Model

If you have a trained model for DRY category:

```bash
python mvp_predict.py --config config/config_DRY.yaml --start-date 2025-01-01 --end-date 2025-02-28
```

**What to Check:**
- Day 1 predictions should be significantly lower than before
- Look for a clear "Hard Reset" pattern at the start of each month
- Compare predictions vs actuals (if available)

### Option B: Retrain and Predict

For best results, retrain the model to learn from the corrected feature dynamics:

```bash
# 1. Train model with fixes
python train_by_brand.py --category DRY

# 2. Run predictions
python mvp_predict.py --config config/config_DRY.yaml --start-date 2025-01-01 --end-date 2025-02-28
```

---

## Step 3: Evaluate Results

### Visual Inspection

Create a plot comparing predictions vs actuals:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
predictions = pd.read_csv('predictions_DRY.csv')  # Adjust path as needed
predictions['date'] = pd.to_datetime(predictions['date'])

# Load actuals (if available)
actuals = pd.read_csv('actuals_DRY.csv')  # Adjust path as needed
actuals['date'] = pd.to_datetime(actuals['date'])

# Merge
df = predictions.merge(actuals, on='date', how='left')

# Plot
plt.figure(figsize=(15, 6))
plt.plot(df['date'], df['predicted'], label='Predicted', linewidth=2)
plt.plot(df['date'], df['actual'], label='Actual', linewidth=2, alpha=0.7)

# Highlight Days 1-5 of each month
for month_start in pd.date_range('2025-01-01', '2025-02-28', freq='MS'):
    plt.axvspan(month_start, month_start + pd.Timedelta(days=4), 
                alpha=0.2, color='yellow', label='Days 1-5' if month_start == pd.Timestamp('2025-01-01') else '')

plt.xlabel('Date')
plt.ylabel('Total CBM')
plt.title('DRY Category: Predictions vs Actuals (Days 1-5 Highlighted)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('early_month_fix_validation.png', dpi=300)
plt.show()
```

### Quantitative Metrics

Calculate early-month MAPE:

```python
import numpy as np

# Filter to Days 1-5 of each month
df['day_of_month'] = df['date'].dt.day
early_month = df[df['day_of_month'] <= 5].copy()

# Calculate MAPE
early_month['ape'] = np.abs((early_month['predicted'] - early_month['actual']) / early_month['actual']) * 100
early_month_mape = early_month['ape'].mean()

print(f"Early Month MAPE (Days 1-5): {early_month_mape:.2f}%")
print(f"Target: <20%")

# Calculate by day
for day in range(1, 6):
    day_data = early_month[early_month['day_of_month'] == day]
    day_mape = day_data['ape'].mean()
    print(f"  Day {day} MAPE: {day_mape:.2f}%")
```

**Success Criteria:**
- Early Month MAPE < 20% (down from 50-100% before fixes)
- Day 1 MAPE < 20%
- Consistent improvement across Days 1-5

---

## Step 4: Compare Before vs After

### Before Fixes (Expected Behavior)
- **Pattern:** Gradual decay from EOM momentum
- **Day 1:** 50-100% over-prediction
- **Days 2-4:** 30-60% over-prediction
- **Root Cause:** Penalty features not dynamically updated

### After Fixes (Target Behavior)
- **Pattern:** Sharp "Hard Reset" at Day 1
- **Day 1:** <20% error
- **Days 2-4:** <15% error
- **Root Cause:** Fixed via dynamic feature updates + LSTM state reset

---

## Troubleshooting

### Issue: Validation Script Fails

**Solution:**
1. Check that you're in the project root directory
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify that `src/predict/predictor.py` has been updated with fixes

### Issue: Predictions Still Too High

**Possible Causes:**
1. Model not retrained with fixes → **Retrain model**
2. Loss weight too low → **Increase `early_month_loss_weight` in config**
3. Penalty features not strong enough → **Review feature values in preprocessing.py**

### Issue: Predictions Now Too Low

**Possible Causes:**
1. Penalty features too aggressive → **Reduce `early_month_low_tier` from -10 to -5**
2. LSTM state reset too strong → **Reduce amplification zone from 3 days to 1 day**
3. Loss weight too high → **Reduce `early_month_loss_weight` in config**

---

## Key Files Reference

- **Fixes:** `src/predict/predictor.py` (lines 595-687)
- **Validation:** `validate_early_month_fixes.py`
- **Documentation:** `EARLY_MONTH_HARD_RESET_FIXES.md`
- **Config:** `config/config_DRY.yaml`
- **Preprocessing:** `src/data/preprocessing.py` (lines 717-813)

---

## Next Steps After Validation

1. ✅ **Validation Passed** → Proceed to Step 2 (Test Predictions)
2. ✅ **Predictions Improved** → Deploy to production
3. ✅ **Monitor Results** → Track early-month MAPE over time
4. ❌ **Validation Failed** → Review fixes, check code implementation
5. ❌ **Predictions Not Improved** → Consider retraining or adjusting parameters

---

**Quick Command Summary:**

```bash
# 1. Validate fixes
python validate_early_month_fixes.py

# 2. Retrain model (optional but recommended)
python train_by_brand.py --category DRY

# 3. Generate predictions
python mvp_predict.py --config config/config_DRY.yaml --start-date 2025-01-01 --end-date 2025-02-28

# 4. Evaluate results (see Python code above)
```

---

**Status Checklist:**

- [ ] Validation script passed
- [ ] Predictions generated for test period
- [ ] Early month MAPE calculated
- [ ] Visual inspection shows "Hard Reset"
- [ ] Results documented

---

For detailed technical explanations, see `EARLY_MONTH_HARD_RESET_FIXES.md`.
