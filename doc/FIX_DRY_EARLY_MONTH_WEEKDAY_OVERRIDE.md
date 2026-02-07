# FIX: DRY Category 4x Over-Prediction on Early-Month High-Volume Weekdays

**Issue Date:** 2026-02-07  
**Affected Category:** DRY  
**Problem:** Predictions are 4x actual on Oct 1-5, 2025 (and similar early-month Wed/Fri dates)

---

## ROOT CAUSE ANALYSIS

### The Problem
DRY category predictions are **4x higher than actual** on early-month high-volume weekdays:
- **Oct 1, 2025 (Wednesday)** - HIGH VOLUME DAY
- **Oct 3, 2025 (Friday)** - HIGH VOLUME DAY

### Why This Happens
The model has two competing signals:

**Signal 1: Weekday Boost** (Strong ✓)
- Wednesday/Friday have 4x loss weight during training
- Model learns: "Wed/Fri = 4x higher volume"

**Signal 2: Early Month Suppression** (Weak ✗)
- Days 1-5 have 30x penalty weight during training
- `is_high_vol_weekday_AND_early_month = -2` (suppression feature)
- Model learns: "Early month = low volume"

**The Conflict:**
When Oct 1 (Wed, day 1) arrives:
- Weekday boost says: "Predict 4x higher!" 
- Early month suppression says: "Predict low!"
- **Weekday boost WINS** → 4x over-prediction

### The Lunar Calendar Connection
The anchor `2025: Oct 6 = Lunar 08-15` (Mid-Autumn Festival) is **NOT directly causing** this issue:
- Oct 1-5, 2025 = Lunar 08-10 to 08-14 (days before Mid-Autumn)
- DRY is non-seasonal and shouldn't use lunar features
- The real issue is the **weekday vs early-month conflict**

---

## THE FIX

### Strategy: Strengthen Early-Month Suppression (Retrain Required)

We need to make the early-month suppression **stronger** than the weekday boost:

### Change 1: Strengthen Interaction Feature (preprocessing.py)
**File:** `src/data/preprocessing.py` lines 805-811

**Before:**
```python
df['is_high_vol_weekday_AND_early_month'] = 0
df.loc[... & (day_of_month >= 6) & (day_of_month <= 10), ...] = -1  # Days 6-10
df.loc[... & (day_of_month <= 5), ...] = -2  # Days 1-5
```

**After:**
```python
df['is_high_vol_weekday_AND_early_month'] = 0
df.loc[... & (day_of_month >= 6) & (day_of_month <= 10), ...] = -2  # Days 6-10 (INCREASED from -1)
df.loc[... & (day_of_month <= 5), ...] = -5  # Days 1-5 (INCREASED from -2)
```

**Impact:** The -5 signal now **dominates** the 4x weekday boost during early month.

---

### Change 2: Increase Early Month Penalty Weight (config_DRY.yaml)
**File:** `config/config_DRY.yaml` line 73

**Before:**
```yaml
dynamic_early_month_base_weight: 30.0  # Days 1-5 penalty
```

**After:**
```yaml
dynamic_early_month_base_weight: 50.0  # Days 1-5 penalty (INCREASED from 30x to 50x)
```

**Impact:** During training, early-month days get 50x loss weight (vs 4x for weekdays), making the model **prioritize early-month suppression over weekday boost**.

---

## WHY THIS IS THE BEST SOLUTION

### ✓ Advantages of Combined Approach (Option 1 + Option 2)
1. **Addresses Root Cause:** Retrains the model to learn correct patterns
2. **Generalizable:** Fixes the issue for ALL months, not just October 2025
3. **Maintains Model Quality:** Uses proper training signals, not hard-coded rules
4. **Preserves Weekday Patterns:** Wed/Fri still show higher volume in mid/late month
5. **Clean Code:** No technical debt or hard-coded business logic

### ✗ Why Option 4 (Hard-Coded Suppression) is BAD
```python
# DON'T DO THIS:
if category == "DRY" and pred_date.day <= 5 and pred_date.weekday() in [0, 2, 4]:
    pred_value = pred_value * 0.25  # Band-aid fix
```

**Problems:**
- ❌ Doesn't fix the model's learned behavior
- ❌ Creates technical debt (hard-coded logic)
- ❌ Only fixes prediction, not training
- ❌ Difficult to maintain and tune
- ❌ Doesn't scale to other categories/patterns

---

## NEXT STEPS

### 1. Retrain DRY Model ⚠️ REQUIRED
```bash
python train.py --config config/config_DRY.yaml --category DRY
```

**Why:** The changes only affect **training**, not prediction. You must retrain to see the fix.

### 2. Validate on Oct 2025
After retraining, run prediction for Oct 2025:
```bash
python mvp_predict.py --start 2025-10-01 --end 2025-10-31
```

**Expected Result:**
- Oct 1 (Wed) and Oct 3 (Fri) should now predict **~1x actual** (not 4x)
- Days 1-5 remain low volume, even on high-volume weekdays
- Days 11+ show normal Wed/Fri weekday boost

### 3. Cross-Validate on Other Months
Check that the fix doesn't break other months:
- Jan 2025 (days 1-5)
- Feb 2025 (days 1-5)
- Mar 2025 (days 1-5)

---

## TECHNICAL DETAILS

### Feature Engineering Logic
The interaction feature `is_high_vol_weekday_AND_early_month` explicitly tells the model:
- "This is a Monday/Wednesday/Friday, BUT it's early month, so SUPPRESS the weekday boost"
- Value = -5 (days 1-5), -2 (days 6-10), 0 (other days)
- This creates a **strong negative signal** that overrides the weekday pattern

### Loss Weighting Logic
During training, the loss function applies:
- Early month (days 1-5): 50x penalty for over-predictions
- Wednesday/Friday: 4x penalty for under-predictions
- **Result:** 50x > 4x, so early-month suppression dominates

### Prediction Behavior
After retraining, when predicting Oct 1, 2025 (Wed, day 1):
1. Model sees: `is_high_volume_weekday = 1` → wants to boost
2. Model sees: `is_first_5_days = 1` → wants to suppress
3. Model sees: `is_high_vol_weekday_AND_early_month = -5` → **strong suppress signal**
4. Model sees: `early_month_low_tier = -10` → **extreme low volume**
5. **Combined effect:** Early-month suppression WINS → predicts low volume ✓

---

## MONITORING

After deploying the retrained model, monitor these metrics:

### Success Criteria
- ✓ Oct 1-5, 2025: Predictions ≤ 1.5x actual (acceptable error)
- ✓ Oct 1/3 (Wed/Fri): No 4x over-prediction
- ✓ Oct 11+ (Wed/Fri): Normal weekday boost maintained
- ✓ Monthly total: Within ±10% of actual

### Failure Indicators
- ✗ Early-month Wed/Fri still 3-4x over-predicted → Increase to 70x or -7
- ✗ Mid-month Wed/Fri now under-predicted → Reduce to 40x or -4
- ✗ Overall monthly volume drops significantly → Adjust mean_error_weight

---

## APPENDIX: Why the Lunar Anchor is Mentioned

The line `2025: Oct 6 = Lunar 08-15` appears in the code because:
1. It's used by MOONCAKE category for seasonal masking
2. It's also used by `_solar_to_lunar_date()` function for lunar calendar conversion
3. **For DRY:** This anchor is NOT causing the issue (DRY shouldn't use lunar features)
4. **For MOONCAKE:** Oct 6 is Mid-Autumn Festival, so Oct 1-5 would be pre-peak days

The 4x over-prediction issue is **specific to DRY category** and caused by:
- Weekday pattern override (not lunar calendar)
- Insufficient early-month suppression signals
- Training loss weight imbalance (30x early-month vs 4x weekday)

---

**Author:** AI Assistant  
**Date:** 2026-02-07  
**Version:** 1.0  
**Status:** Changes implemented, retraining required
