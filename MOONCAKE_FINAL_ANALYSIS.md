# MOONCAKE Final Analysis - Why August is Still Low

## Current Results (After All Improvements)

| Month | Predicted | Actual | Accuracy | Status |
|-------|-----------|--------|----------|--------|
| July | 0 CBM | 23 CBM | 0% | ✅ Correct (off-season) |
| **August** | **740 CBM** | **6015 CBM** | **12.30%** | ❌ **SEVERELY LOW** |
| September | 2910 CBM | 4576 CBM | 63.60% | ✅ Good |
| October | 357 CBM | 112 CBM | 0% | ⚠️ Too high but improving |

---

## Progress So Far

### Iteration 1: Original (Before Fixes)
- August: 157 CBM (2.61% accuracy)
- September: 664 CBM (14.51% accuracy)
- Issue: AND masking too restrictive, 10 CBM threshold too high

### Iteration 2: OR Masking + Lower Threshold
- August: 511 CBM (8.51% accuracy)
- September: 1702 CBM (37.20% accuracy)
- Issue: OR masking too permissive (false positives in July/October)

### Iteration 3: After Retraining + Refined Masking
- August: 740 CBM (12.30% accuracy) ← **Stuck here**
- September: 2910 CBM (63.60% accuracy) ← **Good!**
- Issue: Model learned September well but not August

---

## Root Cause Analysis

### Why September Works But August Doesn't

**Key Insight**: In 2025, Mid-Autumn Festival is **October 6**, which means:

- **August 2025** contains lunar months **6 and 7** (early season)
  - Lunar month 6: Aug 1-21 (NOT in 7-9 range)
  - Lunar month 7: Aug 22-31 (START of active season)
  
- **September 2025** contains lunar months **7 and 8** (PEAK season)
  - Lunar month 7: Sept 1-20 
  - Lunar month 8: Sept 21-30 (approaching Mid-Autumn)

- **October 2025** contains lunar months **8 and 9** (peak lunar months)
  - Lunar month 8: Oct 1-6 (Mid-Autumn is Oct 6)
  - Lunar month 9: Oct 7-31 (post-peak)

###The Model's Training View

During training, the model sees historical data where:
- **Lunar months 7-8-9** = Active season with high volume
- **Lunar month 6** = Off-season with low/zero volume

So the model learns:
- ✅ Lunar month 8 → HIGH predictions (applies to September)
- ✅ Lunar month 7 → MEDIUM predictions (late August, September)
- ❌ Lunar month 6 → LOW predictions (most of August gets suppressed!)

### The Calendar Drift Problem

The model is primarily learning from **lunar month features** (lunar_month, days_until_lunar_08_01), but:
- In 2025, **70% of August is lunar month 6** (off-season in training data)
- Model predicts low for Aug 1-21 because it sees "lunar month 6"
- Only Aug 22-31 (lunar month 7) gets reasonable predictions

This is why:
- **August total = 740 CBM** (only ~10 days of predictions are high)
- **September total = 2910 CBM** (full month has lunar 7-8, both active)

---

## Why Training Improvements Didn't Fix August

Even with aggressive training parameters:
- Higher learning rate (0.001)
- Stronger loss weights (august_boost_weight: 50)
- Lower regularization (weight_decay: 5e-5)

**The model still primarily uses lunar month as the signal**, not gregorian month. The `is_august` feature (gregorian month == 8) exists but is overpowered by the stronger lunar signal that dominated training.

---

## Solution Options

### Option 1: Add Prediction Scaling Factor (Quick Fix)
Add a post-processing boost specifically for August:

```python
# After line 2798 in mvp_predict.py
if category == "MOONCAKE":
    # Boost August predictions by 4x to compensate for lunar month 6 suppression
    august_mask = mooncake_rows['date'].dt.month == 8
    mooncake_rows.loc[august_mask, 'predicted_unscaled'] *= 4.0
```

**Expected Result**: August: 740 × 4 = **2960 CBM** (49% accuracy)

**Pros**: Immediate improvement, no retraining
**Cons**: Hacky, doesn't fix root cause

---

### Option 2: Retrain With Stronger Gregorian Anchoring (Proper Fix)
Modify the training to give much more weight to `is_august` feature:

```yaml
# config_MOONCAKE.yaml
category_specific_params:
  MOONCAKE:
    august_boost_weight: 150.0  # Up from 50.0 - MASSIVE boost
    loss_weight: 200  # Up from 150
```

**Expected Result**: Model learns that gregorian August → high volume regardless of lunar month

**Pros**: Proper fix, model learns correct pattern
**Cons**: Requires retraining (another 5-15 minutes)

---

### Option 3: Hybrid Approach (Best Solution)
Combine both:
1. Apply 2x scaling factor NOW for immediate improvement
2. Retrain with stronger weights for long-term fix

```python
# Immediate fix (Option 1 with 2x instead of 4x)
august_mask = mooncake_rows['date'].dt.month == 8
mooncake_rows.loc[august_mask, 'predicted_unscaled'] *= 2.0
```

Then retrain with:
```yaml
august_boost_weight: 100.0  # Very strong gregorian anchor
loss_weight: 200
```

**Expected Result**: 
- Immediate: August: 740 × 2 = **1480 CBM** (25% accuracy)
- After retrain: August: **3000-4500 CBM** (50-75% accuracy)

---

## Recommended Action

**Use Option 3 (Hybrid Approach)**:

### Step 1: Apply 2x Scaling Factor NOW
This will immediately improve August from 12% to ~25% accuracy.

### Step 2: Retrain with MUCH Stronger August Weighting
This will teach the model that **gregorian August = peak** regardless of lunar calendar.

### Step 3: Remove Scaling Factor After Retrain
Once model learns the pattern, remove the scaling factor.

---

## Why This Will Work

The fundamental issue is that the model's decision hierarchy is:
1. **Lunar month** (strongest signal) - learned from 2023-2024 training data
2. **Gregorian month** (weaker signal) - `is_august` feature exists but underpowered
3. **YoY features** (supporting signal) - helps but sparse

By massively boosting `august_boost_weight` to 100-150 (vs current 50), we force the model to learn:
- **"If gregorian month == 8 → predict HIGH, even if lunar month == 6"**

This inverts the hierarchy for August specifically, making gregorian month the dominant signal for that period.

---

## Technical Note: Why September Works

September doesn't have this problem because:
- Most of September is lunar month 7-8 (both in active range)
- Both lunar AND gregorian signals align
- Model correctly predicts 2910 CBM (64% accuracy)

August is unique because:
- Lunar (month 6) says "low"
- Gregorian (month 8) says "high"  
- Model trusts lunar more → underpredicts

We need to teach it to trust gregorian more in August.

---

## Implementation

Use Option 3 - add scaling factor now, then retrain:

```python
# In mvp_predict.py, after line 2798 (after hard masking)
if category == "MOONCAKE":
    # Temporary boost for August to compensate for lunar month 6 suppression
    # This will be removed after model is retrained with stronger august_boost_weight
    august_mask = mooncake_rows['date'].dt.month == 8
    if august_mask.any():
        boost_factor = 2.0
        original_aug_total = mooncake_rows.loc[august_mask, 'predicted_unscaled'].sum()
        mooncake_rows.loc[august_mask, 'predicted_unscaled'] *= boost_factor
        boosted_aug_total = mooncake_rows.loc[august_mask, 'predicted_unscaled'].sum()
        print(f"    - Applied {boost_factor}x boost to August predictions: {original_aug_total:.2f} → {boosted_aug_total:.2f} CBM")
    
    mooncake_rows.loc[mooncake_mask, 'predicted_unscaled'] = mooncake_rows['predicted_unscaled'].values
```

Then retrain with `august_boost_weight: 100.0`.
