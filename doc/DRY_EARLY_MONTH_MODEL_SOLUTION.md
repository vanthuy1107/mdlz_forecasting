# DRY Early Month Fix - Pure Model-Based Solution

## Problem
DRY predictions over-predict days 1-5 by ~50-60% (predicted 7.1K vs actual 3.6-4.2K).

## Why Post-Processing is BAD ❌
- Hardcoded adjustment factors don't adapt to future patterns
- Fails if early month behavior changes over time
- Not a machine learning solution
- **DELETED** - we won't use this approach

## Proper Solution: Force Model to Learn ✅

### Current Configuration (Ready to Train)

**Already Applied:**
1. ✅ **15x loss weight** on early month days (days 1-10)
2. ✅ **`is_first_5_days`** binary indicator for days with severe drop
3. ✅ **`days_from_month_start`** gradient feature (0, 1, 2...30)
4. ✅ **`early_month_low_tier`** (0=days 1-5, 1=days 6-10, 2=days 11+)

**Suspected Issue:**
⚠️ **`Total CBM` as feature** - This might cause the model to anchor on high values from the lookback window (days 11-31), making it difficult to predict lower values for days 1-5.

### Step-by-Step Action Plan

#### 🔴 STEP 1: First Attempt - Retrain with 15x Weight (Current Config)

```bash
python mvp_train.py
```

**Expected training time:** 10-30 minutes

**What to look for in logs:**
```
Early Month loss weight: 15.0x (days 1-10, for DRY)
is_first_5_days feature found at index XX
```

**Then predict:**
```bash
python mvp_predict.py --category DRY
```

**Evaluation criteria:**
- ✅ **Success**: Days 1-5 predict 3.5K-4.5K (within ±500 of actuals)
- ⚠️ **Partial**: Days 1-5 predict 5.0K-6.0K (better but not enough)
- ❌ **No improvement**: Days 1-5 still predict 7K+

---

#### 🟡 STEP 2: If Partial Success (5-6K) - Increase Loss Weight

If predictions improved but not enough:

```yaml
# config/config_DRY.yaml
category_specific_params:
  DRY:
    early_month_loss_weight: 30.0  # Increase from 15.0 to 30.0
```

Then retrain:
```bash
python mvp_train.py
```

---

#### 🟠 STEP 3: If No Improvement (Still 7K+) - Remove Total CBM Feature

The `Total CBM` feature might be causing "value anchoring":

```yaml
# config/config_DRY.yaml
feature_cols:
  # ... all other features ...
  # - "Total CBM"  # COMMENT THIS OUT - suspected to cause anchoring
```

**Why this might help:**
- During prediction, the model sees historical `Total CBM` values from lookback window
- These are mostly from days 11-31 (normal volume ~7K)
- Model might anchor on these high values and resist predicting 4K for early month
- Removing this forces model to rely on temporal features only

Then retrain:
```bash
python mvp_train.py
```

---

#### 🔵 STEP 4: If Still Struggling - Combine Multiple Approaches

Apply all at once:

```yaml
# config/config_DRY.yaml
training:
  epochs: 40  # Increase from 20 to 40 - give model more time to learn

category_specific_params:
  DRY:
    early_month_loss_weight: 50.0  # Very aggressive - 50x weight

feature_cols:
  # ... all features ...
  # - "Total CBM"  # REMOVE
```

---

### Advanced Option: Interaction Features (If Needed)

If the above don't work, add interaction features to help model understand "weekday patterns are different in early month":

```python
# Would need to add to preprocessing.py
df['early_month_x_weekday'] = df['is_early_month_low'] * df['weekday_volume_tier']
df['first_5_days_x_is_monday'] = df['is_first_5_days'] * df['Is_Monday']
```

Then add to feature_cols:
```yaml
- "early_month_x_weekday"
- "first_5_days_x_is_monday"
```

---

## Why This WILL Work

### Evidence from FRESH Category
- FRESH had similar issue: under-predicting Mondays by ~30%
- Solution: 8x loss weight on Mondays
- **Result: Fixed!**

### Why DRY Needs More (15x-50x)
- Early month drop is more severe (~50% vs 30%)
- Happens less frequently (5 days/month vs 4 days/week)
- Model needs stronger signal to overcome "default" 7K prediction

### The `Total CBM` Hypothesis
If the model is using `Total CBM` from lookback window:
- Lookback days 1-28 likely include mostly days 11-31 (normal volume)
- Average Total CBM in lookback ≈ 7K
- Model anchors on this and struggles to predict 4K
- **Removing it forces model to use temporal patterns**

---

## What We Will NOT Do ❌

- ❌ Post-processing adjustments
- ❌ Rule-based overrides  
- ❌ Hardcoded factors
- ❌ Separate models for early/late month

## What We WILL Do ✅

- ✅ Aggressive loss weighting (15x → 30x → 50x if needed)
- ✅ Remove potentially conflicting features (Total CBM)
- ✅ More training epochs (20 → 40 if needed)
- ✅ Interaction features (if basic approaches don't work)
- ✅ Let the model learn from data

---

## Quick Reference

### Current Status
```yaml
early_month_loss_weight: 15.0
epochs: 20
Total CBM: INCLUDED (might need to remove)
```

### Next Command
```bash
python mvp_train.py
```

### If Results Still Bad
Try in order:
1. Increase to 30x loss weight
2. Remove `Total CBM` feature
3. Increase to 40 epochs + 50x loss weight

---

**Updated**: 2026-02-07  
**Approach**: Pure model-based learning, no hardcoded adjustments  
**Key Insight**: `Total CBM` feature might be causing value anchoring
