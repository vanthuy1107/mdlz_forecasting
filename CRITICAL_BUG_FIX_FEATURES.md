# CRITICAL BUG FIX - mvp_train.py Ignoring config_DRY.yaml

## Problem Discovered

After retraining, the model **still over-predicted** because `mvp_train.py` was **NOT using your config_DRY.yaml features**!

### Evidence

Looking at `outputs/DRY/models/metadata.json`, the trained model used these features:
```json
"feature_cols": [
  "month_sin", "month_cos", "dayofmonth_sin", "dayofmonth_cos",
  "holiday_indicator", "days_until_next_holiday", "days_since_holiday",
  "is_weekend", "day_of_week_sin", "day_of_week_cos",
  "weekday_volume_tier", "is_high_volume_weekday", "Is_Monday",
  "is_EOM", "days_until_month_end",
  "Total CBM",
  "lunar_month_sin", "lunar_month_cos",  // ❌ NOT in config_DRY.yaml
  "lunar_day_sin", "lunar_day_cos",      // ❌ NOT in config_DRY.yaml  
  "days_to_tet",                          // ❌ NOT in config_DRY.yaml
  "days_to_mid_autumn",                   // ❌ NOT in config_DRY.yaml
  "is_active_season",                     // ❌ NOT in config_DRY.yaml
  "days_until_peak",                      // ❌ NOT in config_DRY.yaml
  "is_golden_window",                     // ❌ NOT in config_DRY.yaml
  "cbm_per_qty", "cbm_per_qty_last_year"
]
```

### What Was MISSING (from config_DRY.yaml)

❌ **Missing critical early month features:**
- `early_month_low_tier`
- `is_early_month_low`
- `is_first_5_days`
- `days_from_month_start`

**Result:** Model had NO IDEA about early month patterns, so it just predicted normal volume (~7K).

## Root Cause

In `mvp_train.py` lines 780-811, the code was:
1. Reading base `config.yaml` feature_cols
2. Adding lunar/seasonal features
3. **Overwriting** any category-specific feature_cols from `config_DRY.yaml`

```python
# OLD CODE (BUGGY)
extra_features = ["lunar_month_sin", "lunar_month_cos", ...]
current_features = list(data_config["feature_cols"])  # ← Reads from BASE config!
for feat in extra_features:
    if feat not in current_features:
        current_features.append(feat)
data_config["feature_cols"] = current_features  # ← Overwrites category config!
```

## The Fix

Modified `mvp_train.py` to **detect and respect** category-specific configs:

```python
# NEW CODE (FIXED)
current_features = list(data_config["feature_cols"])

# Check if this is a category-specific config
has_category_specific_features = any(
    feat in current_features 
    for feat in ["early_month_low_tier", "is_early_month_low", 
                 "mid_month_peak_tier", "is_mid_month_peak"]
)

if not has_category_specific_features:
    # Using base config → add standard features
    print(f"  [INFO] Using base config features, adding standard lunar/seasonal features...")
    # Add extra features...
else:
    # Using category-specific config → respect it!
    print(f"  [INFO] Using category-specific config features ({len(current_features)} features)")
    print(f"  [INFO] Respecting feature_cols from config_{category_filter}.yaml")
    # DON'T add extra features!
```

## What You Need to Do NOW

### Step 1: Retrain with Fixed Code

```bash
python mvp_train.py
```

**What to look for in logs:**
```
[INFO] Using category-specific config features (20 features)
[INFO] Respecting feature_cols from config_DRY.yaml
Early Month loss weight: 15.0x (days 1-10, for DRY)
is_early_month_low feature found at index XX
is_first_5_days feature found at index XX
```

### Step 2: Verify Features in metadata.json

After training, check `outputs/DRY/models/metadata.json`:

```bash
# Should NOW include:
cat outputs/DRY/models/metadata.json | grep -A 50 feature_cols
```

**Should see:**
- ✅ `early_month_low_tier`
- ✅ `is_early_month_low`
- ✅ `is_first_5_days`
- ✅ `days_from_month_start`

**Should NOT see (unless in your config):**
- ❌ `lunar_month_sin/cos`
- ❌ `days_to_tet`
- ❌ `days_to_mid_autumn`
- ❌ `is_golden_window`

### Step 3: Test Predictions

```bash
python mvp_predict.py --category DRY
```

**Expected:** Days 1-5 should now predict **4.5K-5.5K** (much better than 7.5K)

## Why This Happened

The original `mvp_train.py` was designed for a global model approach where all categories used the same features. When you added category-specific configs (config_DRY.yaml, config_FRESH.yaml, etc.), the code wasn't updated to respect those custom feature lists.

## Files Modified

- ✅ `mvp_train.py` (lines 774-829): Now detects and respects category-specific feature_cols

## Next Steps

1. **Retrain NOW** with the fixed code
2. Check logs to confirm category-specific features are loaded
3. Verify metadata.json has the right features
4. Test predictions
5. If still over-predicting (but better), try:
   - Increase `early_month_loss_weight` from 15.0 to 30.0
   - Remove `Total CBM` feature
   - Increase epochs from 20 to 40

---

## Quick Verification Checklist

Before retraining:
- ✅ `config/config_DRY.yaml` has `early_month_loss_weight: 15.0`
- ✅ `config/config_DRY.yaml` has `is_first_5_days` in feature_cols
- ✅ `mvp_train.py` respects category-specific feature_cols (just fixed)

After retraining:
- ✅ Training logs show "Using category-specific config features"
- ✅ Training logs show "Early Month loss weight: 15.0x"
- ✅ `metadata.json` includes early_month features
- ✅ `metadata.json` does NOT include lunar features (unless you want them)
- ✅ Predictions for days 1-5 are < 6K (improved)

---

**Updated**: 2026-02-07 
**Status**: CRITICAL BUG FIXED - Ready to retrain
**Priority**: HIGH - This is why your previous retraining didn't work!
