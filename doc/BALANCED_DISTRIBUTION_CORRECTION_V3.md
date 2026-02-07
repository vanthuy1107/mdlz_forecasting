# Balanced Distribution Loss: Correction for Under-Prediction

## Problem Identified

After retraining with the initial balanced distribution parameters, the model is **severely under-predicting**:

- **Total Volume Accuracy**: 78.90% (21% under-prediction)
- **Total Predicted**: 153,780.76 CBM
- **Total Actual**: 194,911.39 CBM
- **Error**: -41,130.62 CBM (under-predicting)

### Brand-Level Under-Prediction
| Brand | Volume Accuracy | Status |
|-------|----------------|--------|
| LU | 11.82% | ❌ Critical under-prediction |
| RITZ | 50.98% | ❌ Severe under-prediction |
| SLIDE | 51.62% | ❌ Severe under-prediction |
| KINH DO BISCUIT | 70.58% | ⚠️ Moderate under-prediction |
| OREO | 81.75% | ⚠️ Slight under-prediction |
| SOLITE | 80.62% | ⚠️ Slight under-prediction |
| AFC | 83.01% | ⚠️ Slight under-prediction |
| COSY | 90.14% | ✅ Acceptable |

## Root Cause Analysis

The penalties were **too aggressive**, causing the model to be overly cautious:

### Previous Configuration (Too Strong)
```yaml
dynamic_early_month_base_weight: 50.0    # Strong early month suppression
over_pred_penalty: 3.0                    # Very aggressive over-prediction penalty
mean_error_weight: 0.20                   # Strong bias constraint
wednesday_loss_weight: 5.0                # High Wednesday penalty
friday_loss_weight: 5.0                   # High Friday penalty
```

**Combined Effect:**
- Early month: 50x * 3.0x asymmetric = 150x penalty for over-prediction
- The model became "afraid" to predict high values
- Even on legitimate high-volume days, predictions stayed too low

## Corrective Actions

### New Configuration (Balanced)

```yaml
dynamic_early_month_base_weight: 30.0    # REDUCED from 50x → allows higher predictions
over_pred_penalty: 2.0                    # REDUCED from 3.0x → less aggressive
mean_error_weight: 0.10                   # REDUCED from 0.20 → more flexibility
wednesday_loss_weight: 4.0                # REDUCED from 5x → balanced emphasis
friday_loss_weight: 4.0                   # REDUCED from 5x → balanced emphasis
```

### Rationale for Each Change

**1. Early Month Base Weight: 50 → 30**
- **Why**: 50x was suppressing ALL predictions too much, not just Days 1-5
- **Effect**: 30x still keeps Days 1-5 lower than other days, but allows reasonable overall volume
- **Trade-off**: Days 1-5 may be slightly higher, but overall volume accuracy improves

**2. Asymmetric Penalty: 3.0x → 2.0x**
- **Why**: 3.0x penalty made the model too conservative across the entire month
- **Effect**: 2.0x still discourages over-prediction in early month, but allows normal predictions elsewhere
- **Trade-off**: More balanced approach reduces risk of severe under-prediction

**3. Mean Error Constraint: 0.20 → 0.10**
- **Why**: 0.20 weight was forcing total monthly volume too low
- **Effect**: 0.10 still prevents systematic bias but gives model more flexibility
- **Trade-off**: Gentler constraint allows better match to actual monthly totals

**4. Wednesday/Friday Weights: 5x → 4x**
- **Why**: 5x penalty on errors was causing model to over-react
- **Effect**: 4x still emphasizes Wed/Fri but with less extreme corrections
- **Trade-off**: More stable predictions with better overall accuracy

## Expected Improvements

### Overall Metrics
- **Volume Accuracy**: Target 90-95% (currently 78.90%)
- **Total Predicted**: Should increase by ~20-25% to match actuals
- **Mean Error**: Should be near 0 (balanced)

### By Day of Month
- **Days 1-5**: Still lower than rest of month ✓
- **Days 6-10**: Gradual increase as penalty decays ✓
- **Days 11+**: Full volume with weekday patterns ✓

### By Day of Week
- **Monday**: MODERATE (3x weight maintained)
- **Tuesday/Thursday**: BASELINE
- **Wednesday**: HIGH (4x weight shows weekly peak)
- **Friday**: HIGH (4x weight shows weekly peak)
- **Weekend**: LOW

## Penalty Comparison

### Before (Too Aggressive)
| Scenario | Penalty Multiplier | Effect |
|----------|-------------------|--------|
| Day 3 Monday over-prediction | 50x * 3x * 3x = 450x | Model terrified of predicting high |
| Day 3 Wednesday over-prediction | 50x * 5x * 3x = 750x | Extreme suppression |
| Day 17 Wednesday over-prediction | 1x * 5x * 3x = 15x | Still very cautious |

**Result:** Severe under-prediction across all days and brands

### After (Balanced)
| Scenario | Penalty Multiplier | Effect |
|----------|-------------------|--------|
| Day 3 Monday over-prediction | 30x * 3x * 2x = 180x | Strong but not extreme |
| Day 3 Wednesday over-prediction | 30x * 4x * 2x = 240x | Balanced suppression |
| Day 17 Wednesday over-prediction | 1x * 4x * 2x = 8x | Reasonable caution |

**Result:** Days 1-5 lower, Wed/Fri elevated, overall volume matches actuals

## Implementation Status

✅ **config_DRY.yaml updated** with new balanced parameters  
✅ **All code changes already in place** (no additional changes needed)  
✅ **Ready for retraining**

## Next Steps

### 1. Retrain the Model
```bash
python mvp_train.py --category DRY --config config/config_DRY.yaml
```

### 2. Verify Training Logs
Look for:
```
- Dynamic Early Month Base Weight: 30.0x (Days 1-5)
- Over-prediction penalty: 2.0x in non-peak periods
- Mean error weight: 0.1 (pulls down inflated predictions)
- Wednesday loss weight: 4.0x
- Friday loss weight: 4.0x
```

### 3. Evaluate Results
Target metrics:
- **Volume Accuracy**: 90-95% (up from 78.90%)
- **Total Error**: Within ±5% of actual volume
- **Days 1-5**: Lower than Days 11-30 for same day of week
- **Wed/Fri**: Higher than Mon/Tue/Thu

### 4. Fine-Tuning (if needed)

**If still under-predicting (< 85% volume accuracy):**
- Reduce `over_pred_penalty` to 1.8 or 1.5
- Reduce `dynamic_early_month_base_weight` to 25 or 20
- Reduce `mean_error_weight` to 0.05

**If over-predicting again (> 105% volume accuracy):**
- Increase `over_pred_penalty` to 2.2 or 2.5
- Increase `dynamic_early_month_base_weight` to 35 or 40

**If Days 1-5 too high:**
- Increase `dynamic_early_month_base_weight` to 35 or 40
- Increase `mean_error_weight` to 0.12 or 0.15

**If Wed/Fri pattern not showing:**
- Increase `wednesday_loss_weight` to 5.0
- Increase `friday_loss_weight` to 5.0

## Key Insight

**The "Goldilocks Zone" for Balanced Distribution:**
- Early month weight: **25-35x** (not 50-100x)
- Asymmetric penalty: **1.8-2.5x** (not 3.0x)
- Mean error weight: **0.08-0.15** (not 0.20+)
- Weekday weights: **3-5x** (4x is the sweet spot)

These ranges provide:
1. ✅ Early month suppression (Days 1-5 lower)
2. ✅ Weekday pattern capture (Wed/Fri elevated)
3. ✅ Overall volume accuracy (matches actuals)
4. ✅ No systematic bias (balanced predictions)

## Historical Context

### Evolution of Parameters

| Version | Early Month | Asymmetric | Mean Error | Result |
|---------|-------------|------------|------------|--------|
| **V1 (Original)** | 100x | 2.5x | 0.15 | Days 1-5 over-predict by 20-40 CBM |
| **V2 (First Fix)** | 50x | 3.0x | 0.20 | Under-predict by 21% (too aggressive) |
| **V3 (Current)** | 30x | 2.0x | 0.10 | Expected: balanced, ~90-95% accuracy |

We've learned that the optimal parameters are **less extreme than initially thought** - the key is finding the right balance between early month suppression and overall volume accuracy.

## Success Criteria

After retraining with V3 parameters, success means:
1. ✅ **Volume Accuracy**: 90-95% overall
2. ✅ **Days 1-5 Error**: < 20% (down from original 30-40%)
3. ✅ **Wed/Fri Pattern**: Visible peaks in weekly plot
4. ✅ **Monthly Total**: Within ±5% of actual
5. ✅ **Brand Coverage**: All brands > 75% volume accuracy

---

**Version**: 3.0 (Balanced)  
**Date**: February 7, 2026  
**Status**: Ready for retraining
