# Balanced Distribution Loss: Quick Reference

## What Changed?

**Problem:** Model systematically over-predicts by 20-40 CBM in Days 1-5 of each month due to "Peak-Defense" loss function bias.

**Solution:** Recalibrated loss function with two new mechanisms:
1. **Asymmetric Penalty**: Over-predictions penalized 2.5x more in non-peak periods
2. **Mean Error Constraint**: Forces monthly totals to align with historical averages

## Configuration Changes

### Before (Peak-Defense Mode)
```yaml
category_specific_params:
  DRY:
    use_dynamic_early_month_weight: true  # 100x penalty for Days 1-5
```

### After (Balanced Distribution Mode)
```yaml
category_specific_params:
  DRY:
    # Asymmetric Penalty
    use_asymmetric_penalty: true
    over_pred_penalty: 2.5              # 2.5x penalty for over-prediction
    under_pred_penalty: 1.0             # 1.0x penalty for under-prediction
    
    # Mean Error Constraint
    apply_mean_error_constraint: true
    mean_error_weight: 0.15             # 0.15 weight for mean error constraint
    
    # Works with existing dynamic early month weighting
    use_dynamic_early_month_weight: true
```

## Key Metrics

### Validation Results
- ✅ **Test 1**: Asymmetric penalty ratio = 1.86x (expected: 1.8-2.7x)
- ✅ **Test 2**: Peak protection maintained (spike detection still active)
- ✅ **Test 3**: Mean error constraint working (15% penalty increase for bias)
- ✅ **Test 4**: Combined mode = 96.4% penalty increase over Peak-Defense
- ✅ **Test 5**: Days 1-5 penalty = 250x for over-prediction (100x * 2.5)

### Expected Improvements
- **Days 1-5 error**: Reduce from 30-40% to <20%
- **Monthly totals**: Align within ±5% of historical averages
- **Systematic bias**: Eliminate upward drift (+18 CBM → near 0)

## Usage

### Training
```bash
python mvp_train.py --category DRY --config config/config_DRY.yaml
```

### Validation
```bash
python validate_balanced_distribution_loss.py
# Expected: 5/5 tests passed
```

### Evaluation
```python
# Compare predictions vs actuals for Jan-Feb 2025
# Check: Days 1-5 error, monthly totals, mean bias
```

## Parameter Tuning

### If model under-predicts:
- **Reduce** `over_pred_penalty`: 2.5 → 2.0
- **Reduce** `mean_error_weight`: 0.15 → 0.10

### If upward bias persists:
- **Increase** `over_pred_penalty`: 2.5 → 3.0
- **Increase** `mean_error_weight`: 0.15 → 0.20

## Files Modified

1. **`src/utils/losses.py`** - Added asymmetric penalty & mean error constraint
2. **`mvp_train.py`** - Integrated new parameters into training pipeline
3. **`config/config_DRY.yaml`** - Enabled Balanced Distribution Mode
4. **`validate_balanced_distribution_loss.py`** - Validation script (NEW)
5. **`doc/BALANCED_DISTRIBUTION_LOSS_GUIDE.md`** - Full documentation (NEW)

## Backward Compatibility

✅ **Fully backward compatible** - defaults to Peak-Defense Mode if new parameters not specified

## Status

✅ **Implementation Complete**  
✅ **Validation Passed** (5/5 tests)  
✅ **Documentation Complete**  
🔄 **Ready for Training & Evaluation**

---

**Version:** 1.0  
**Date:** February 7, 2026  
**Implementation Time:** ~2 hours
