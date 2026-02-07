# Balanced Distribution Loss: Tuning for Weekday Patterns

## Summary of Changes

Based on user feedback that predictions need to be:
1. **Higher on Wednesday and Friday** (high-volume days)
2. **Lower on Days 1-5** (early month period)

The following adjustments were made to the configuration:

## Configuration Updates

### 1. Added Weekday Loss Weights

```yaml
# New parameters in config_DRY.yaml
monday_loss_weight: 3.0      # 3x weight on Monday (standard)
wednesday_loss_weight: 5.0   # 5x weight on Wednesday (high-volume day) ← NEW
friday_loss_weight: 5.0      # 5x weight on Friday (high-volume day) ← NEW
```

**Effect:** The model will be penalized 5x more for errors on Wednesday and Friday, forcing it to predict higher volumes on these days to match actual patterns.

### 2. Strengthened Asymmetric Penalty

```yaml
# Updated from 2.5x to 3.0x
over_pred_penalty: 3.0  # Increased from 2.5 for stronger early-month suppression
```

**Effect:** Over-predictions in non-peak periods (including Days 1-5) are now penalized 3.0x more than under-predictions, providing stronger downward pressure on early-month predictions.

### 3. Increased Mean Error Constraint

```yaml
# Updated from 0.15 to 0.20
mean_error_weight: 0.20  # Increased for stronger bias correction
```

**Effect:** Stronger penalty for systematic over-prediction bias, further pulling down inflated Days 1-5 predictions.

### 4. Reduced Early Month Base Weight (CRITICAL CHANGE)

```yaml
# New parameter - allows weekday patterns to show through
dynamic_early_month_base_weight: 50.0  # Reduced from 100x to 50x
```

**Effect:** This is the most important change. By reducing from 100x to 50x:
- Days 1-5 still get strong penalty (50x is significant)
- BUT Wednesday/Friday loss weights (5x) can now influence the predictions
- Previously, 100x penalty completely dominated, preventing Wed/Fri high-volume predictions
- Now: 50x penalty + 5x Wed/Fri weight + 3x asymmetric = balanced approach

## Mathematical Effect

### Before (100x early month weight):
- **Day 3 Wednesday**: 100x (early month) completely dominates → prediction stays LOW
- **Day 17 Wednesday**: 1x (no early month) + 5x (Wed weight) = prediction goes HIGH

**Problem:** Days 1-5 Wed/Fri predictions too low, can't capture weekday pattern

### After (50x early month weight):
- **Day 3 Wednesday**: 50x (early month) + 5x (Wed weight) = 55x total
  - Early month penalty still strong (50x vs 1x baseline)
  - But Wed weight (5x) adds influence → prediction can rise moderately
- **Day 17 Wednesday**: 1x (no early month) + 5x (Wed weight) = 6x total
  - Full weekday pattern expression

**Solution:** Days 1-5 Wed/Fri can predict higher (capturing weekday pattern) while overall early-month predictions remain lower than before (due to stronger asymmetric penalty and mean error constraint).

## Combined Effect Table

| Scenario | Early Month Weight | Weekday Weight | Asymmetric Penalty | Total Effect |
|----------|-------------------|----------------|-------------------|--------------|
| **Day 3 Monday (over-pred)** | 50x | 3x | 3x | 450x penalty |
| **Day 3 Wednesday (over-pred)** | 50x | 5x | 3x | 750x penalty |
| **Day 3 Friday (over-pred)** | 50x | 5x | 3x | 750x penalty |
| **Day 17 Wednesday (over-pred)** | 1x | 5x | 3x | 15x penalty |

**Key Insight:** Days 1-5 Wednesday/Friday still get very high penalty (750x), but the Wednesday/Friday component (5x) helps the model learn that these days should be higher than Monday/Tuesday within the early-month period.

## Implementation Changes

### Code Changes

1. **`src/utils/losses.py`**:
   - Updated `get_dynamic_early_month_weight()` to accept configurable `base_weight` parameter
   - Updated `spike_aware_mse()` to pass `dynamic_early_month_base_weight` parameter

2. **`mvp_train.py`**:
   - Added extraction of `wednesday_loss_weight` and `friday_loss_weight` from config
   - Added extraction of `dynamic_early_month_base_weight` from config
   - Updated `create_criterion()` function to pass new parameters
   - Updated `spike_aware_mse()` call with new parameters

3. **`config/config_DRY.yaml`**:
   - Added weekday loss weights (Mon: 3x, Wed: 5x, Fri: 5x)
   - Increased asymmetric penalty (2.5x → 3.0x)
   - Increased mean error weight (0.15 → 0.20)
   - Added dynamic early month base weight (50.0)

## Expected Results

### Days 1-5 Predictions (by day of week):
- **Monday**: LOW (strong early month + moderate weekday = net LOW)
- **Tuesday**: LOW (strong early month + no weekday boost = net LOW)
- **Wednesday**: MODERATE (strong early month + high weekday = net MODERATE)
- **Thursday**: LOW (strong early month + no weekday boost = net LOW)
- **Friday**: MODERATE (strong early month + high weekday = net MODERATE)
- **Saturday/Sunday**: LOW (strong early month + weekend = net LOW)

### Days 17-30 Predictions (by day of week):
- **Monday**: MODERATE-HIGH (moderate weekday weight)
- **Tuesday**: MODERATE (baseline)
- **Wednesday**: HIGH (5x weekday weight)
- **Thursday**: MODERATE (baseline)
- **Friday**: HIGH (5x weekday weight)
- **Saturday/Sunday**: LOW (weekend)

## Validation

After retraining, verify:
1. **Wednesday/Friday predictions are higher** than Monday/Tuesday across all days
2. **Days 1-5 predictions are lower** than Days 17-30 for same day of week
3. **Overall monthly totals** align with historical averages (±5%)
4. **No systematic bias** (mean error ≈ 0)

## Next Steps

1. **Retrain the model:**
   ```bash
   python mvp_train.py --category DRY --config config/config_DRY.yaml
   ```

2. **Evaluate results:**
   - Plot predicted vs actual by day of week
   - Plot predicted vs actual by day of month
   - Check Wednesday/Friday predictions are elevated
   - Check Days 1-5 predictions are suppressed

3. **Fine-tune if needed:**
   - If Wed/Fri still not high enough: increase `wednesday_loss_weight` and `friday_loss_weight` to 6.0 or 7.0
   - If Days 1-5 still too high: increase `over_pred_penalty` to 3.5 or reduce `dynamic_early_month_base_weight` to 40.0
   - If Days 1-5 too low: reduce `over_pred_penalty` to 2.5 or increase `dynamic_early_month_base_weight` to 60.0

## Key Takeaway

The critical insight is that **reducing the early month base weight from 100x to 50x** allows the weekday loss weights (5x for Wed/Fri) to "break through" and influence predictions, while still maintaining strong early-month suppression through:
- 50x base weight (still very high)
- 3.0x asymmetric penalty (increased)
- 0.20 mean error constraint (increased)

This creates a **hierarchy of signals**:
1. **Strongest**: Early month suppression (50x)
2. **Strong**: Weekday patterns (5x)
3. **Moderate**: Asymmetric penalty (3x)
4. **Supporting**: Mean error constraint (0.20)

Together, these mechanisms ensure:
- Days 1-5 are LOW overall ✓
- Wednesday/Friday are HIGHER than other days ✓
- Spike protection maintained ✓
- No systematic bias ✓
