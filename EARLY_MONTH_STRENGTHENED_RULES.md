# Early Month Rules Strengthened - Override Weekday Volume Signal

## Problem
Days 1-5 of each month were still predicting high values when they fell on high-volume weekdays (Monday/Wednesday/Friday), even though these days should always be low volume regardless of weekday.

The model was being influenced by `weekday_volume_tier` (Monday=6, Wed/Fri=5) which was overpowering the early month low volume signal.

## Solution - Three-Level Strengthening

### 1. Feature-Level Strengthening

#### A. More Extreme Tier Values (`early_month_low_tier`)
**Before:**
- Days 1-5: `0` (very low)
- Days 6-10: `1` (transitioning)
- Other days: `2` (normal)

**After (STRENGTHENED):**
- Days 1-5: **`-10`** (EXTREME low - much stronger signal to override weekday_volume_tier)
- Days 6-10: `1` (transitioning)
- Other days: `2` (normal)

**Impact:** The `-10` value creates a much stronger negative signal that directly counteracts the positive weekday_volume_tier values (6 for Monday, 5 for Wed/Fri).

#### B. Stronger Interaction Feature (`is_high_vol_weekday_AND_early_month`)
**Before:**
- Binary flag: `1` when both high-volume weekday AND early month (days 1-10)
- Value: 0 or 1

**After (STRENGTHENED):**
- Days 1-5 on high-volume weekday: **`-2`** (STRONG active suppression)
- Days 6-10 on high-volume weekday: **`-1`** (moderate suppression)
- Other days: `0` (no suppression)

**Impact:** Instead of just flagging the collision (value=1), we now actively suppress it with negative values. The `-2` for days 1-5 tells the model "CANCEL the weekday boost."

### 2. Loss Function Strengthening (Dynamic Weighting)

**Before:**
- Days 1-5: 50x loss weight
- Days 6-10: Exponential decay from 50x to 1x
- Days 11+: 1x

**After (STRENGTHENED):**
- Days 1-5: **100x loss weight** (DOUBLED from 50x)
- Days 6-10: Exponential decay from 100x to 1x (adjusted lambda=0.92)
  - Day 6: 39.9x
  - Day 7: 15.9x
  - Day 8: 6.3x
  - Day 9: 2.5x
  - Day 10: 1.0x
- Days 11+: 1x

**Impact:** The 100x penalty during training makes any error on days 1-5 extremely expensive, forcing the model to predict LOW values even when weekday features suggest high volume.

## How It Works Together

### Example: Monday, January 5th (High-Volume Weekday + Early Month)

**Input Features:**
1. `weekday_volume_tier` = 6 (Monday is high volume)
2. `Is_Monday` = 1 (binary flag)
3. `early_month_low_tier` = **-10** (EXTREME low signal)
4. `is_first_5_days` = 1
5. `is_high_vol_weekday_AND_early_month` = **-2** (active suppression)

**Loss Function:**
- Base MSE loss
- × 100 (dynamic early month weight for day 5)
- = **100x penalty** if prediction is too high

**Result:** The model learns to predict LOW volume because:
1. The `-10` tier value strongly counteracts the `+6` weekday signal
2. The `-2` interaction feature explicitly suppresses the Monday boost
3. The 100x loss penalty makes it very expensive to get this wrong during training

## Files Modified

1. **`src/data/preprocessing.py`**
   - `add_early_month_low_volume_features()`: Changed `early_month_low_tier` for days 1-5 from `0` to `-10`
   - Same function: Changed `is_high_vol_weekday_AND_early_month` from binary (0/1) to tri-level (0/-1/-2)

2. **`src/utils/losses.py`**
   - `get_dynamic_early_month_weight()`: Increased loss weight from 50x to 100x for days 1-5
   - Adjusted lambda decay from 0.78 to 0.92 to maintain proper exponential curve

3. **`config/config_DRY.yaml`**
   - Updated documentation to reflect the strengthened rules

## Testing

To verify the changes work:

```bash
# Train the model with the strengthened rules
python mvp_train.py --category DRY --config config/config_DRY.yaml

# Evaluate early month predictions
python evaluate_early_month.py

# Check that days 1-5 predictions are LOW even on Mon/Wed/Fri
python diagnose_test_data.py
```

## Expected Behavior

After retraining with these strengthened rules:
- **Days 1-5**: Predictions should be LOW (under 20-30 CBM) **REGARDLESS** of weekday
- **Monday, Jan 5th**: Should predict ~15-25 CBM (not 40+ CBM)
- **Wednesday, Jan 3rd**: Should predict ~10-20 CBM (not 35+ CBM)
- **Days 6-10**: Gradual transition from low to normal volume
- **Days 11+**: Normal weekday patterns apply (Mon/Wed/Fri higher)

## Why This Works Better

1. **Direct Opposition**: `-10` tier value directly fights against `+6` weekday value in the model's learned coefficients
2. **Explicit Suppression**: `-2` interaction tells the model "cancel the weekday boost" rather than hoping it figures it out
3. **Training Pressure**: 100x loss weight makes early month accuracy more important than everything else during optimization
4. **Balanced Gradient**: The extreme penalties don't hurt later days because decay brings weight back to 1x by day 11

## Rollback Instructions

If this causes issues, you can revert by:

1. Change `early_month_low_tier` values back to `0` for days 1-5
2. Change `is_high_vol_weekday_AND_early_month` back to binary (0/1)
3. Change loss weight back to `50.0` in `get_dynamic_early_month_weight()`

Or simply revert the commits to these three files.
