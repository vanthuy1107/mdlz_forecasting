# Before vs After: Early Month Rule Strengthening

## The Core Issue

**User Request:** "I want the rule stronger - days 1-5 should be LOW even when weekday_volume_tier is HIGH"

**Problem:** Model was influenced by `weekday_volume_tier` (Mon=6, Wed/Fri=5) and predicting high values on days 1-5 when they fell on high-volume weekdays.

---

## Changes Summary Table

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| **early_month_low_tier** (days 1-5) | `0` | **`-10`** | 10x stronger negative signal |
| **is_high_vol_weekday_AND_early_month** (days 1-5) | `1` (flag) | **`-2`** (suppress) | Active cancellation vs passive flag |
| **is_high_vol_weekday_AND_early_month** (days 6-10) | `1` (flag) | **`-1`** (suppress) | Moderate suppression |
| **Loss weight** (days 1-5) | `50x` | **`100x`** | 2x stronger training penalty |
| **Loss weight** (day 6) | `~23x` | **`~40x`** | Steeper decay maintains pressure |

---

## Detailed Before/After Comparison

### 1. Feature: `early_month_low_tier`

#### BEFORE
```python
# src/data/preprocessing.py (OLD)
df[early_month_low_tier_col] = 2  # Default
df.loc[(day_of_month >= 6) & (day_of_month <= 10), early_month_low_tier_col] = 1
df.loc[day_of_month <= 5, early_month_low_tier_col] = 0  # Days 1-5
```

**Values:**
- Days 1-5: `0`
- Days 6-10: `1`  
- Days 11+: `2`

**Problem:** Value of `0` is weak compared to weekday_volume_tier of `6` (Monday).

---

#### AFTER
```python
# src/data/preprocessing.py (NEW)
df[early_month_low_tier_col] = 2  # Default
df.loc[(day_of_month >= 6) & (day_of_month <= 10), early_month_low_tier_col] = 1
df.loc[day_of_month <= 5, early_month_low_tier_col] = -10  # Days 1-5 STRENGTHENED
```

**Values:**
- Days 1-5: **`-10`** ← EXTREME negative signal
- Days 6-10: `1`
- Days 11+: `2`

**Solution:** Value of `-10` creates strong negative force that directly opposes `+6` from weekday tier.

---

### 2. Feature: `is_high_vol_weekday_AND_early_month`

#### BEFORE
```python
# src/data/preprocessing.py (OLD)
df['is_high_vol_weekday_AND_early_month'] = (
    (df['is_high_volume_weekday'] == 1) & (day_of_month <= 10)
).astype(int)
```

**Values:**
- High-volume weekday on days 1-10: `1`
- All other cases: `0`

**Problem:** Binary flag (0/1) only signals collision, doesn't actively suppress. Model must learn complex interaction.

---

#### AFTER
```python
# src/data/preprocessing.py (NEW)
df['is_high_vol_weekday_AND_early_month'] = 0  # Default
df.loc[(df['is_high_volume_weekday'] == 1) & (day_of_month >= 6) & (day_of_month <= 10), 
       'is_high_vol_weekday_AND_early_month'] = -1  # Days 6-10
df.loc[(df['is_high_volume_weekday'] == 1) & (day_of_month <= 5), 
       'is_high_vol_weekday_AND_early_month'] = -2  # Days 1-5
```

**Values:**
- High-volume weekday on days 1-5: **`-2`** ← STRONG suppression
- High-volume weekday on days 6-10: **`-1`** ← Moderate suppression
- All other cases: `0`

**Solution:** Negative values explicitly tell model "CANCEL the weekday boost" instead of hoping it learns the interaction.

---

### 3. Loss Function: Dynamic Weight Schedule

#### BEFORE
```python
# src/utils/losses.py (OLD)
def get_dynamic_early_month_weight(day_of_month: torch.Tensor) -> torch.Tensor:
    # Days 1-5: MAXIMUM PENALTY (50x)
    mask_days_1_5 = (day_of_month >= 1) & (day_of_month <= 5)
    weights = torch.where(mask_days_1_5, torch.tensor(50.0, device=...), weights)
    
    # Days 6-10: Exponential decay from 50
    lambda_decay = 0.78
    exponential_decay = 50.0 * torch.exp(-lambda_decay * (day_of_month - 5))
    ...
```

**Weight Schedule:**
- Day 1-5: `50x`
- Day 6: `22.9x`
- Day 7: `10.5x`
- Day 8: `4.8x`
- Day 9: `2.2x`
- Day 10: `1.0x`

**Problem:** 50x penalty might not be enough when weekday signal is very strong (Monday).

---

#### AFTER
```python
# src/utils/losses.py (NEW)
def get_dynamic_early_month_weight(day_of_month: torch.Tensor) -> torch.Tensor:
    # Days 1-5: MAXIMUM PENALTY (100x) - STRENGTHENED
    mask_days_1_5 = (day_of_month >= 1) & (day_of_month <= 5)
    weights = torch.where(mask_days_1_5, torch.tensor(100.0, device=...), weights)
    
    # Days 6-10: Exponential decay from 100
    lambda_decay = 0.92  # Adjusted for 100x base
    exponential_decay = 100.0 * torch.exp(-lambda_decay * (day_of_month - 5))
    ...
```

**Weight Schedule:**
- Day 1-5: **`100x`** ← DOUBLED from 50x
- Day 6: **`39.9x`**
- Day 7: **`15.9x`**
- Day 8: **`6.3x`**
- Day 9: **`2.5x`**
- Day 10: **`1.0x`**

**Solution:** 100x penalty makes early month errors EXTREMELY expensive during training, forcing model to prioritize this rule above all else.

---

## Example: Monday, January 5th

### Feature Contributions (Simplified)

| Feature | BEFORE | AFTER | Change |
|---------|--------|-------|--------|
| `weekday_volume_tier` | +6 | +6 | (same) |
| `Is_Monday` | +1 | +1 | (same) |
| `early_month_low_tier` | 0 | **-10** | **Much stronger negative** |
| `is_first_5_days` | +1 | +1 | (same) |
| `is_high_vol_weekday_AND_early_month` | +1 | **-2** | **Active suppression** |

### Training Loss (if prediction = 40, actual = 15)

| Phase | BEFORE | AFTER |
|-------|--------|-------|
| Base MSE | (40-15)² = 625 | (40-15)² = 625 |
| × Weight | × 50 | × **100** |
| **Total Loss** | **31,250** | **62,500** |

**Impact:** 2x larger gradient pushes model MORE strongly toward low predictions.

---

## Combined Effect Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  WEEKDAY SIGNAL (Monday)                    │
│                  weekday_volume_tier = +6                   │
│                          ↓                                  │
│                      Pushes UP                              │
│                          ↓                                  │
├─────────────────────────────────────────────────────────────┤
│                   BEFORE (Weak Rules)                       │
│                                                             │
│  early_month_low_tier = 0     → weak negative              │
│  interaction = +1             → passive flag                │
│  loss_weight = 50x            → moderate penalty            │
│                                                             │
│  Net Result: ↑↑↑ (weekday wins)                            │
│  Prediction: ~40 CBM ❌                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  WEEKDAY SIGNAL (Monday)                    │
│                  weekday_volume_tier = +6                   │
│                          ↓                                  │
│                      Pushes UP                              │
│                          ↓                                  │
├─────────────────────────────────────────────────────────────┤
│                  AFTER (Strong Rules)                       │
│                                                             │
│  early_month_low_tier = -10   → STRONG negative ↓↓↓        │
│  interaction = -2             → active suppression ↓↓       │
│  loss_weight = 100x           → EXTREME penalty             │
│                                                             │
│  Net Result: ↓↓↓ (early month wins)                        │
│  Prediction: ~15 CBM ✓                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Files Modified

1. **`src/data/preprocessing.py`**
   - Line ~769: Changed days 1-5 tier from `0` to `-10`
   - Lines ~809-811: Changed interaction from binary to tri-level (`0/-1/-2`)

2. **`src/utils/losses.py`**
   - Line ~52: Changed weight from `50.0` to `100.0`
   - Line ~59: Changed lambda from `0.78` to `0.92`
   - Line ~60: Changed base from `50.0` to `100.0`

3. **`config/config_DRY.yaml`**
   - Lines 33-41: Updated comments to reflect strengthened rules
   - Lines 69-75: Updated feature descriptions

---

## Testing Recommendation

```bash
# 1. Retrain the model
python mvp_train.py --category DRY --config config/config_DRY.yaml

# 2. Evaluate early month performance
python evaluate_early_month.py

# 3. Check specific days
python diagnose_test_data.py  # Look for days 1-5 on Mon/Wed/Fri
```

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Days 1-5 on Monday | ~35-40 CBM | **~15-25 CBM** |
| Days 1-5 on Wednesday | ~35-38 CBM | **~12-20 CBM** |
| Days 1-5 on Friday | ~38-42 CBM | **~18-25 CBM** |
| Days 11+ on Monday | ~45 CBM ✓ | ~45 CBM ✓ (unchanged) |

**Key Success Criteria:**
✓ Days 1-5 predictions are LOW regardless of weekday  
✓ Weekday patterns still work correctly after day 10  
✓ No negative side effects on other parts of the month  

---

## Rollback Instructions

If needed, revert these specific changes:

```bash
# In preprocessing.py line 769:
df.loc[day_of_month <= 5, early_month_low_tier_col] = 0  # Change back to 0

# In preprocessing.py lines 809-811:
df['is_high_vol_weekday_AND_early_month'] = (
    (df['is_high_volume_weekday'] == 1) & (day_of_month <= 10)
).astype(int)  # Change back to binary

# In losses.py lines 52, 59-60:
weights = torch.where(mask_days_1_5, torch.tensor(50.0, ...), weights)
lambda_decay = 0.78
exponential_decay = 50.0 * torch.exp(...)
```
