# Visual Summary: Early Month Rules Strengthened

## Problem Scenario
```
Date: Monday, January 5th
├─ weekday_volume_tier = 6 (Monday is high volume)
├─ Is_Monday = 1
└─ RESULT BEFORE: Model predicts 40+ CBM ❌ (TOO HIGH)
```

## Solution Applied

### 1. Feature Engineering Strengthening

```
┌─────────────────────────────────────────────────────────────┐
│ early_month_low_tier (Days 1-5)                             │
│                                                              │
│ BEFORE: value = 0                                           │
│ AFTER:  value = -10  ← EXTREME negative signal             │
│                                                              │
│ Impact: -10 strongly counteracts +6 from weekday_volume_tier│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ is_high_vol_weekday_AND_early_month                         │
│                                                              │
│ BEFORE: 0 or 1 (binary flag)                                │
│ AFTER:  0 / -1 / -2 (tri-level suppression)                │
│         ├─ Days 1-5:   -2 (STRONG suppression)             │
│         ├─ Days 6-10:  -1 (moderate suppression)           │
│         └─ Days 11+:    0 (no suppression)                 │
│                                                              │
│ Impact: -2 actively cancels Monday/Wed/Fri boost            │
└─────────────────────────────────────────────────────────────┘
```

### 2. Loss Function Strengthening

```
┌───────────────────────────────────────────────────────────────┐
│ Dynamic Early Month Loss Weight Schedule                     │
│                                                               │
│ Weight                                                        │
│  100x │████████████████████ Days 1-5 (STRENGTHENED)         │
│   90x │                                                       │
│   80x │                                                       │
│   70x │                                                       │
│   60x │                                                       │
│   50x │                    ← BEFORE: started at 50x          │
│   40x │                    ╲                                 │
│   30x │                     ╲                                │
│   20x │                      ╲ Days 6-10                    │
│   10x │                       ╲ (exponential decay)         │
│    1x │________________________╲________________________     │
│       1   2   3   4   5   6   7   8   9  10  11  12...      │
│                    Day of Month                              │
│                                                               │
│ Day 1-5:  100x penalty (was 50x)                            │
│ Day 6:    39.9x                                              │
│ Day 7:    15.9x                                              │
│ Day 8:     6.3x                                              │
│ Day 9:     2.5x                                              │
│ Day 10:    1.0x                                              │
│ Day 11+:   1.0x (normal)                                     │
└───────────────────────────────────────────────────────────────┘
```

## How It Works: Monday, January 5th Example

```
┌────────────────────────────────────────────────────────────────┐
│ Input Features at Inference                                    │
├────────────────────────────────────────────────────────────────┤
│ weekday_volume_tier              = +6   (Monday boost)        │
│ Is_Monday                         = +1   (Monday flag)         │
│ early_month_low_tier             = -10  (EXTREME low signal)  │
│ is_first_5_days                  = +1   (penalty zone)        │
│ is_high_vol_weekday_AND_early_month = -2   (active suppression)│
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ Model Internal Calculation (Simplified)                       │
├────────────────────────────────────────────────────────────────┤
│ Base prediction = f(historical data)                          │
│                                                                 │
│ + (weekday_volume_tier × learned_weight_1)                    │
│   = +6 × w1                                                    │
│                                                                 │
│ + (early_month_low_tier × learned_weight_2)                   │
│   = -10 × w2  ← STRONG negative contribution                  │
│                                                                 │
│ + (is_high_vol_weekday_AND_early_month × learned_weight_3)    │
│   = -2 × w3   ← Active suppression                            │
│                                                                 │
│ NET EFFECT: Strong push toward LOW prediction                 │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ During Training (Loss Calculation)                            │
├────────────────────────────────────────────────────────────────┤
│ If prediction = 40 CBM, actual = 15 CBM:                      │
│                                                                 │
│ Base MSE = (40 - 15)² = 625                                   │
│ × Early Month Weight = × 100                                  │
│ = 62,500 (HUGE PENALTY!)                                      │
│                                                                 │
│ Gradient descent will STRONGLY push weights to predict LOW    │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ Result After Training                                          │
├────────────────────────────────────────────────────────────────┤
│ Model learns:                                                  │
│ "When day ≤ 5, IGNORE weekday boost, predict LOW"            │
│                                                                 │
│ Monday, Jan 5: Predict ~15-25 CBM ✓                          │
│ (Not 40+ CBM)                                                  │
└────────────────────────────────────────────────────────────────┘
```

## Three-Pronged Attack

```
                    EARLY MONTH RULE
                    (Days 1-5 = LOW)
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
    FEATURE           FEATURE           LOSS
    SIGNAL 1          SIGNAL 2          FUNCTION
         │                │                │
    tier = -10       interaction = -2   weight = 100x
         │                │                │
         └────────────────┴────────────────┘
                          │
                          ▼
              DOMINATES weekday_volume_tier
              (Even when Monday = +6)
```

## Expected Results

```
Day of Month │ Weekday  │ OLD Prediction │ NEW Prediction │
─────────────┼──────────┼────────────────┼────────────────┤
     1       │ Monday   │    35 CBM ❌   │    12 CBM ✓   │
     2       │ Tuesday  │    20 CBM ⚠️   │    10 CBM ✓   │
     3       │ Wednesday│    38 CBM ❌   │    15 CBM ✓   │
     5       │ Friday   │    40 CBM ❌   │    20 CBM ✓   │
     7       │ Sunday   │    18 CBM ✓    │    18 CBM ✓   │
    11       │ Monday   │    45 CBM ✓    │    45 CBM ✓   │  ← Normal after day 10
    15       │ Friday   │    50 CBM ✓    │    50 CBM ✓   │  ← Weekday boost OK here
```

## Key Takeaways

✓ **Days 1-5 are ALWAYS low**, regardless of weekday  
✓ **-10 tier value** provides strong counterforce to weekday signal  
✓ **-2 interaction** explicitly cancels weekday boost  
✓ **100x loss weight** makes training prioritize early month accuracy  
✓ **Days 11+** maintain normal weekday patterns  

## Training Command

```bash
python mvp_train.py --category DRY --config config/config_DRY.yaml
```

## Files Changed

- `src/data/preprocessing.py` (feature engineering)
- `src/utils/losses.py` (loss function)
- `config/config_DRY.yaml` (documentation)
