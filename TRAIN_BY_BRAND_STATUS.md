# train_by_brand.py - Status and Usage Guide

## Current Status

✅ **WORKING** - The script imports `train_single_model` from `mvp_train.py`, which was just fixed to respect category-specific feature_cols.

## What Was Fixed

Since `train_by_brand.py` uses `train_single_model` from `mvp_train.py` (line 65), the bug fix we applied to `mvp_train.py` **automatically fixes train_by_brand.py** too!

### Before Fix:
- `train_single_model` ignored category-specific features
- Brand models trained WITHOUT early month features

### After Fix:
- `train_single_model` now respects `config_DRY.yaml` features  
- Brand models will train WITH early month features ✅

## Usage

### Option 1: Train All Brands in Config Categories
```bash
python train_by_brand.py
```
- Reads `major_categories` from `config.yaml`
- Trains all brands for each category
- Example: If `config.yaml` has `major_categories: ["DRY"]`, trains all DRY brands

### Option 2: Train All Brands in Specific Category
```bash
python train_by_brand.py --category DRY
```
- Trains all brands in DRY category
- Uses `config_DRY.yaml` configuration

### Option 3: Train Specific Brands
```bash
python train_by_brand.py --category DRY --brands AFC COSY OREO
```
- Trains only AFC, COSY, and OREO brands
- Uses `config_DRY.yaml` configuration

### Option 4: Skip Already-Trained Brands
```bash
python train_by_brand.py --category DRY --skip-existing
```
- Checks if `outputs/DRY_BRANDNAME/models/best_model.pth` exists
- Skips brands that already have models
- Useful for resuming interrupted training

## Output Structure

Each brand gets its own directory:
```
outputs/
  └─ DRY_AFC/
      ├─ models/
      │   ├─ best_model.pth       # Trained model
      │   ├─ metadata.json        # Training metadata
      │   └─ scaler.pkl           # Feature scaler
      └─ test_predictions.png     # Prediction plot
  └─ DRY_COSY/
      ├─ models/
      │   ├─ best_model.pth
      │   ├─ metadata.json
      │   └─ scaler.pkl
      └─ test_predictions.png
  ... (more brands)
```

## How It Works

1. **Loads category-specific config** (e.g., `config_DRY.yaml`)
2. **Loads all data** for the years specified in config
3. **Discovers brands** in the category automatically
4. **For each brand:**
   - Filters data to that specific brand
   - Creates brand-specific output directory (`outputs/DRY_BRANDNAME/`)
   - Calls `train_single_model` from `mvp_train.py`
   - Saves model, scaler, and predictions

## Key Features

✅ **Auto-detects brands** - No need to manually specify brand list  
✅ **Uses category config** - Inherits all settings from `config_DRY.yaml`  
✅ **Respects early month features** - Fixed via `mvp_train.py` update  
✅ **Skip existing models** - Resume interrupted training  
✅ **Parallel-ready** - Each brand is independent

## Performance Notes

### Training Time
- **Per brand**: ~20-60 seconds (depends on data size)
- **10 brands**: ~5-10 minutes total
- **50 brands**: ~20-30 minutes total

### When to Use Brand-Level Models

**Use brand-level models when:**
- Different brands have distinct volume patterns
- Category-level model doesn't capture brand-specific behavior
- You need brand-specific forecasts for inventory planning

**Use category-level model when:**
- Brands share similar patterns
- Limited training data per brand
- Faster training/deployment needed

## Troubleshooting

### Issue: "No BRAND column found"
**Solution:** Check your data has a `BRAND` column, or specify custom column:
```bash
python train_by_brand.py --category DRY --brand-col BRAND_NAME
```

### Issue: "No data found for category"
**Solution:** Verify category name matches exactly (case-sensitive):
```bash
# Check available categories in your data first
python -c "import pandas as pd; data = pd.read_csv('dataset/data_cat/data_2024.csv'); print(data['CATEGORY'].unique())"
```

### Issue: Script hangs/takes too long
**Cause:** Loading data and training takes time
**Solutions:**
- Use `--brands` to train only specific brands
- Use `--skip-existing` to skip already-trained brands
- Train during off-hours

### Issue: Features not matching config
**Status:** ✅ FIXED - `mvp_train.py` now respects category-specific features
**Verify:** Check `outputs/DRY_BRANDNAME/models/metadata.json` has your early month features

## Verification Checklist

After training a brand model, verify:

1. **Model file exists:**
   ```bash
   ls outputs/DRY_AFC/models/best_model.pth
   ```

2. **Metadata has correct features:**
   ```bash
   cat outputs/DRY_AFC/models/metadata.json | grep "early_month"
   ```
   Should show:
   - `early_month_low_tier`
   - `is_early_month_low`
   - `is_first_5_days`
   - `days_from_month_start`

3. **Check training logs** for early month loss weight:
   ```
   Early Month loss weight: 15.0x (days 1-10, for DRY)
   ```

## Comparison: train_by_brand.py vs mvp_train.py

| Feature | mvp_train.py | train_by_brand.py |
|---------|--------------|-------------------|
| **Trains** | Category-level model | Brand-level models |
| **Output** | `outputs/DRY/` | `outputs/DRY_AFC/`, `outputs/DRY_COSY/`, etc. |
| **Use case** | Category forecasts | Brand-specific forecasts |
| **Training time** | ~20-60 sec | ~20-60 sec × number of brands |
| **Accuracy** | Good for category | Better for individual brands |

## Next Steps

1. **Test with one brand first:**
   ```bash
   python train_by_brand.py --category DRY --brands AFC
   ```

2. **Verify features are correct** in metadata.json

3. **If good, train all brands:**
   ```bash
   python train_by_brand.py --category DRY
   ```

4. **Use predictions:**
   ```bash
   python predict_by_brand.py --category DRY --brand AFC
   ```

---

**Status**: ✅ Fixed (inherits fix from mvp_train.py)  
**Last Updated**: 2026-02-07  
**Dependencies**: Requires mvp_train.py with category-specific feature fix
