# Brand-Level Prediction with HybridLSTM

## Overview

The `predict_hybrid_lstm.py` script now supports **brand-level predictions** using the category-level HybridLSTM model. This allows you to:
- Predict for specific brands within a category
- Predict for multiple brands at once
- Get brand-level breakdowns of forecasts

## Usage Examples

### 1. Predict for a Single Brand

```bash
# Predict for OREO brand only
python predict_hybrid_lstm.py --category DRY --brand OREO --save

# With custom date range
python predict_hybrid_lstm.py --category DRY --brand OREO \
    --start 2025-01-01 --end 2025-12-31 --save
```

**Output:**
- `outputs/hybrid_lstm/DRY/predictions_OREO_20260128_145539.csv`

### 2. Predict for Multiple Brands

```bash
# Predict for OREO and AFC brands
python predict_hybrid_lstm.py --category DRY --brands OREO AFC --save

# Predict for all brands (automatically detects)
python predict_hybrid_lstm.py --category DRY --save
```

**Output:**
- `outputs/hybrid_lstm/DRY/predictions_all_brands_20260128_145539.csv` (detailed predictions)
- `outputs/hybrid_lstm/DRY/summary_by_brand_20260128_145539.csv` (summary statistics)

### 3. Category-Level Prediction (No Brand Filtering)

```bash
# Predict for entire category (aggregate)
python predict_hybrid_lstm.py --category DRY --save
```

If the data doesn't have a BRAND column, this mode is used automatically.

## Output Format

### Detailed Predictions CSV

| Column | Description |
|--------|-------------|
| `brand` | Brand name (only present in brand-level mode) |
| `prediction_date` | Date when prediction was made (window end date) |
| `forecast_date` | Date being forecasted |
| `day_ahead` | Days ahead (1-30 for 30-day horizon) |
| `predicted_cbm` | Forecasted volume in CBM |

### Brand Summary CSV (Multi-Brand Mode Only)

| Column | Description |
|--------|-------------|
| `brand` | Brand name |
| `num_predictions` | Total number of predictions for this brand |
| `avg_cbm` | Average predicted CBM across all forecasts |
| `total_cbm` | Sum of all predicted CBM |

## How It Works

### Category-Level Model, Brand-Level Predictions

The HybridLSTM model is trained on **category-level aggregated data** (all brands combined), but can make predictions on **individual brand data** because:

1. **Feature Engineering is Brand-Agnostic**: The expanding window features (rolling means, momentum, etc.) are calculated per brand but use the same methodology
2. **Temporal Patterns Transfer**: Many temporal patterns (weekday effects, holidays, seasonality) are shared across brands
3. **Statistical Normalization**: The model learns relative patterns that apply to different volume scales

### Comparison with Train-by-Brand Approach

| Aspect | predict_hybrid_lstm.py (Brand Filtering) | train_by_brand.py + predict_by_brand.py |
|--------|------------------------------------------|------------------------------------------|
| **Training** | One model per category | One model per brand |
| **Model Count** | 1 model (DRY) | N models (DRY_OREO, DRY_AFC, etc.) |
| **Training Time** | Fast (train once) | Slow (train N times) |
| **Prediction** | Apply category model to brand data | Load brand-specific model |
| **Accuracy** | Good for brands with similar patterns | Best for brands with unique patterns |
| **Data Requirement** | Category needs sufficient data | Each brand needs sufficient data |
| **Best For** | Categories with homogeneous brands | Categories with heterogeneous brands |

## When to Use Each Approach

### Use `predict_hybrid_lstm.py` (Brand Filtering) When:
- ✅ You want **quick brand-level breakdowns**
- ✅ Brands within the category have **similar patterns**
- ✅ Some brands have **limited historical data**
- ✅ You want **fast training** (one model for all brands)
- ✅ You need **consistent predictions** across brands

### Use `train_by_brand.py` + `predict_by_brand.py` When:
- ✅ Each brand has **unique patterns** (e.g., different seasonality)
- ✅ Each brand has **sufficient historical data** (1000+ samples)
- ✅ You need **maximum accuracy per brand**
- ✅ You can afford **longer training time**

## Example: Full Workflow

### Step 1: Train Category-Level Model

```bash
# Train HybridLSTM for DRY category (aggregates all brands)
python train_hybrid_lstm.py --category DRY --epochs 20
```

**Output:** `outputs/hybrid_lstm/DRY/models/best_model.pth`

### Step 2: Check Available Brands

```bash
# List all brands in the dataset
python -c "
import pandas as pd
df = pd.read_csv('dataset/data_cat/data_2023.csv')
dry = df[df['CATEGORY'] == 'DRY']
print('Available brands:', sorted(dry['BRAND'].unique()))
print('Sample counts:', dry['BRAND'].value_counts())
"
```

### Step 3: Make Brand-Level Predictions

```bash
# Option A: Predict for specific brands
python predict_hybrid_lstm.py --category DRY --brands OREO AFC COSY --save

# Option B: Predict for all brands
python predict_hybrid_lstm.py --category DRY --save

# Option C: Predict for one brand with date range
python predict_hybrid_lstm.py --category DRY --brand OREO \
    --start 2025-01-01 --end 2025-12-31 --save
```

### Step 4: Analyze Results

```python
import pandas as pd

# Load predictions
df = pd.read_csv('outputs/hybrid_lstm/DRY/predictions_all_brands_20260128_145539.csv')

# Aggregate by brand and month
df['forecast_date'] = pd.to_datetime(df['forecast_date'])
df['month'] = df['forecast_date'].dt.to_period('M')

monthly_by_brand = df.groupby(['brand', 'month'])['predicted_cbm'].sum().reset_index()
print(monthly_by_brand.head(20))

# Compare brand volumes
brand_totals = df.groupby('brand')['predicted_cbm'].sum().sort_values(ascending=False)
print("\nBrand Total Forecasts:")
print(brand_totals)
```

## Advanced Options

### Custom Brand Column Name

If your data uses a different column name for brands:

```bash
python predict_hybrid_lstm.py --category DRY \
    --brand-col "PRODUCT_NAME" \
    --brands "Product A" "Product B" \
    --save
```

### Custom Model Path

Use a specific model file:

```bash
python predict_hybrid_lstm.py --category DRY \
    --model-path outputs/hybrid_lstm/DRY_custom/models/best_model.pth \
    --brands OREO AFC \
    --save
```

### Custom Data File

Use external test data:

```bash
python predict_hybrid_lstm.py --category DRY \
    --data-path dataset/test/future_data_2025.csv \
    --brands OREO AFC \
    --save
```

## Limitations

### 1. Model Was Trained on Aggregated Data
The model learned from **category-level patterns** (all brands combined), so it may not capture **brand-specific anomalies** or **unique brand behaviors**.

**Solution:** If brands have very different patterns, use `train_by_brand.py` to train separate models.

### 2. Brand Data Must Have Sufficient History
For brand-level predictions, each brand needs at least `input_size` days of historical data (default: 30 days).

**Minimum Data Requirements:**
- `input_size` rows per brand (e.g., 30 days)
- Better results with 90+ days of history

### 3. Feature Engineering Uses All Data
The expanding window features (rolling means, etc.) are calculated on the **full category data first**, then filtered by brand. This ensures consistency but means brand-specific expanding features are not re-calculated.

## Performance Comparison

Based on testing with DRY category (2 brands):

| Metric | Value |
|--------|-------|
| Brands Predicted | 2 (OREO, AFC) |
| Total Predictions | 41,940 |
| Prediction Time | ~26 seconds |
| Predictions per Brand | 20,970 |
| Average CBM per Brand | ~22.7 |

**Comparison:**
- **Category-level prediction**: ~32 seconds for all brands combined
- **Brand-level prediction (2 brands)**: ~26 seconds with individual breakdowns
- **train_by_brand approach**: ~2 minutes training per brand + prediction time

## Troubleshooting

### Issue: "Not enough data (X rows < 30 required)"

**Cause:** A brand doesn't have enough historical data to create prediction windows.

**Solution:**
- Exclude that brand from predictions
- Reduce `--input-size` (not recommended, affects accuracy)
- Train a separate category-level model for low-volume brands

### Issue: Predictions are identical for all brands

**Cause:** The model is trained on aggregated data and may produce similar predictions for brands with similar patterns.

**Solution:** This is expected behavior. If you need more brand-specific predictions, use `train_by_brand.py`.

### Issue: Missing brand column

**Error:** `KeyError: 'BRAND'`

**Solution:** Specify the correct brand column name:
```bash
python predict_hybrid_lstm.py --category DRY --brand-col "YOUR_COLUMN_NAME"
```

Or train and predict at category level (no brand filtering):
```bash
python predict_hybrid_lstm.py --category DRY --save
```

## Summary

The updated `predict_hybrid_lstm.py` now supports:

✅ **Single brand prediction**: `--brand OREO`
✅ **Multiple brand prediction**: `--brands OREO AFC COSY`
✅ **Auto-detect all brands**: No brand arguments (automatically detects)
✅ **Brand summary statistics**: Automatic summary CSV for multi-brand predictions
✅ **Custom brand column**: `--brand-col` for flexibility

This provides a **fast, flexible way** to get brand-level forecasts using a single category-level model!
