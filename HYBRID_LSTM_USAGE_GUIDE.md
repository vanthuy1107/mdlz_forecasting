# HybridLSTM Training and Prediction Guide

## Overview

This guide shows you how to train and use the **HybridLSTM model** (the newer dual-branch architecture) instead of the older RNNWithCategory model used by `predict_by_brand.py`.

## Key Differences

### Architecture Comparison

| Script | Model | Level | Architecture |
|--------|-------|-------|--------------|
| `train_by_brand.py` + `predict_by_brand.py` | RNNWithCategory | Brand-level | Single LSTM path |
| `train_hybrid_lstm.py` + `predict_hybrid_lstm.py` | HybridLSTM | Category-level | Dual-branch (LSTM + Dense) |

### Feature Engineering

**HybridLSTM** uses **leak-free expanding window features**:
- `rolling_mean_7d_expanding` - No look-ahead bias
- `weekday_avg_expanding` - Only uses past data
- `momentum_3d_vs_14d_expanding` - Causal momentum indicators

**RNNWithCategory** uses standard rolling features that may contain data leakage.

## Usage

### 1. Training HybridLSTM Model

Train a category-level model (e.g., DRY):

```bash
# Basic training (20 epochs by default)
python train_hybrid_lstm.py --category DRY

# Custom hyperparameters
python train_hybrid_lstm.py --category DRY \
    --epochs 50 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --input-size 30 \
    --horizon 30

# With asymmetric loss (recommended for FMCG)
python train_hybrid_lstm.py --category DRY \
    --loss asymmetric_mse \
    --under-penalty 3.0 \
    --over-penalty 1.0
```

**Output:**
- Model: `outputs/hybrid_lstm/DRY/models/best_model.pth`
- Metrics: `outputs/hybrid_lstm/DRY/test_metrics.json`

### 2. Making Predictions with HybridLSTM

#### Option A: Category-Level Predictions (All Brands Aggregated)

```bash
# Predict using training data (2023-2024)
python predict_hybrid_lstm.py --category DRY --save
```

#### Option B: Brand-Level Predictions (NEW! 🎉)

```bash
# Predict for a single brand
python predict_hybrid_lstm.py --category DRY --brand OREO --save

# Predict for multiple specific brands
python predict_hybrid_lstm.py --category DRY --brands OREO AFC COSY --save

# Predict for ALL brands automatically
python predict_hybrid_lstm.py --category DRY --save
# (automatically detects brands if BRAND column exists)
```

**Output:**
- Single brand: `predictions_OREO_20260128.csv`
- Multiple brands: `predictions_all_brands_20260128.csv` + `summary_by_brand_20260128.csv`

#### Option C: Predict for Specific Date Range

```bash
# Category-level for 2025
python predict_hybrid_lstm.py --category DRY \
    --start 2025-01-01 \
    --end 2025-12-31 \
    --save

# Brand-level for 2025
python predict_hybrid_lstm.py --category DRY --brand OREO \
    --start 2025-01-01 \
    --end 2025-12-31 \
    --save
```

#### Option D: Predict on Custom Data File

```bash
# Category-level
python predict_hybrid_lstm.py --category DRY \
    --data-path dataset/test/data_2025.csv \
    --save

# Brand-level
python predict_hybrid_lstm.py --category DRY --brands OREO AFC \
    --data-path dataset/test/data_2025.csv \
    --save
```

**Output:**
- Predictions: `outputs/hybrid_lstm/DRY/predictions_*.csv`
- Brand summary (if multi-brand): `summary_by_brand_*.csv`

**📖 See [BRAND_LEVEL_PREDICTION_GUIDE.md](BRAND_LEVEL_PREDICTION_GUIDE.md) for detailed brand-level prediction documentation.**

### 3. Prediction Output Format

The prediction CSV contains:

| Column | Description |
|--------|-------------|
| `prediction_date` | Date when prediction was made (window end date) |
| `forecast_date` | Date being forecasted |
| `day_ahead` | Days ahead (1-30 for 30-day horizon) |
| `predicted_cbm` | Forecasted volume in CBM |

Example:
```csv
prediction_date,forecast_date,day_ahead,predicted_cbm
2023-01-05,2023-01-05,1,21.22
2023-01-05,2023-01-06,2,21.69
2023-01-05,2023-01-07,3,22.82
...
```

## Comparison: predict_by_brand.py vs predict_hybrid_lstm.py

### predict_by_brand.py (OLD)
```bash
# Predicts using brand-level RNNWithCategory models
python predict_by_brand.py --category DRY --start 2025-01-01 --end 2025-12-31
```

**Requires:** Models trained with `train_by_brand.py`
- `outputs/DRY_AFC/models/best_model.pth`
- `outputs/DRY_OREO/models/best_model.pth`
- etc. (one model per brand)

**Features:**
- ❌ Standard rolling windows (may leak data)
- ✅ Brand-specific predictions
- ✅ Google Sheets upload support

### predict_hybrid_lstm.py (NEW)
```bash
# Predicts using category-level HybridLSTM model
python predict_hybrid_lstm.py --category DRY --start 2025-01-01 --end 2025-12-31 --save
```

**Requires:** Model trained with `train_hybrid_lstm.py`
- `outputs/hybrid_lstm/DRY/models/best_model.pth` (one model per category)

**Features:**
- ✅ Leak-free expanding window features
- ✅ Dual-branch architecture (LSTM + Dense)
- ✅ Asymmetric loss function (penalizes under-forecasting)
- ✅ **Brand-level predictions** (NEW! Using category model with brand filtering)
- ✅ Auto-detect brands or specify custom brands
- ❌ No Google Sheets upload yet

## When to Use Which Model?

### Use HybridLSTM (`train_hybrid_lstm.py` + `predict_hybrid_lstm.py`) when:
- ✅ You want **leak-free feature engineering** (production-ready)
- ✅ You need **category-level forecasts** (aggregate volume)
- ✅ You want **asymmetric loss** (penalize stock-outs more than overstock)
- ✅ You have **sufficient category data** (thousands of samples)

### Use RNNWithCategory (`train_by_brand.py` + `predict_by_brand.py`) when:
- ✅ You need **brand-level forecasts** (per-brand breakdown)
- ✅ You want **Google Sheets integration** (upload predictions automatically)
- ✅ You have **brand-specific patterns** to capture

## Configuration

Both models use `config/config.yaml` and category-specific overrides:
- `config/config_DRY.yaml`
- `config/config_TET.yaml`
- `config/config_MOONCAKE.yaml`

### Key Configuration Sections

```yaml
# HybridLSTM feature groups (config.yaml)
model:
  hybrid:
    temporal_features:  # Goes to LSTM branch
      - "Total CBM"
      - "rolling_mean_7d_expanding"
      - "rolling_mean_30d_expanding"
      - "momentum_3d_vs_14d_expanding"
    
    static_features:  # Goes to Dense branch
      - "month_sin"
      - "month_cos"
      - "holiday_indicator"
      - "days_since_holiday"
      - "lunar_month"
      - "lunar_day"
      # ... more features
    
    # Architecture
    dense_hidden_sizes: [64, 32]  # Dense branch MLP
    fusion_hidden_sizes: [64]     # Fusion layer after concatenation
    
    # Loss function
    loss_function: "asymmetric_mse"
    asymmetric_loss:
      under_penalty: 3.0  # Stock-out penalty
      over_penalty: 1.0   # Overstock penalty
```

## Model Performance

After 3 epochs of training on DRY category:

| Metric | Value |
|--------|-------|
| Training Loss | 23,607 → 17,807 (continued training) |
| Validation Loss | 3,429 (best) |
| Test MAE | 84.66 CBM |
| Test RMSE | 119.11 CBM |

## Troubleshooting

### Issue: "Missing static features: ['days_since_holiday', 'lunar_month', 'lunar_day']"
**Solution:** This has been fixed in the latest version. The features are now automatically generated during data preparation.

### Issue: "RuntimeError: size mismatch for output_layer.weight"
**Solution:** This occurs when using an older model format. Retrain the model with the latest `train_hybrid_lstm.py` which saves model configuration in the checkpoint.

### Issue: Model loads but predictions are wrong
**Solution:** Ensure the input data has the same feature engineering as training:
- Same expanding window calculations
- Same holiday calendar
- Same lunar date calculations

## Next Steps

1. **For Production:** Use `train_hybrid_lstm.py` + `predict_hybrid_lstm.py`
2. **For Brand Analysis:** Use `train_by_brand.py` + `predict_by_brand.py`
3. **For Comparison:** Train both models and compare results

## Files Summary

| File | Purpose |
|------|---------|
| `train_hybrid_lstm.py` | Train HybridLSTM model (category-level, leak-free) |
| `predict_hybrid_lstm.py` | Predict using HybridLSTM model (category or brand-level) |
| `train_by_brand.py` | Train RNNWithCategory models (brand-level) |
| `predict_by_brand.py` | Predict using RNNWithCategory models |
| `validate_no_leakage.py` | Verify expanding features are leak-free |
| `src/data/expanding_features.py` | Leak-free feature engineering functions |
| `BRAND_LEVEL_PREDICTION_GUIDE.md` | **Detailed brand-level prediction guide** |
| `HYBRID_LSTM_USAGE_GUIDE.md` | This file - HybridLSTM overview |

---

**Note:** The HybridLSTM model is the **recommended approach for production** due to its leak-free feature engineering and dual-branch architecture.
