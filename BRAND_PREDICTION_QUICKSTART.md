# Brand-Level Prediction - Quick Start Guide

## 🎯 Goal
Train and predict for each BRAND (AFC, COSY, OREO, etc.) within the DRY category, using the DRY configuration.

## 📋 Prerequisites

1. Your data has a `BRAND` column (added by `combine_data.py` using masterdata)
2. You have `config/config_DRY.yaml` configuration file
3. Data files exist in `dataset/data_cat/`

## 🚀 Quick Start (3 Commands)

### Step 1: Check Available Brands
```bash
python check_brands.py --category DRY
```

**Expected Output:**
```
Found 9 brand(s):
 1. AFC
    Samples: 1,234
    Date range: 2023-01-01 to 2024-12-31
    Total volume: 5,678.90 CBM

 2. COSY
    Samples: 2,345
    ...
```

### Step 2: Train Brand Models
```bash
python train_by_brand.py --category DRY
```

**What it does:**
- Filters data for DRY category + each brand (AFC, COSY, etc.)
- Uses `config_DRY.yaml` settings for all brands
- Trains separate model for each brand
- Saves to `outputs/DRY_AFC/`, `outputs/DRY_COSY/`, etc.

**Expected Output:**
```
BRAND 1/9: DRY - AFC
================================================================================
  - Filtered to 1,234 samples for DRY - AFC
  - Training...
  - Model saved to: outputs/DRY_AFC/models/best_model.pth
  ✓ SUCCESS

BRAND 2/9: DRY - COSY
...
```

### Step 3: Make Predictions
```bash
python predict_by_brand.py --category DRY --start 2025-01-01 --end 2025-12-31 --combine-output
```

**What it does:**
- Loads each brand's trained model
- Makes predictions for 2025
- Saves individual predictions to each brand's folder
- Combines all into `outputs/DRY_all_brands_predictions_2025.csv`

**Expected Output:**
```
BRAND 1/9: DRY - AFC
  - Predicted 365 days
  - Mean prediction: 12.34
  - Total volume: 4,504.10

Combined predictions saved to: outputs/DRY_all_brands_predictions_2025.csv
Brands: ['AFC', 'COSY', 'KINH DO BISCUIT', 'LU', 'OREO', 'OTHER CAKE', 'RITZ', 'SLIDE', 'SOLITE']
```

## 📊 Output Structure

```
outputs/
├── DRY_AFC/
│   ├── models/
│   │   └── best_model.pth          # Trained model
│   ├── scaler.pkl                   # Feature scaler
│   ├── predictions_2025.csv         # Brand predictions
│   ├── train_loss.png
│   └── val_loss.png
├── DRY_COSY/
│   └── ... (same structure)
├── DRY_OREO/
│   └── ... (same structure)
└── DRY_all_brands_predictions_2025.csv  # Combined predictions
```

## 🎨 Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load combined predictions
df = pd.read_csv('outputs/DRY_all_brands_predictions_2025.csv')
df['date'] = pd.to_datetime(df['date'])

# Plot by brand
fig, ax = plt.subplots(figsize=(15, 6))
for brand in df['BRAND'].unique():
    brand_df = df[df['BRAND'] == brand]
    ax.plot(brand_df['date'], brand_df['predicted'], label=brand)

ax.legend()
ax.set_title('DRY Category: Brand-Level Predictions')
plt.tight_layout()
plt.savefig('outputs/DRY_brand_comparison.png')

# Summary by brand
print(df.groupby('BRAND')['predicted'].agg(['mean', 'sum']).round(2))
```

## ⚙️ Advanced Options

### Train Specific Brands Only
```bash
python train_by_brand.py --category DRY --brands AFC COSY OREO
```

### Skip Already Trained Brands
```bash
python train_by_brand.py --category DRY --skip-existing
```

### Predict Specific Brands
```bash
python predict_by_brand.py --category DRY --brands AFC COSY
```

### Custom Date Range
```bash
python predict_by_brand.py --category DRY --start 2025-06-01 --end 2025-08-31
```

## 🔧 Troubleshooting

### "BRAND column not found"
**Solution:** Run `combine_data.py` to add BRAND column:
```bash
python combine_data.py
```

### "Model not found for brand 'AFC'"
**Solution:** Train the model first:
```bash
python train_by_brand.py --category DRY --brands AFC
```

### "No data found for category 'DRY' and brand 'AFC'"
**Solution:** Check if brand exists:
```bash
python check_brands.py --category DRY
```

## 📝 Key Points

1. **Same Config for All Brands**: All brands in DRY use `config_DRY.yaml` settings
   - Window size: 28 days
   - Horizon: 20 days
   - Learning rate: 0.0005
   - Epochs: 30

2. **Separate Models**: Each brand gets its own trained model
   - `DRY_AFC/models/best_model.pth`
   - `DRY_COSY/models/best_model.pth`
   - etc.

3. **Independent Predictions**: Each brand is predicted separately
   - Uses its own historical data
   - Uses its own trained model
   - Results can be combined for comparison

## 🎯 Next Steps

1. **Check brands**: `python check_brands.py --category DRY`
2. **Train models**: `python train_by_brand.py --category DRY`
3. **Make predictions**: `python predict_by_brand.py --category DRY --combine-output`
4. **Analyze results**: Use the combined CSV file

## 📚 Full Documentation

See `BRAND_PREDICTION_README.md` for complete details and advanced usage.

## ❓ Questions

Run with `--help` for more options:
```bash
python train_by_brand.py --help
python predict_by_brand.py --help
python check_brands.py --help
```
