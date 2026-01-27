# Brand-Level Prediction System

This document explains how to train and predict for individual BRANDS within a category (e.g., DRY) using the category's configuration.

## Overview

The brand-level prediction system allows you to:
1. **Train separate models** for each BRAND (AFC, COSY, OREO, etc.) within a category
2. **Use the category's configuration** (e.g., `config_DRY.yaml`) for all brands
3. **Make predictions** for each brand independently
4. **Save outputs** to brand-specific directories (e.g., `outputs/DRY_AFC/`, `outputs/DRY_COSY/`)

## Setup

### 1. Ensure Your Data Has BRAND Column

Your data should be grouped by:
- `ACTUALSHIPDATE` (date)
- `WHSEID` (warehouse)
- `CATEGORY` (e.g., DRY, FRESH)
- `BRAND` (e.g., AFC, COSY, OREO)

The `combine_data.py` script already handles this if you have the masterdata file at:
```
masterdata/masterdata category-brand.xlsx
```

### 2. Verify Category Configuration

Make sure you have a category-specific config file:
```
config/config_DRY.yaml
```

This configuration will be used for ALL brands within that category.

## Usage

### Step 1: Train Brand Models

Train models for all brands in DRY category:
```bash
python train_by_brand.py --category DRY
```

Train models for specific brands only:
```bash
python train_by_brand.py --category DRY --brands AFC COSY OREO
```

Skip brands that already have trained models:
```bash
python train_by_brand.py --category DRY --skip-existing
```

### Step 2: Make Predictions

Predict for all brands with trained models:
```bash
python predict_by_brand.py --category DRY
```

Predict for specific brands:
```bash
python predict_by_brand.py --category DRY --brands AFC COSY
```

Predict for a specific date range:
```bash
python predict_by_brand.py --category DRY --start 2025-01-01 --end 2025-12-31
```

Combine all brand predictions into a single file:
```bash
python predict_by_brand.py --category DRY --combine-output
```

## Output Structure

Each brand gets its own directory:
```
outputs/
├── DRY_AFC/
│   ├── models/
│   │   └── best_model.pth          # Trained model
│   ├── scaler.pkl                   # Feature scaler
│   ├── predictions_2025.csv         # Predictions
│   ├── train_loss.png               # Training plots
│   └── val_loss.png
├── DRY_COSY/
│   ├── models/
│   │   └── best_model.pth
│   ├── scaler.pkl
│   └── predictions_2025.csv
├── DRY_OREO/
│   └── ...
└── DRY_all_brands_predictions_2025.csv  # Combined (if --combine-output used)
```

## Example Workflow

### Full Pipeline for DRY Category Brands

```bash
# 1. Train all brands in DRY category
python train_by_brand.py --category DRY

# Expected output:
# MDLZ FORECASTING: Brand-Level Training Pipeline
# Category: DRY
# ================================================================================
# 
# [1/4] Discovering brands in category 'DRY'...
#   - Found 9 brand(s) in DRY:
#     1. AFC (1,234 samples)
#     2. COSY (2,345 samples)
#     3. KINH DO BISCUIT (890 samples)
#     4. LU (3,456 samples)
#     5. OREO (5,678 samples)
#     6. OTHER CAKE (456 samples)
#     7. RITZ (2,890 samples)
#     8. SLIDE (1,567 samples)
#     9. SOLITE (789 samples)
# 
# [2/4] Will train 9 brand model(s):
#   1. AFC -> outputs/DRY_AFC/
#   2. COSY -> outputs/DRY_COSY/
#   ...

# 2. Make predictions for 2025
python predict_by_brand.py --category DRY --start 2025-01-01 --end 2025-12-31 --combine-output

# Expected output:
# MDLZ FORECASTING: Brand-Level Prediction Pipeline
# Category: DRY
# ================================================================================
# 
# [1/4] Loading configuration for category 'DRY'...
#   - Prediction period: 2025-01-01 to 2025-12-31
#   - Total days: 365
# 
# [2/4] Loading data...
#   - Loaded 50,000 samples
# 
# [3/4] Discovering brands with trained models...
#   - Found 9 brand(s) with trained models:
#     1. AFC (outputs/DRY_AFC/)
#     2. COSY (outputs/DRY_COSY/)
#     ...
# 
# [4/4] Will predict 9 brand(s)
# 
# [SUCCESS] Brand prediction completed for DRY - AFC
#   - Predicted 365 days
#   - Mean prediction: 12.34
#   - Total volume: 4,504.10
# ...
# 
# ================================================================================
# Combining predictions...
# ================================================================================
#   - Combined predictions saved to: outputs/DRY_all_brands_predictions_2025.csv
#   - Total rows: 3,285
#   - Brands: ['AFC', 'COSY', 'KINH DO BISCUIT', 'LU', 'OREO', 'OTHER CAKE', 'RITZ', 'SLIDE', 'SOLITE']
```

## Advanced Options

### Custom Data File

Use a specific data file for prediction:
```bash
python predict_by_brand.py --category DRY --data-file dataset/test/data_2025.csv
```

### Custom Brand Column Name

If your data uses a different column name for brands:
```bash
python train_by_brand.py --category DRY --brand-col BRAND_NAME
python predict_by_brand.py --category DRY --brand-col BRAND_NAME
```

## Checking Available Brands

To see which brands are available in your data:
```bash
python -c "
import pandas as pd
data = pd.read_csv('dataset/data_cat/data_2024.csv')
dry_data = data[data['CATEGORY'] == 'DRY']
brands = dry_data['BRAND'].dropna().unique()
print('Brands in DRY category:')
for brand in sorted(brands):
    count = len(dry_data[dry_data['BRAND'] == brand])
    print(f'  - {brand}: {count:,} rows')
"
```

## Troubleshooting

### Error: "Model not found for brand 'AFC'"

**Solution:** Train the brand model first:
```bash
python train_by_brand.py --category DRY --brands AFC
```

### Error: "BRAND column not found in data"

**Solution:** Make sure you've run `combine_data.py` with the masterdata file to add BRAND column:
```bash
python combine_data.py --data-dir "G:/My Drive/26. MDLZ/WMS_history" --output-dir "./dataset/data_cat"
```

### Error: "No data found for category 'DRY' and brand 'AFC'"

**Solution:** Check if the brand exists in your data:
```bash
python -c "
import pandas as pd
data = pd.read_csv('dataset/data_cat/data_2024.csv')
print(data[data['CATEGORY'] == 'DRY']['BRAND'].unique())
"
```

## Configuration Details

Each brand uses the **same configuration** from `config_DRY.yaml`:
- `window.input_size`: 28 days
- `window.horizon`: 20 days
- `model.hidden_size`: 128
- `model.n_layers`: 2
- `training.learning_rate`: 0.0005
- `training.epochs`: 30

If you want different settings for specific brands, you can:
1. Create brand-specific config files: `config_DRY_AFC.yaml`
2. Or modify the base `config_DRY.yaml` before training

## Comparing Brand Predictions

To analyze and compare predictions across brands:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load combined predictions
df = pd.read_csv('outputs/DRY_all_brands_predictions_2025.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Plot predictions by brand
fig, ax = plt.subplots(figsize=(15, 6))
for brand in df['BRAND'].unique():
    brand_df = df[df['BRAND'] == brand]
    ax.plot(brand_df['date'], brand_df['predicted'], label=brand)

ax.set_xlabel('Date')
ax.set_ylabel('Predicted Volume (CBM)')
ax.set_title('DRY Category: Brand-Level Predictions for 2025')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/DRY_brand_comparison.png', dpi=150)
plt.show()

# Summary statistics by brand
summary = df.groupby('BRAND')['predicted'].agg(['mean', 'sum', 'std']).round(2)
print("\nBrand Summary Statistics:")
print(summary)
```

## Next Steps

1. **Train brand models** for your category:
   ```bash
   python train_by_brand.py --category DRY
   ```

2. **Make predictions**:
   ```bash
   python predict_by_brand.py --category DRY --start 2025-01-01 --end 2025-12-31 --combine-output
   ```

3. **Analyze results** using the combined output file

## Questions?

- Check that your data has the BRAND column
- Verify the category-specific config exists (e.g., `config_DRY.yaml`)
- Look at the output directories to see trained models and predictions
- Use `--help` flag for more options:
  ```bash
  python train_by_brand.py --help
  python predict_by_brand.py --help
  ```
