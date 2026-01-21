# MDLZ Warehouse Forecasting System

A production-ready deep learning system for predicting warehouse shipment volumes (Total CBM) using category-aware LSTM models with Vietnamese holiday and lunar calendar features.

## Overview

This system forecasts future shipment volumes for MDLZ warehouse outbound flows, per product category, with special handling for:
- **Vietnamese holiday spikes** (Tet, Mid-Autumn Festival, etc.)
- **Lunar calendar seasonality**
- **Structural priors** (CBM/QTY density, last-year density)
- **High- and low-volume categories** with different modeling strategies

## Project Structure

```
mdlz_forecasting/
├── config/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   └── config.yaml        # Configuration file
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py      # Data loading utilities
│   │   ├── dataset.py     # PyTorch Dataset classes
│   │   └── preprocessing.py # Data preprocessing and feature engineering
│   ├── models.py          # Model architectures (RNNWithCategory, RNNForecastor)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py     # OOP Trainer class
│   └── utils/
│       ├── __init__.py
│       ├── losses.py      # Custom loss functions (spike_aware_mse)
│       ├── visualization.py # Plotting utilities
│       ├── saving.py      # Saving utilities
│       └── google_sheets.py # Google Sheets integration
├── mvp_test.py            # Training script (MVP test pipeline)
├── mvp_predict_2025.py   # Prediction script (inference pipeline)
├── combine_data.py        # Data combination utilities
├── MODEL_LOGIC.md         # Detailed model documentation
├── PRODUCTION_READINESS_ASSESSMENT.md
└── requirements.txt       # Python dependencies
```

## Features

- **Vietnamese Holiday Support**: Handles Tet, Mid-Autumn Festival, Independence Day, and Labor Day
- **Lunar Calendar Features**: Approximates Vietnamese lunar calendar with cyclical encodings
- **Category-Aware Models**: Separate handling for major (DRY, FRESH) and minor (POSM, OTHER) categories
- **Residual Learning**: Optional residual target learning against causal baselines
- **Spike-Aware Loss**: Custom loss function that weights high-volume periods (Tet spikes) more heavily
- **Multiple Training Modes**: Support for global, single-category, or per-category models
- **Two Prediction Modes**: Teacher Forcing (evaluation) and Recursive (production forecasting)
- **Modular Architecture**: Clean separation of concerns (data, models, training, utils)
- **Configuration-Driven**: All hyperparameters and paths in `config/config.yaml`

## Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

Key dependencies:
- `torch` - PyTorch for deep learning
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Scaling and utilities
- `matplotlib` - Visualization
- `pyyaml` - Configuration management
- `gspread` (optional) - Google Sheets integration

## Quick Start

### Training

Train models on historical data (2023-2024):

```bash
python mvp_test.py
```

This will:
- Load data from `config/data.data_dir` for years 2023-2024
- Apply Vietnamese holiday and lunar calendar features
- Train one or more models based on `data.category_mode`
- Save checkpoints, scalers, plots, and metadata to `outputs/mvp_test*/`

### Prediction

Generate forecasts for future dates:

```bash
python mvp_predict_2025.py
```

This will:
- Load trained model(s) from `outputs/mvp_test*/models/`
- Load prediction data from `config.inference.prediction_data_path`
- Generate predictions for the period specified in `config.inference`
- Save results to `outputs/predictions/` with timestamps

## Configuration

All settings are managed through `config/config.yaml`:

### Data Configuration

```yaml
data:
  data_dir: "dataset/data_cat"
  years: [2023, 2024]
  target_col: "Total CBM"  # Target variable to predict
  cat_col: "CATEGORY"
  category_mode: "single"    # "all" | "single" | "both" | "each"
  category_filter: "DRY"     # Used when category_mode == "single"
  major_categories: ["DRY", "FRESH"]
  minor_categories: ["POSM", "OTHER"]
  use_residual_target: true
  baseline_source_col: "rolling_mean_30d"
```

### Model Configuration

```yaml
model:
  name: "RNNWithCategory"
  cat_emb_dim: 8
  input_dim: 20  # Dynamically set based on feature_cols
  hidden_size: 64
  n_layers: 2
  output_dim: 1
  use_layer_norm: true
```

### Training Configuration

```yaml
training:
  batch_size: 64
  epochs: 20
  learning_rate: 0.001
  loss: "spike_aware_mse"  # Forced to spike_aware_mse in mvp_test.py
  device: "auto"  # "auto" | "cuda" | "cpu"
```

### Inference Configuration

```yaml
inference:
  prediction_data_path: "dataset/test/data_2025.csv"
  prediction_start: "2025-01-01"
  prediction_end: "2025-01-31"
```

### Window Configuration

```yaml
window:
  input_size: 30   # Look-back window length
  horizon: 1        # Prediction horizon (1 day ahead)
```

## Category Modes

The system supports four category training modes:

1. **`"all"`**: Train one global model on all major categories
   - Output: `outputs/mvp_test_all/`
   - Suitable for shared representations across categories

2. **`"single"`**: Train one model on a single category
   - Output: `outputs/mvp_test_{category}/`
   - Example: `outputs/mvp_test_DRY/`
   - Suitable for category-specific patterns

3. **`"both"`**: Train one global model plus one model per category
   - Outputs: `outputs/mvp_test_all/` + `outputs/mvp_test_{category}/` for each
   - Allows comparison between global and per-category approaches

4. **`"each"`**: Train one model per category (uses `major_categories` if specified)
   - Output: `outputs/mvp_test_{category}/` for each category
   - Suitable for independent category modeling

## Prediction Modes

`mvp_predict_2025.py` supports two prediction modes:

1. **Teacher Forcing (Test Evaluation)**
   - Uses actual ground truth values from the prediction period as input features
   - Suitable for model evaluation when historical data is available
   - Not suitable for true production forecasting

2. **Recursive (Production Forecast)**
   - Uses the model's own predictions as inputs for subsequent time steps
   - Simulates true production forecasting where future values are unknown
   - Starts from a historical window and recursively generates predictions day-by-day

## Data Pipeline

The system expects data in CSV format with the following columns:

- **Required columns**:
  - `ACTUALSHIPDATE`: Date column for temporal sorting
  - `CATEGORY`: Product category (e.g., "DRY", "FRESH", "POSM", "OTHER")
  - `Total CBM`: Target volume to predict (configurable via `data.target_col`)
  - `Total QTY`: Quantity (used for density features)

- **Generated features** (automatically created during preprocessing):
  - Temporal: `month_sin`, `month_cos`, `dayofmonth_sin`, `dayofmonth_cos`
  - Weekend: `is_weekend`, `day_of_week`
  - Lunar: `lunar_month`, `lunar_day`, `lunar_month_sin`, `lunar_month_cos`, `lunar_day_sin`, `lunar_day_cos`
  - Holidays: `holiday_indicator`, `days_until_next_holiday`, `days_since_holiday`, `days_to_tet`
  - Rolling: `rolling_mean_7d`, `rolling_mean_30d`, `momentum_3d_vs_14d`
  - Density: `cbm_per_qty`, `cbm_per_qty_last_year`

Data files should be located at: `{data_dir}/data_{year}.csv`

## Output Structure

### Training Outputs (`mvp_test.py`)

```
outputs/
└── mvp_test*/              # * = "_all", "_DRY", "_FRESH", etc.
    ├── models/
    │   ├── best_model.pth  # Best model checkpoint
    │   ├── scaler.pkl      # Fitted StandardScaler
    │   └── metadata.json   # Training metadata and config
    └── test_predictions.png # Test set predictions visualization
```

### Prediction Outputs (`mvp_predict_2025.py`)

```
outputs/
└── predictions/
    └── run_YYYYMMDD_HHMMSS/
        ├── predictions_teacher_forcing.csv
        ├── predictions_recursive.csv
        ├── comparison_table.txt
        └── metrics.json
```

## Model Architectures

### RNNWithCategory (Default)

Category-aware LSTM that:
- Embeds category information into a dense vector
- Initializes LSTM hidden state from category embedding
- Concatenates category embedding to input at each timestep
- Uses layer normalization for stability
- Supports multiple categories with shared LSTM weights

### RNNForecastor

Basic RNN without category information (for comparison/baseline).

## Key Components

### Data Module (`src/data/`)

- `DataReader`: Loads CSV data files by year or file pattern
- `ForecastDataset`: PyTorch Dataset for time series windows
- `slicing_window_category`: Creates sliding windows grouped by category
- `encode_categories`: Encodes categories to integer IDs
- `split_data`: Temporal data splitting (train/val/test)
- `add_temporal_features`: Creates cyclical temporal encodings
- `add_cbm_density_features`: Computes CBM/QTY density and last-year prior
- `aggregate_daily`: Aggregates transaction-level data to daily totals

### Models Module (`src/models.py`)

- `RNNWithCategory`: Category-aware LSTM model (primary)
- `RNNForecastor`: Basic RNN model (baseline)

### Training Module (`src/training/`)

- `Trainer`: Complete training and evaluation class with:
  - Training loop with validation
  - Checkpointing and best model saving
  - Learning rate scheduling
  - Evaluation and prediction methods

### Utils Module (`src/utils/`)

- `spike_aware_mse`: Custom loss function (3× weight for top 20% values)
- `plot_difference`: Visualization of predictions vs actuals
- `upload_to_google_sheets`: Optional Google Sheets integration

## Feature Engineering

The system includes comprehensive feature engineering:

1. **Temporal Features**: Cyclical encodings for month and day-of-month
2. **Weekend Features**: Day-of-week and weekend indicators
3. **Lunar Calendar**: Approximate Vietnamese lunar calendar with cyclical encodings
4. **Vietnamese Holidays**: Tet, Mid-Autumn Festival, Independence Day, Labor Day
5. **Tet Countdown**: Continuous countdown feature for Tet anticipation
6. **Rolling Statistics**: 7-day and 30-day rolling means, momentum indicators
7. **CBM Density**: Current and last-year CBM per QTY ratios
8. **Residual Targets**: Optional residual learning against causal baselines

See `MODEL_LOGIC.md` for detailed documentation of all features.

## Documentation

- **`MODEL_LOGIC.md`**: Comprehensive documentation of the model logic, feature engineering, training pipeline, and prediction workflow
- **`PRODUCTION_READINESS_ASSESSMENT.md`**: Assessment of production readiness and recommendations

## Usage Examples

### Basic Training

```bash
# Train with default config (single category, DRY)
python mvp_test.py
```

### Training All Categories

Edit `config/config.yaml`:
```yaml
data:
  category_mode: "all"
```

Then run:
```bash
python mvp_test.py
```

### Prediction

Edit `config/config.yaml`:
```yaml
inference:
  prediction_data_path: "dataset/test/data_2025.csv"
  prediction_start: "2025-01-01"
  prediction_end: "2025-01-31"
```

Then run:
```bash
python mvp_predict_2025.py
```

### Programmatic Usage

```python
from config import load_config
from src.models import RNNWithCategory
from src.training import Trainer
import torch

# Load config
config = load_config()

# Build model
model = RNNWithCategory(
    num_categories=config.model.num_categories,
    cat_emb_dim=config.model.cat_emb_dim,
    input_dim=config.model.input_dim,
    hidden_size=config.model.hidden_size,
    n_layers=config.model.n_layers
)

# Setup trainer
trainer = Trainer(
    model=model,
    criterion=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    save_dir='outputs/models'
)

# Train
train_losses, val_losses = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=config.training.epochs
)
```

## Vietnamese Holiday Calendar

The system includes hardcoded Vietnamese holidays for 2023-2025:

- **Tet (Lunar New Year)**: 7-day windows (varies by year)
- **Mid-Autumn Festival**: Single day (varies by year)
- **Independence Day**: September 2 (fixed)
- **Labor Day**: April 30 - May 1 (fixed)

Holiday dates are defined in `mvp_test.py` and `mvp_predict_2025.py` in the `VIETNAM_HOLIDAYS_BY_YEAR` dictionary.

## License

Internal use - MDLZ Warehouse Forecasting System
