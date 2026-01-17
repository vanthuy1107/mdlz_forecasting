# MDLZ Warehouse Quantity Prediction System

A production-ready deep learning system for predicting warehouse shipment quantities using category-aware LSTM models.

## Project Structure

```
mdlz_wh_prediction/
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
│   │   └── preprocessing.py # Data preprocessing and windowing
│   ├── models/
│   │   ├── __init__.py
│   │   └── rnn_model.py   # Model architectures
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py     # OOP Trainer class
│   └── utils/
│       ├── __init__.py
│       ├── losses.py      # Custom loss functions
│       ├── visualization.py # Plotting utilities
│       └── saving.py      # Saving utilities
├── main.py                # Entry point for training/inference
└── MODEL_LOGIC.md         # Detailed model documentation
```

## Features

- **Modular Architecture**: Clean separation of concerns (data, models, training, utils)
- **Configuration-Driven**: All hyperparameters and paths in `config/config.yaml`
- **OOP Trainer**: Robust `Trainer` class with checkpointing and best model tracking
- **Standardized Data Pipeline**: Proper Dataset and DataLoader flow
- **Production-Ready**: Clean entry point with training and inference modes

## Installation

```bash
# Install required packages
pip install torch numpy pandas scikit-learn matplotlib pyyaml
```

## Quick Start

### Training

```bash
# Train with default config
python main.py --mode train

# Train with custom config
python main.py --mode train --config config/custom_config.yaml
```

### Inference

```bash
# Predict with trained model
python main.py --mode predict --model-path outputs/models/best_model.pth
```

## Configuration

All settings are managed through `config/config.yaml`:

```yaml
# Data Configuration
data:
  data_dir: "../dataset/data_cat"
  years: [2022]
  feature_cols: ["month_sin", "month_cos", "dayofmonth_sin", "dayofmonth_cos", "QTY"]
  train_size: 0.7
  val_size: 0.1
  test_size: 0.2

# Model Configuration
model:
  name: "RNNWithCategory"
  cat_emb_dim: 4
  input_dim: 5
  hidden_size: 32
  n_layers: 2

# Training Configuration
training:
  batch_size: 64
  epochs: 20
  learning_rate: 0.003
  optimizer: "Adam"
  loss: "MSE"
```

## Usage Examples

### Programmatic Usage

```python
from config import load_config
from main import train, predict

# Training
train(config_path='config/config.yaml')

# Inference
y_true, y_pred = predict(
    model_path='outputs/models/best_model.pth',
    config_path='config/config.yaml'
)
```

### Using the Trainer Class

```python
from config import load_config
from src.models import RNNWithCategory
from src.training import Trainer
import torch
import torch.nn as nn

# Load config
config = load_config()

# Build model
model = RNNWithCategory(
    num_categories=10,
    cat_emb_dim=4,
    input_dim=5,
    hidden_size=32,
    n_layers=2
)

# Setup trainer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    save_dir='outputs/models'
)

# Train
train_losses, val_losses = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20
)

# Evaluate
test_loss, y_true, y_pred = trainer.evaluate(
    test_loader,
    return_predictions=True
)
```

## Data Pipeline

The system expects data in CSV format with the following columns:
- `ACTUALSHIPDATE`: Date column for temporal sorting
- `CATEGORY`: Product category (will be encoded to `CATEGORY_ID`)
- `QTY`: Target quantity to predict
- Temporal features: `month_sin`, `month_cos`, `dayofmonth_sin`, `dayofmonth_cos`

Data files should be located at: `{data_dir}/data_{year}.csv`

## Output

Training produces:
- `outputs/models/best_model.pth`: Best model checkpoint
- `outputs/learning_curve.png`: Training/validation loss curves
- `outputs/test_predictions.png`: Test set predictions visualization
- `outputs/train_fit.png`: Training set predictions visualization
- `outputs/predictions.txt`: Numerical predictions

## Model Architectures

### RNNWithCategory (Default)
Category-aware LSTM that:
- Embeds category information
- Initializes LSTM hidden state from category embedding
- Concatenates category embedding to input at each timestep

### RNNForecastor
Basic RNN without category information (for comparison).

## Key Components

### Data Module (`src/data/`)
- `DataReader`: Loads CSV data files
- `ForecastDataset`: PyTorch Dataset for time series
- `slicing_window_category`: Creates sliding windows grouped by category
- `encode_categories`: Encodes categories to integer IDs
- `split_data`: Temporal data splitting

### Models Module (`src/models/`)
- `RNNWithCategory`: Category-aware LSTM model
- `RNNForecastor`: Basic RNN model

### Training Module (`src/training/`)
- `Trainer`: Complete training and evaluation class with:
  - Training loop with validation
  - Checkpointing and best model saving
  - Learning rate scheduling
  - Evaluation and prediction methods

### Utils Module (`src/utils/`)
- `spike_aware_mse`: Custom loss for handling demand spikes
- `plot_difference`: Visualization of predictions vs actuals
- `plot_learning_curve`: Training curves
- `save_pred_actual_txt`: Save predictions to text file

## Migration from Old Structure

The old flat structure has been refactored as follows:

- `data_utils.py` → `src/data/` (loader.py, dataset.py, preprocessing.py)
- `model.py` → `src/models/rnn_model.py`
- `solver.py` + `train.py` → `src/training/trainer.py` + `main.py`
- `loss.py` → `src/utils/losses.py`
- `visualize.py` → `src/utils/visualization.py`
- `save.py` → `src/utils/saving.py`

## License

Internal use - MDLZ Warehouse Prediction System

