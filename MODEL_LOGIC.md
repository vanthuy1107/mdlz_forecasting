# Model Logic Documentation

## Project Overview
This is a **Warehouse Quantity Prediction System** for MDLZ (Mondelēz International) that uses deep learning to forecast warehouse shipment quantities based on historical data and product categories.

---

## 1. Problem Definition

**Objective**: Predict future shipment quantities (`QTY`) for different product categories based on:
- Historical quantity patterns
- Temporal features (month, day of month) - cyclical encoding
- Holiday features (holiday indicators, days until next holiday)
- Product category information

**Type**: Time Series Forecasting with Category Information and Holiday Context

---

## 2. Data Pipeline

### 2.1 Data Preparation (`combine_data.py`)

Before training, raw data files are combined into yearly files:
- Input: Files like `Outboundreports_YYYYMMDD_YYYYMMDD.csv` in `dataset/data_cat/`
- Output: Combined files `data_{year}.csv` (e.g., `data_2023.csv`, `data_2024.csv`)
- Process: Groups files by year, concatenates, sorts by date, removes duplicates
- Usage: Run `python combine_data.py` before training

### 2.2 Data Loading (`DataReader`)

Located in `src/data/loader.py`:
- Loads CSV files from `dataset/data_cat/data_{year}.csv`
- Supports multiple years concatenation
- Fallback: Can load by file pattern if combined files don't exist
- Methods: `load(years)`, `load_year(year)`, `load_by_file_pattern(years, file_prefix)`

### 2.3 Feature Engineering

#### Temporal Features (Cyclical Encoding):
```
- month_sin: sin(2π × (month - 1) / 12)
- month_cos: cos(2π × (month - 1) / 12)
- dayofmonth_sin: sin(2π × (day - 1) / 31)
- dayofmonth_cos: cos(2π × (day - 1) / 31)
```

**Why Cyclical Encoding?**
- Captures periodic patterns (e.g., December is close to January)
- Prevents artificial ordering (month 12 ≠ 12× month 1)
- Maintains continuity in temporal space

#### Holiday Features:
```
- holiday_indicator: Binary (0 or 1) indicating if date is a US holiday
- days_until_next_holiday: Number of days until the next holiday
```

**Holidays Included**: New Year's Day, MLK Day, Presidents' Day, Memorial Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas

#### Quantity Feature:
```
- QTY: Historical quantity values (included in input window)
```

**Total Features**: 7 features per timestep (4 temporal + 2 holiday + 1 quantity)

#### Category Encoding:
Located in `src/data/preprocessing.py`:
```python
categories = sorted(data["CATEGORY"].unique())
cat2id = {cat: i for i, cat in enumerate(categories)}
data["CATEGORY_ID"] = data["CATEGORY"].map(cat2id)
```

### 2.4 Data Splitting
```python
Train: 70% (temporal split)
Validation: 10%
Test: 20%
```

**Important**: Uses temporal split (not random) to avoid data leakage

### 2.5 Window Slicing (`slicing_window_category`)

Located in `src/data/preprocessing.py`. Creates overlapping sequences for time series prediction:

```
For each category group:
  Input window: [t-30, t-29, ..., t-2, t-1]
  Target: [t]
  
  Features per timestep: [month_sin, month_cos, dayofmonth_sin, dayofmonth_cos, 
                          holiday_indicator, days_until_next_holiday, QTY]
```

**Parameters** (from `config/config.yaml`):
- `input_size = 30`: Look back 30 time steps
- `horizon = 1`: Predict 1 step ahead
- `feature_cols`: 7 features (4 temporal + 2 holiday + 1 quantity)
- `target_col`: QTY to predict

**Output Shapes**:
```
X: (N_samples, 30, 7)  # N samples, 30 timesteps, 7 features
y: (N_samples, 1)      # N samples, 1 prediction
cat: (N_samples,)      # N samples, category ID
```

**Dataset Creation**:
- Uses `ForecastDataset` class from `src/data/dataset.py`
- Returns tuples of (X, cat, y) for each sample

---

## 3. Model Architecture

### 3.1 RNNWithCategory (Currently Used)

Located in `src/models/rnn_model.py`.

**Architecture Components**:

```
Input: 
  - x_seq: (Batch, Time=30, Features=7)
  - x_cat: (Batch, 1) or (Batch,)

Layer 1: Category Embedding
  - Embedding(num_categories, emb_dim=4)
  - Maps category ID → dense vector
  
Layer 2: Hidden State Initialization
  - h0_fc: Linear(emb_dim=4, hidden_size=32)
  - Initializes LSTM hidden state from category embedding
  - Repeated across num_layers (2 layers)
  - Cell state c0: zeros
  
Layer 3: Feature Concatenation
  - Concatenate time-series features with category embedding
  - Input shape: (Batch, Time=30, Features=7+4=11)
  
Layer 4: LSTM
  - input_size: 11 (7 time features + 4 category embedding)
  - hidden_size: 32
  - num_layers: 2
  - batch_first: True
  
Layer 5: Output Layer
  - Takes last timestep output: out[:, -1, :]
  - fc: Linear(hidden_size=32, output_dim=1)
  
Output: (Batch, 1) - predicted quantity
```

**Forward Pass**:
```python
1. x_cat → cat_vec (B, 4)
2. cat_vec → h0 (num_layers=2, B, 32) via h0_fc
3. cat_vec expanded over time → cat_seq (B, T=30, 4)
4. [x_seq, cat_seq] concatenated → x (B, T=30, 11)
5. LSTM(x, (h0, c0)) → out (B, T=30, 32)
6. Take last timestep: out[:, -1, :] → (B, 32)
7. fc → prediction (B, 1)
```

**Key Design Choices**:
- **Category embedding in hidden state**: Allows LSTM to condition its processing on product category from the start
- **Category embedding concatenated to input**: Provides category context at every timestep
- **Dual category integration**: Both initialization and concatenation ensure category information flows throughout

### 3.2 RNNForecastor (Alternative Baseline)

Located in `src/models/rnn_model.py`. Simpler baseline model without category information:

```
Input: (Batch, Time, Features)
  ↓
RNN(input_size=embedding_dim, hidden_size=32, layers=2)
  ↓
Take last timestep output
  ↓
LayerNorm (optional, currently commented out)
  ↓
Dropout(p=0.2)
  ↓
Linear(hidden_size → output_dim)
  ↓
Output: (Batch, output_dim)
```

**Note**: Currently not used as default model; can be selected via config.

---

## 4. Configuration System

The project uses YAML-based configuration via `config/config.yaml` and `config/config.py`.

### 4.1 Configuration File Structure

```yaml
data:
  data_dir: "./dataset/data_cat"
  file_pattern: "data_{year}.csv"
  years: [2023, 2024]
  feature_cols: [month_sin, month_cos, dayofmonth_sin, dayofmonth_cos, 
                 holiday_indicator, days_until_next_holiday, QTY]
  target_col: "QTY"
  cat_col: "CATEGORY"
  train_size: 0.7
  val_size: 0.1
  test_size: 0.2

window:
  input_size: 30
  horizon: 1

model:
  name: "RNNWithCategory"
  num_categories: null  # Set dynamically
  cat_emb_dim: 4
  input_dim: 7
  hidden_size: 32
  n_layers: 2
  output_dim: 1

training:
  batch_size: 64
  val_batch_size: 16
  test_batch_size: 16
  epochs: 20
  learning_rate: 0.003
  optimizer: "Adam"
  loss: "MSE"  # or "spike_aware_mse"
  device: "auto"  # auto, cuda, or cpu
```

### 4.2 Hyperparameters (Current Defaults)

```python
# Model
num_categories: Dynamic (based on data)
cat_emb_dim: 4
input_dim: 7  # 4 temporal + 2 holiday + 1 quantity
hidden_size: 32
n_layers: 2
output_dim: 1
dropout_prob: 0.2 (only in RNNForecastor)

# Window
input_size: 30  # Look back 30 timesteps
horizon: 1      # Predict 1 step ahead

# Training
batch_size: 64 (train), 16 (val/test)
learning_rate: 0.003
epochs: 20
optimizer: Adam
```

### 4.2 Loss Function

**MSELoss** (Mean Squared Error):
```
Loss = (1/N) Σ(predicted - actual)²
```

**Alternative Available**: `spike_aware_mse` (in `src/utils/losses.py`)
- Assigns 3x weight to top 20% values (spikes)
- Useful for handling sudden demand surges
- Can be selected via config: `training.loss: "spike_aware_mse"`

### 4.3 Learning Rate Scheduler

**ReduceLROnPlateau**:
```python
mode: "min"           # Minimize validation loss
factor: 0.5           # Reduce LR by half
patience: 3           # Wait 3 epochs before reduction
min_lr: 1e-5          # Minimum learning rate
```

**Behavior**:
- Monitors validation loss
- If no improvement for 3 epochs → LR × 0.5
- Prevents overfitting and helps convergence

---

## 5. Training Process

### 5.1 Trainer Class (`src/training/trainer.py`)

The project uses an OOP `Trainer` class for training and evaluation.

**Initialization**:
```python
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    log_interval=5,
    save_dir="../outputs/models"
)
```

**Features**:
- Automatic best model tracking and saving
- Learning curve history
- Checkpoint saving/loading
- Progress logging

### 5.2 Training Loop (`fit` method)

```python
train_losses, val_losses = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    save_best=True,
    verbose=True
)
```

**Process**:
1. For each epoch:
   - Train one epoch (`train_epoch`)
   - Evaluate on validation set
   - Update learning rate scheduler
   - Save best model if validation loss improves
   - Log metrics every `log_interval` epochs

2. After training:
   - Load best model state based on validation loss
   - Return training and validation loss histories

### 5.3 Evaluation (`evaluate` method)

```python
avg_loss, y_true, y_pred = trainer.evaluate(
    dataloader=test_loader,
    return_predictions=True
)
```

**Process**:
- Set model to eval mode
- Disable gradient computation (with `torch.no_grad()`)
- For each batch:
  - Forward pass
  - Compute loss
  - Collect predictions and labels (if `return_predictions=True`)
- Return average loss and optionally predictions

### 5.4 Prediction (`predict` method)

```python
y_true, y_pred = trainer.predict(dataloader)
```

Returns predictions and true labels for visualization or analysis.

---

## 6. Prediction & Visualization

### 6.1 Inference

```python
model.eval()
with torch.no_grad():
    outputs = model(inputs, cat)
```

### 6.2 Evaluation Metrics

- **Primary**: MSE (Mean Squared Error)
- **Visual**: Actual vs Predicted plots

### 6.3 Output Files

Located in `outputs/` directory (configurable via `config.output.output_dir`):

```
outputs/
  ├── learning_curve.png        # Train/val loss over epochs
  ├── train_fit.png             # Training set predictions
  ├── test_predictions.png      # Test set predictions
  ├── predictions.txt           # Numerical predictions (actual vs predicted)
  └── models/
      └── best_model.pth        # Best model checkpoint (if save_model=True)
```

**Visualization Functions** (in `src/utils/visualization.py`):
- `plot_learning_curve()`: Plots training/validation loss over epochs
- `plot_difference()`: Plots actual vs predicted values

---

## 7. Model Workflow Summary

```
1. Data Preparation (combine_data.py)
   Raw files → Combined yearly files (data_YYYY.csv)
   
2. Main Pipeline (main.py)
   ├── Load Configuration (config/config.yaml)
   ├── Data Pipeline (prepare_data_pipeline)
   │   ├── Load Data (DataReader)
   │   ├── Add Temporal Features (month_sin, month_cos, etc.)
   │   ├── Add Holiday Features
   │   ├── Encode Categories
   │   ├── Temporal Split (70/10/20)
   │   └── Create Windows (input_size=30)
   ├── Build Model (RNNWithCategory)
   ├── Build Criterion, Optimizer, Scheduler
   ├── Train (Trainer.fit)
   │   ├── Train Epoch
   │   ├── Validate
   │   ├── Update LR Scheduler
   │   └── Save Best Model
   ├── Evaluate on Test Set
   └── Generate Visualizations
       ├── Learning Curve
       ├── Test Predictions
       ├── Train Predictions
       └── Save Predictions Text
```

**Entry Point**: `python main.py --mode train --config config/config.yaml`

---

## 8. Key Technical Decisions

### 8.1 Why LSTM over RNN?
- **LSTM** (RNNWithCategory): Better at capturing long-term dependencies
- Can handle vanishing gradient problem better
- Has cell state for maintaining long-term memory

### 8.2 Why Category Embedding?
- Different product categories have different demand patterns
- Learned embedding captures category-specific characteristics
- Enables transfer learning across similar categories

### 8.3 Why Batch First?
- `batch_first=True` → shape: (Batch, Time, Features)
- More intuitive for processing
- Matches typical data organization

### 8.4 Why Take Last Timestep?
```python
out = out[:, -1, :]  # Take output at last timestep
```
- Sequence-to-one prediction (not sequence-to-sequence)
- Last hidden state contains aggregated information from entire sequence
- Predicting 1 step ahead based on 30 historical steps

### 8.5 Why Configuration File?
- Centralized hyperparameter management
- Easy experiment tracking and reproducibility
- No code changes needed for different configurations
- Supports dynamic values (e.g., `num_categories` set from data)

---

## 9. Potential Improvements

### 9.1 Model Enhancements
- [ ] Add attention mechanism to weigh important timesteps
- [ ] Try GRU instead of LSTM (fewer parameters)
- [ ] Add residual connections for deeper networks
- [ ] Implement multi-horizon prediction (predict multiple steps ahead)

### 9.2 Feature Engineering
- [x] Include holiday indicators (implemented)
- [ ] Add day-of-week features (cyclical)
- [ ] Add seasonal trend decomposition
- [ ] Incorporate external variables (promotions, weather)
- [ ] Add lag features (previous day/week values)

### 9.3 Training Improvements
- [ ] Use `spike_aware_mse` for imbalanced data
- [ ] Implement early stopping
- [ ] Add gradient clipping for stability
- [ ] Use cross-validation for hyperparameter tuning

### 9.4 Data Processing
- [ ] Normalize/standardize input features
- [ ] Handle missing values explicitly
- [ ] Add data augmentation (noise injection)
- [ ] Implement stratified sampling by category

---

## 10. File Structure

```
mdlz_wh_prediction/
├── main.py                    # Main entry point (train/predict)
├── combine_data.py            # Data preparation script
├── config/
│   ├── config.py              # Configuration management class
│   └── config.yaml            # Configuration file (hyperparameters)
├── src/
│   ├── data/
│   │   ├── loader.py          # DataReader class for loading CSV files
│   │   ├── preprocessing.py   # Feature engineering, windowing, splitting
│   │   └── dataset.py         # ForecastDataset PyTorch dataset class
│   ├── models/
│   │   └── rnn_model.py       # RNNWithCategory, RNNForecastor models
│   ├── training/
│   │   └── trainer.py         # Trainer class for training/evaluation
│   └── utils/
│       ├── losses.py          # Custom loss functions (spike_aware_mse)
│       ├── visualization.py   # Plotting utilities
│       └── saving.py          # Save predictions and results
├── dataset/
│   ├── data_cat/              # Input data directory
│   │   ├── data_2023.csv      # Combined yearly files
│   │   ├── data_2024.csv
│   │   └── ...
│   └── data_2025.csv
├── outputs/                   # Output directory (generated)
│   ├── models/                # Saved model checkpoints
│   ├── learning_curve.png
│   ├── train_fit.png
│   ├── test_predictions.png
│   └── predictions.txt
├── requirements.txt           # Python dependencies
├── README.md                  # Project README
└── MODEL_LOGIC.md             # This documentation
```

---

## 11. Usage Example

### 11.1 Command Line Usage

**Step 1: Prepare Data**
```bash
python combine_data.py
```

**Step 2: Train Model**
```bash
python main.py --mode train --config config/config.yaml
```

**Step 3: Make Predictions**
```bash
python main.py --mode predict --config config/config.yaml --model-path outputs/models/best_model.pth
```

### 11.2 Programmatic Usage

```python
from config import load_config
from main import train, predict

# Train with default config
trainer, config, data_dict = train()

# Or with custom config
trainer, config, data_dict = train(config_path="config/config.yaml")

# Make predictions
y_true, y_pred = predict(
    model_path="outputs/models/best_model.pth",
    config_path="config/config.yaml"
)
```

### 11.3 Custom Configuration

```python
from config import load_config

# Load config
config = load_config("config/config.yaml")

# Modify programmatically
config.set('training.epochs', 50)
config.set('training.learning_rate', 0.001)
config.set('data.years', [2023, 2024, 2025])

# Use in training
trainer, _, _ = train()
```

---

## 12. Dependencies

See `requirements.txt` for full list. Key dependencies:

```
torch          # PyTorch for deep learning
numpy          # Numerical operations
pandas         # Data manipulation
scikit-learn   # Preprocessing utilities (train_test_split)
matplotlib     # Visualization (plotting)
pyyaml         # YAML configuration parsing
```

**Installation**:
```bash
pip install -r requirements.txt
```

---

## Conclusion

This warehouse prediction system leverages **category-aware LSTM** architecture to forecast shipment quantities. The model:
- Integrates product category information via embeddings
- Uses cyclical time features to capture temporal patterns
- Incorporates holiday features to account for seasonal demand variations
- Employs a robust OOP training pipeline with learning rate scheduling
- Uses YAML-based configuration for easy hyperparameter management
- Provides comprehensive evaluation and visualization

**Key Improvements Over Previous Version**:
- Extended input window from 6 to 30 timesteps for better context
- Added holiday features (2 additional features)
- Restructured into modular package structure (`src/` directory)
- Introduced configuration management system
- Added data preparation script (`combine_data.py`)
- Implemented OOP Trainer class with checkpoint management

The dual integration of category information (initialization + concatenation) allows the model to specialize its predictions for different product categories while learning from shared temporal patterns. The extended window size and holiday features enable better capture of long-term trends and seasonal variations.

