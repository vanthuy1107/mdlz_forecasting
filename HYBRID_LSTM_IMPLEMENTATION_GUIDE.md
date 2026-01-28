
# Multi-Input Hybrid LSTM Implementation Guide
## Transitioning to Dual-Branch Architecture with Leak-Free Feature Engineering

**Version**: 1.0  
**Date**: January 2026  
**Status**: ✅ Implementation Complete - Ready for Testing

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Components](#implementation-components)
4. [Anti-Leakage Protocol](#anti-leakage-protocol)
5. [Usage Guide](#usage-guide)
6. [Configuration](#configuration)
7. [Training Pipeline](#training-pipeline)
8. [Validation & Testing](#validation--testing)
9. [Migration from Legacy Models](#migration-from-legacy-models)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### Motivation

The legacy single-input LSTM architecture treats all features uniformly, which has several limitations:

1. **Feature Heterogeneity**: Temporal sequences (CBM history) and static metadata (brand ID, day-of-week stats) have fundamentally different characteristics but are mixed in the same input tensor.

2. **Loss of Domain Knowledge**: Hand-crafted features like "average Monday volume" or "holiday proximity" are treated the same as raw time-series data, losing their interpretive value.

3. **Data Leakage Risk**: Computing statistical features (e.g., "average volume for this weekday") using the entire dataset introduces future information into past predictions.

4. **Under-Forecasting Bias**: Standard MSE loss treats under-forecasting and over-forecasting equally, but in FMCG supply chain, stock-outs (under-forecasting) are typically 3-5x more costly than excess inventory.

### Solution: Multi-Input Hybrid LSTM

The **Hybrid LSTM** architecture addresses these issues through:

- **Dual-Branch Processing**: Separate pathways for temporal sequences (LSTM) and static metadata (Dense MLP).
- **Feature Fusion**: Intelligent combination of temporal trends and operational constraints.
- **Expanding Window Features**: Strict time-series validation using only past data for feature computation.
- **Asymmetric Loss**: Penalizes under-forecasting more heavily than over-forecasting.

---

## Architecture

### Conceptual Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Input: Temporal Sequence (30 days x D features)            │
│         + Static Features (Brand, Stats, Holidays)          │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
     Branch A: LSTM                Branch B: Dense
     (Temporal Path)               (Static/Meta Path)
             │                            │
     [LSTM Layers]                  [MLP Layers]
     Hidden State H             Contextual Embedding C
             │                            │
             └────────────┬───────────────┘
                          │
                [Concatenate: H || C]
                          │
                [Fusion Dense Layers]
                          │
                     [Output Layer]
                          │
                Forecast (30 days)
```

### Branch A: Temporal Sequence Processor (LSTM)

**Purpose**: Capture short-term fluctuations, momentum, and recent trends.

**Input**: Historical CBM time-series (e.g., 30-day sliding window)
- `Total CBM` (historical values)
- `rolling_mean_7d_expanding` (7-day rolling average)
- `rolling_mean_30d_expanding` (30-day rolling average)
- `momentum_3d_vs_14d_expanding` (momentum indicator)

**Architecture**:
- 2-layer Stacked LSTM
- Hidden size: 128 (configurable per category)
- Category embedding for hidden state initialization
- LayerNorm for stabilization
- Dropout for regularization (0.2)

**Output**: Hidden state $H$ ∈ $\mathbb{R}^{128}$

### Branch B: Contextual Knowledge Base (Dense Stream)

**Purpose**: Provide domain knowledge and operational constraints that raw LSTM might struggle to learn from noisy daily data.

**Input**: Static/meta features (1D vector)

**Cyclical Temporal Encoding**:
- `month_sin`, `month_cos` (seasonal patterns)
- `dayofmonth_sin`, `dayofmonth_cos` (monthly cycles)
- `day_of_week_sin`, `day_of_week_cos` (weekly patterns)

**Event Proximity Triggers**:
- `holiday_indicator` (binary: is today a holiday?)
- `days_until_next_holiday` (countdown feature)
- `days_since_holiday` (recency feature)

**Operational Constraints**:
- `is_weekend` (binary: Sundays typically have zero volume)
- `Is_Monday` (binary: Mondays often have peaks for FRESH)
- `is_EOM` (binary: end-of-month surge indicator)
- `days_until_month_end` (countdown to EOM)
- `weekday_volume_tier` (tiered encoding: Wed > Fri > Tue/Thu)
- `is_high_volume_weekday` (binary: Wed or Fri)

**Statistical Metadata (Expanding Window - NO LEAKAGE)**:
- `weekday_avg_expanding` (average volume for this weekday, using only past data)
- `category_avg_expanding` (category average, using only past data)

**Architecture**:
- Multi-layer MLP: [64, 32]
- ReLU activations
- LayerNorm between layers
- Dropout (0.2)

**Output**: Contextual embedding $C$ ∈ $\mathbb{R}^{32}$

### Feature Fusion Layer

**Purpose**: Learn complex interactions between temporal trends ($H$) and operational reality ($C$).

**Input**: Concatenated vector $[H || C]$ ∈ $\mathbb{R}^{160}$

**Architecture**:
- Fusion layers: [64]
- ReLU activations
- LayerNorm
- Dropout (0.2)

**Output**: Fused representation $F$ ∈ $\mathbb{R}^{64}$

**Inference Logic**:
The fusion layer acts as a decision engine:
- If temporal trend suggests high volume BUT it's a Sunday → Output low volume (operational constraint overrides)
- If temporal trend suggests low volume BUT there's an upcoming holiday → Output higher volume (event anticipation)

### Output Layer

**Purpose**: Direct multi-step forecasting to avoid exposure bias.

**Architecture**: Single linear layer $\mathbb{R}^{64} \rightarrow \mathbb{R}^{30}$

**Output**: 30-day forecast horizon (all steps predicted simultaneously)

---

## Implementation Components

### 1. Model Architecture

**File**: `src/models.py`

**Class**: `HybridLSTM`

**Key Parameters**:
```python
HybridLSTM(
    num_categories=1,           # Single category per model (category-specific training)
    cat_emb_dim=5,              # Category embedding dimension
    temporal_input_dim=4,       # Number of temporal features (CBM, rolling, momentum)
    static_input_dim=15,        # Number of static features (cyclical, holidays, etc.)
    hidden_size=128,            # LSTM hidden size
    n_layers=2,                 # Number of LSTM layers
    dense_hidden_sizes=[64, 32],# Dense branch MLP layers
    fusion_hidden_sizes=[64],   # Fusion layer sizes
    output_dim=30,              # Forecast horizon (30 days)
    dropout_prob=0.2,           # Dropout probability
    use_layer_norm=True,        # Apply LayerNorm
)
```

**Forward Pass Signature**:
```python
def forward(
    self,
    x_temporal: Tensor,  # (batch, 30, 4) - temporal sequences
    x_static: Tensor,    # (batch, 15) - static features
    x_cat: Tensor,       # (batch,) - category IDs
) -> Tensor:
    # Returns: (batch, 30) - 30-day forecast
```

### 2. Leak-Free Feature Engineering

**File**: `src/data/expanding_features.py`

**Functions**:

#### `add_expanding_statistical_features()`
Computes statistical features (mean, std) using ONLY past data.

**Example**:
```python
# BAD: Uses entire dataset (introduces future information)
df['weekday_avg'] = df.groupby('day_of_week')['Total CBM'].transform('mean')

# GOOD: Uses expanding window (only past data)
df = add_expanding_statistical_features(
    df,
    target_col='Total CBM',
    features_to_add=['weekday_avg', 'category_avg'],
)
# Creates: weekday_avg_expanding, category_avg_expanding
```

**How it Works**:
For each row at time $t$:
1. Filter data to times $< t$ (strictly before)
2. Compute statistics using only this past data
3. Assign to current row

**Anti-Leakage Guarantee**: No future information is used.

#### `add_expanding_rolling_features()`
Computes rolling means using `.shift(1)` to exclude current value.

**Example**:
```python
df = add_expanding_rolling_features(
    df,
    target_col='Total CBM',
    windows=[7, 30],  # 7-day and 30-day rolling means
)
# Creates: rolling_mean_7d_expanding, rolling_mean_30d_expanding
```

#### `add_expanding_momentum_features()`
Computes momentum (short-term vs long-term trend).

**Example**:
```python
df = add_expanding_momentum_features(
    df,
    target_col='Total CBM',
    short_window=3,   # 3-day average
    long_window=14,   # 14-day average
)
# Creates: momentum_3d_vs_14d_expanding
```

#### `add_brand_tier_feature()`
Computes brand's contribution to category volume using expanding window.

**Example**:
```python
df = add_brand_tier_feature(
    df,
    target_col='Total CBM',
    brand_col='BRAND',
    cat_col='CATEGORY',
)
# Creates: brand_tier_expanding
```

### 3. Dataset Class

**File**: `src/data/dataset.py`

**Class**: `HybridForecastDataset`

**Purpose**: Separate temporal and static features for dual-branch architecture.

**Usage**:
```python
from src.data.dataset import HybridForecastDataset

# X_temporal: (N, 30, 4) - temporal sequences
# X_static: (N, 15) - static features
# y: (N, 30) - targets (30-day forecast)

dataset = HybridForecastDataset(
    X_temporal=X_temporal,
    X_static=X_static,
    y=y,
    cat=category_ids,  # optional
)

# Returns: (X_temporal, X_static, cat, y) for each sample
```

### 4. Asymmetric Loss Function

**File**: `src/utils/losses.py`

**Class**: `AsymmetricMSELoss`

**Purpose**: Penalize under-forecasting more than over-forecasting.

**Mathematical Formula**:
$$
L(y, \hat{y}) = \begin{cases}
\alpha \cdot (y - \hat{y})^2 & \text{if } y > \hat{y} \text{ (under-forecast)} \\
\beta \cdot (\hat{y} - y)^2 & \text{if } y \leq \hat{y} \text{ (over-forecast)}
\end{cases}
$$

**Recommended Parameters**:
- $\alpha = 3.0$ (under-forecast penalty)
- $\beta = 1.0$ (over-forecast penalty)
- Ratio: 3:1 (under-forecasting is 3x more costly)

**Usage**:
```python
from src.utils.losses import AsymmetricMSELoss

loss_fn = AsymmetricMSELoss(
    under_penalty=3.0,
    over_penalty=1.0,
    peak_season_boost=2.0,  # 2x weight during TET/Mid-Autumn
)

loss = loss_fn(y_pred, y_true, is_peak_season=peak_indicator)
```

**Why Asymmetric?**
In FMCG supply chain:
- **Stock-out** (under-forecast) → Lost sales, customer dissatisfaction, reputation damage
- **Over-stock** (over-forecast) → Holding costs, potential waste (for perishables)

Typically, stock-out cost is 3-5x higher than over-stock cost.

---

## Anti-Leakage Protocol

### The Problem: Data Leakage

**Data Leakage** occurs when the model has access to information from the future during training, making it appear more accurate than it would be in production.

**Example Violation**:
```python
# WRONG: Calculate average Monday volume using entire dataset
monday_mask = df['day_of_week'] == 0
df['avg_monday_volume'] = df[monday_mask]['Total CBM'].mean()

# This introduces future information!
# For a Monday in January 2023, we're using data from December 2024.
```

**Why It's Critical**:
1. **Overestimated Performance**: Model appears to have 95% accuracy in testing but fails in production (60% accuracy).
2. **Invalid Predictions**: Model cannot be deployed because required features don't exist for future dates.
3. **Regulatory/Audit Issues**: For financial/critical applications, data leakage invalidates the entire model.

### The Solution: Expanding Windows

**Expanding Window** computes features at time $t$ using ONLY data from times $< t$.

**Correct Implementation**:
```python
# RIGHT: Calculate average Monday volume using expanding window
df = df.sort_values(['CATEGORY', 'ACTUALSHIPDATE'])

for idx in range(len(df)):
    current_date = df.iloc[idx]['ACTUALSHIPDATE']
    past_data = df.iloc[:idx]  # Only rows BEFORE current
    
    if df.iloc[idx]['day_of_week'] == 0:  # Monday
        past_mondays = past_data[past_data['day_of_week'] == 0]
        avg_monday_volume = past_mondays['Total CBM'].mean()
        df.at[idx, 'avg_monday_volume_expanding'] = avg_monday_volume
```

### Validation Protocol

**Step 1**: Run the validation script BEFORE training.

```bash
python validate_no_leakage.py --data-path dataset/data_cat/data_2024.csv
```

**Tests Performed**:
1. **Expanding Window Stability**: Early rows have reasonable feature values (not wildly different from later rows).
2. **No Look-Ahead Bias**: Features at time $t$ match manually calculated values using only past data.
3. **Window Slicing Correctness**: Input window $X$ ends before target window $y$ starts (no overlap).
4. **Temporal Split Validity**: Train dates < Val dates < Test dates (no time travel).

**Expected Output**:
```
============================================================
VALIDATION SUMMARY
============================================================

Total tests: 5
Passed: 5
Failed: 0

Detailed results:
  [PASS] expanding_window_stability
  [PASS] no_lookahead_bias
  [PASS] no_window_overlap
  [PASS] correct_gap
  [PASS] temporal_order

============================================================
✓ ALL TESTS PASSED - No data leakage detected!
============================================================
```

**Step 2**: During training, use temporal splits (no shuffling).

```python
# Temporal split (chronological)
train_df = df[df['ACTUALSHIPDATE'] < '2024-07-01']
val_df = df[(df['ACTUALSHIPDATE'] >= '2024-07-01') & 
            (df['ACTUALSHIPDATE'] < '2024-10-01')]
test_df = df[df['ACTUALSHIPDATE'] >= '2024-10-01']
```

**Step 3**: For recursive prediction (future dates), use only model's own predictions.

```python
# For January 2025 prediction, use historical data up to 2024-12-31
initial_window = df[df['ACTUALSHIPDATE'] <= '2024-12-31'].tail(30)

# For each future day, use model's prediction (not actual value)
predictions = []
current_window = initial_window.copy()

for date in pd.date_range('2025-01-01', '2025-01-31'):
    pred = model.predict(current_window)
    predictions.append(pred)
    
    # Update window: remove oldest, add prediction
    current_window = current_window.iloc[1:].append(pred)
```

---

## Usage Guide

### Quick Start (5 Steps)

#### Step 1: Prepare Data with Leak-Free Features

```python
from train_hybrid_lstm import prepare_data_with_expanding_features
import pandas as pd

# Load raw data
df = pd.read_csv('dataset/data_cat/data_2024.csv')
df = df[df['CATEGORY'] == 'DRY']  # Filter to DRY category

# Add all features (expanding windows, no leakage)
df_prepared = prepare_data_with_expanding_features(
    df,
    target_col='Total CBM',
    time_col='ACTUALSHIPDATE',
    cat_col='CATEGORY',
)
```

**Output**:
```
============================================================
PREPARING DATA WITH EXPANDING WINDOW FEATURES
============================================================

1. Adding calendar-based features (no leakage)...
   ✓ Added: month_sin, month_cos, dayofmonth_sin, dayofmonth_cos
   ✓ Added: holiday_indicator, days_until_next_holiday
   ✓ Added: weekday_volume_tier, is_high_volume_weekday
   ✓ Added: is_EOM, days_until_month_end
   ✓ Added: day_of_week_sin, day_of_week_cos, is_weekend, Is_Monday

2. Adding expanding window statistical features (LEAK-FREE)...
   ✓ Added: weekday_avg_expanding, category_avg_expanding
   ✓ Added: rolling_mean_7d_expanding, rolling_mean_30d_expanding
   ✓ Added: momentum_3d_vs_14d_expanding

3. Feature preparation complete!
   Total features: 25
   Total rows: 2000
```

#### Step 2: Define Feature Groups

```python
# Temporal features (for LSTM branch)
temporal_features = [
    'Total CBM',
    'rolling_mean_7d_expanding',
    'rolling_mean_30d_expanding',
    'momentum_3d_vs_14d_expanding',
]

# Static features (for Dense branch)
static_features = [
    # Cyclical encodings
    'month_sin', 'month_cos',
    'dayofmonth_sin', 'dayofmonth_cos',
    'day_of_week_sin', 'day_of_week_cos',
    # Holiday features
    'holiday_indicator', 'days_until_next_holiday', 'days_since_holiday',
    # Operational features
    'is_weekend', 'Is_Monday', 'is_EOM', 'days_until_month_end',
    'weekday_volume_tier', 'is_high_volume_weekday',
    # Statistical metadata
    'weekday_avg_expanding', 'category_avg_expanding',
]
```

#### Step 3: Create Datasets

```python
from train_hybrid_lstm import create_hybrid_datasets

train_dataset, val_dataset, test_dataset = create_hybrid_datasets(
    df_prepared,
    temporal_features=temporal_features,
    static_features=static_features,
    target_col='Total CBM',
    input_size=30,  # 30-day input window
    horizon=30,     # 30-day forecast horizon
    train_size=0.7,
    val_size=0.1,
)
```

#### Step 4: Initialize Model

```python
from src.models import HybridLSTM

model = HybridLSTM(
    num_categories=1,                      # Single category (DRY)
    cat_emb_dim=5,
    temporal_input_dim=len(temporal_features),  # 4 temporal features
    static_input_dim=len(static_features),      # 17 static features
    hidden_size=128,                       # LSTM hidden size
    n_layers=2,                            # 2 LSTM layers
    dense_hidden_sizes=[64, 32],           # Dense branch: [64, 32]
    fusion_hidden_sizes=[64],              # Fusion layer: [64]
    output_dim=30,                         # 30-day forecast
    dropout_prob=0.2,
    use_layer_norm=True,
)
```

#### Step 5: Train Model

```python
from torch.utils.data import DataLoader
from train_hybrid_lstm import train_hybrid_model

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train model
model, train_losses, val_losses = train_hybrid_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    learning_rate=0.001,
    loss_fn=None,  # Default: AsymmetricMSELoss(under_penalty=3.0, over_penalty=1.0)
    device='cuda',
    save_path='outputs/hybrid_lstm/DRY/best_model.pth',
)
```

**Training Output**:
```
============================================================
TRAINING HYBRID LSTM MODEL
============================================================

Using Asymmetric MSE Loss (under_penalty=3.0, over_penalty=1.0)

Training for 20 epochs...
Device: cuda
Learning rate: 0.001
Optimizer: Adam
Scheduler: ReduceLROnPlateau

Epoch 1/20 - Train Loss: 1250.3456, Val Loss: 1180.2345, LR: 0.001000 [BEST]
Epoch 2/20 - Train Loss: 980.1234, Val Loss: 950.5678, LR: 0.001000 [BEST]
Epoch 3/20 - Train Loss: 850.4567, Val Loss: 820.9012, LR: 0.001000 [BEST]
...
Epoch 20/20 - Train Loss: 420.1234, Val Loss: 450.5678, LR: 0.000050

✓ Training complete! Best model saved to: outputs/hybrid_lstm/DRY/best_model.pth
✓ Best validation loss: 420.1234
```

---

## Configuration

### Base Configuration

**File**: `config/config.yaml`

**Key Section**: `model.hybrid`

```yaml
model:
  name: "HybridLSTM"  # Change from "RNNWithCategory" to use new architecture
  
  # Common parameters
  num_categories: null  # Set dynamically
  cat_emb_dim: 5
  hidden_size: 128
  n_layers: 2
  output_dim: 30
  dropout_prob: 0.2
  use_layer_norm: true
  
  # HybridLSTM-specific configuration
  hybrid:
    # Feature split: which features go to which branch
    temporal_features:
      - "Total CBM"
      - "rolling_mean_7d_expanding"
      - "rolling_mean_30d_expanding"
      - "momentum_3d_vs_14d_expanding"
    
    static_features:
      - "month_sin"
      - "month_cos"
      - "dayofmonth_sin"
      - "dayofmonth_cos"
      - "day_of_week_sin"
      - "day_of_week_cos"
      - "holiday_indicator"
      - "days_until_next_holiday"
      - "days_since_holiday"
      - "is_weekend"
      - "Is_Monday"
      - "is_EOM"
      - "days_until_month_end"
      - "weekday_volume_tier"
      - "is_high_volume_weekday"
      - "weekday_avg_expanding"
      - "category_avg_expanding"
    
    # Branch B (Dense Path) architecture
    dense_hidden_sizes: [64, 32]
    
    # Fusion layer architecture
    fusion_hidden_sizes: [64]
    
    # Loss function
    loss_function: "asymmetric_mse"
    
    # Asymmetric loss parameters
    asymmetric_loss:
      under_penalty: 3.0
      over_penalty: 1.0
      peak_season_boost: 2.0
```

### Category-Specific Configuration

**File**: `config/config_DRY.yaml` (example for DRY category)

```yaml
# DRY-specific overrides
model:
  hidden_size: 128
  n_layers: 2
  dropout_prob: 0.2
  
  hybrid:
    dense_hidden_sizes: [64, 32]
    fusion_hidden_sizes: [64]
    
    asymmetric_loss:
      under_penalty: 2.5  # DRY is stable, lower penalty ratio
      over_penalty: 1.0

training:
  epochs: 30
  learning_rate: 0.0005
  batch_size: 64
```

**File**: `config/config_TET.yaml` (example for TET category)

```yaml
# TET-specific overrides
model:
  hidden_size: 256  # Larger hidden size for sparse, high-variance category
  n_layers: 3
  dropout_prob: 0.3
  
  hybrid:
    dense_hidden_sizes: [128, 64]
    fusion_hidden_sizes: [128]
    
    asymmetric_loss:
      under_penalty: 5.0  # TET is critical, higher penalty ratio
      over_penalty: 1.0
      peak_season_boost: 3.0  # 3x weight during TET season

training:
  epochs: 60
  learning_rate: 0.002
  batch_size: 32
```

---

## Training Pipeline

### Full Training Script

**File**: `train_hybrid_lstm.py`

**Usage**:

```bash
# Train DRY category
python train_hybrid_lstm.py --category DRY --epochs 30 --learning-rate 0.001

# Train TET category with custom loss
python train_hybrid_lstm.py --category TET --epochs 60 --loss asymmetric_mse --under-penalty 5.0

# Train MOONCAKE with peak season boosting
python train_hybrid_lstm.py --category MOONCAKE --epochs 50 --under-penalty 4.0 --peak-season-boost 3.0
```

### Training Loop

```python
# Training loop (simplified)
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        X_temporal, X_static, y = batch
        X_temporal = X_temporal.to(device)
        X_static = X_static.to(device)
        y = y.to(device)
        
        # Dummy category IDs (single category per model)
        x_cat = torch.zeros(X_temporal.size(0), dtype=torch.long, device=device)
        
        # Forward pass
        y_pred = model(X_temporal, X_static, x_cat)
        
        # Compute loss
        loss = loss_fn(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            X_temporal, X_static, y = batch
            X_temporal = X_temporal.to(device)
            X_static = X_static.to(device)
            y = y.to(device)
            
            x_cat = torch.zeros(X_temporal.size(0), dtype=torch.long, device=device)
            y_pred = model(X_temporal, X_static, x_cat)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
```

---

## Validation & Testing

### Pre-Training Validation

**Run the validation script BEFORE training**:

```bash
python validate_no_leakage.py --data-path dataset/data_cat/data_2024.csv
```

**What It Checks**:
1. Expanding window features use only past data
2. Window slicing has no overlap between X and y
3. Temporal splits maintain chronological order

**Expected Result**: All tests should pass.

### Post-Training Testing

**Evaluate on Test Set**:

```python
# Load best model
model.load_state_dict(torch.load('outputs/hybrid_lstm/DRY/best_model.pth'))
model.eval()

# Evaluate on test set
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_loader:
        X_temporal, X_static, y = batch
        X_temporal = X_temporal.to(device)
        X_static = X_static.to(device)
        y = y.to(device)
        
        x_cat = torch.zeros(X_temporal.size(0), dtype=torch.long, device=device)
        y_pred = model(X_temporal, X_static, x_cat)
        
        loss = loss_fn(y_pred, y)
        test_loss += loss.item()
        
        predictions.append(y_pred.cpu().numpy())
        actuals.append(y.cpu().numpy())

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Calculate metrics
from src.utils.evaluation import calculate_metrics

metrics = calculate_metrics(
    y_true=np.concatenate(actuals),
    y_pred=np.concatenate(predictions),
)
print(f"Test MAE: {metrics['mae']:.2f}")
print(f"Test RMSE: {metrics['rmse']:.2f}")
print(f"Test MAPE: {metrics['mape']:.2f}%")
```

---

## Migration from Legacy Models

### Step-by-Step Migration

#### Step 1: Identify Current Model

**Current**: `RNNWithCategory` in `config/config.yaml`

```yaml
model:
  name: "RNNWithCategory"
  num_categories: 1
  cat_emb_dim: 5
  input_dim: 15
  hidden_size: 128
  n_layers: 2
  output_dim: 30
```

#### Step 2: Update Configuration

**New**: Change to `HybridLSTM` and add hybrid section

```yaml
model:
  name: "HybridLSTM"  # Changed
  
  # Add hybrid-specific configuration
  hybrid:
    temporal_features: [...]
    static_features: [...]
    dense_hidden_sizes: [64, 32]
    fusion_hidden_sizes: [64]
    loss_function: "asymmetric_mse"
    asymmetric_loss:
      under_penalty: 3.0
      over_penalty: 1.0
```

#### Step 3: Update Feature Engineering

**Legacy** (potential leakage):
```python
# mvp_train.py or mvp_predict.py
df['rolling_mean_7d'] = df.groupby('CATEGORY')['Total CBM'].rolling(7).mean()
df['rolling_mean_30d'] = df.groupby('CATEGORY')['Total CBM'].rolling(30).mean()
```

**New** (leak-free):
```python
from src.data.expanding_features import add_expanding_rolling_features

df = add_expanding_rolling_features(
    df,
    target_col='Total CBM',
    windows=[7, 30],
)
# Creates: rolling_mean_7d_expanding, rolling_mean_30d_expanding
```

#### Step 4: Update Dataset Creation

**Legacy**:
```python
from src.data.dataset import ForecastDataset

dataset = ForecastDataset(X, y, cat)
# X: (N, T, D) - all features mixed together
```

**New**:
```python
from src.data.dataset import HybridForecastDataset

dataset = HybridForecastDataset(X_temporal, X_static, y, cat)
# X_temporal: (N, T, D_temporal) - temporal sequences
# X_static: (N, D_static) - static features
```

#### Step 5: Update Training Script

**Legacy**:
```python
from src.models import RNNWithCategory

model = RNNWithCategory(
    num_categories=1,
    cat_emb_dim=5,
    input_dim=15,
    hidden_size=128,
    n_layers=2,
    output_dim=30,
)

# Forward pass
y_pred = model(X, cat)
```

**New**:
```python
from src.models import HybridLSTM

model = HybridLSTM(
    num_categories=1,
    cat_emb_dim=5,
    temporal_input_dim=4,   # Number of temporal features
    static_input_dim=17,    # Number of static features
    hidden_size=128,
    n_layers=2,
    dense_hidden_sizes=[64, 32],
    fusion_hidden_sizes=[64],
    output_dim=30,
)

# Forward pass
y_pred = model(X_temporal, X_static, cat)
```

#### Step 6: Update Loss Function

**Legacy**:
```python
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)
```

**New**:
```python
from src.utils.losses import AsymmetricMSELoss

loss_fn = AsymmetricMSELoss(under_penalty=3.0, over_penalty=1.0)
loss = loss_fn(y_pred, y_true)
```

---

## Performance Optimization

### 1. Feature Selection

**Temporal Branch** (Keep Only Essential):
- Historical CBM values (required)
- Short-term rolling mean (7-day)
- Long-term rolling mean (30-day)
- Momentum indicator (3d vs 14d)

**Static Branch** (Add Domain Knowledge):
- Cyclical encodings (month, day, weekday)
- Holiday features (indicator, proximity)
- Operational constraints (weekend, EOM)
- Statistical metadata (weekday avg, category avg)

### 2. Hyperparameter Tuning

**LSTM Branch**:
- `hidden_size`: 128 (baseline), 256 (high-variance categories like TET)
- `n_layers`: 2 (baseline), 3 (complex seasonal patterns)
- `dropout_prob`: 0.2 (baseline), 0.3 (high-variance categories)

**Dense Branch**:
- `dense_hidden_sizes`: [64, 32] (baseline), [128, 64] (TET/MOONCAKE)
- Use LayerNorm between layers for stability

**Fusion Layer**:
- `fusion_hidden_sizes`: [64] (baseline), [128] (complex interactions)

### 3. Loss Function Tuning

**Asymmetric Loss Penalties**:
- **DRY (stable)**: under_penalty=2.5, over_penalty=1.0 (2.5:1 ratio)
- **FRESH (moderate)**: under_penalty=3.0, over_penalty=1.0 (3:1 ratio)
- **TET (critical)**: under_penalty=5.0, over_penalty=1.0 (5:1 ratio)
- **MOONCAKE (critical)**: under_penalty=4.0, over_penalty=1.0, peak_season_boost=3.0

### 4. Training Optimization

**Learning Rate Schedule**:
- Initial LR: 0.001 (baseline), 0.002 (high-variance categories)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Min LR: 1e-5

**Batch Size**:
- 32 (baseline, TET, MOONCAKE - small datasets)
- 64 (DRY, FRESH - large datasets)

**Epochs**:
- 20 (quick testing)
- 30-50 (production training, DRY/FRESH)
- 60-100 (high-variance categories, TET/MOONCAKE)

---

## Troubleshooting

### Issue 1: Validation Loss Not Decreasing

**Symptoms**:
- Training loss decreases, but validation loss stays flat or increases.

**Causes**:
1. **Overfitting**: Model memorizing training data.
2. **Data Leakage**: Features contain future information (validation set "leaks" into training).
3. **Learning Rate Too High**: Model overshoots minimum.

**Solutions**:
1. **Increase Dropout**: Set `dropout_prob=0.3` or higher.
2. **Run Validation Script**: `python validate_no_leakage.py` to check for leakage.
3. **Reduce Learning Rate**: Set `learning_rate=0.0005` or lower.
4. **Add L2 Regularization**: Add `weight_decay=1e-5` to optimizer.

### Issue 2: Model Predicts Flat Line (No Variation)

**Symptoms**:
- All predictions are similar (e.g., always predicts average value).

**Causes**:
1. **Loss Function Too Conservative**: Standard MSE encourages predicting the mean.
2. **Static Features Dominating**: Dense branch overrides temporal trends.
3. **Insufficient Training**: Model hasn't learned temporal patterns yet.

**Solutions**:
1. **Use Asymmetric Loss**: Penalizes under-forecasting, encourages bolder predictions.
   ```python
   loss_fn = AsymmetricMSELoss(under_penalty=3.0, over_penalty=1.0)
   ```
2. **Balance Branch Sizes**: Ensure LSTM hidden size >= Dense output size.
   ```yaml
   hybrid:
     # LSTM output: 128
     dense_hidden_sizes: [64, 32]  # Dense output: 32
     # 128 > 32, so LSTM dominates (good)
   ```
3. **Train Longer**: Increase epochs to 50-100.

### Issue 3: Large Prediction Errors During Peak Seasons

**Symptoms**:
- Model performs well during normal periods but fails during TET/Mid-Autumn.

**Causes**:
1. **Insufficient Peak Season Data**: Model doesn't see enough high-volume examples.
2. **No Peak Season Weighting**: Loss function treats all days equally.
3. **Missing Event Features**: Holiday proximity features not included.

**Solutions**:
1. **Add Peak Season Boosting**: Increase loss weight during critical periods.
   ```python
   loss_fn = AsymmetricMSELoss(
       under_penalty=5.0,
       over_penalty=1.0,
       peak_season_boost=3.0,  # 3x weight during peak season
   )
   ```
2. **Ensure Holiday Features Exist**: Check that `days_until_next_holiday`, `holiday_indicator` are in static features.
3. **Use Seasonal Active Window**: For MOONCAKE, add `is_active_season` feature.

### Issue 4: "Shape Mismatch" Error During Training

**Symptoms**:
```
RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.FloatTensor instead
```

**Causes**:
- Category IDs (`x_cat`) not converted to `torch.long`.

**Solution**:
```python
# Ensure category IDs are long tensors
x_cat = x_cat.long()

# Or create dummy category IDs if single category
x_cat = torch.zeros(batch_size, dtype=torch.long, device=device)
```

### Issue 5: Memory Error (Out of CUDA Memory)

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB
```

**Causes**:
1. **Batch Size Too Large**: Model can't fit entire batch in GPU memory.
2. **Model Too Large**: Hidden sizes too large for GPU.

**Solutions**:
1. **Reduce Batch Size**: Set `batch_size=16` or `batch_size=8`.
2. **Reduce Model Size**:
   ```yaml
   hybrid:
     hidden_size: 64  # Reduced from 128
     dense_hidden_sizes: [32, 16]  # Reduced from [64, 32]
   ```
3. **Use Gradient Accumulation**:
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(train_loader):
       loss = loss / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

---

## Conclusion

The **Multi-Input Hybrid LSTM** architecture provides a robust, production-ready solution for MDLZ forecasting with:

1. ✅ **Dual-Branch Processing**: Separate pathways for temporal and static features.
2. ✅ **Leak-Free Feature Engineering**: Expanding windows guarantee no future information.
3. ✅ **Asymmetric Loss**: Penalizes under-forecasting more heavily (FMCG-specific).
4. ✅ **Category-Specific Training**: Independent models per category (DRY, TET, etc.).
5. ✅ **Comprehensive Validation**: Scripts to verify no data leakage before deployment.

### Next Steps

1. **Validate**: Run `python validate_no_leakage.py` to ensure no data leakage.
2. **Train**: Use `python train_hybrid_lstm.py --category DRY` to train your first model.
3. **Evaluate**: Compare performance against legacy `RNNWithCategory` model.
4. **Tune**: Adjust hyperparameters per category (use category-specific configs).
5. **Deploy**: Integrate trained model into production pipeline.

---

**Implementation Status**: ✅ **COMPLETE - Ready for Testing**

**Files Created**:
- `src/models.py` - HybridLSTM model class
- `src/data/expanding_features.py` - Leak-free feature engineering
- `src/data/dataset.py` - HybridForecastDataset class
- `src/utils/losses.py` - AsymmetricMSELoss
- `train_hybrid_lstm.py` - Training script
- `validate_no_leakage.py` - Validation script
- `config/config.yaml` - Updated configuration

**Documentation**:
- This guide (HYBRID_LSTM_IMPLEMENTATION_GUIDE.md)

---

**Questions or Issues?**

Please refer to the troubleshooting section or open an issue in the repository.

**Good luck with your forecasting! 🚀**
