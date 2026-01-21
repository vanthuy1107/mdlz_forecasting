## Model Logic Documentation

This document describes the **current end‑to‑end logic** of the MDLZ warehouse
forecasting MVP as implemented in:

- **Training**: `mvp_test.py` — trains models on historical data
- **Prediction**: `mvp_predict_2025.py` — generates forecasts using trained models
- **Core modules**: `src/data/preprocessing.py`, `src/models.py`, and
  `src/training/trainer.py`

The previous version of this document referred to an older US‑holiday,
`QTY`–only pipeline driven by `main.py`. That entrypoint has been removed and
replaced by the MVP scripts described here.

---

## 1. Problem Definition

- **Objective**: Forecast future shipment volume (currently `Total CBM`) for
  MDLZ warehouse outbound flows, per product category, while:
  - Capturing **Vietnamese holiday spikes** (Tet, Mid‑Autumn, etc.)
  - Respecting **lunar calendar seasonality**
  - Using **structural priors** such as CBM/QTY density and last‑year density
  - Handling high‑ and low‑volume categories differently
- **Type**: Multivariate time‑series forecasting with:
  - Temporal and calendar features (solar + lunar)
  - Holiday countdown and “days‑since‑holiday” features
  - Category embeddings and category‑aware LSTM
  - Optional **residual target** (model learns deviations from a causal baseline)

The **target column** is currently `Total CBM` (see `config/config.yaml`), but
the pipeline is written so that another numeric target could be used via
configuration.

---

## 2. End‑to‑End MVP Test Pipeline (`mvp_test.py`)

The main entrypoint is `mvp_test.py`. Its `main()` function runs a complete
experiment with fixed, reproducible overrides:

1. **Configuration**
   - Loads `config/config.yaml` via `load_config()`.
   - Overrides for MVP test:
     - `data.years = [2023, 2024]`
     - `training.epochs = 20`
     - `training.loss = "spike_aware_mse"` (forced in code)
     - `output.output_dir = "outputs/mvp_test"`
     - `output.model_dir = "outputs/mvp_test/models"`
   - Reads **category behavior switches** from `data` config:
     - `category_mode`: `"all"`, `"single"`, `"both"`, or `"each"`
       - `"all"`: Train one global model on all major categories
       - `"single"`: Train one model on a single category (`category_filter`)
       - `"both"`: Train one global model plus one model per category
       - `"each"`: Train one model per category (uses `major_categories` if specified)
     - `category_filter`: used when `category_mode == "single"` (default `"DRY"`)
     - `major_categories`: categories allowed into the LSTM when training
       a global head (default: `["DRY", "FRESH"]`)
     - `minor_categories`: low‑volume categories handled by simple heuristics
       (default: `["POSM", "OTHER"]`)

2. **Data Loading (`DataReader`)**
   - Uses `DataReader` from `src/data/loader.py`:
     - Primary path: `data_reader.load(years=data_config["years"])`
       (expects `data_YYYY.csv` under `dataset/data_cat/`).
     - Fallback path if combined files are missing:
       `load_by_file_pattern(years, file_prefix="Outboundreports")`.
   - Fixes `DtypeWarning` by casting the first and fifth columns to `str` to
     avoid mixed types.

3. **Category Discovery and Training Tasks**
   - Uses `data.cat_col` (typically `CATEGORY`) to list available categories.
   - Builds a list of **training tasks** based on `category_mode`:
     - `category_mode == "all"` → single model on **all major categories**
       (suffix: `"_all"`).
     - `category_mode == "single"` → single model on `category_filter`
       (e.g. `"DRY"`, suffix: `"_{category_filter}"`).
     - `category_mode == "both"` → one global model (`ALL CATEGORIES`, suffix:
       `"_all"`) plus one model per individual category (suffix: `"_{category}"`).
     - `category_mode == "each"` → one model per category, using `major_categories`
       if specified, otherwise all available categories (suffix: `"_{category}"`).
   - Each task calls `train_single_model(...)` with a `category_filter` and an
     `output_suffix`, resulting in separate `outputs/mvp_test*/` directories.

---

## 3. Feature Engineering in the MVP Pipeline

Most of the feature engineering happens inside `train_single_model` in
`mvp_test.py`, using utilities from `src/data/preprocessing.py` plus some
additional MVP‑specific logic.

### 3.1 Base Configuration of Features

In `config/config.yaml` (section `data.feature_cols`) the **base feature set**
is:

- `month_sin`, `month_cos`
- `dayofmonth_sin`, `dayofmonth_cos`
- `holiday_indicator`
- `days_until_next_holiday`
- `days_since_holiday`
- `is_weekend`, `day_of_week`
- `lunar_month`, `lunar_day`
- `rolling_mean_7d`, `rolling_mean_30d`
- `momentum_3d_vs_14d`
- `Total CBM` (current numeric target used also as an input feature)

At runtime, `train_single_model` **extends** this list with:

- `lunar_month_sin`, `lunar_month_cos`
- `lunar_day_sin`, `lunar_day_cos`
- `days_to_tet` (continuous countdown to next Tet window)
- `cbm_per_qty`
- `cbm_per_qty_last_year`

The combined list is written back into the config at
`data.feature_cols` and the model’s `input_dim` is updated dynamically to match
the final feature count.

### 3.2 Temporal Features (`add_temporal_features`)

Function: `add_temporal_features` in `src/data/preprocessing.py`.

Given the time column `ACTUALSHIPDATE`, it creates:

- `month_sin = sin(2π × (month − 1) / 12)`
- `month_cos = cos(2π × (month − 1) / 12)`
- `dayofmonth_sin = sin(2π × (day − 1) / 31)`
- `dayofmonth_cos = cos(2π × (day − 1) / 31)`

These encodings capture **yearly** and **monthly** periodicity while treating
the calendar as cyclic rather than linear.

### 3.3 Weekend and Day‑of‑Week Features (`add_weekend_features`)

Defined in `mvp_test.py`:

- `day_of_week`: integer \(0 = Monday, ..., 6 = Sunday\).
- `is_weekend`: 1 if `day_of_week ∈ {5, 6}` (Saturday or Sunday), else 0.

This lets the model distinguish weekday vs weekend behavior and capture
weekly seasonality.

### 3.4 Vietnamese Holiday Features

`mvp_test.py` defines:

- `VIETNAM_HOLIDAYS_BY_YEAR`: a map of years → Tet window,
  Mid‑Autumn, Independence Day (2 Sep) and Labor Day (30 Apr–1 May).
- `get_vietnam_holidays(start_date, end_date)`: retrieves all configured
  holiday dates in a given range.
- `add_holiday_features_vietnam(df, ...)`: builds:
  - `holiday_indicator`: 1 if the date is in the Vietnamese holiday set,
    otherwise 0.
  - `days_until_next_holiday`: days until the **next** Vietnamese holiday,
    up to an extended horizon.
  - `days_since_holiday`: days since the **last** Vietnamese holiday.
- `add_days_since_holiday(df, ...)`: an additional helper (not always used by
  the MVP pipeline) that separately computes “days since holiday” using an
  extended look‑back window.

Compared to the older US‑holiday version, the MVP now uses **Vietnam‑specific
holidays** throughout the pipeline, which is critical for Tet‑driven demand.

### 3.5 Lunar Calendar and Cyclical Lunar Features

MVP logic approximates the Vietnamese lunar calendar:

- `solar_to_lunar_date(solar_date)`: converts a Gregorian date into an
  approximate `(lunar_month, lunar_day)`.
- `add_lunar_calendar_features(df, ...)`:
  - `lunar_month` ∈ [1, 12]
  - `lunar_day` ∈ [1, 30]
- `add_lunar_cyclical_features(df, ...)`:
  - `lunar_month_sin`, `lunar_month_cos` from `lunar_month`
  - `lunar_day_sin`, `lunar_day_cos` from `lunar_day`

These features allow the model to see **lunar periodicity** (e.g. Tet timing)
without hardcoding specific Gregorian dates beyond the holiday table.

### 3.6 Tet Countdown Feature (`add_days_to_tet_feature`)

`add_days_to_tet_feature(df, ...)` in `mvp_test.py`:

- Uses the **start dates** of Tet windows (first day in each Tet period) across
  years 2023–2025.
- For each date, computes `days_to_tet` as the number of days until the next
  Tet start (0 if already inside the Tet window).
- Provides a **smooth countdown signal** that allows the model to ramp up
  expectations before the actual spike.

### 3.7 Daily Aggregation (`aggregate_daily`)

Function: `aggregate_daily` in `src/data/preprocessing.py`.

Purpose: convert transaction‑level rows to **daily totals per category** while
preserving temporal / holiday features.

- Groups by:
  - `date_only = normalize(ACTUALSHIPDATE)`
  - `CATEGORY`
- Aggregates:
  - `target_col` (e.g. `Total CBM`): sum
  - Optionally `Total QTY`: sum
  - Temporal and holiday features (`month_sin`, `holiday_indicator`, etc.):
    first value per day (should be identical within each day).
- Renames `date_only` back to `ACTUALSHIPDATE` and sorts by
  `[CATEGORY, ACTUALSHIPDATE]`.

The result is **one row per date per category**, which is what the LSTM sees.

### 3.8 CBM Density and Last‑Year Density Prior (`add_cbm_density_features`)

Function: `add_cbm_density_features` in `src/data/preprocessing.py`.

Assumes:

- Target column (`cbm_col`) is e.g. `Total CBM` (daily total per category).
- `qty_col` is `Total QTY` (daily quantity).

Creates:

- `cbm_per_qty = Total CBM / max(Total QTY, eps)` (per‑day density).
- `cbm_per_qty_last_year`: density for the **same category and same calendar
  date one year earlier**, using:
  - A shifted copy of the data where `ACTUALSHIPDATE` is moved by +1 year,
    then left‑joined.
  - If no exact match exists (e.g. first year of history), falls back to a
    **category‑level median density**.

This functions as a **structural prior** for how “bulky” shipments are expected
to be around the same lunar/seasonal period.

### 3.9 Rolling Means and Momentum

Function: `add_rolling_and_momentum_features` in `mvp_test.py`.

Per category and date (after daily aggregation), it computes:

- `rolling_mean_7d`: 7‑day rolling mean of the target.
- `rolling_mean_30d`: 30‑day rolling mean.
- `momentum_3d_vs_14d = rolling_mean_3d − rolling_mean_14d`.

These features:

- Provide a **short‑ and medium‑term pace signal**.
- Reduce the need for the LSTM to re‑learn simple local averages.

### 3.10 Residual Target and Causal Baseline

Configured in `config.yaml` under `data`:

- `use_residual_target: true`
- `baseline_source_col: "rolling_mean_30d"`
- `baseline_col: "baseline_for_target"`
- `residual_col: "target_residual"`

When `use_residual_target` is true:

1. For each `(CATEGORY, date)` row, build a **causal baseline**:
   - Start from `baseline_source_col` (e.g. `rolling_mean_30d`).
   - Shift by 1 step **within each category** so that the baseline for day *t*
     only uses information up to day *t − 1*.
   - Fill missing initial baseline values with the unshifted source.
2. Compute residual target:
   - `target_residual_t = target_t − baseline_for_target_t`.
3. The model is trained to predict `target_residual` rather than the raw
   `Total CBM`.
4. After inverse‑scaling, predictions are **reconstructed** during evaluation as:
   - `y_true_original = y_true_residual + baseline`
   - `y_pred_original = y_pred_residual + baseline`

This makes training **scale‑invariant** and focuses the LSTM on deviations from
the statistical baseline.

---

## 4. Category Handling and Multi‑Target Decomposition

Inside `train_single_model`:

- **Single‑category mode** (`category_filter` not `None`):
  - Filters the full dataset to rows where `CATEGORY == category_filter`.
  - Trains one model whose outputs apply to that category only.
- **Global mode (no `category_filter`)**:
  - Restricts the training data to `major_categories` only (e.g. `DRY`, `FRESH`).
  - **Excludes minor categories** from the LSTM, to avoid backpropagating
    noisy, low‑volume signals into the shared hidden state.
  - Minor categories are intended to be modeled by simple baselines such as
    `moving_average_forecast_by_category` in `src/data/preprocessing.py`.

The category column is encoded via `encode_categories`:

- Creates `CATEGORY_ID` and a `cat2id` mapping.
- Number of categories is written to `model.num_categories` at runtime.

---

## 5. Data Splitting, Scaling and Windowing

### 5.1 Temporal Split (`split_data`)

Function: `split_data` in `src/data/preprocessing.py`.

- Uses **temporal** (sequential) split, not random, to avoid leakage:
  - `train_size = 0.7`
  - `val_size = 0.1`
  - `test_size = 0.2`
- Returns `train_data`, `val_data`, `test_data`, preserving chronological
  ordering.

### 5.2 Target Scaling

- `fit_scaler(train_data, target_col=target_col_for_model)` fits a
  `StandardScaler` on the **training target** (residual or absolute).
- `apply_scaling` applies the same scaler to train/val/test target columns.
- The scaler is saved to disk as `scaler.pkl` alongside the trained model so
  that inference and plotting can invert the transformation later.

Inverse scaling during evaluation is performed with
`inverse_transform_scaling(...)` from `src/data/preprocessing.py`.

### 5.3 Sliding Windows (`slicing_window_category`)

Function: `slicing_window_category` in `src/data/preprocessing.py`.

Per category group:

- Input window length: `input_size = 30` timesteps.
- Horizon: `horizon = 1` step ahead.

Shapes:

- `X`: \((N_\text{samples}, 30, n_\text{features})\) with
  `n_features = len(data.feature_cols)` after all dynamic extensions.
- `y`: \((N_\text{samples}, 1)\) containing the (scaled) supervised target
  (residual or absolute).
- `cats`: \((N_\text{samples},)\) with integer `CATEGORY_ID`s.

These are wrapped into PyTorch datasets (`ForecastDataset`) and then into
`DataLoader`s for train/val/test.

---

## 6. Model Architecture (`RNNWithCategory` and Baseline)

### 6.1 `RNNWithCategory` (Primary Model)

Location: `src/models.py`.

**Inputs**:

- `x_seq`: \((B, T, D)\) feature sequence with `D = model.input_dim`.
- `x_cat`: category IDs, shape \((B,)\) or \((B, 1)\).

**Layers**:

1. **Category Embedding**
   - `Embedding(num_categories, cat_emb_dim)`.
   - Produces `cat_vec` of shape \((B, cat_emb_dim)\).
2. **Hidden State Initialization**
   - Linear layer `h0_fc(cat_vec)` → \((B, hidden_size)\).
   - Tanh non‑linearity, then repeat across `n_layers` to build initial
     hidden state `h0`.
   - Cell state `c0` initialized to zeros.
3. **Category‑Augmented Input**
   - Expand `cat_vec` over time into `cat_seq` \((B, T, cat_emb_dim)\).
   - Concatenate with `x_seq` along feature axis:
     - `x = concat([x_seq, cat_seq])` → \((B, T, D + cat_emb_dim)\).
4. **LSTM**
   - `nn.LSTM(input_size=D + cat_emb_dim, hidden_size=hidden_size,
     num_layers=n_layers, batch_first=True)`.
   - Produces `out` of shape \((B, T, hidden_size)\).
5. **Layer Normalization (Optional)**
   - If `use_layer_norm` is true, applies `LayerNorm(hidden_size)` to the last
     timestep output.
6. **Output Layer**
   - Takes `last_out = out[:, -1, :]`.
   - Passes through a linear layer `fc(last_out)` → prediction of shape
     \((B, output_dim)\).

This architecture injects category information both into the **initial hidden
state** and into every timestep’s input.

### 6.2 `RNNForecastor` (Baseline Without Category Embedding)

Also in `src/models.py`:

- Plain LSTM with input size `embedding_dim`, hidden size `hidden_size`,
  `n_layers` layers and optional dropout before the final linear head.
- API is compatible with `RNNWithCategory` but ignores `x_cat`.
- Can be selected via `model.name = "RNNForecastor"` in the config if a
  simpler baseline is desired.

---

## 7. Training Loop and Loss

### 7.1 Trainer (`src/training/trainer.py`)

`mvp_test.py` builds a `Trainer` instance with:

- `model`: `RNNWithCategory` (or `RNNForecastor`).
- `criterion`: **always** `spike_aware_mse` for the MVP test, even if
  `training.loss` is set differently in the YAML.
- `optimizer`: Adam with `learning_rate` from config.
- Optional `ReduceLROnPlateau` scheduler if configured.
- Device: `"auto"`, `"cuda"`, or `"cpu"` as per config.

The `fit(...)` method:

- Trains for `training.epochs` epochs.
- Tracks and saves the **best model** by validation loss.
- Records training and validation loss histories.

The `evaluate(...)` method:

- Runs the model on `test_loader` without gradients.
- Returns mean test loss plus flattened `y_true` and `y_pred` (in scaled space).

### 7.2 `spike_aware_mse` Loss

Location: `src/utils/losses.py`.

- Extends standard MSE by giving **3× weight** to the top ~20% of target values.
- Ensures Tet and other extreme peaks contribute more strongly to the loss and
  gradient signal.
- In the MVP test, this loss is **forced** regardless of the YAML `loss`
  setting, to prioritize spike accuracy on Vietnamese data.

---

## 8. Evaluation, Inverse Scaling and Plots

After training:

1. **Test Evaluation**
   - `trainer.evaluate(test_loader, return_predictions=True)` computes
     test loss in scaled space.
2. **Inverse Scaling**
   - Uses `inverse_transform_scaling` and the saved `StandardScaler` to
     convert `y_true` and `y_pred` back to the **original target scale**.
   - If residual learning is enabled, adds back the aligned baseline as
     described earlier to reconstruct absolute `Total CBM`.
3. **Plotting**
   - Takes up to `n_samples = min(100, len(y_true_original))`.
   - Calls `plot_difference(y_true_plot, y_pred_plot, save_path=...)` from
     `src/utils/visualization.py`.
   - Saves `test_predictions.png` under the task‑specific
     `outputs/mvp_test*_*/` directory.
4. **Metadata**
   - Writes a `metadata.json` file under `model_dir` with:
     - Model, data, window and training configs
     - Best validation and test loss
     - Training time
     - Category mapping (`cat2id`) and category filter used
   - Saves the fitted scaler as `scaler.pkl` next to the best model checkpoint.

---

## 9. Workflow Summary and Usage

### 9.1 Standard MVP Test Run

From the project root:

```bash
python mvp_test.py
```

This will:

- Load years 2023–2024.
- Apply Vietnam‑specific calendar and holiday features.
- Train one or more models according to `data.category_mode`.
- Save checkpoints, scalers, plots and metadata into
  `outputs/mvp_test*/` subdirectories.

### 9.2 Configuration Tweaks

Typical changes are made in `config/config.yaml`, for example:

- Switch between **global** and **per‑category** training:
  - `data.category_mode: "all" | "single" | "both" | "each"`.
  - `data.category_filter: "DRY"` when using `"single"`.
  - `data.major_categories: ["DRY", "FRESH"]` when using `"each"` or `"all"`.
- Enable/disable residual learning:
  - `data.use_residual_target: true | false`.
- Adjust window and horizon:
  - `window.input_size`, `window.horizon`.
- Adjust model size:
  - `model.hidden_size`, `model.n_layers`, `model.cat_emb_dim`.

Note: `mvp_test.py` may still override some defaults (e.g. years, epochs,
loss) to keep the MVP experiment reproducible; check the top of `main()` if you
need to change those behaviors.

### 9.3 Prediction Workflow (`mvp_predict_2025.py`)

After training models with `mvp_test.py`, use `mvp_predict_2025.py` to generate
forecasts for future dates. This script supports two prediction modes:

**1. Teacher Forcing (Test Evaluation)**
- Uses actual ground truth target values from the prediction period as input
  features.
- Suitable for model evaluation when historical data is available.
- Not suitable for true production forecasting (requires future values).

**2. Recursive (Production Forecast)**
- Uses the model's own predictions as inputs for subsequent time steps.
- Simulates true production forecasting where future values are unknown.
- Starts from a historical window (e.g., last 30 days of 2024) and recursively
  generates predictions day-by-day.

#### 9.3.1 Prediction Configuration

The prediction window and data path are configured in `config/config.yaml`
under the `inference` section:

- `inference.prediction_data_path`: Path to CSV file containing prediction
  period data (e.g., `"dataset/test/data_2025.csv"`).
- `inference.prediction_start`: Start date for predictions (e.g., `"2025-01-01"`).
- `inference.prediction_end`: End date for predictions (e.g., `"2025-01-31"`).

The script respects the same `data.category_mode` setting as training:
- `"single"`: Predicts one category (uses model from `outputs/mvp_test_{category}/`).
- `"all"`: Predicts all categories using a global model (uses model from
  `outputs/mvp_test_all/`).
- `"each"`: Predicts each category separately using per-category models (uses
  models from `outputs/mvp_test_{category}/` for each category).

#### 9.3.2 Prediction Pipeline

The prediction workflow mirrors the training pipeline:

1. **Load Historical Data**: Loads reference data (e.g., 2024) to establish
   category mappings and extract the historical window for recursive prediction.

2. **Load Prediction Data**: Loads the target period data (e.g., January 2025)
   from the configured path.

3. **Load Trained Model**: Loads the trained model checkpoint, scaler, and
   metadata from the appropriate `outputs/mvp_test*/models/` directory based on
   `category_mode`.

4. **Feature Engineering**: Applies the same feature engineering pipeline as
   training:
   - Temporal features (month/day cyclical encodings)
   - Weekend and day-of-week features
   - Lunar calendar features (lunar_month, lunar_day)
   - Lunar cyclical encodings (sine/cosine)
   - Vietnamese holiday features
   - Tet countdown feature (`days_to_tet`)
   - Daily aggregation by category
   - CBM density features (including last-year prior)
   - Rolling means and momentum features

5. **Category Mapping**: Remaps prediction data categories to match the
   training-time category IDs using the `trained_cat2id` mapping from model
   metadata.

6. **Scaling**: Applies the same `StandardScaler` used during training to
   normalize target values.

7. **Teacher Forcing Mode** (if enabled):
   - Creates prediction windows using actual target values from the prediction
     period.
   - Runs model inference on these windows.
   - Computes evaluation metrics (MSE, MAE, RMSE, accuracy).

8. **Recursive Mode**:
   - Extracts the last `input_size` days from historical data as the initial
     window.
   - For each prediction date:
     - Creates an input window from the most recent `input_size` days.
     - Runs model inference to predict the next day.
     - Updates rolling features (rolling_mean_7d, rolling_mean_30d, momentum)
       using the predicted value.
     - Appends the prediction to the window for the next iteration.
   - Handles residual targets by reconstructing absolute values from residuals
     + baseline.

9. **Inverse Scaling**: Converts predictions back to original scale using the
   saved scaler.

10. **Residual Reconstruction**: If residual learning was used during training,
    reconstructs absolute target values by adding back the baseline.

11. **Output**: Saves predictions to CSV files and generates comparison plots.
    Optionally uploads results to Google Sheets if configured.

#### 9.3.3 Running Predictions

From the project root:

```bash
python mvp_predict_2025.py
```

The script will:
- Load the prediction window from `config.inference`.
- Load the appropriate trained model(s) based on `data.category_mode`.
- Generate predictions in both Teacher Forcing and Recursive modes.
- Save results to `outputs/predictions/` with timestamps.
- Generate comparison tables and accuracy metrics.

---

## 10. Key Differences vs. Legacy Version

Compared with the earlier version of this project (US‑holiday, `QTY`‑only,
`main.py` pipeline), the MVP implemented here:

- Switches to **Vietnamese holidays** and adds a **Tet countdown** feature.
- Incorporates **lunar calendar + cyclical lunar features**.
- Predicts `Total CBM` and adds **CBM/QTY density + last‑year density prior**.
- Uses **residual learning** against a causal rolling‑mean baseline.
- Applies **spike‑aware MSE** by default to focus on Tet and other spikes.
- Supports **multi‑task category configurations** and separates major from
  minor categories for more stable shared representations.

