import pandas as pd
import numpy as np
import torch

from src.data import add_history_features, add_calendar_features

def build_next_row_features(
    hist_brand,
    new_date,
    new_demand,
    input_size,
    time_col,
    brand_col,
    brand_id_col,
    target_col,
    baseline=None,
):
    """
    Compute baseline + residual + calendar for one new row only.
    Assumes hist_brand already contains full features.
    """

    brand = hist_brand[brand_col].iloc[-1]

    # --- baseline from last N real demands (or use provided baseline) ---
    if baseline is None:
        last_n = hist_brand[target_col].iloc[-input_size:].values
        baseline = float(np.mean(last_n)) if len(last_n) > 0 else 0.0

    # --- calendar ---
    row = pd.DataFrame({
        time_col: [new_date],
        brand_col: [brand],
        target_col: [new_demand],
    })

    row = add_calendar_features(row, time_col)

    # pre-holiday zeroing rule
    if row["is_pre_off_holiday"].iloc[0] == 1:
        baseline = 0.0

    row["baseline"] = baseline
    row["residual"] = new_demand - baseline
    row[brand_id_col] = hist_brand[brand_id_col].iloc[-1]  # Add brand_id

    return row



def forecast_commit_month(
    model,
    feature_engineer,
    history,
    forecast_start,
    forecast_end,
    time_col,
    brand_col,
    brand_id_col,
    feature_cols,
    target_col,
    input_size,
    device,
    verbose: bool = False
):
    """
    Multi-step + Recursive forecasting: 14 days -> predict 7 days, no overlap.
    - Uses 14-day windows to predict 7-day horizons
    - Recursive: uses predictions to extend history for next predictions
    - No overlap: each 7-day prediction covers distinct periods
    """

    model.eval()
    results = []

    # Ensure history already contains full features
    history = history.copy().sort_values([brand_col, time_col])

    # Generate forecast periods (7-day chunks, no overlap)
    forecast_periods = []
    current_start = forecast_start
    while current_start <= forecast_end:
        period_end = min(current_start + pd.Timedelta(days=6), forecast_end)
        forecast_periods.append((current_start, period_end))
        current_start = period_end + pd.Timedelta(days=1)

    for period_start, period_end in forecast_periods:
        if verbose:
            print(f"  Predicting period: {period_start} → {period_end}")

        new_feature_rows = []

        # Group by brand for this period
        for brand, hist_brand in history.groupby(brand_col, sort=False):

            hist_brand = hist_brand[hist_brand[time_col] < period_start]

            if len(hist_brand) < input_size:
                if verbose:
                    print(f"    Skipping {brand}: insufficient history ({len(hist_brand)} < {input_size})")
                continue

            # -------------------------
            # Build model input window
            # -------------------------
            X = hist_brand[feature_cols].iloc[-input_size:].values
            baseline = float(hist_brand["baseline"].iloc[-1])
            brand_id = int(hist_brand[brand_id_col].iloc[-1])

            # -------------------------
            # Multi-step Inference (predict 7 days at once)
            # -------------------------
            with torch.no_grad():
                x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
                b_t = torch.tensor([brand_id], dtype=torch.long).to(device)

                y_scaled = model(x_t, b_t).cpu().numpy().reshape(-1)  # Shape: (7,)

                y_residuals = feature_engineer.inverse_transform_y(
                    y_scaled,
                    np.array([brand_id] * len(y_scaled))
                )

                y_residuals = np.asarray(y_residuals).reshape(-1)

            # -------------------------
            # Generate predictions for each day in the 7-day period
            # -------------------------
            period_dates = pd.date_range(period_start, period_end, freq="D")

            # Create a temporary history for this period to update baselines recursively
            temp_history = hist_brand.copy()

            for i, d in enumerate(period_dates):
                # Update baseline using the most recent data (including previous predictions in this period)
                recent_data = temp_history[temp_history[time_col] < d]
                if len(recent_data) >= input_size:
                    baseline = float(recent_data[target_col].iloc[-input_size:].mean())
                else:
                    baseline = float(recent_data[target_col].mean()) if len(recent_data) > 0 else 0.0

                y_residual = float(y_residuals[i])

                if not np.isfinite(y_residual):
                    if verbose:
                        print(f"[WARN] Invalid residual at {d} brand={brand}")
                    y_residual = 0.0

                y_pred = max(0.0, y_residual + baseline)

                # -------------------------
                # Build incremental features for this prediction
                # -------------------------
                new_row = build_next_row_features(
                    temp_history,  # Use temp_history for other calculations
                    new_date=d,
                    new_demand=y_pred,
                    input_size=input_size,
                    time_col=time_col,
                    brand_col=brand_col,
                    brand_id_col=brand_id_col,
                    target_col=target_col,
                    baseline=baseline,  # Use the recursively computed baseline
                )

                # -------------------------
                # Save output
                # -------------------------
                results.append({
                    "date": d,
                    "brand": brand,
                    "predicted": y_pred,
                    "residual": float(new_row["residual"].iloc[0]),
                    "baseline": float(new_row["baseline"].iloc[0]),
                })

                # -------------------------
                # Add to temporary history for next baseline calculation
                # -------------------------
                temp_history = pd.concat([temp_history, new_row], ignore_index=True)

                new_feature_rows.append(new_row)

        # -------------------------
        # Append all new rows for this period
        # -------------------------
        if new_feature_rows:
            history = pd.concat(
                [history] + new_feature_rows,
                ignore_index=True
            )

    return (
        pd.DataFrame(results)
        .sort_values(["brand", "date"])
        .reset_index(drop=True)
    )







# def forecast_teacher_forcing(
#     model,
#     scaler,
#     history_with_actuals,   # contains real actuals up to D-1
#     time_col,
#     feature_cols,
#     target_col,
#     brand_id_col,
#     input_size,
#     start_date,
#     horizon_days=30,
#     device=None,
# ):
#     """
#     Daily teacher-forcing forecast.
#     - Uses actuals up to D−1
#     - Recomputed daily
#     - Overwrites previous results
#     """

#     model.eval()
#     results = []

#     forecast_dates = pd.date_range(
#         start_date,
#         start_date + pd.Timedelta(days=horizon_days - 1),
#         freq="D"
#     )

#     for d in forecast_dates:
#         for brand in history_with_actuals[brand_id_col].unique():
#             hist_brand = history_with_actuals[
#                 history_with_actuals[brand_id_col] == brand
#             ]

#             window = build_single_window(
#                 hist_brand,
#                 d,
#                 input_size,
#                 time_col,
#                 feature_cols,
#                 target_col,
#                 brand_id_col,
#             )
            
#             if window is None:
#                 continue

#             X, baseline, brand_id = window
#             assert baseline != 0, f"Baseline zero at {d} brand={brand_id}"

#             with torch.no_grad():
#                 x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
#                 b_t = torch.tensor([brand_id]).to(device)

#                 y_scaled = model(x_t, b_t).cpu().numpy()[0, 0]
#                 y_unscaled = scaler.inverse_transform_y(
#                     np.asarray(y_scaled).reshape(-1),
#                     np.asarray(brand_id).reshape(-1),
#                 )

#                 # ensure scalar
#                 y_unscaled = float(np.asarray(y_unscaled).reshape(-1)[0])

#                 y_pred = max(0.0, float(y_unscaled + baseline))

#             results.append({
#                 "date": d,
#                 "brand": brand_id,
#                 "predicted": y_pred,
#                 "baseline": baseline,
#             })

#     return pd.DataFrame(results)
