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
    target_col,
):
    """
    Compute baseline + residual + calendar for one new row only.
    Assumes hist_brand already contains full features.
    """

    brand = hist_brand[brand_col].iloc[-1]

    # --- baseline from last N real demands ---
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
    Production-ready monthly recursive forecast.
    - No full history recomputation
    - Incremental baseline
    - Stable scaler usage
    - O(N) growth
    """

    model.eval()
    results = []

    # Ensure history already contains full features
    history = history.copy().sort_values([brand_col, time_col])

    forecast_dates = pd.date_range(forecast_start, forecast_end, freq="D")

    for d in forecast_dates:

        new_feature_rows = []

        # Group once per day
        for brand, hist_brand in history.groupby(brand_col, sort=False):

            hist_brand = hist_brand[hist_brand[time_col] < d]

            if len(hist_brand) < input_size:
                continue

            # -------------------------
            # Build model input window
            # -------------------------
            X = hist_brand[feature_cols].iloc[-input_size:].values
            baseline = float(hist_brand["baseline"].iloc[-1])
            brand_id = int(hist_brand[brand_id_col].iloc[-1])

            # -------------------------
            # Inference
            # -------------------------
            with torch.no_grad():

                x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
                b_t = torch.tensor([brand_id], dtype=torch.long).to(device)

                y_scaled = model(x_t, b_t).cpu().numpy().reshape(-1)

                y_residual = feature_engineer.inverse_transform_y(
                    y_scaled,
                    np.array([brand_id])
                )

                y_residual = float(np.asarray(y_residual).reshape(-1)[0])

                if not np.isfinite(y_residual):
                    if verbose:
                        print(f"[WARN] Invalid residual at {d} brand={brand}")
                    y_residual = 0.0

                y_pred = max(0.0, y_residual + baseline)

            # -------------------------
            # Build incremental features
            # -------------------------
            new_row = build_next_row_features(
                hist_brand,
                new_date=d,
                new_demand=y_pred,
                input_size=input_size,
                time_col=time_col,
                brand_col=brand_col,
                target_col=target_col,
            )

            new_row[brand_id_col] = brand_id

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

            new_feature_rows.append(new_row)

        # -------------------------
        # Append all new rows
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
