import torch
import pandas as pd
import numpy as np


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
    input_size,
    device,
    horizon=7
):

    model.eval()
    results = []

    history = history.sort_values([brand_col, time_col])
    forecast_dates = pd.date_range(forecast_start, forecast_end, freq="D")

    brand_states = {}

    # Initialize states
    for brand, df_brand in history.groupby(brand_col):

        if len(df_brand) < input_size:
            continue

        state = feature_engineer.init_brand_state(df_brand)

        X_window = df_brand[feature_cols].iloc[-input_size:].values.astype(np.float32)

        state["window"] = X_window
        brand_states[brand] = state

    current_idx = 0

    # ======================================================
    # 🔥 BLOCK FORECAST (7 ngày mỗi lần)
    # ======================================================
    while current_idx < len(forecast_dates):

        block_dates = forecast_dates[current_idx: current_idx + horizon]

        for brand, state in brand_states.items():

            X = state["window"]
            brand_id = state["brand_id"]

            with torch.no_grad():
                x_t = torch.tensor(X).unsqueeze(0).to(device)
                b_t = torch.tensor([brand_id], dtype=torch.long).to(device)

                y_scaled_vec = model(x_t, b_t).squeeze(0).cpu().numpy()

            # iterate 7 outputs
            for i, d in enumerate(block_dates):

                y_scaled = y_scaled_vec[i]

                is_holiday = feature_engineer.is_holiday(d, brand)
                if is_holiday:
                    y_pred = 0.0

                    # 🔥 IMPORTANT: compute scaled version of zero
                    zero_scaled = feature_engineer.scaler.transform_value(0.0, brand)
                    y_scaled_for_state = zero_scaled

                else:
                    y_pred = feature_engineer.scaler.inverse(y_scaled, brand)
                    y_pred = max(0.0, float(y_pred))
                    y_scaled_for_state = y_scaled

                # build next feature using predicted scaled target
                new_features = feature_engineer.build_next_features(
                    state=state,
                    new_date=d,
                    new_scaled_target=y_scaled_for_state
                )

                # slide window
                state["window"][:-1] = state["window"][1:]
                state["window"][-1] = new_features

                results.append({
                    time_col: d,
                    brand_col: brand,
                    "predicted": y_pred
                })

        current_idx += horizon

    return pd.DataFrame(results).sort_values([brand_col, time_col])