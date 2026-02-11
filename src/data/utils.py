import numpy as np

def slicing_window(
    df,
    input_size,
    horizon,
    feature_cols,
    target_col,
    baseline_col,
    brand_id_col,
    time_col,
    off_holiday_col=None,
    label_start_date=None,
    label_end_date=None,
    return_dates=False,
    return_off_holiday=False,
):
    X, y, baselines, brands, dates, off_flags = [], [], [], [], [], []

    for brand_id, g in df.groupby(brand_id_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)

        X_data = g[feature_cols].values
        y_data = g[target_col].values.squeeze()
        b_data = g[baseline_col].values
        time_vals = g[time_col].values

        for i in range(len(g) - input_size - horizon + 1):
            label_date = time_vals[i + input_size]

            if label_start_date and label_date < label_start_date:
                continue
            if label_end_date and label_date >= label_end_date:
                continue

            X.append(X_data[i : i + input_size])
            y.append(y_data[i + input_size : i + input_size + horizon])
            baselines.append(b_data[i + input_size : i + input_size + horizon])
            brands.append(brand_id)
            dates.append(label_date)

            if off_holiday_col is not None:
                off_flags.append(
                    g.loc[i + input_size, off_holiday_col]
                )

    X = np.array(X)
    y = np.array(y)
    baselines = np.array(baselines)
    brands = np.array(brands)
    dates = np.array(dates) if return_dates else None
    off_flags = np.array(off_flags) if return_off_holiday else None

    if return_dates and return_off_holiday:
        return X, y, baselines, brands, dates, off_flags
    elif return_dates:
        return X, y, baselines, brands, dates
    elif return_off_holiday:
        return X, y, baselines, brands, off_flags
    else:
        return X, y, baselines, brands
