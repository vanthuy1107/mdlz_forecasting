import pandas as pd
import numpy as np

class MonthlyAdaptiveScaler:
    def __init__(self, n_months=12, min_std=1e-3):
        self.stats = {}
        self.n_months = n_months
        self.min_std = min_std

    def fit(self, df, brand_col, target_col, time_col):

        for brand, g in df.groupby(brand_col):

            g = g.sort_values(time_col)
            max_date = g[time_col].max()
            cutoff = max_date - pd.DateOffset(months=self.n_months)

            g_recent = g[g[time_col] >= cutoff]
            if len(g_recent) < 10:
                g_recent = g

            y_log = np.log1p(g_recent[target_col].values)

            mean = y_log.mean()
            std = y_log.std()

            if std < self.min_std:
                std = 1.0

            self.stats[brand] = {
                "mean": mean,
                "std": std
            }

    def transform(self, df, brand_col, target_col):

        df = df.copy()

        for brand, s in self.stats.items():
            mask = df[brand_col] == brand
            if not mask.any():
                continue

            y_log = np.log1p(df.loc[mask, target_col].values)
            df.loc[mask, target_col] = (y_log - s["mean"]) / s["std"]

        return df

    def inverse(self, value, brand):

        s = self.stats[brand]
        y_log = value * s["std"] + s["mean"]
        return np.expm1(y_log)
    
    def transform_value(self, value, brand):
        s = self.stats[brand]
        return (value - s["mean"]) / s["std"]