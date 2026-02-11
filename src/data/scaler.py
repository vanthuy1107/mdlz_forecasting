"""
Rolling Group-wise Robust Scaler
"""
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np


class RollingGroupScaler:
    def __init__(
        self,
        group_col,
        time_col,
        feature_cols,
        lookback_months=6,
        scaler_cls=RobustScaler,
        scaler_kwargs=None,
    ):
        self.group_col = group_col
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.lookback_months = lookback_months
        self.scaler_cls = scaler_cls
        self.scaler_kwargs = scaler_kwargs or {"quantile_range": (1, 99)}

        self.scalers = {}
        self.valid_groups_ = set()

    # ---------------- core ---------------- #

    def fit(self, df: pd.DataFrame, origin: pd.Timestamp):
        self.scalers.clear()
        self.valid_groups_.clear()

        start = origin - pd.DateOffset(months=self.lookback_months)
        window = df[(df[self.time_col] >= start) & (df[self.time_col] < origin)]

        if window.empty:
            raise ValueError("No data in scaler lookback window")

        for g, gdf in window.groupby(self.group_col):
            if len(gdf) < 5:
                continue

            X = self._to_2d(gdf[self.feature_cols].values)
            scaler = self.scaler_cls(**self.scaler_kwargs).fit(X)

            self.scalers[g] = scaler
            self.valid_groups_.add(g)

        if not self.valid_groups_:
            raise ValueError("No valid groups after fitting scaler")

        return self

    def transform(self, df: pd.DataFrame):
        df = self._filter_valid(df)
        df = df.copy()

        for g, scaler in self.scalers.items():
            mask = df[self.group_col] == g
            if mask.any():
                X = self._to_2d(df.loc[mask, self.feature_cols].values)
                df.loc[mask, self.feature_cols] = scaler.transform(X)

        return df

    # ---------------- inverse ---------------- #

    def inverse_transform_y(self, y_scaled, cat_ids):
        y_scaled = np.asarray(y_scaled)
        y_scaled = self._to_2d(y_scaled)

        y_inv = np.zeros_like(y_scaled, dtype=float)

        for i, g in enumerate(cat_ids):
            scaler = self.scalers.get(int(g))
            if scaler is None:
                continue

            y_inv[i] = y_scaled[i] * scaler.scale_[0] + scaler.center_[0]

        return y_inv.squeeze()

    # ---------------- helpers ---------------- #

    def _filter_valid(self, df):
        before = len(df)
        df = df[df[self.group_col].isin(self.valid_groups_)].copy()
        dropped = before - len(df)

        if dropped:
            print(f"[Scaler] Dropped {dropped} rows (invalid groups)")

        return df

    @staticmethod
    def _to_2d(x):
        return x.reshape(-1, 1) if x.ndim == 1 else x

