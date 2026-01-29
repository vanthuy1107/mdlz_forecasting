"""
Rolling Group-wise Robust Scaler
"""
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
        scaler_kwargs=None
    ):
        self.group_col = group_col
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.lookback_months = lookback_months
        self.scaler_cls = scaler_cls
        self.scaler_kwargs = scaler_kwargs or {"quantile_range": (10, 90)}

        self.scalers = {}            # group_id -> scaler
        self.valid_groups_ = set()   # ✅ groups actually fitted
        self.origin = None

    def fit(self, df, origin : pd.Timestamp):

        self.scalers = {}
        self.valid_groups_ = set()
        self.origin = origin

        start_date = origin - pd.DateOffset(months=self.lookback_months)

        mask = (
            (df[self.time_col] >= start_date) &
            (df[self.time_col] < origin)
        )

        df_window = df.loc[mask]

        if df_window.empty:
            raise ValueError("No data to fit scaler in lookback window")

        for g, gdf in df_window.groupby(self.group_col):
            if len(gdf) < 5:
                continue  # ❌ too little data → invalid group

            scaler = self.scaler_cls(**self.scaler_kwargs)

            X = gdf[self.feature_cols].values
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            scaler.fit(X)

            self.scalers[g] = scaler
            self.valid_groups_.add(g)   # ✅ mark valid

        
        # print(f"[Scaler.fit] origin={origin}")
        # print(f"[Scaler.fit] valid_groups_={self.valid_groups_}")

        if not self.valid_groups_:
            raise ValueError("No valid groups after fitting scaler")

        return self

    def filter_invalid_groups(self, df):
        """
        Drop rows belonging to categories without fitted scaler
        """
        if not hasattr(self, "valid_groups_"):
            raise RuntimeError("Scaler has not been fitted")

        before = len(df)
        df_filtered = df[df[self.group_col].isin(self.valid_groups_)].copy()
        after = len(df_filtered)

        dropped = before - after
        if dropped > 0:
            print(
                f"[Scaler] Dropped {dropped} rows "
                f"from {before} due to missing scaler "
                f"(kept {len(self.valid_groups_)} groups)"
            )
        # print(f"[filter] using valid_groups_={self.valid_groups_}")

        return df_filtered
    
    def transform(self, df):
        if not hasattr(self, "valid_groups_"):
            raise RuntimeError("Call fit() before transform()")

        # ❗ Loại category không hợp lệ trước
        df = self.filter_invalid_groups(df)

        df = df.copy()
        for g, scaler in self.scalers.items():
            mask = df[self.group_col] == g
            if mask.any():
                X = df.loc[mask, self.feature_cols].values
                if X.ndim == 1:
                    X = X.reshape(-1, 1)

                df.loc[mask, self.feature_cols] = scaler.transform(X)

        return df
    
    def inverse_transform_y(self, y_scaled, cat_ids):
        """
        Inverse transform target using group-specific RobustScaler

        y_scaled: np.ndarray shape (N,) or (N, horizon)
        cat_ids : np.ndarray shape (N,)
        """
        if not hasattr(self, "scalers"):
            raise RuntimeError("Scaler has not been fitted")

        y_scaled = np.asarray(y_scaled)

        if y_scaled.ndim == 1:
            y_scaled = y_scaled[:, None]

        y_inv = np.zeros_like(y_scaled, dtype=float)

        for i, cat in enumerate(cat_ids):
            scaler = self.scalers.get(int(cat))
            if scaler is None:
                continue  # should not happen if filtered correctly

            center = scaler.center_[0]
            scale = scaler.scale_[0]

            y_inv[i] = y_scaled[i] * scale + center

        return y_inv.squeeze()

