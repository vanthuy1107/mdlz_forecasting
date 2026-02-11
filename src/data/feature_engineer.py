import pandas as pd
import numpy as np
from typing import Tuple
from src.data import RollingGroupScaler
from config import load_holidays

HOLIDAYS = load_holidays(None)
OFF_HOLIDAYS = {d for y in HOLIDAYS.values() for d in y.get("off", [])}
OTHER_HOLIDAYS = {d for y in HOLIDAYS.values() for d in y.get("other", [])}


def encode_brands(df, brand_col):

    if df[brand_col].isna().any():
        raise ValueError(f"NaN detected in {brand_col} before encoding")

    df[brand_col] = df[brand_col].astype(str)

    categories = sorted(df[brand_col].unique())
    brand2id = {b: i for i, b in enumerate(categories)}

    df["brand_id"] = df[brand_col].map(brand2id)

    return df, brand2id, len(brand2id)


def add_calendar_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[time_col])
    dates = dt.dt.date

    # --- cyclical features (unchanged) ---
    month = dt.dt.month
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    phase = (dt.dt.day - 1) / dt.dt.days_in_month
    df["dayofmonth_sin"] = np.sin(2 * np.pi * phase)
    df["dayofmonth_cos"] = np.cos(2 * np.pi * phase)

    dow = dt.dt.dayofweek
    df["is_weekend"] = (dow >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- holidays ---
    df["is_off_holiday"] = dates.isin(OFF_HOLIDAYS).astype(int)
    df["is_other_holiday"] = dates.isin(OTHER_HOLIDAYS).astype(int)

    # ✅ calendar-correct pre-holiday
    next_dates = dates + pd.Timedelta(days=1)
    df["is_pre_off_holiday"] = next_dates.isin(OFF_HOLIDAYS).astype(int)
    df["is_pre_other_holiday"] = next_dates.isin(OTHER_HOLIDAYS).astype(int)

    return df



def add_baseline(df, target_col, time_col, brand_col):
    df = df.sort_values([brand_col, time_col]).copy()
    df["baseline"] = (
        df.groupby(brand_col)[target_col]
          .rolling(7, min_periods=1)
          .mean()
          .shift(1)
          .reset_index(level=0, drop=True)
    )
    df["baseline"] = df.groupby(brand_col)["baseline"].ffill().fillna(0)
    df.loc[df["is_pre_off_holiday"] == 1, "baseline"] = 0.0
    return df


def add_history_features(df, time_col, brand_col, target_col):
    df = add_calendar_features(df, time_col)
    df = add_baseline(df, target_col, time_col, brand_col)
    df["residual"] = df[target_col] - df["baseline"]
    return df


class FeatureEngineer:
    def __init__(self, config):
        self.config = config

        # Normalize config column names once
        self.time_col = config.data["time_col"].lower()
        self.brand_col = config.data["brand_col"].lower()
        self.target_col = config.data["target_col"].lower()
        self.brand_id_col = config.data["brand_id_col"].lower()

        self.scaler: RollingGroupScaler | None = None
        self.brand2id = None
        self.num_brands = None

    # ---------- public API ----------

    def fit_transform(self, df: pd.DataFrame, fit_end_date: pd.Timestamp):
        df = self._clean(df)
        df = self._features(df)
        df = self._encode(df, fit=True)
        return self._scale(df, fit_end_date)

    def transform(self, df: pd.DataFrame):
        self._check_fitted()
        df = self._clean(df)
        df = self._features(df)
        df = self._encode(df, fit=False)
        return self.scaler.transform(df)

    def inverse_transform_y(self, y_scaled, cat_ids):
        self._check_fitted()
        return self.scaler.inverse_transform_y(y_scaled, cat_ids)

    # ---------- internals ----------

    def _check_fitted(self):
        if self.scaler is None:
            raise RuntimeError("FeatureEngineer not fitted")

    def _clean(self, df):
        df = df.copy()

        # Ensure dataframe columns are lowercase
        df.columns = df.columns.str.lower()

        df[self.time_col] = pd.to_datetime(df[self.time_col])

        return df.sort_values(self.time_col).reset_index(drop=True)

    def _features(self, df):
        return add_history_features(
            df,
            self.time_col,
            self.brand_col,
            self.target_col,
        )

    def _encode(self, df, fit):

        if fit:
            df, self.brand2id, self.num_brands = encode_brands(df, self.brand_col)

            # Update model config
            self.config.set("model.num_brands", self.num_brands)

        else:
            unknown = set(df[self.brand_col]) - set(self.brand2id)
            if unknown:
                raise ValueError(f"Unknown brands at inference: {unknown}")

            df[self.brand_id_col] = df[self.brand_col].map(self.brand2id)

        return df

    def _scale(self, df, fit_end_date):

        self.scaler = RollingGroupScaler(
            group_col=self.brand_id_col,
            time_col=self.time_col,
            feature_cols="residual",
            lookback_months=1,
        )

        self.scaler.fit(df, fit_end_date)

        return self.scaler.transform(df)


