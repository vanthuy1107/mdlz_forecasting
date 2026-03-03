from __future__ import annotations
import pandas as pd
import numpy as np
from collections import deque
from config import load_holidays
from src.utils import MonthlyAdaptiveScaler
from typing import Any, Dict, List
from abc import ABC, abstractmethod

HOLIDAYS = load_holidays(None)

TET_DATES = set()
for year, groups in HOLIDAYS.items():
    if "Tet" in groups:
        for d in groups["Tet"]:
            TET_DATES.add(pd.Timestamp(d).date())

OFF_HOLIDAYS = {
    pd.Timestamp(d).date()
    for y in HOLIDAYS.values()
    for g in y.values()
    for d in g
}

# ==========================================================
# Feature Engineer
# ==========================================================
"""
FeatureEngineer V3 — decorator-based auto-discovery.

Để thêm feature mới:
    1. Thêm 1 class với @register_feature("tên_feature") ở phần REGISTRY
    2. Uncomment tên đó trong config.yaml
    → XONG, không cần chạm gì khác.

Để xóa feature:
    1. Xóa class tương ứng (hoặc giữ lại, chỉ comment trong config)

Naming convention cho dynamic features (không cần class):
    lag_<N>           e.g. lag_1, lag_7
    rolling_mean_<N>  e.g. rolling_mean_7
    rolling_std_<N>   e.g. rolling_std_7
"""

# ===========================================================================
# DECORATOR & BASE CLASS
# ===========================================================================

FEATURE_REGISTRY: Dict[str, "BaseFeature"] = {}


def register_feature(name: str):
    """Decorator — đăng ký feature vào registry theo tên config."""
    def decorator(cls):
        FEATURE_REGISTRY[name] = cls()
        return cls
    return decorator


class BaseFeature(ABC):
    """Mỗi feature implement 2 method: vec (batch) và single (inference)."""

    @abstractmethod
    def vec(self, df: pd.DataFrame, dt: pd.Series, cfg: dict) -> pd.DataFrame:
        """Vectorized — dùng trong fit_transform / transform."""
        ...

    @abstractmethod
    def single(self, ts: pd.Timestamp, cfg: dict) -> Any:
        """Single-row — dùng trong recursive inference."""
        ...


# ===========================================================================
# SHARED VECTOR HELPERS
# ===========================================================================

def _tet_nearest_vec(dates_d64: np.ndarray, tet_d64: np.ndarray) -> np.ndarray:
    pos      = np.searchsorted(tet_d64, dates_d64)
    prev_idx = np.clip(pos - 1, 0, len(tet_d64) - 1)
    next_idx = np.clip(pos,     0, len(tet_d64) - 1)
    dist_prev = (dates_d64 - tet_d64[prev_idx]).astype(int)
    dist_next = (tet_d64[next_idx] - dates_d64).astype(int)
    return np.where(np.abs(dist_prev) <= np.abs(dist_next), -dist_prev, dist_next)


def _holiday_dists_vec(dates_d64: np.ndarray, hol_d64: np.ndarray):
    pos      = np.searchsorted(hol_d64, dates_d64)
    prev_idx = np.clip(pos - 1, 0, len(hol_d64) - 1)
    next_idx = np.clip(pos,     0, len(hol_d64) - 1)
    dist_prev = (dates_d64 - hol_d64[prev_idx]).astype(int)
    dist_next = (hol_d64[next_idx] - dates_d64).astype(int)
    return dist_prev, dist_next


def _d64(dt: pd.Series) -> np.ndarray:
    return dt.values.astype("datetime64[D]")


def _tet_nearest_single(ts: pd.Timestamp, cfg: dict):
    if not cfg["tet_dates_list"]:
        return None
    return min(((ts.date() - t).days for t in cfg["tet_dates_list"]), key=abs)


# ===========================================================================
# ██████████████  FEATURE DEFINITIONS  ██████████████
# Để thêm feature mới: copy 1 block dưới, đổi tên class + logic.
# ===========================================================================

# ── Calendar ────────────────────────────────────────────────────────────────

@register_feature("month_sin")
class MonthSin(BaseFeature):
    def vec(self, df, dt, cfg):
        df["month_sin"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)
        return df
    def single(self, ts, cfg):
        return np.sin(2 * np.pi * (ts.month - 1) / 12)


@register_feature("month_cos")
class MonthCos(BaseFeature):
    def vec(self, df, dt, cfg):
        df["month_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)
        return df
    def single(self, ts, cfg):
        return np.cos(2 * np.pi * (ts.month - 1) / 12)


@register_feature("dayofmonth_sin")
class DayOfMonthSin(BaseFeature):
    def vec(self, df, dt, cfg):
        df["dayofmonth_sin"] = np.sin(2 * np.pi * (dt.dt.day - 1) / dt.dt.days_in_month)
        return df
    def single(self, ts, cfg):
        return np.sin(2 * np.pi * (ts.day - 1) / ts.days_in_month)


@register_feature("dayofmonth_cos")
class DayOfMonthCos(BaseFeature):
    def vec(self, df, dt, cfg):
        df["dayofmonth_cos"] = np.cos(2 * np.pi * (dt.dt.day - 1) / dt.dt.days_in_month)
        return df
    def single(self, ts, cfg):
        return np.cos(2 * np.pi * (ts.day - 1) / ts.days_in_month)


@register_feature("wom_sin")
class WomSin(BaseFeature):
    def vec(self, df, dt, cfg):
        df["wom_sin"] = np.sin(2 * np.pi * ((dt.dt.day - 1) // 7) / 4)
        return df
    def single(self, ts, cfg):
        return np.sin(2 * np.pi * ((ts.day - 1) // 7) / 4)


@register_feature("wom_cos")
class WomCos(BaseFeature):
    def vec(self, df, dt, cfg):
        df["wom_cos"] = np.cos(2 * np.pi * ((dt.dt.day - 1) // 7) / 4)
        return df
    def single(self, ts, cfg):
        return np.cos(2 * np.pi * ((ts.day - 1) // 7) / 4)


@register_feature("dow_sin")
class DowSin(BaseFeature):
    def vec(self, df, dt, cfg):
        df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        return df
    def single(self, ts, cfg):
        return np.sin(2 * np.pi * ts.dayofweek / 7)


@register_feature("dow_cos")
class DowCos(BaseFeature):
    def vec(self, df, dt, cfg):
        df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        return df
    def single(self, ts, cfg):
        return np.cos(2 * np.pi * ts.dayofweek / 7)


@register_feature("is_weekend")
class IsWeekend(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.dayofweek >= 5)


@register_feature("is_month_start")
class IsMonthStart(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_month_start"] = (dt.dt.day <= 3).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.day <= 3)


# ── Day-of-week binary flags ─────────────────────────────────────────────────

@register_feature("is_monday")
class IsMonday(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_monday"] = (dt.dt.dayofweek == 0).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.dayofweek == 0)


@register_feature("is_tuesday")
class IsTuesday(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_tuesday"] = (dt.dt.dayofweek == 1).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.dayofweek == 1)


@register_feature("is_wednesday")
class IsWednesday(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_wednesday"] = (dt.dt.dayofweek == 2).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.dayofweek == 2)


@register_feature("is_thursday")
class IsThursday(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_thursday"] = (dt.dt.dayofweek == 3).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.dayofweek == 3)


@register_feature("is_friday")
class IsFriday(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_friday"] = (dt.dt.dayofweek == 4).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.dayofweek == 4)


@register_feature("is_saturday")
class IsSaturday(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_saturday"] = (dt.dt.dayofweek == 5).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.dayofweek == 5)


@register_feature("is_sunday")
class IsSunday(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_sunday"] = (dt.dt.dayofweek == 6).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.dayofweek == 6)


# ── Holiday ──────────────────────────────────────────────────────────────────

@register_feature("is_holiday")
class IsHoliday(BaseFeature):
    def vec(self, df, dt, cfg):
        df["is_holiday"] = np.isin(_d64(dt), cfg["holidays_np"]).astype(int)
        return df
    def single(self, ts, cfg):
        return int(ts.date() in OFF_HOLIDAYS)


@register_feature("holiday_distance")
class HolidayDistance(BaseFeature):
    def vec(self, df, dt, cfg):
        hol = cfg["holidays_np"]
        if len(hol) == 0:
            df["holiday_distance"] = 0
            return df
        dp, dn = _holiday_dists_vec(_d64(dt), hol)
        nearest = np.where(np.abs(dp) <= np.abs(dn), -dp, dn)
        df["holiday_distance"] = np.clip(nearest, -28, 28)
        return df
    def single(self, ts, cfg):
        hol = cfg["holidays_np"]
        if len(hol) == 0:
            return 0
        dates = np.array([ts.date()], dtype="datetime64[D]")
        dp, dn = _holiday_dists_vec(dates, hol)
        nearest = np.where(np.abs(dp) <= np.abs(dn), -dp, dn)
        return int(np.clip(nearest[0], -28, 28))


@register_feature("days_since_last_holiday")
class DaysSinceLastHoliday(BaseFeature):
    def vec(self, df, dt, cfg):
        hol = cfg["holidays_np"]
        if len(hol) == 0:
            df["days_since_last_holiday"] = 28
            return df
        dp, _ = _holiday_dists_vec(_d64(dt), hol)
        df["days_since_last_holiday"] = np.clip(np.maximum(dp, 0), 0, 28)
        return df
    def single(self, ts, cfg):
        hol = cfg["holidays_np"]
        if len(hol) == 0:
            return 28
        dates = np.array([ts.date()], dtype="datetime64[D]")
        dp, _ = _holiday_dists_vec(dates, hol)
        return int(np.clip(max(dp[0], 0), 0, 28))


@register_feature("days_to_next_holiday")
class DaysToNextHoliday(BaseFeature):
    def vec(self, df, dt, cfg):
        hol = cfg["holidays_np"]
        if len(hol) == 0:
            df["days_to_next_holiday"] = 28
            return df
        _, dn = _holiday_dists_vec(_d64(dt), hol)
        df["days_to_next_holiday"] = np.clip(np.maximum(dn, 0), 0, 28)
        return df
    def single(self, ts, cfg):
        hol = cfg["holidays_np"]
        if len(hol) == 0:
            return 28
        dates = np.array([ts.date()], dtype="datetime64[D]")
        _, dn = _holiday_dists_vec(dates, hol)
        return int(np.clip(max(dn[0], 0), 0, 28))


# ── Tet ──────────────────────────────────────────────────────────────────────

@register_feature("pre_tet")
class PreTet(BaseFeature):
    def vec(self, df, dt, cfg):
        tet = cfg["tet_dates_np"]
        if len(tet) == 0:
            df["pre_tet"] = 0
            return df
        nearest = _tet_nearest_vec(_d64(dt), tet)
        df["pre_tet"] = np.where((nearest < 0) & (nearest >= -15), -nearest, 0)
        return df
    def single(self, ts, cfg):
        n = _tet_nearest_single(ts, cfg)
        if n is None:
            return 0
        return -n if -15 <= n < 0 else 0


@register_feature("post_tet")
class PostTet(BaseFeature):
    def vec(self, df, dt, cfg):
        tet = cfg["tet_dates_np"]
        if len(tet) == 0:
            df["post_tet"] = 0
            return df
        nearest = _tet_nearest_vec(_d64(dt), tet)
        df["post_tet"] = np.where((nearest > 0) & (nearest <= 7), nearest, 0)
        return df
    def single(self, ts, cfg):
        n = _tet_nearest_single(ts, cfg)
        if n is None:
            return 0
        return n if 0 < n <= 7 else 0


@register_feature("is_tet_window")
class IsTetWindow(BaseFeature):
    def vec(self, df, dt, cfg):
        tet = cfg["tet_dates_np"]
        if len(tet) == 0:
            df["is_tet_window"] = 0
            return df
        nearest = _tet_nearest_vec(_d64(dt), tet)
        df["is_tet_window"] = ((nearest >= -15) & (nearest <= 7)).astype(int)
        return df
    def single(self, ts, cfg):
        n = _tet_nearest_single(ts, cfg)
        if n is None:
            return 0
        return int(-15 <= n <= 7)


# ===========================================================================
# FeatureEngineer
# ===========================================================================

class FeatureEngineer:

    def __init__(self, config):
        self.config       = config
        self.time_col     = config.data["time_col"].lower()
        self.brand_col    = config.data["brand_col"].lower()
        self.target_col   = config.data["target_col"].lower()
        self.brand_id_col = config.data["brand_id_col"].lower()
        self.feature_cols: List[str] = config.data["feature_cols"]

        self.brand2id   = None
        self.num_brands = None
        self.scaler     = MonthlyAdaptiveScaler(n_months=6)

        self._parse_feature_meta()
        self._validate_feature_cols()
        self._cfg = self._build_feature_cfg()
        self.required_history = self._compute_required_history(
            config.window["input_size"]
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._clean(df)
        self.scaler.fit(df, brand_col=self.brand_col,
                        target_col=self.target_col, time_col=self.time_col)
        df = self.scaler.transform(df, brand_col=self.brand_col,
                                   target_col=self.target_col)
        df = self._add_features_vectorized(df)
        return self._encode(df, fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._clean(df)
        df = self.scaler.transform(df, brand_col=self.brand_col,
                                   target_col=self.target_col)
        df = self._add_features_vectorized(df)
        return self._encode(df, fit=False)

    # ── Recursive inference ─────────────────────────────────────────────────

    def init_brand_state(self, df_brand: pd.DataFrame) -> dict:
        df_brand = df_brand.sort_values(self.time_col)
        brand    = df_brand[self.brand_col].iloc[-1]
        return {
            "brand":         brand,
            "brand_id":      self.brand2id[brand],
            "target_buffer": deque(
                df_brand[self.target_col].values[-self.required_history:],
                maxlen=self.required_history + 5
            ),
        }

    def build_next_features(self, state: dict, new_date, new_scaled_target: float) -> np.ndarray:
        """No pandas. No concat. No groupby."""
        buffer = state["target_buffer"]
        buffer.append(new_scaled_target)

        feats: Dict[str, Any] = {}
        ts = pd.Timestamp(new_date)

        for name, lag in self.lag_features.items():
            feats[name] = buffer[-lag - 1] if len(buffer) > lag else 0.0

        for name, window in self.rolling_mean_features.items():
            vals = list(buffer)[-window - 1:-1]
            feats[name] = float(np.mean(vals)) if vals else 0.0

        for name, window in self.rolling_std_features.items():
            vals = list(buffer)[-window - 1:-1]
            feats[name] = float(np.std(vals)) if vals else 0.0

        for fc in self.feature_cols:
            if fc in FEATURE_REGISTRY:
                feats[fc] = FEATURE_REGISTRY[fc].single(ts, self._cfg)

        return np.array(
            [feats.get(fc, 0.0) for fc in self.feature_cols],
            dtype=np.float32
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def is_holiday(self, date, brand=None) -> bool:
        return pd.Timestamp(date).date() in OFF_HOLIDAYS

    # ── Internals ───────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.lower()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        return df.sort_values([self.brand_col, self.time_col]).reset_index(drop=True)

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            categories      = sorted(df[self.brand_col].unique())
            self.brand2id   = {b: i for i, b in enumerate(categories)}
            self.num_brands = len(self.brand2id)
            self.config.set("model.num_brands", self.num_brands)
        df[self.brand_id_col] = df[self.brand_col].map(self.brand2id)
        return df

    def _add_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dt = df[self.time_col]

        for fc in self.feature_cols:
            if fc in FEATURE_REGISTRY:
                df = FEATURE_REGISTRY[fc].vec(df, dt, self._cfg)

        grp = df.groupby(self.brand_col, sort=False)

        for name, lag in self.lag_features.items():
            df[name] = grp[self.target_col].shift(lag)

        for name, window in self.rolling_mean_features.items():
            df[name] = grp[self.target_col].shift(1).rolling(window, min_periods=1).mean()

        for name, window in self.rolling_std_features.items():
            df[name] = (grp[self.target_col].shift(1)
                          .rolling(window, min_periods=1).std().fillna(0))

        df.fillna(0, inplace=True)
        return df

    def _parse_feature_meta(self):
        self.lag_features          = {}
        self.rolling_mean_features = {}
        self.rolling_std_features  = {}
        for fc in self.feature_cols:
            if fc.startswith("lag_"):
                self.lag_features[fc] = int(fc.split("_")[1])
            elif fc.startswith("rolling_mean_"):
                self.rolling_mean_features[fc] = int(fc.split("_")[2])
            elif fc.startswith("rolling_std_"):
                self.rolling_std_features[fc] = int(fc.split("_")[2])

    def _validate_feature_cols(self):
        """Fail-fast: báo lỗi ngay nếu tên feature trong config không tồn tại."""
        dynamic_prefixes = ("lag_", "rolling_mean_", "rolling_std_")
        unknown = [
            fc for fc in self.feature_cols
            if fc not in FEATURE_REGISTRY
            and not any(fc.startswith(p) for p in dynamic_prefixes)
        ]
        if unknown:
            raise ValueError(
                f"\n[FeatureEngineer] Unknown feature(s) in config: {unknown}"
                f"\nAvailable: {sorted(FEATURE_REGISTRY.keys())}"
                f"\nDynamic prefixes: {list(dynamic_prefixes)}"
            )

    def _compute_required_history(self, input_size: int) -> int:
        max_lb = input_size
        for v in (*self.lag_features.values(),
                  *self.rolling_mean_features.values(),
                  *self.rolling_std_features.values()):
            max_lb = max(max_lb, v)
        return max_lb + 1

    def _build_feature_cfg(self) -> dict:
        return {
            "holidays_np":    np.array(sorted(pd.to_datetime(list(OFF_HOLIDAYS))),
                                       dtype="datetime64[D]"),
            "tet_dates_np":   np.array(sorted(pd.to_datetime(list(TET_DATES))),
                                       dtype="datetime64[D]"),
            "tet_dates_list": sorted(TET_DATES),
        }