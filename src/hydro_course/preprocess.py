"""Preprocessing helpers for hydrometric daily series.

All functions are trailing-only: rolling statistics never include the current
day, so they are safe to use as features without leaking the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resample_daily(
    raw: pd.DataFrame,
    *,
    time_col: str = "Time_Tag",
    value_col: str = "MeasuredValue",
) -> pd.DataFrame:
    """Aggregate 15-minute discharge to daily mean / max / min."""
    s = raw.sort_values(time_col).set_index(time_col)[value_col]
    out = pd.DataFrame({
        "q_mean": s.resample("D").mean(),
        "q_max": s.resample("D").max(),
        "q_min": s.resample("D").min(),
    })
    return out


def short_gap_interpolate(series: pd.Series, max_gap: int = 2) -> pd.Series:
    """Linearly interpolate runs of NaN whose length is <= max_gap days.

    Longer gaps are left as NaN so downstream code can split the series
    into independent runs.
    """
    s = series.copy()
    is_na = s.isna()
    gid = (is_na != is_na.shift()).cumsum()
    sizes = is_na.groupby(gid).transform("sum")
    fillable = is_na & (sizes <= max_gap)
    interp = s.interpolate(method="time")
    s.loc[fillable] = interp.loc[fillable]
    return s


def trailing_zscore(series: pd.Series, window: int = 14, min_periods: int = 7) -> pd.Series:
    """Rolling-z using only past values (shift by 1 before the window)."""
    rm = series.shift(1).rolling(window, min_periods=min_periods).mean()
    rs = series.shift(1).rolling(window, min_periods=min_periods).std()
    return (series - rm) / rs


def build_features(
    q: pd.Series,
    *,
    lags: tuple[int, ...] = (1, 2, 7),
    rolling: tuple[tuple[int, str], ...] = ((7, "mean"), (7, "std"), (7, "max"), (30, "mean"), (30, "max")),
    add_seasonal: bool = True,
) -> pd.DataFrame:
    """Build the canonical feature DataFrame used by Modules 4-7."""
    feats = pd.DataFrame(index=q.index)
    feats["q"] = q
    for lag in lags:
        feats[f"q_lag{lag}"] = q.shift(lag)
    for window, stat in rolling:
        roll = q.shift(1).rolling(window, min_periods=max(2, window // 2))
        feats[f"q_roll{window}_{stat}"] = getattr(roll, stat)()
    if add_seasonal:
        doy = feats.index.dayofyear.values
        feats["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        feats["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
        feats["month"] = feats.index.month
    return feats


def time_split(index: pd.DatetimeIndex, train_frac: float = 0.70, val_frac: float = 0.15) -> pd.Series:
    """Tag each timestamp as 'train' / 'val' / 'test' by chronological position."""
    n = len(index)
    t_train_end = index[int(train_frac * n)]
    t_val_end = index[int((train_frac + val_frac) * n)]
    out = pd.Series(index=index, dtype="object")
    out.loc[: t_train_end] = "train"
    out.loc[t_train_end:t_val_end] = "val"
    out.loc[t_val_end:] = "test"
    return out
