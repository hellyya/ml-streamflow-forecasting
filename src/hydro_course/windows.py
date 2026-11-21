"""Sliding-window helpers for hydrometric forecasting notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_lag_features(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Build lag columns for a univariate series (index = time)."""
    out = pd.DataFrame(index=series.index)
    for lag in lags:
        out[f"lag_{lag}"] = series.shift(lag)
    out["target"] = series
    return out.dropna()


def make_sliding_windows(
    values: np.ndarray,
    window_size: int,
    forecast_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) arrays for supervised forecasting.

    X[i] = values[i : i+window_size]
    y[i] = values[i+window_size+forecast_horizon-1]  (one-step if horizon=1)
    """
    values = np.asarray(values, dtype=float).ravel()
    X, y = [], []
    end = len(values) - window_size - forecast_horizon + 1
    for i in range(end):
        X.append(values[i : i + window_size])
        y.append(values[i + window_size + forecast_horizon - 1])
    return np.array(X), np.array(y)
