"""Regression metrics for the Hydrology ML course.

Training helpers (MAE, MSE, RMSE, R²) remain for loss diagnostics and sklearn
cross-checks. **Reporting and model-comparison tables** use the four professor
metrics: RMAE, MAPE, NSE, KGE — see ``hydrology_metrics``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, MSE, RMSE, and R² for 1-D arrays (internal / training diagnostics)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def metrics_by_hand(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Same sklearn-style metrics computed explicitly (for teaching loss / R²)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash–Sutcliffe Efficiency (Nash & Sutcliffe, 1970).

    NSE = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²

    Range: (−∞, 1]. 1 = perfect; 0 = skill of predicting the mean; negative =
    worse than the mean. Sensitive to the variance of the test period — a flat
    low-flow test set can inflate NSE even when peak timing is poor.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_true - y_pred
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kling–Gupta Efficiency (Gupta et al., 2009; Kling et al., 2012).

    KGE = 1 − √[(r − 1)² + (α − 1)² + (β − 1)²]

    where r = Pearson correlation, α = σ_ŷ / σ_y, β = μ_ŷ / μ_y.

    Range: (−∞, 1]. Decomposes skill into correlation, variability, and bias.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) < 2:
        return float("nan")
    mu_y, mu_p = float(np.mean(y_true)), float(np.mean(y_pred))
    sigma_y, sigma_p = float(np.std(y_true, ddof=0)), float(np.std(y_pred, ddof=0))
    if sigma_y == 0 or sigma_p == 0:
        r = 0.0 if sigma_y == 0 and sigma_p == 0 else float("nan")
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = sigma_p / sigma_y if sigma_y > 0 else float("nan")
    beta = mu_p / mu_y if mu_y != 0 else float("nan")
    if any(np.isnan(v) for v in (r, alpha, beta)):
        return float("nan")
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_obs: float = 1e-6,
) -> float:
    """Mean Absolute Percentage Error (%).

    MAPE = mean(|y − ŷ| / |y|) × 100

    **Zero-flow assumption:** observations with |y| ≤ ``min_obs`` m³/s are
    excluded from the average. On HY024, true zero flow is rare but very low
    flows make MAPE unstable; document the threshold in your report.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.abs(y_true) > min_obs
    if not np.any(mask):
        return float("nan")
    pct = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(pct) * 100.0)


def rmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative Mean Absolute Error (dimensionless).

    **Course definition:** RMAE = MAE / mean(y)

    i.e. mean absolute error normalised by the mean observed discharge. Values
    near 0 are better; interpret as "typical error as a fraction of typical flow."
    (Some papers use MAE / range instead — we use mean(y) consistently.)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mu = float(np.mean(y_true))
    if mu == 0:
        return float("nan")
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return float(mae / mu)


def hydrology_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mape_min_obs: float = 1e-6,
) -> dict[str, float]:
    """Return RMAE, MAPE, NSE, and KGE — the four professor reporting metrics."""
    return {
        "RMAE": rmae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred, min_obs=mape_min_obs),
        "NSE": nse(y_true, y_pred),
        "KGE": kge(y_true, y_pred),
    }


def hydrology_metrics_by_hand(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mape_min_obs: float = 1e-6,
) -> dict[str, float]:
    """Explicit implementations of RMAE, MAPE, NSE, KGE (for teaching)."""
    return hydrology_metrics(y_true, y_pred, mape_min_obs=mape_min_obs)
