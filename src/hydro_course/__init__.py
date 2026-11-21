"""Shared helpers for the Hydrology ML course notebooks."""

from hydro_course.metrics import hydrology_metrics, regression_metrics
from hydro_course.windows import make_lag_features, make_sliding_windows
from hydro_course.paths import (
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    ensure_module_dirs,
    module_figures_dir,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_PROCESSED",
    "FIGURES_DIR",
    "OUTPUTS_DIR",
    "MODELS_DIR",
    "ensure_module_dirs",
    "module_figures_dir",
    "regression_metrics",
    "hydrology_metrics",
    "make_sliding_windows",
    "make_lag_features",
]
