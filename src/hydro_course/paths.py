"""Project path helpers — keeps notebooks portable across Colab, Kaggle, and local runs."""

from __future__ import annotations

from pathlib import Path


def _find_project_root(start: Path | None = None) -> Path:
    """Walk upward until we find README.md (course root)."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "README.md").exists() and (candidate / "data").is_dir():
            return candidate
    # Fallback: assume notebook lives under notebooks/module_XX/
    return here.parents[2] if len(here.parents) >= 3 else here


PROJECT_ROOT = _find_project_root()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_module_dirs(module_id: str) -> dict[str, Path]:
    """Create figures/outputs/models folders for a module (e.g. 'module_01')."""
    fig = FIGURES_DIR / module_id
    out = OUTPUTS_DIR / module_id
    mdl = MODELS_DIR / module_id
    for path in (fig, out, mdl):
        path.mkdir(parents=True, exist_ok=True)
    return {"figures": fig, "outputs": out, "models": mdl}


def module_figures_dir(module_id: str) -> Path:
    return ensure_module_dirs(module_id)["figures"]
