"""Shared path helpers for local scripts and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DEPLOY_DIR = PROJECT_ROOT / "deploy"
DEPLOY_MODELS_DIR = DEPLOY_DIR / "models"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_WEIGHT_NAME = "LOL_v2_synthetic.pth"
DEFAULT_ONNX_NAME = "retinexformer.onnx"


def resolve_project_path(path_like: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def candidate_weight_paths(filename: str = DEFAULT_WEIGHT_NAME) -> list[Path]:
    return [
        MODELS_DIR / filename,
        DEPLOY_MODELS_DIR / filename,
    ]


def resolve_weights_path(
    weights_path: str | Path | None = None,
    filename: str = DEFAULT_WEIGHT_NAME,
) -> Path:
    if weights_path:
        return resolve_project_path(weights_path)

    existing = first_existing(candidate_weight_paths(filename))
    if existing is not None:
        return existing

    return candidate_weight_paths(filename)[0]


def resolve_onnx_path(
    onnx_path: str | Path | None = None,
    filename: str = DEFAULT_ONNX_NAME,
) -> Path:
    if onnx_path:
        return resolve_project_path(onnx_path)
    return DEPLOY_MODELS_DIR / filename


def standard_directories() -> list[Path]:
    return [
        MODELS_DIR,
        DEPLOY_MODELS_DIR,
        PROJECT_ROOT / "deploy" / "verification",
    ]


def format_path_list(paths: Iterable[Path]) -> str:
    return ", ".join(str(path) for path in paths)
