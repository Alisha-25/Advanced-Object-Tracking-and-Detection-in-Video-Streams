"""Configuration and reproducibility helpers shared by command-line scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML configuration and resolve relative output paths from its folder."""
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping.")
    return config, config_file.parent


def resolve_path(value: str | Path, config_folder: Path) -> Path:
    """Resolve a path, preserving absolute Windows or POSIX locations."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_folder / path).resolve()


def get_device(requested: str) -> torch.device:
    """Select CUDA when available unless the configuration explicitly requests CPU."""
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU.")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Seed common RNGs so the sampled training frames are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a readable UTF-8 JSON result file, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
