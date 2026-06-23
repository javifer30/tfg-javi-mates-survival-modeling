"""Shared helpers for dynamic_landmark models."""

from __future__ import annotations

import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = "auto") -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def create_nn(input_dim, output_dim, layers=None, dropout=0.1, activation="ReLU", no_activation_last=False):
    layers = layers or [64]
    act = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "SeLU": nn.SELU}.get(activation, nn.ReLU)
    modules = []
    prev = input_dim
    for hidden in layers:
        modules.append(nn.Linear(prev, hidden))
        modules.append(act())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        prev = hidden
    modules.append(nn.Linear(prev, output_dim))
    if not no_activation_last:
        modules.append(act())
    return nn.Sequential(*modules)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_yaml(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def metric(metrics: dict, split: str, name: str) -> float:
    try:
        return float(metrics["splits"][split][name])
    except Exception:
        return float("nan")
