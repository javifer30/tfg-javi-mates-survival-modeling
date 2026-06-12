"""Data loading for dynamic_72h model experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.dynamic_72h.discretization import discretize_duration_event


@dataclass
class Dynamic72hSplit:
    name: str
    patient_ids: np.ndarray
    x_seq: np.ndarray
    m_seq: np.ndarray
    x_static: np.ndarray
    duration_eval_days: np.ndarray
    duration_rel_days: np.ndarray
    event_eval: np.ndarray
    t_idx: np.ndarray
    event: np.ndarray


def load_split(dataset_dir: str | Path, split: str, sample_size: int | None = None) -> Dynamic72hSplit:
    path = Path(dataset_dir) / f"{split}_dynamic_72h.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    n = arrays["patient_ids"].shape[0]
    if sample_size is not None:
        n = min(int(sample_size), n)
        arrays = {key: value[:n] for key, value in arrays.items()}
    t_idx, event = discretize_duration_event(arrays["duration_eval_days"], arrays["event_eval"])
    return Dynamic72hSplit(
        name=split,
        patient_ids=arrays["patient_ids"],
        x_seq=arrays["X_seq"].astype("float32", copy=False),
        m_seq=arrays["M_seq"].astype("float32", copy=False),
        x_static=arrays["X_static"].astype("float32", copy=False),
        duration_eval_days=arrays["duration_eval_days"].astype("float32", copy=False),
        duration_rel_days=arrays["duration_rel_days"].astype("float32", copy=False),
        event_eval=arrays["event_eval"].astype("int64", copy=False),
        t_idx=t_idx,
        event=event,
    )


def build_model_input(split: Dynamic72hSplit, input_mode: str) -> np.ndarray:
    if input_mode == "values_only":
        return split.x_seq
    if input_mode == "values_plus_mask":
        return np.concatenate([split.x_seq, split.m_seq], axis=2).astype("float32")
    if input_mode == "values_plus_mask_plus_static":
        repeated_static = np.repeat(split.x_static[:, None, :], split.x_seq.shape[1], axis=1)
        return np.concatenate([split.x_seq, split.m_seq, repeated_static], axis=2).astype("float32")
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def input_metadata(split: Dynamic72hSplit, input_mode: str) -> dict:
    n_temporal = int(split.x_seq.shape[2])
    n_mask = n_temporal if "mask" in input_mode else 0
    n_static = int(split.x_static.shape[1]) if "static" in input_mode else 0
    return {
        "n_temporal_features": n_temporal,
        "n_mask_features": n_mask,
        "n_static_features": n_static,
        "n_model_input_features": n_temporal + n_mask + n_static,
        "input_mode": input_mode,
    }


class Dynamic72hDataset(Dataset):
    def __init__(self, split: Dynamic72hSplit, input_mode: str):
        self.split = split
        self.x = build_model_input(split, input_mode)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.x[idx]),
            "x_recon": torch.from_numpy(self.x[idx]),
            "t_idx": torch.tensor(self.split.t_idx[idx], dtype=torch.long),
            "event": torch.tensor(self.split.event[idx], dtype=torch.long),
            "duration": torch.tensor(self.split.duration_eval_days[idx], dtype=torch.float32),
            "patient_id": str(self.split.patient_ids[idx]),
        }


def make_loader(split: Dynamic72hSplit, input_mode: str, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(
        Dynamic72hDataset(split, input_mode),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=False,
    )


def validate_splits(train: Dynamic72hSplit, val: Dynamic72hSplit, test: Dynamic72hSplit | None = None) -> dict:
    train_ids = set(map(str, train.patient_ids))
    val_ids = set(map(str, val.patient_ids))
    test_ids = set(map(str, test.patient_ids)) if test is not None else set()
    checks = {
        "train_validation_no_overlap": len(train_ids & val_ids) == 0,
        "train_test_no_overlap": len(train_ids & test_ids) == 0 if test is not None else True,
        "validation_test_no_overlap": len(val_ids & test_ids) == 0 if test is not None else True,
        "train_t_idx_min": int(train.t_idx.min()),
        "train_t_idx_max": int(train.t_idx.max()),
        "validation_t_idx_min": int(val.t_idx.min()),
        "validation_t_idx_max": int(val.t_idx.max()),
    }
    if test is not None:
        checks["test_t_idx_min"] = int(test.t_idx.min())
        checks["test_t_idx_max"] = int(test.t_idx.max())
    if not all(v for k, v in checks.items() if k.endswith("overlap")):
        raise ValueError(f"Split overlap check failed: {checks}")
    for split in [train, val] + ([test] if test is not None else []):
        if split.t_idx.min() < 0 or split.t_idx.max() > 9:
            raise ValueError(f"{split.name} has t_idx outside [0, 9]")
    return checks

