"""Training and evaluation for the isolated static-only DySurv pipeline."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.landmark_survival_metrics import (
    eval_surv_metrics,
    horizon_c_index_rows,
    mean_horizon_c_index,
    metric_integration_grid,
)
from src.models.landmark_dynamic.common import resolve_device, save_json, save_yaml, set_seed
from src.models.landmark_dynamic.discretization import discretize_duration_event
from src.models.landmark_dynamic.dysurv_static_faithful import DySurvStaticFaithful72h, kl_divergence
from src.models.landmark_dynamic.losses import hazards_to_survival, logistic_hazard_nll
from src.models.landmark_dynamic.predict import survival_df_from_array
from src.models.landmark_dynamic.train_dysurv_faithful import (
    _distribution_stats,
    _faithful_split_files,
    _save_curve_examples,
    collapse_flags,
    effective_kl_weight,
)


@dataclass
class StaticFaithfulSplit:
    name: str
    patient_ids: np.ndarray
    x_static: np.ndarray
    durations: np.ndarray
    events: np.ndarray
    t_idx: np.ndarray


def load_static_faithful_split(
    dataset_dir: str | Path,
    split: str,
    sample_size: int | None = None,
    split_files: dict[str, str] | None = None,
) -> StaticFaithfulSplit:
    file_split = "val" if split == "validation" else split
    filename = (split_files or _faithful_split_files()).get(split, f"{file_split}_dynamic_landmark.npz")
    path = Path(dataset_dir) / filename
    with np.load(path) as data:
        patient_ids = data["patient_ids"]
        x_static = data["X_static"].astype("float32", copy=False)
        durations = data["duration_eval_days"].astype("float32", copy=False)
        raw_events = data["event_eval"].astype("int64", copy=False)
    n = len(patient_ids) if sample_size is None else min(len(patient_ids), int(sample_size))
    patient_ids = patient_ids[:n]
    x_static = x_static[:n]
    durations = durations[:n]
    raw_events = raw_events[:n]
    t_idx, events = discretize_duration_event(durations, raw_events)
    return StaticFaithfulSplit(split, patient_ids, x_static, durations, events, t_idx)


class StaticFaithfulDataset(Dataset):
    def __init__(self, split: StaticFaithfulSplit):
        self.split = split

    def __len__(self) -> int:
        return len(self.split.patient_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x_static": torch.from_numpy(self.split.x_static[index]),
            "t_idx": torch.tensor(self.split.t_idx[index], dtype=torch.long),
            "event": torch.tensor(self.split.events[index], dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
        }


def make_loader(
    split: StaticFaithfulSplit,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        StaticFaithfulDataset(split),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=False,
    )


def validate_static_splits(
    train: StaticFaithfulSplit,
    validation: StaticFaithfulSplit,
    test: StaticFaithfulSplit | None,
    dataset_dir: str | Path,
) -> dict:
    ids = {
        "train": set(map(str, train.patient_ids)),
        "validation": set(map(str, validation.patient_ids)),
        "test": set(map(str, test.patient_ids)) if test is not None else set(),
    }
    metadata_path = Path(dataset_dir) / "preprocessing_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    checks = {
        "dataset_name": metadata.get("dataset"),
        "same_faithful_dataset_files": True,
        "train_validation_no_overlap": not bool(ids["train"] & ids["validation"]),
        "train_test_no_overlap": not bool(ids["train"] & ids["test"]),
        "validation_test_no_overlap": not bool(ids["validation"] & ids["test"]),
        "test_loaded": test is not None,
        "input_mode": "static_only",
        "temporal_input_loaded": False,
        "mask_input_loaded": False,
        "target_not_in_input": True,
        "static_preprocessing": metadata.get("static_preprocessing"),
        "preprocessing_fit_split": metadata.get("imputation_fit_split"),
        "static_feature_count": int(train.x_static.shape[1]),
    }
    split_list = [train, validation] + ([test] if test is not None else [])
    for split in split_list:
        if split.x_static.ndim != 2 or not np.isfinite(split.x_static).all():
            raise ValueError(f"{split.name} must contain finite X_static [N, F]")
        if split.x_static.shape[1] != train.x_static.shape[1]:
            raise ValueError("Static feature dimension differs across splits")
        if len(split.patient_ids) != len(split.durations) or len(split.patient_ids) != len(split.events):
            raise ValueError(f"{split.name} patient/target alignment mismatch")
    overlap_checks = [
        checks["train_validation_no_overlap"],
        checks["train_test_no_overlap"],
        checks["validation_test_no_overlap"],
    ]
    if not all(overlap_checks):
        raise ValueError(f"Split overlap detected: {checks}")
    if not str(metadata.get("dataset", "")).startswith("dysurv_faithful_"):
        raise ValueError("Static-only pipeline must reuse a dysurv_faithful landmark dataset")
    return checks


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, loader, device, config, epoch: int, optimizer=None, collect_survival: bool = False):
    training = optimizer is not None
    model.train(training)
    params = config["params"]
    w_surv = float(params["w_surv"])
    w_recon = float(params["w_recon"])
    w_kl = effective_kl_weight(float(params["w_kl"]), epoch, int(params.get("kl_warmup_epochs", 0)))
    totals = {"total_loss": 0.0, "survival_loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
    latent_mu, latent_logvar, all_logits, all_risk10, all_survival, all_indices = [], [], [], [], [], []
    n = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = _batch_to_device(batch, device)
            if training:
                optimizer.zero_grad()
            output = model(batch["x_static"].float())
            loss_surv = logistic_hazard_nll(output["logits"], batch["t_idx"], batch["event"])
            loss_recon = torch.mean((output["reconstruction"] - batch["x_static"].float()) ** 2)
            loss_kl = kl_divergence(output["mu"], output["logvar"])
            loss = w_surv * loss_surv + w_recon * loss_recon + w_kl * loss_kl
            if training:
                loss.backward()
                optimizer.step()

            deterministic_logits = model.survival_head(output["mu"])
            survival = hazards_to_survival(deterministic_logits)
            risk10 = 1.0 - survival[:, -1]
            batch_size = batch["x_static"].shape[0]
            n += batch_size
            for name, value in [
                ("total_loss", loss),
                ("survival_loss", loss_surv),
                ("reconstruction_loss", loss_recon),
                ("kl_loss", loss_kl),
            ]:
                totals[name] += float(value.detach().cpu()) * batch_size
            latent_mu.append(output["mu"].detach().cpu().numpy())
            latent_logvar.append(output["logvar"].detach().cpu().numpy())
            all_logits.append(deterministic_logits.detach().cpu().numpy())
            all_risk10.append(risk10.detach().cpu().numpy())
            if collect_survival:
                all_survival.append(survival.detach().cpu().numpy())
                all_indices.append(batch["row_index"].detach().cpu().numpy())

    diagnostics = {name: value / max(n, 1) for name, value in totals.items()}
    diagnostics["effective_w_kl"] = w_kl
    diagnostics.update(
        _distribution_stats(
            latent_mu,
            latent_logvar,
            all_logits,
            all_risk10,
            config["collapse"].get("active_unit_variance_threshold", 0.01),
        )
    )
    if not collect_survival:
        return diagnostics, None
    indices = np.concatenate(all_indices)
    survival = np.concatenate(all_survival, axis=0)
    return diagnostics, survival[np.argsort(indices)]


def compute_survival_metrics(survival: np.ndarray, split: StaticFaithfulSplit, config: dict) -> dict:
    surv_df = survival_df_from_array(survival)
    grid = metric_integration_grid(
        surv_df,
        split.durations,
        config["experiment"]["max_horizon_days"],
        config["evaluation"].get("metric_integration_num_points", 100),
    )
    metrics = eval_surv_metrics(surv_df, split.durations, split.events, grid)
    rows = horizon_c_index_rows(
        "dysurv_static_faithful_72h",
        split.name,
        surv_df,
        split.durations,
        split.events,
        config["evaluation"]["horizon_times"],
    )
    metrics["mean_horizon_c_index"] = mean_horizon_c_index(rows)
    metrics["horizon_c_index"] = {str(int(row["horizon_day"])): row["c_index"] for row in rows}
    return metrics


def _objective_key(metrics: dict) -> tuple[float, float]:
    ctd = float(metrics.get("ctd_antolini", math.nan))
    ibll = float(metrics.get("ibll", math.nan))
    return (
        ctd if math.isfinite(ctd) else -math.inf,
        -ibll if math.isfinite(ibll) else -math.inf,
    )


def _objective_better(metrics: dict, best: tuple[float, float] | None) -> bool:
    key = _objective_key(metrics)
    return math.isfinite(key[0]) and (best is None or key > best)


def _save_predictions(output_dir: Path, split: StaticFaithfulSplit, survival: np.ndarray) -> None:
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "patient_id": split.patient_ids.astype(str),
        "duration_eval_days": split.durations,
        "event_eval": split.events,
        "risk10": 1.0 - survival[:, -1],
    }
    for index in range(survival.shape[1]):
        payload[f"survival_day_{index + 1}"] = survival[:, index]
    pd.DataFrame(payload).to_parquet(predictions_dir / f"{split.name}_survival_predictions.parquet", index=False)


def train_dysurv_static_faithful(config: dict, logger) -> dict:
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    include_test = bool(config.get("include_test", False))
    if config.get("phase") == "tuning" and include_test:
        raise ValueError("Test data cannot be loaded during tuning")
    dataset_dir = config["paths"]["prepared_dataset_dir"]
    sample_size = config.get("sample_size")
    split_files = _faithful_split_files(config)
    train = load_static_faithful_split(dataset_dir, "train", sample_size, split_files)
    validation = load_static_faithful_split(dataset_dir, "validation", sample_size, split_files)
    test = load_static_faithful_split(dataset_dir, "test", sample_size, split_files) if include_test else None
    checks = validate_static_splits(train, validation, test, dataset_dir)
    params = config["params"]
    fixed = config["model_fixed"]
    model = DySurvStaticFaithful72h(
        input_dim=train.x_static.shape[1],
        latent_dim=int(fixed["latent_dim"]),
        encoder_multiplier=fixed["encoder_multiplier"],
        decoder_multiplier=fixed["decoder_multiplier"],
        survival_multiplier=fixed["survival_multiplier"],
        decoder_activation=fixed.get("decoder_activation", "none"),
        dropout=float(params["dropout"]),
        num_durations=int(fixed["num_durations"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    num_workers = config["data"].get("num_workers", 0)
    train_loader = make_loader(train, params["batch_size"], True, num_workers)
    val_loader = make_loader(validation, params["batch_size"], False, num_workers)
    output_dir = Path(config["paths"]["output_dir"])
    for folder in ["checkpoints", "metrics", "audit"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    save_yaml(output_dir / "config_snapshot.yaml", config)
    save_yaml(output_dir / "config_used.yaml", config)
    save_json(output_dir / "audit" / "data_and_leakage_checks.json", checks)

    history = []
    metric_best = {"state": None, "objective": None, "epoch": None, "metrics": None, "diagnostics": None}
    noncollapsed_best = {"state": None, "objective": None, "epoch": None, "metrics": None, "diagnostics": None}
    best_val_loss = math.inf
    patience_count = 0
    low_kl_epochs = 0
    epochs = int(params["epochs"])
    for epoch in range(1, epochs + 1):
        train_diag, _ = run_epoch(model, train_loader, device, config, epoch, optimizer=optimizer)
        val_diag, val_survival = run_epoch(model, val_loader, device, config, epoch, collect_survival=True)
        val_metrics = compute_survival_metrics(val_survival, validation, config)
        low_kl_epochs = low_kl_epochs + 1 if val_diag["kl_loss"] < float(config["collapse"]["kl_threshold"]) else 0
        flags = collapse_flags(val_diag, config, low_kl_epochs)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_diag.items()})
        row.update({f"validation_{key}": value for key, value in val_diag.items()})
        row.update({f"validation_{key}": value for key, value in val_metrics.items() if key != "horizon_c_index"})
        row.update({f"validation_c_index_day_{day}": value for day, value in val_metrics["horizon_c_index"].items()})
        row.update(flags)
        history.append(row)

        if _objective_better(val_metrics, metric_best["objective"]):
            metric_best = {
                "state": copy.deepcopy(model.state_dict()),
                "objective": _objective_key(val_metrics),
                "epoch": epoch,
                "metrics": copy.deepcopy(val_metrics),
                "diagnostics": {**val_diag, **flags},
            }
        if not flags["collapse_suspected"] and _objective_better(val_metrics, noncollapsed_best["objective"]):
            noncollapsed_best = {
                "state": copy.deepcopy(model.state_dict()),
                "objective": _objective_key(val_metrics),
                "epoch": epoch,
                "metrics": copy.deepcopy(val_metrics),
                "diagnostics": {**val_diag, **flags},
            }
        if val_diag["total_loss"] < best_val_loss - 1e-7:
            best_val_loss = val_diag["total_loss"]
            patience_count = 0
        else:
            patience_count += 1
        logger.info(
            "DySurv static faithful epoch %d/%d val_loss=%.5f val_ctd=%.5f val_ibs=%.5f val_ibll=%.5f "
            "risk10_std=%.6f active_units=%d kl=%.6f collapse=%s",
            epoch,
            epochs,
            val_diag["total_loss"],
            val_metrics["ctd_antolini"],
            val_metrics["ibs"],
            val_metrics["ibll"],
            val_diag["std_risk10"],
            val_diag["active_units"],
            val_diag["kl_loss"],
            flags["collapse_suspected"],
        )
        if patience_count >= int(params["patience"]):
            break

    if metric_best["state"] is None:
        raise RuntimeError("No finite validation Ctd was produced")
    selected_noncollapsed = noncollapsed_best["state"] is not None
    selected = noncollapsed_best if selected_noncollapsed else metric_best
    last_state = copy.deepcopy(model.state_dict())
    last_epoch = int(history[-1]["epoch"])
    model.load_state_dict(selected["state"])
    pd.DataFrame(history).to_csv(output_dir / "metrics" / "epoch_metrics.csv", index=False)
    torch.save(
        {"model_state_dict": selected["state"], "best_epoch": selected["epoch"], "config": config},
        output_dir / "checkpoints" / "best_model.pt",
    )
    torch.save(
        {"model_state_dict": last_state, "last_epoch": last_epoch, "config": config},
        output_dir / "checkpoints" / "last_model.pt",
    )

    split_metrics = {}
    for split in [validation] + ([test] if test is not None else []):
        loader = make_loader(split, params["batch_size"], False, num_workers)
        diagnostics, survival = run_epoch(model, loader, device, config, int(selected["epoch"]), collect_survival=True)
        metrics = compute_survival_metrics(survival, split, config)
        split_metrics[split.name] = {**metrics, "diagnostics": diagnostics}
        _save_predictions(output_dir, split, survival)
        _save_curve_examples(output_dir, split, survival, config["evaluation"].get("n_example_patients_per_group", 3))

    result = {
        "model": "dysurv_static_faithful_72h",
        "seed": int(config["seed"]),
        "best_epoch": int(selected["epoch"]),
        "metric_best_epoch": int(metric_best["epoch"]),
        "selection_used_noncollapsed_epoch": bool(selected_noncollapsed),
        "metric_best_validation": metric_best["metrics"],
        "metric_best_collapse": metric_best["diagnostics"],
        "best_validation_loss": float(best_val_loss),
        "splits": split_metrics,
        "collapse": selected["diagnostics"],
        "test_used_in_tuning": bool(config.get("phase") == "tuning" and include_test),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "metrics" / "metrics.json", result)
    save_json(output_dir / "audit" / "collapse_summary.json", selected["diagnostics"])
    save_json(
        output_dir / "audit" / "epoch_selection.json",
        {
            "selected_epoch": int(selected["epoch"]),
            "selected_noncollapsed_epoch": bool(selected_noncollapsed),
            "selected_validation_metrics": selected["metrics"],
            "selected_collapse": selected["diagnostics"],
            "metric_best_epoch": int(metric_best["epoch"]),
            "metric_best_validation_metrics": metric_best["metrics"],
            "metric_best_collapse": metric_best["diagnostics"],
        },
    )
    return result
