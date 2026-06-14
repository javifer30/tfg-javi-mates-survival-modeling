"""Training and evaluation for the isolated DySurv-faithful 72h pipeline."""

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

from src.evaluation.static_72h_metrics import (
    eval_surv_metrics,
    horizon_c_index_rows,
    mean_horizon_c_index,
    metric_integration_grid,
)
from src.models.dynamic_72h.common import resolve_device, save_json, save_yaml, set_seed
from src.models.dynamic_72h.discretization import discretize_duration_event
from src.models.dynamic_72h.dysurv_faithful import DySurvFaithful72h, kl_divergence
from src.models.dynamic_72h.losses import hazards_to_survival, logistic_hazard_nll
from src.models.dynamic_72h.predict import survival_df_from_array


@dataclass
class FaithfulSplit:
    name: str
    patient_ids: np.ndarray
    x_seq: np.ndarray
    m_seq: np.ndarray
    x_static: np.ndarray
    durations: np.ndarray
    events: np.ndarray
    t_idx: np.ndarray


def load_faithful_split(dataset_dir: str | Path, split: str, sample_size: int | None = None) -> FaithfulSplit:
    file_split = "val" if split == "validation" else split
    with np.load(Path(dataset_dir) / f"{file_split}_dynamic_72h.npz") as data:
        arrays = {key: data[key] for key in data.files}
    n = len(arrays["patient_ids"])
    if sample_size is not None:
        n = min(n, int(sample_size))
        arrays = {key: value[:n] for key, value in arrays.items()}
    t_idx, events = discretize_duration_event(arrays["duration_eval_days"], arrays["event_eval"])
    return FaithfulSplit(
        name=split,
        patient_ids=arrays["patient_ids"],
        x_seq=arrays["X_seq"].astype("float32", copy=False),
        m_seq=arrays["M_seq"].astype("float32", copy=False),
        x_static=arrays["X_static"].astype("float32", copy=False),
        durations=arrays["duration_eval_days"].astype("float32", copy=False),
        events=events,
        t_idx=t_idx,
    )


def build_faithful_input(x_seq: np.ndarray, x_static: np.ndarray, input_mode: str) -> np.ndarray:
    if input_mode == "temporal_only":
        return x_seq.astype("float32", copy=False)
    if input_mode == "temporal_plus_static_repeated":
        repeated = np.broadcast_to(x_static[:, None, :], (x_seq.shape[0], x_seq.shape[1], x_static.shape[1]))
        return np.concatenate([x_seq, repeated], axis=2).astype("float32")
    raise ValueError(f"Unsupported faithful input_mode: {input_mode}")


class FaithfulDataset(Dataset):
    def __init__(self, split: FaithfulSplit, input_mode: str):
        self.split = split
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.split.patient_ids)

    def __getitem__(self, index: int) -> dict:
        x_temporal = self.split.x_seq[index]
        if self.input_mode == "temporal_only":
            x_input = x_temporal
        elif self.input_mode == "temporal_plus_static_repeated":
            repeated_static = np.broadcast_to(self.split.x_static[index], (x_temporal.shape[0], self.split.x_static.shape[1]))
            x_input = np.concatenate([x_temporal, repeated_static], axis=1).astype("float32")
        else:
            raise ValueError(f"Unsupported faithful input_mode: {self.input_mode}")
        return {
            "x_input": torch.from_numpy(x_input),
            "x_temporal": torch.from_numpy(x_temporal),
            "m_temporal": torch.from_numpy(self.split.m_seq[index]),
            "t_idx": torch.tensor(self.split.t_idx[index], dtype=torch.long),
            "event": torch.tensor(self.split.events[index], dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
        }


def make_loader(split: FaithfulSplit, input_mode: str, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        FaithfulDataset(split, input_mode),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=False,
    )


def validate_faithful_splits(train: FaithfulSplit, validation: FaithfulSplit, test: FaithfulSplit | None) -> dict:
    train_ids = set(map(str, train.patient_ids))
    val_ids = set(map(str, validation.patient_ids))
    test_ids = set(map(str, test.patient_ids)) if test is not None else set()
    checks = {
        "train_validation_no_overlap": not bool(train_ids & val_ids),
        "train_test_no_overlap": not bool(train_ids & test_ids),
        "validation_test_no_overlap": not bool(val_ids & test_ids),
        "test_loaded": test is not None,
        "input_hours": int(train.x_seq.shape[1]),
        "target_not_in_input": True,
        "mask_not_in_input": True,
    }
    if train.x_seq.shape[1] != 72 or validation.x_seq.shape[1] != 72:
        raise ValueError("Faithful inputs must contain exactly the first 72 hourly steps")
    if test is not None and test.x_seq.shape[1] != 72:
        raise ValueError("Faithful test input must contain exactly 72 hourly steps")
    if not checks["train_validation_no_overlap"] or not checks["train_test_no_overlap"] or not checks["validation_test_no_overlap"]:
        raise ValueError(f"Split overlap detected: {checks}")
    return checks


def effective_kl_weight(base_weight: float, epoch: int, warmup_epochs: int) -> float:
    if int(warmup_epochs) <= 0:
        return float(base_weight)
    return float(base_weight) * min(1.0, float(epoch) / float(warmup_epochs))


def reconstruction_loss(prediction, target, mask, mode: str, observed_weight: float) -> torch.Tensor:
    squared = (prediction - target) ** 2
    if mode == "full_imputed_mse":
        return squared.mean()
    if mode == "observed_weighted":
        weights = 1.0 + (float(observed_weight) - 1.0) * mask.float()
        return (squared * weights).sum() / weights.sum().clamp_min(1.0)
    raise ValueError(f"Unsupported reconstruction_weighting: {mode}")


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _distribution_stats(mu, logvar, logits, risk10) -> dict:
    mu = np.concatenate(mu, axis=0)
    logvar = np.concatenate(logvar, axis=0)
    logits = np.concatenate(logits, axis=0)
    risk10 = np.concatenate(risk10, axis=0)
    return {
        "std_mu": float(np.std(mu)),
        "mean_abs_mu": float(np.mean(np.abs(mu))),
        "std_logvar": float(np.std(logvar)),
        "mean_logvar": float(np.mean(logvar)),
        "std_logits": float(np.std(logits)),
        "std_risk10": float(np.std(risk10)),
        "min_risk10": float(np.min(risk10)),
        "max_risk10": float(np.max(risk10)),
        "range_risk10": float(np.ptp(risk10)),
        "number_unique_risk10_rounded_6": int(np.unique(np.round(risk10, 6)).size),
    }


def run_epoch(model, loader, device, config, epoch: int, optimizer=None, collect_survival: bool = False):
    training = optimizer is not None
    model.train(training)
    params = config["params"]
    data_cfg = config["data"]
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
            output = model(batch["x_input"].float())
            loss_surv = logistic_hazard_nll(output["logits"], batch["t_idx"], batch["event"])
            loss_recon = reconstruction_loss(
                output["reconstruction"],
                batch["x_temporal"].float(),
                batch["m_temporal"],
                data_cfg["reconstruction_weighting"],
                data_cfg.get("observed_reconstruction_weight", 2.0),
            )
            loss_kl = kl_divergence(output["mu"], output["logvar"])
            loss = w_surv * loss_surv + w_recon * loss_recon + w_kl * loss_kl
            if training:
                loss.backward()
                optimizer.step()

            deterministic_logits = model.survival_head(output["mu"])
            survival = hazards_to_survival(deterministic_logits)
            risk10 = 1.0 - survival[:, -1]
            batch_size = batch["x_input"].shape[0]
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

    result = {name: value / max(n, 1) for name, value in totals.items()}
    result["effective_w_kl"] = w_kl
    result.update(_distribution_stats(latent_mu, latent_logvar, all_logits, all_risk10))
    if not collect_survival:
        return result, None
    indices = np.concatenate(all_indices)
    survival = np.concatenate(all_survival, axis=0)
    return result, survival[np.argsort(indices)]


def compute_survival_metrics(survival: np.ndarray, split: FaithfulSplit, config: dict) -> dict:
    surv_df = survival_df_from_array(survival)
    grid = metric_integration_grid(
        surv_df,
        split.durations,
        config["experiment"]["max_horizon_days"],
        config["evaluation"].get("metric_integration_num_points", 100),
    )
    metrics = eval_surv_metrics(surv_df, split.durations, split.events, grid)
    rows = horizon_c_index_rows(
        "dysurv_faithful_72h",
        split.name,
        surv_df,
        split.durations,
        split.events,
        config["evaluation"]["horizon_times"],
    )
    metrics["mean_horizon_c_index"] = mean_horizon_c_index(rows)
    metrics["horizon_c_index"] = {str(int(row["horizon_day"])): row["c_index"] for row in rows}
    return metrics


def collapse_flags(diagnostics: dict, config: dict, low_kl_epochs: int) -> dict:
    cfg = config["collapse"]
    flags = {
        "low_std_risk10": diagnostics["std_risk10"] < float(cfg["std_risk10_threshold"]),
        "low_range_risk10": diagnostics["range_risk10"] < float(cfg["range_risk10_threshold"]),
        "low_std_mu": diagnostics["std_mu"] < float(cfg["std_mu_threshold"]),
        "low_unique_risk10": diagnostics["number_unique_risk10_rounded_6"] < int(cfg["min_unique_risk10_rounded_6"]),
        "persistent_low_kl": low_kl_epochs >= int(cfg["kl_consecutive_epochs"]),
    }
    flags["collapse_suspected"] = bool(
        flags["low_std_risk10"]
        or flags["low_range_risk10"]
        or flags["low_unique_risk10"]
        or (flags["low_std_mu"] and flags["persistent_low_kl"])
    )
    return flags


def _objective_better(metrics: dict, best: tuple[float, float] | None) -> bool:
    ctd = float(metrics.get("ctd_antolini", math.nan))
    ibll = float(metrics.get("ibll", math.nan))
    if math.isnan(ctd):
        return False
    key = (ctd, -ibll if not math.isnan(ibll) else -math.inf)
    return best is None or key > best


def _save_predictions(output_dir: Path, split: FaithfulSplit, survival: np.ndarray) -> None:
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


def _save_curve_examples(output_dir: Path, split: FaithfulSplit, survival: np.ndarray, per_group: int) -> None:
    n = min(int(per_group), len(split.patient_ids))
    if n <= 0:
        return
    risk = 1.0 - survival[:, -1]
    low = np.argsort(risk)[:n]
    high = np.argsort(risk)[-n:]
    early_event_candidates = np.where(split.events == 1)[0]
    early = early_event_candidates[np.argsort(split.durations[early_event_candidates])[:n]] if early_event_candidates.size else np.asarray([], dtype=int)
    long_censored_candidates = np.where(split.events == 0)[0]
    long_censored = long_censored_candidates[np.argsort(-split.durations[long_censored_candidates])[:n]] if long_censored_candidates.size else np.asarray([], dtype=int)
    groups = [("low_risk", low), ("high_risk", high), ("early_event", early), ("long_censored", long_censored)]
    rows = []
    for group, indices in groups:
        for index in indices:
            row = {
                "group": group,
                "patient_id": str(split.patient_ids[index]),
                "duration_eval_days": float(split.durations[index]),
                "event_eval": int(split.events[index]),
                "risk10": float(risk[index]),
            }
            row.update({f"survival_day_{day}": float(survival[index, day - 1]) for day in range(1, survival.shape[1] + 1)})
            rows.append(row)
    pd.DataFrame(rows).drop_duplicates(["group", "patient_id"]).to_csv(output_dir / "predictions" / f"{split.name}_curve_examples.csv", index=False)


def train_dysurv_faithful(config: dict, logger) -> dict:
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    include_test = bool(config.get("include_test", False))
    if config.get("phase") == "tuning" and include_test:
        raise ValueError("Test data cannot be loaded during tuning")
    dataset_dir = config["paths"]["prepared_dataset_dir"]
    sample_size = config.get("sample_size")
    train = load_faithful_split(dataset_dir, "train", sample_size)
    validation = load_faithful_split(dataset_dir, "validation", sample_size)
    test = load_faithful_split(dataset_dir, "test", sample_size) if include_test else None
    checks = validate_faithful_splits(train, validation, test)
    input_mode = config["data"]["input_mode"]
    input_dim = train.x_seq.shape[2] + (train.x_static.shape[1] if input_mode == "temporal_plus_static_repeated" else 0)
    reconstruction_dim = train.x_seq.shape[2]
    params = config["params"]
    fixed = config["model_fixed"]

    model = DySurvFaithful72h(
        input_dim=input_dim,
        reconstruction_dim=reconstruction_dim,
        seq_len=train.x_seq.shape[1],
        rnn_hidden_dim=fixed.get("rnn_hidden_dim"),
        latent_dim=int(fixed["latent_dim"]),
        encoder_mlp=fixed["encoder_mlp"],
        survival_mlp=fixed["survival_mlp"],
        dropout=float(params["dropout"]),
        num_durations=int(fixed["num_durations"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    train_loader = make_loader(train, input_mode, params["batch_size"], True, config["data"].get("num_workers", 0))
    val_loader = make_loader(validation, input_mode, params["batch_size"], False, config["data"].get("num_workers", 0))
    output_dir = Path(config["paths"]["output_dir"])
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "audit").mkdir(parents=True, exist_ok=True)
    save_yaml(output_dir / "config_snapshot.yaml", config)
    save_json(output_dir / "audit" / "data_and_leakage_checks.json", checks)

    history = []
    metric_best_state = None
    metric_best_objective = None
    metric_best_epoch = None
    metric_best_metrics = None
    metric_best_diagnostics = None
    noncollapsed_best_state = None
    noncollapsed_best_objective = None
    noncollapsed_best_epoch = None
    noncollapsed_best_metrics = None
    noncollapsed_best_diagnostics = None
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

        if _objective_better(val_metrics, metric_best_objective):
            ctd = float(val_metrics["ctd_antolini"])
            ibll = float(val_metrics.get("ibll", math.nan))
            metric_best_objective = (ctd, -ibll if not math.isnan(ibll) else -math.inf)
            metric_best_state = copy.deepcopy(model.state_dict())
            metric_best_epoch = epoch
            metric_best_metrics = copy.deepcopy(val_metrics)
            metric_best_diagnostics = {**val_diag, **flags}
        if not flags["collapse_suspected"] and _objective_better(val_metrics, noncollapsed_best_objective):
            ctd = float(val_metrics["ctd_antolini"])
            ibll = float(val_metrics.get("ibll", math.nan))
            noncollapsed_best_objective = (ctd, -ibll if not math.isnan(ibll) else -math.inf)
            noncollapsed_best_state = copy.deepcopy(model.state_dict())
            noncollapsed_best_epoch = epoch
            noncollapsed_best_metrics = copy.deepcopy(val_metrics)
            noncollapsed_best_diagnostics = {**val_diag, **flags}

        if val_diag["total_loss"] < best_val_loss - 1e-7:
            best_val_loss = val_diag["total_loss"]
            patience_count = 0
        else:
            patience_count += 1
        logger.info(
            "DySurv faithful epoch %d/%d val_loss=%.5f val_ctd=%.5f risk10_std=%.6f collapse=%s",
            epoch,
            epochs,
            val_diag["total_loss"],
            val_metrics["ctd_antolini"],
            val_diag["std_risk10"],
            flags["collapse_suspected"],
        )
        if patience_count >= int(params["patience"]):
            break

    if metric_best_state is None:
        raise RuntimeError("No finite validation Ctd was produced")
    selection_used_noncollapsed = noncollapsed_best_state is not None
    best_state = noncollapsed_best_state if selection_used_noncollapsed else metric_best_state
    best_epoch = noncollapsed_best_epoch if selection_used_noncollapsed else metric_best_epoch
    best_metrics = noncollapsed_best_metrics if selection_used_noncollapsed else metric_best_metrics
    best_diagnostics = noncollapsed_best_diagnostics if selection_used_noncollapsed else metric_best_diagnostics
    last_state = copy.deepcopy(model.state_dict())
    last_epoch = int(history[-1]["epoch"])
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(output_dir / "metrics" / "epoch_metrics.csv", index=False)
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch, "config": config}, output_dir / "checkpoints" / "best_model.pt")
    torch.save({"model_state_dict": last_state, "last_epoch": last_epoch, "config": config}, output_dir / "checkpoints" / "last_model.pt")

    split_metrics = {}
    prediction_splits = [validation] + ([test] if test is not None else [])
    for split in prediction_splits:
        loader = make_loader(split, input_mode, params["batch_size"], False, config["data"].get("num_workers", 0))
        diagnostics, survival = run_epoch(model, loader, device, config, best_epoch, collect_survival=True)
        metrics = compute_survival_metrics(survival, split, config)
        split_metrics[split.name] = {**metrics, "diagnostics": diagnostics}
        _save_predictions(output_dir, split, survival)
        _save_curve_examples(output_dir, split, survival, config["evaluation"].get("n_example_patients_per_group", 3))

    result = {
        "model": "dysurv_faithful_72h",
        "seed": int(config["seed"]),
        "best_epoch": int(best_epoch),
        "metric_best_epoch": int(metric_best_epoch),
        "selection_used_noncollapsed_epoch": bool(selection_used_noncollapsed),
        "metric_best_validation": metric_best_metrics,
        "metric_best_collapse": metric_best_diagnostics,
        "best_validation_loss": float(best_val_loss),
        "splits": split_metrics,
        "collapse": best_diagnostics,
        "test_used_in_tuning": bool(config.get("phase") == "tuning" and include_test),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "metrics" / "metrics.json", result)
    save_json(output_dir / "audit" / "collapse_summary.json", best_diagnostics)
    save_json(
        output_dir / "audit" / "epoch_selection.json",
        {
            "selected_epoch": int(best_epoch),
            "selected_noncollapsed_epoch": bool(selection_used_noncollapsed),
            "selected_validation_metrics": best_metrics,
            "selected_collapse": best_diagnostics,
            "metric_best_epoch": int(metric_best_epoch),
            "metric_best_validation_metrics": metric_best_metrics,
            "metric_best_collapse": metric_best_diagnostics,
        },
    )
    return result
