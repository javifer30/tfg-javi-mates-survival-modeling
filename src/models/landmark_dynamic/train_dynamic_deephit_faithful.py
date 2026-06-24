"""Training and evaluation for isolated Dynamic-DeepHit faithful landmark runs."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.landmark_survival_metrics import (
    eval_surv_metrics,
    horizon_c_index_rows,
    mean_horizon_c_index,
    metric_integration_grid,
)
from src.models.landmark_dynamic.common import resolve_device, save_json, save_yaml, set_seed
from src.models.landmark_dynamic.dynamic_deephit_faithful import DynamicDeepHitFaithful72h
from src.models.landmark_dynamic.losses import deephit_ranking_loss, pmf_nll
from src.models.landmark_dynamic.predict import survival_df_from_array
from src.models.landmark_dynamic.train_dysurv_faithful import (
    _faithful_split_files,
    FaithfulSplit,
    _save_curve_examples,
    load_faithful_split,
    make_loader,
    validate_faithful_splits,
)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _loss_weights(params: dict) -> tuple[float, float, float]:
    alpha = float(params["alpha_ranking"])
    beta = float(params["beta_nll"])
    if alpha < 0 or beta < 0 or alpha + beta > 1.0:
        raise ValueError(f"Dynamic-DeepHit requires alpha_ranking + beta_nll <= 1, got {alpha + beta}")
    return 1.0 - alpha - beta, alpha, beta


def _prediction_stats(pmf_full: np.ndarray, attention: np.ndarray, num_durations: int) -> dict:
    pmf = pmf_full[:, :num_durations]
    survival = 1.0 - np.cumsum(pmf, axis=1)
    risk10 = 1.0 - survival[:, -1]
    pmf_sums = pmf_full.sum(axis=1)
    tail = pmf_full[:, num_durations:].sum(axis=1) if pmf_full.shape[1] > num_durations else np.zeros(len(pmf_full))
    attention_safe = np.clip(attention, 1e-12, 1.0)
    attention_entropy = -np.sum(attention_safe * np.log(attention_safe), axis=1)
    return {
        "mean_risk10": float(np.mean(risk10)),
        "std_risk10": float(np.std(risk10)),
        "min_risk10": float(np.min(risk10)),
        "max_risk10": float(np.max(risk10)),
        "range_risk10": float(np.ptp(risk10)),
        "number_unique_risk10_rounded_6": int(np.unique(np.round(risk10, 6)).size),
        "pmf_sum_min": float(np.min(pmf_sums)),
        "pmf_sum_max": float(np.max(pmf_sums)),
        "pmf_sum_mean": float(np.mean(pmf_sums)),
        "mean_tail_probability": float(np.mean(tail)),
        "min_tail_probability": float(np.min(tail)),
        "max_tail_probability": float(np.max(tail)),
        "mean_attention_entropy": float(np.mean(attention_entropy)),
        "std_attention_entropy": float(np.std(attention_entropy)),
    }


def run_epoch(model, loader, device, config, optimizer=None, collect_predictions=False):
    training = optimizer is not None
    model.train(training)
    params = config["params"]
    fixed = config["model_fixed"]
    num_durations = int(fixed["num_durations"])
    w_long, w_rank, w_nll = _loss_weights(params)
    totals = {"total_loss": 0.0, "longitudinal_loss": 0.0, "ranking_loss": 0.0, "nll_loss": 0.0}
    all_pmf, all_attention, all_indices = [], [], []
    n = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = _batch_to_device(batch, device)
            if training:
                optimizer.zero_grad()
            output = model(batch["x_input"].float())
            pmf_eval = output["pmf"][:, :num_durations]
            loss_long = F.mse_loss(
                output["longitudinal_prediction"][:, :-1, :],
                batch["x_temporal"][:, 1:, :].float(),
            )
            loss_rank = deephit_ranking_loss(pmf_eval, batch["t_idx"], batch["event"], params["sigma"])
            loss_nll = pmf_nll(output["pmf"], batch["t_idx"], batch["event"])
            loss = w_long * loss_long + w_rank * loss_rank + w_nll * loss_nll
            if training:
                loss.backward()
                optimizer.step()
            batch_size = batch["x_input"].shape[0]
            n += batch_size
            for name, value in [
                ("total_loss", loss),
                ("longitudinal_loss", loss_long),
                ("ranking_loss", loss_rank),
                ("nll_loss", loss_nll),
            ]:
                totals[name] += float(value.detach().cpu()) * batch_size
            all_pmf.append(output["pmf"].detach().cpu().numpy())
            all_attention.append(output["attention"].detach().cpu().numpy())
            all_indices.append(batch["row_index"].detach().cpu().numpy())

    indices = np.concatenate(all_indices)
    order = np.argsort(indices)
    pmf_full = np.concatenate(all_pmf, axis=0)[order]
    attention = np.concatenate(all_attention, axis=0)[order]
    diagnostics = {name: value / max(n, 1) for name, value in totals.items()}
    diagnostics.update({"w_longitudinal": w_long, "w_ranking": w_rank, "w_nll": w_nll})
    diagnostics.update(_prediction_stats(pmf_full, attention, num_durations))
    if not collect_predictions:
        return diagnostics, None
    survival = 1.0 - np.cumsum(pmf_full[:, :num_durations], axis=1)
    return diagnostics, {"survival": np.clip(survival, 0.0, 1.0), "pmf": pmf_full, "attention": attention}


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
        "dynamic_deephit_faithful_72h",
        split.name,
        surv_df,
        split.durations,
        split.events,
        config["evaluation"]["horizon_times"],
    )
    metrics["mean_horizon_c_index"] = mean_horizon_c_index(rows)
    metrics["horizon_c_index"] = {str(int(row["horizon_day"])): row["c_index"] for row in rows}
    return metrics


def collapse_flags(diagnostics: dict, config: dict) -> dict:
    cfg = config["collapse"]
    tolerance = float(cfg["pmf_sum_tolerance"])
    flags = {
        "low_std_risk10": diagnostics["std_risk10"] < float(cfg["std_risk10_threshold"]),
        "low_range_risk10": diagnostics["range_risk10"] < float(cfg["range_risk10_threshold"]),
        "low_unique_risk10": diagnostics["number_unique_risk10_rounded_6"] < int(cfg["min_unique_risk10_rounded_6"]),
        "invalid_pmf_sum": diagnostics["pmf_sum_min"] < 1.0 - tolerance or diagnostics["pmf_sum_max"] > 1.0 + tolerance,
        "near_zero_tail_probability": diagnostics["mean_tail_probability"] < float(cfg["min_mean_tail_probability"]),
    }
    flags["collapse_suspected"] = bool(
        flags["low_std_risk10"]
        or flags["low_range_risk10"]
        or flags["low_unique_risk10"]
        or flags["invalid_pmf_sum"]
        or flags["near_zero_tail_probability"]
    )
    return flags


def _objective_better(metrics: dict, best: tuple[float, float] | None) -> bool:
    ctd = float(metrics.get("ctd_antolini", math.nan))
    ibll = float(metrics.get("ibll", math.nan))
    if math.isnan(ctd):
        return False
    key = (ctd, -ibll if not math.isnan(ibll) else -math.inf)
    return best is None or key > best


def _objective_key(metrics: dict) -> tuple[float, float]:
    ctd = float(metrics.get("ctd_antolini", math.nan))
    ibll = float(metrics.get("ibll", math.nan))
    return (
        ctd if math.isfinite(ctd) else -math.inf,
        -ibll if math.isfinite(ibll) else -math.inf,
    )


def _save_predictions(output_dir: Path, split: FaithfulSplit, predictions: dict, num_durations: int) -> None:
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    survival = predictions["survival"]
    pmf = predictions["pmf"]
    payload = {
        "patient_id": split.patient_ids.astype(str),
        "duration_eval_days": split.durations,
        "event_eval": split.events,
        "risk10": 1.0 - survival[:, -1],
        "tail_probability": pmf[:, num_durations:].sum(axis=1),
    }
    for index in range(num_durations):
        payload[f"survival_day_{index + 1}"] = survival[:, index]
        payload[f"pmf_day_{index + 1}"] = pmf[:, index]
    pd.DataFrame(payload).to_parquet(predictions_dir / f"{split.name}_survival_predictions.parquet", index=False)


def train_dynamic_deephit_faithful(config: dict, logger) -> dict:
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    include_test = bool(config.get("include_test", False))
    if config.get("phase") == "tuning" and include_test:
        raise ValueError("Test data cannot be loaded during tuning")
    dataset_dir = config["paths"]["prepared_dataset_dir"]
    sample_size = config.get("sample_size")
    split_files = _faithful_split_files(config)
    train = load_faithful_split(dataset_dir, "train", sample_size, split_files)
    validation = load_faithful_split(dataset_dir, "validation", sample_size, split_files)
    test = load_faithful_split(dataset_dir, "test", sample_size, split_files) if include_test else None
    checks = validate_faithful_splits(train, validation, test)
    checks["longitudinal_target_excludes_static"] = True
    input_mode = config["data"]["input_mode"]
    input_dim = train.x_seq.shape[2] + (train.x_static.shape[1] if input_mode == "temporal_plus_static_repeated" else 0)
    temporal_dim = train.x_seq.shape[2]
    params = config["params"]
    fixed = config["model_fixed"]
    num_durations = int(fixed["num_durations"])
    include_tail = bool(fixed.get("include_tail_category", True))
    output_dim = num_durations + (1 if include_tail else 0)
    network_params = lambda layers: {
        "layers": layers,
        "dropout": float(params["dropout"]),
        "activation": fixed.get("activation", "ReLU"),
    }
    model = DynamicDeepHitFaithful72h(
        input_dim=input_dim,
        temporal_dim=temporal_dim,
        output_dim=output_dim,
        layers_rnn=int(fixed["layers_rnn"]),
        hidden_rnn=int(fixed["hidden_rnn"]),
        typ=fixed.get("typ", "LSTM"),
        long_param=network_params(fixed["long_layers"]),
        att_param=network_params(fixed["attention_layers"]),
        cs_param=network_params(fixed["cause_specific_layers"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    train_loader = make_loader(train, input_mode, params["batch_size"], True, config["data"].get("num_workers", 0))
    val_loader = make_loader(validation, input_mode, params["batch_size"], False, config["data"].get("num_workers", 0))
    output_dir = Path(config["paths"]["output_dir"])
    for directory in [output_dir / "checkpoints", output_dir / "metrics", output_dir / "audit", output_dir / "predictions"]:
        directory.mkdir(parents=True, exist_ok=True)
    save_yaml(output_dir / "config_snapshot.yaml", config)
    save_yaml(output_dir / "config_used.yaml", config)
    save_json(output_dir / "audit" / "data_and_leakage_checks.json", checks)

    history = []
    metric_best = {"state": None, "objective": None, "epoch": None, "metrics": None, "diagnostics": None}
    individualized_best = {"state": None, "objective": None, "epoch": None, "metrics": None, "diagnostics": None}
    best_val_loss = math.inf
    patience_count = 0
    for epoch in range(1, int(params["epochs"]) + 1):
        train_diag, _ = run_epoch(model, train_loader, device, config, optimizer=optimizer)
        val_diag, val_predictions = run_epoch(model, val_loader, device, config, collect_predictions=True)
        val_metrics = compute_survival_metrics(val_predictions["survival"], validation, config)
        flags = collapse_flags(val_diag, config)
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
        if not flags["collapse_suspected"] and _objective_better(val_metrics, individualized_best["objective"]):
            individualized_best = {
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
            "Dynamic-DeepHit faithful epoch %d/%d val_loss=%.5f val_ctd=%.5f val_ibs=%.5f val_ibll=%.5f "
            "risk10_std=%.6f tail_mean=%.6f collapse=%s",
            epoch,
            int(params["epochs"]),
            val_diag["total_loss"],
            val_metrics["ctd_antolini"],
            val_metrics["ibs"],
            val_metrics["ibll"],
            val_diag["std_risk10"],
            val_diag["mean_tail_probability"],
            flags["collapse_suspected"],
        )
        if patience_count >= int(params["patience"]):
            break

    if metric_best["state"] is None:
        raise RuntimeError("No finite validation Ctd was produced")
    selected = individualized_best if individualized_best["state"] is not None else metric_best
    selected_noncollapsed = individualized_best["state"] is not None
    last_state = copy.deepcopy(model.state_dict())
    last_epoch = int(history[-1]["epoch"])
    model.load_state_dict(selected["state"])
    pd.DataFrame(history).to_csv(output_dir / "metrics" / "epoch_metrics.csv", index=False)
    torch.save({"model_state_dict": selected["state"], "best_epoch": selected["epoch"], "config": config}, output_dir / "checkpoints" / "best_model.pt")
    torch.save({"model_state_dict": last_state, "last_epoch": last_epoch, "config": config}, output_dir / "checkpoints" / "last_model.pt")

    split_metrics = {}
    for split in [validation] + ([test] if test is not None else []):
        loader = make_loader(split, input_mode, params["batch_size"], False, config["data"].get("num_workers", 0))
        diagnostics, predictions = run_epoch(model, loader, device, config, collect_predictions=True)
        metrics = compute_survival_metrics(predictions["survival"], split, config)
        split_metrics[split.name] = {**metrics, "diagnostics": diagnostics}
        _save_predictions(output_dir, split, predictions, num_durations)
        _save_curve_examples(output_dir, split, predictions["survival"], config["evaluation"].get("n_example_patients_per_group", 3))

    result = {
        "model": "dynamic_deephit_faithful_72h",
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
