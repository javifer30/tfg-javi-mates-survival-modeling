"""Training entrypoints for dynamic_72h models."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.dynamic_72h_metrics import evaluate_survival_predictions
from src.models.dynamic_72h.common import resolve_device, save_json, save_yaml, set_seed
from src.models.dynamic_72h.data import input_metadata, load_split, make_loader, validate_splits
from src.models.dynamic_72h.discretization import target_summary
from src.models.dynamic_72h.dynamic_deephit import DynamicDeepHit72h
from src.models.dynamic_72h.dysurv import DySurv72h, kl_loss
from src.models.dynamic_72h.losses import (
    deephit_ranking_loss,
    hazards_to_survival,
    logistic_hazard_nll,
    mse_observed,
    pmf_nll,
)
from src.models.dynamic_72h.predict import survival_df_from_array


def train_dynamic_72h_model(config: dict, logger):
    model_name = config["model"]["name"]
    if model_name == "dysurv":
        return _train_dysurv(config, logger)
    if model_name == "dynamic_deephit":
        return _train_dynamic_deephit(config, logger)
    raise ValueError(f"Unsupported dynamic_72h model: {model_name}")


def _prepare(config):
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    dataset_dir = config["paths"]["dataset_dir"]
    sample_size = config.get("sample_size")
    train = load_split(dataset_dir, "train", sample_size)
    val = load_split(dataset_dir, "val", sample_size)
    include_test = bool(config.get("include_test", False))
    test = load_split(dataset_dir, "test", sample_size) if include_test else None
    checks = validate_splits(train, val, test)
    input_mode = config["data"].get("input_mode", "values_plus_mask_plus_static")
    metadata = input_metadata(train, input_mode)
    return seed, device, train, val, test, input_mode, metadata, checks


def _dirs(config, model_name):
    output_dir = Path(config["paths"]["output_dir"])
    metrics_dir = output_dir / "metrics"
    audit_dir = Path(config["paths"].get("audit_dir", output_dir / "audit"))
    predictions_dir = output_dir / "predictions"
    for path in [output_dir, metrics_dir, audit_dir, predictions_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return output_dir, metrics_dir, audit_dir, predictions_dir


def _write_common_audits(model_name, output_dir, audit_dir, config, train, val, test, metadata, checks):
    save_yaml(output_dir / "config_snapshot.yaml", config)
    save_json(
        audit_dir / f"{model_name}_input_audit.json",
        {
            "dataset_dir": config["paths"]["dataset_dir"],
            "input_metadata": metadata,
            "splits": {
                "train": _split_shape(train),
                "validation": _split_shape(val),
                **({"test": _split_shape(test)} if test is not None else {}),
            },
            "checks": checks,
            "test_used_in_tuning": bool(config.get("phase") == "tuning" and config.get("include_test", False)),
        },
    )
    summaries = [target_summary("train", train.duration_eval_days, train.event, train.t_idx), target_summary("validation", val.duration_eval_days, val.event, val.t_idx)]
    if test is not None:
        summaries.append(target_summary("test", test.duration_eval_days, test.event, test.t_idx))
    target_df = pd.concat(summaries, ignore_index=True)
    target_df.to_csv(audit_dir / "target_discretization_summary.csv", index=False)
    save_json(
        audit_dir / f"{model_name}_target_audit.json",
        {
            "cuts": config["experiment"]["cuts"],
            "t_idx_min": int(min(train.t_idx.min(), val.t_idx.min())),
            "t_idx_max": int(max(train.t_idx.max(), val.t_idx.max())),
            "censored_at_10d_idx9_count_train": int(((train.duration_eval_days >= 10.0) & (train.event == 0) & (train.t_idx == 9)).sum()),
        },
    )


def _split_shape(split):
    return {
        "n_patients": int(split.patient_ids.shape[0]),
        "X_seq_shape": list(split.x_seq.shape),
        "M_seq_shape": list(split.m_seq.shape),
        "X_static_shape": list(split.x_static.shape),
        "event_rate": float(split.event.mean()),
    }


def _batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items() if key not in {"patient_id"}}


def _train_dysurv(config, logger):
    seed, device, train, val, test, input_mode, metadata, checks = _prepare(config)
    model_name = "dysurv"
    output_dir, metrics_dir, audit_dir, predictions_dir = _dirs(config, model_name)
    _write_common_audits(model_name, output_dir, audit_dir, config, train, val, test, metadata, checks)
    cfg = config["model"]["params"]
    train_loader = make_loader(train, input_mode, cfg["batch_size"], True, config["data"].get("num_workers", 0))
    val_loader = make_loader(val, input_mode, cfg["batch_size"], False, config["data"].get("num_workers", 0))
    model = DySurv72h(input_dim=metadata["n_model_input_features"], **_dysurv_model_kwargs(cfg)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    train_log, best_state, best_val = [], None, math.inf
    patience_count = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss = _run_dysurv_epoch(model, train_loader, device, optimizer, cfg)
        val_loss = _run_dysurv_epoch(model, val_loader, device, None, cfg)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_loss.items()}, **{f"val_{k}": v for k, v in val_loss.items()}}
        train_log.append(row)
        if val_loss["loss_total"] < best_val:
            best_val = val_loss["loss_total"]
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= int(cfg.get("patience", 15)):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(train_log).to_csv(output_dir / "train_log.csv", index=False)
    split_surv, split_targets = _predict_dysurv(model, [train, val] + ([test] if test is not None else []), input_mode, cfg["batch_size"], device)
    metrics = evaluate_survival_predictions(model_name, split_surv, split_targets, config, metrics_dir, audit_dir, predictions_dir)
    metrics["best_validation_loss"] = float(best_val)
    save_json(metrics_dir / "metrics.json", metrics)
    return metrics


def _dysurv_model_kwargs(cfg):
    keys = ["hidden_rnn", "layers_rnn", "latent_dim", "encoder_layers", "decoder_layers", "survival_layers", "dropout", "num_durations"]
    out = {key: cfg[key] for key in keys if key in cfg}
    out.setdefault("num_durations", 10)
    return out


def _run_dysurv_epoch(model, loader, device, optimizer, cfg):
    training = optimizer is not None
    model.train(training)
    totals = {"loss_total": 0.0, "loss_surv": 0.0, "loss_recon": 0.0, "loss_kl": 0.0}
    n = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        if training:
            optimizer.zero_grad()
        out = model(batch["x"].float())
        loss_surv = logistic_hazard_nll(out["logits"], batch["t_idx"], batch["event"])
        loss_recon = mse_observed(out["reconstruction"], batch["x_recon"].float())
        loss_kl = kl_loss(out["mu"], out["logvar"])
        loss = float(cfg.get("w_surv", 0.333)) * loss_surv + float(cfg.get("w_recon", 0.333)) * loss_recon + float(cfg.get("w_kl", 0.333)) * loss_kl
        if training:
            loss.backward()
            optimizer.step()
        bs = batch["x"].shape[0]
        for key, value in [("loss_total", loss), ("loss_surv", loss_surv), ("loss_recon", loss_recon), ("loss_kl", loss_kl)]:
            totals[key] += float(value.detach().cpu()) * bs
        n += bs
    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def _predict_dysurv(model, splits, input_mode, batch_size, device):
    model.eval()
    split_surv, split_targets = {}, {}
    for split in splits:
        loader = make_loader(split, input_mode, batch_size, False)
        surv = []
        for batch in loader:
            x = batch["x"].to(device).float()
            logits = model.predict_logits(x)
            surv.append(hazards_to_survival(logits).cpu().numpy())
        name = "validation" if split.name == "val" else split.name
        split_surv[name] = survival_df_from_array(np.concatenate(surv, axis=0))
        split_targets[name] = (split.duration_eval_days, split.event)
    return split_surv, split_targets


def _train_dynamic_deephit(config, logger):
    seed, device, train, val, test, input_mode, metadata, checks = _prepare(config)
    model_name = "dynamic_deephit"
    output_dir, metrics_dir, audit_dir, predictions_dir = _dirs(config, model_name)
    _write_common_audits(model_name, output_dir, audit_dir, config, train, val, test, metadata, checks)
    cfg = config["model"]["params"]
    if float(cfg["alpha"]) + float(cfg["beta"]) > 1.0:
        raise ValueError("Dynamic-DeepHit requires alpha + beta <= 1")
    train_loader = make_loader(train, input_mode, cfg["batch_size"], True, config["data"].get("num_workers", 0))
    val_loader = make_loader(val, input_mode, cfg["batch_size"], False, config["data"].get("num_workers", 0))
    num_durations = int(cfg.get("num_durations", 10))
    output_dim = num_durations + (1 if cfg.get("include_tail_category", True) else 0)
    model = DynamicDeepHit72h(
        input_dim=metadata["n_model_input_features"],
        output_dim=output_dim,
        layers_rnn=int(cfg.get("layers_rnn", 1)),
        hidden_rnn=int(cfg.get("hidden_rnn", 64)),
        typ=cfg.get("typ", "LSTM"),
        long_param=cfg.get("long_param", {"layers": [64], "dropout": 0.1, "activation": "ReLU"}),
        att_param=cfg.get("att_param", {"layers": [64], "dropout": 0.1, "activation": "ReLU"}),
        cs_param=cfg.get("cs_param", {"layers": [64], "dropout": 0.1, "activation": "ReLU"}),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    train_log, best_state, best_val = [], None, math.inf
    patience_count = 0
    for epoch in range(1, int(cfg.get("iters", cfg.get("epochs", 100))) + 1):
        train_loss = _run_ddh_epoch(model, train_loader, device, optimizer, cfg)
        val_loss = _run_ddh_epoch(model, val_loader, device, None, cfg)
        train_log.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_loss.items()}, **{f"val_{k}": v for k, v in val_loss.items()}})
        if val_loss["loss_total"] < best_val:
            best_val = val_loss["loss_total"]
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= int(cfg.get("patience", 10)):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(train_log).to_csv(output_dir / "train_log.csv", index=False)
    split_surv, split_targets = _predict_ddh(model, [train, val] + ([test] if test is not None else []), input_mode, cfg["batch_size"], device, num_durations)
    _write_ddh_probability_audit(model, [train, val] + ([test] if test is not None else []), input_mode, cfg["batch_size"], device, audit_dir, num_durations)
    metrics = evaluate_survival_predictions(model_name, split_surv, split_targets, config, metrics_dir, audit_dir, predictions_dir)
    metrics["best_validation_loss"] = float(best_val)
    save_json(metrics_dir / "metrics.json", metrics)
    return metrics


def _run_ddh_epoch(model, loader, device, optimizer, cfg):
    training = optimizer is not None
    model.train(training)
    totals = {"loss_total": 0.0, "loss_longitudinal": 0.0, "loss_ranking": 0.0, "loss_nll": 0.0}
    n = 0
    num_durations = int(cfg.get("num_durations", 10))
    for batch in loader:
        batch = _batch_to_device(batch, device)
        if training:
            optimizer.zero_grad()
        out = model(batch["x"].float())
        pmf_eval = out["pmf"][:, :num_durations]
        loss_long = F.mse_loss(out["longitudinal_prediction"][:, :-1, :], batch["x"][:, 1:, :].float())
        loss_rank = deephit_ranking_loss(pmf_eval, batch["t_idx"], batch["event"], cfg.get("sigma", 0.1))
        loss_nll = pmf_nll(out["pmf"], batch["t_idx"], batch["event"])
        alpha, beta = float(cfg["alpha"]), float(cfg["beta"])
        loss = (1.0 - alpha - beta) * loss_long + alpha * loss_rank + beta * loss_nll
        if training:
            loss.backward()
            optimizer.step()
        bs = batch["x"].shape[0]
        for key, value in [("loss_total", loss), ("loss_longitudinal", loss_long), ("loss_ranking", loss_rank), ("loss_nll", loss_nll)]:
            totals[key] += float(value.detach().cpu()) * bs
        n += bs
    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def _predict_ddh(model, splits, input_mode, batch_size, device, num_durations):
    model.eval()
    split_surv, split_targets = {}, {}
    for split in splits:
        loader = make_loader(split, input_mode, batch_size, False)
        surv = []
        for batch in loader:
            pmf = model(batch["x"].to(device).float())["pmf"][:, :num_durations]
            s = 1.0 - torch.cumsum(pmf, dim=1)
            surv.append(s.clamp(0.0, 1.0).cpu().numpy())
        name = "validation" if split.name == "val" else split.name
        split_surv[name] = survival_df_from_array(np.concatenate(surv, axis=0))
        split_targets[name] = (split.duration_eval_days, split.event)
    return split_surv, split_targets


@torch.no_grad()
def _write_ddh_probability_audit(model, splits, input_mode, batch_size, device, audit_dir, num_durations):
    model.eval()
    rows = []
    for split in splits:
        loader = make_loader(split, input_mode, batch_size, False)
        pmf_sums, cif_decreases, surv_increases, s10_values = [], [], [], []
        for batch in loader:
            pmf_full = model(batch["x"].to(device).float())["pmf"]
            pmf = pmf_full[:, :num_durations]
            cif = torch.cumsum(pmf, dim=1)
            surv = 1.0 - cif
            pmf_sums.append(pmf_full.sum(dim=1).detach().cpu().numpy())
            cif_decreases.append((torch.diff(cif, dim=1) < -1e-6).any(dim=1).detach().cpu().numpy())
            surv_increases.append((torch.diff(surv, dim=1) > 1e-6).any(dim=1).detach().cpu().numpy())
            s10_values.append(surv[:, -1].detach().cpu().numpy())
        name = "validation" if split.name == "val" else split.name
        pmf_sums = np.concatenate(pmf_sums)
        cif_decreases = np.concatenate(cif_decreases)
        surv_increases = np.concatenate(surv_increases)
        s10_values = np.concatenate(s10_values)
        rows.append(
            {
                "split": name,
                "pmf_sum_min": float(pmf_sums.min()),
                "pmf_sum_max": float(pmf_sums.max()),
                "pmf_sum_mean": float(pmf_sums.mean()),
                "cif_non_decreasing": bool(not cif_decreases.any()),
                "survival_non_increasing": bool(not surv_increases.any()),
                "share_s10_below_1e-6": float(np.mean(s10_values < 1e-6)),
                "mean_s10": float(np.mean(s10_values)),
            }
        )
    pd.DataFrame(rows).to_csv(audit_dir / "dynamic_deephit_probability_audit.csv", index=False)
