"""
TFG adaptation of the original DeepSurv model.

The original implementation is stored in:
src/models_references/DeepSurv/

The original code depends on Theano/Lasagne. This file keeps the same core idea
for the TFG pipeline using PyTorch: X_static -> MLP -> scalar log-risk, optimized
with the Cox partial likelihood and correct censoring handling.
"""

from pathlib import Path

import numpy as np
import pandas as pd
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from src.data.static_dataset import EVENT_COL, ID_COL, SPLIT_COL, TIME_COL
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.time_dependent_survival import horizon_c_index_dict, mean_horizon_c_index, survival_time_dependent_metrics
from src.models.static_common import cap_survival_targets, get_device, load_static_splits, make_time_grid, model_metrics_dir, save_json, split_xy


class DeepSurvNet(nn.Module if nn is not None else object):
    def __init__(self, in_features, hidden_layers, dropout):
        if nn is None:
            raise ImportError("torch is required for DeepSurv")
        super().__init__()
        layers = []
        previous = in_features
        for hidden in hidden_layers:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cox_partial_likelihood_loss(log_risk, time, event):
    order = torch.argsort(time, descending=True)
    log_risk = log_risk[order]
    event = event[order]
    log_cumsum_hazard = torch.logcumsumexp(log_risk, dim=0)
    observed = event == 1
    n_events = torch.clamp(observed.sum(), min=1)
    return -((log_risk - log_cumsum_hazard) * observed).sum() / n_events


def _to_loader(x, time, event, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(x.values, dtype=torch.float32),
        torch.tensor(time.values, dtype=torch.float32),
        torch.tensor(event.values, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_risk(model, x, device):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x.values, dtype=torch.float32, device=device)
        return model(tensor).cpu().numpy()


def _breslow_baseline_cumulative_hazard(time, event, log_risk, time_grid):
    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=int)
    hazard_ratio = np.exp(np.clip(np.asarray(log_risk, dtype=float), -50.0, 50.0))
    event_times = np.sort(np.unique(time_values[event_values == 1]))

    cumulative_hazard = []
    cumulative = 0.0
    for event_time in event_times:
        at_risk = time_values >= event_time
        denominator = hazard_ratio[at_risk].sum()
        if denominator <= 0.0:
            continue
        n_events = np.sum((time_values == event_time) & (event_values == 1))
        cumulative += float(n_events / denominator)
        cumulative_hazard.append((float(event_time), cumulative))

    if not cumulative_hazard:
        return np.zeros(len(time_grid), dtype=float)

    baseline_times = np.asarray([item[0] for item in cumulative_hazard], dtype=float)
    baseline_values = np.asarray([item[1] for item in cumulative_hazard], dtype=float)
    indices = np.searchsorted(baseline_times, np.asarray(time_grid, dtype=float), side="right") - 1
    grid_hazard = np.zeros(len(time_grid), dtype=float)
    valid = indices >= 0
    grid_hazard[valid] = baseline_values[indices[valid]]
    return grid_hazard


def _survival_from_baseline(log_risk, baseline_cumulative_hazard, time_grid):
    hazard_ratio = np.exp(np.clip(np.asarray(log_risk, dtype=float), -50.0, 50.0))
    survival = np.exp(-np.outer(baseline_cumulative_hazard, hazard_ratio))
    return pd.DataFrame(np.clip(survival, 0.0, 1.0), index=time_grid)


def _default_grid(max_horizon_days):
    upper = int(float(max_horizon_days))
    if upper > 1:
        return list(range(1, upper))
    return [float(max_horizon_days)]


def _predict(
    model,
    df,
    split_name,
    device,
    baseline_cumulative_hazard,
    time_grid,
    max_horizon_days,
    evaluation_time_grid,
    censoring_time,
    censoring_event,
):
    x, time, event, ids = split_xy(df)
    time, event = cap_survival_targets(time, event, max_horizon_days)
    risk = _predict_risk(model, x, device)
    survival = _survival_from_baseline(risk, baseline_cumulative_hazard, time_grid)
    preds = ids.copy()
    preds["risk_score"] = risk
    preds["model"] = "deepsurv"
    return (
        preds,
        survival,
        evaluate_predictions(
            time,
            event,
            risk_scores=risk,
            survival=survival,
            risk_metric_name="harrell_c_index",
            evaluation_time_grid=evaluation_time_grid,
            censoring_time=censoring_time,
            censoring_event=censoring_event,
            compute_curve_metrics=split_name != "train",
        ),
        time.to_numpy(dtype=float),
        event.to_numpy(dtype=int),
    )


def _checkpoint_payload(model, optimizer, epoch, best_loss, x_train, model_cfg):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_loss,
        "in_features": x_train.shape[1],
        "feature_columns": list(x_train.columns),
        "model_config": model_cfg,
    }


def _save_checkpoint(path, model, optimizer, epoch, best_loss, x_train, model_cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, optimizer, epoch, best_loss, x_train, model_cfg), path)


def _load_checkpoint(path, model, optimizer, device, logger):
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_loss = float(checkpoint.get("best_val_loss", np.inf))
    logger.info("Resuming DeepSurv from %s at epoch %d", checkpoint_path, start_epoch)
    return start_epoch, best_loss


def train_deepsurv(config, logger):
    if torch is None:
        raise ImportError("torch is required for DeepSurv")
    paths = config["paths"]
    model_cfg = config["model"]
    train, val, test = load_static_splits(paths)
    x_train, time_train, event_train, _ = split_xy(train)
    x_val, time_val, event_val, _ = split_xy(val)
    max_horizon_days = model_cfg.get("max_horizon_days", 10)
    num_durations = model_cfg.get("num_durations", 10)
    eval_time_train, eval_event_train = cap_survival_targets(time_train, event_train, max_horizon_days)

    device = get_device(model_cfg.get("device", "auto"))
    model = DeepSurvNet(
        in_features=x_train.shape[1],
        hidden_layers=model_cfg.get("hidden_layers", [64, 32]),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg.get("learning_rate", 1e-3),
        weight_decay=model_cfg.get("weight_decay", 0.0),
    )
    logger.info("DeepSurv device: %s", device)
    batch_size = model_cfg.get("batch_size", 256)
    loader = _to_loader(x_train, time_train, event_train, batch_size, shuffle=True)
    val_tensors = (
        torch.tensor(x_val.values, dtype=torch.float32, device=device),
        torch.tensor(time_val.values, dtype=torch.float32, device=device),
        torch.tensor(event_val.values, dtype=torch.float32, device=device),
    )

    checkpoints_dir = Path(paths["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_loss = np.inf
    best_state = None
    patience = model_cfg.get("early_stopping_patience", 10)
    epochs_without_improvement = 0
    train_log = []
    start_epoch = 1
    if model_cfg.get("resume_from_checkpoint"):
        start_epoch, best_loss = _load_checkpoint(
            model_cfg["resume_from_checkpoint"],
            model,
            optimizer,
            device,
            logger,
        )

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, model_cfg.get("epochs", 50) + 1):
        last_epoch = epoch
        model.train()
        epoch_losses = []
        for batch_x, batch_time, batch_event in loader:
            batch_x = batch_x.to(device)
            batch_time = batch_time.to(device)
            batch_event = batch_event.to(device)
            optimizer.zero_grad()
            loss = cox_partial_likelihood_loss(model(batch_x), batch_time, batch_event)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(
                cox_partial_likelihood_loss(model(val_tensors[0]), val_tensors[1], val_tensors[2]).cpu()
            )
        train_log.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            if model_cfg.get("save_best_checkpoint", True):
                _save_checkpoint(
                    checkpoints_dir / "deepsurv_best_model.pt",
                    model,
                    optimizer,
                    epoch,
                    best_loss,
                    x_train,
                    model_cfg,
                )
        else:
            epochs_without_improvement += 1
        if model_cfg.get("save_last_checkpoint", True):
            _save_checkpoint(
                checkpoints_dir / "deepsurv_last_model.pt",
                model,
                optimizer,
                epoch,
                best_loss,
                x_train,
                model_cfg,
            )
        save_every = model_cfg.get("save_every_n_epochs")
        if save_every and epoch % int(save_every) == 0:
            _save_checkpoint(
                checkpoints_dir / f"deepsurv_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                best_loss,
                x_train,
                model_cfg,
            )
        if epoch % model_cfg.get("log_every", 10) == 0:
            logger.info("DeepSurv epoch %d val_loss %.5f", epoch, val_loss)
        if epochs_without_improvement >= patience:
            logger.info("DeepSurv early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_state is not None and model_cfg.get("save_best_checkpoint", True):
        _save_checkpoint(checkpoints_dir / "deepsurv_best_model.pt", model, optimizer, last_epoch, best_loss, x_train, model_cfg)
    metrics_dir = model_metrics_dir(paths, "deepsurv")
    pd.DataFrame(train_log).to_csv(metrics_dir / "deepsurv_train_log.csv", index=False)

    time_grid = make_time_grid(max_horizon_days, num_durations)
    train_risk = _predict_risk(model, x_train, device)
    baseline_cumulative_hazard = _breslow_baseline_cumulative_hazard(eval_time_train, eval_event_train, train_risk, time_grid)

    eval_cfg = config.get("evaluation", {})
    evaluation_time_grid = eval_cfg.get("evaluation_time_grid", model_cfg.get("evaluation_time_grid", _default_grid(max_horizon_days)))
    horizon_times = eval_cfg.get("horizon_times", model_cfg.get("horizon_times", _default_grid(max_horizon_days)))
    metrics = {
        "model": "deepsurv",
        "best_val_loss": best_loss,
        "evaluation_time_grid": evaluation_time_grid,
        "horizon_times": horizon_times,
        "splits": {},
    }
    predictions = []
    antolini_rows = []
    weighted_rows = []
    for split_name, split_df in {"train": train, "validation": val, "test": test}.items():
        preds, survival, split_metrics, time_values, event_values = _predict(
            model,
            split_df,
            split_name,
            device,
            baseline_cumulative_hazard,
            time_grid,
            max_horizon_days,
            evaluation_time_grid,
            eval_time_train.to_numpy(dtype=float),
            eval_event_train.to_numpy(dtype=int),
        )
        preds["split"] = split_name
        predictions.append(preds)
        metrics["splits"][split_name] = split_metrics
        antolini, weighted = survival_time_dependent_metrics(
            split_name,
            time_values,
            event_values,
            survival,
            eval_time_train.to_numpy(dtype=float),
            eval_event_train.to_numpy(dtype=int),
            horizon_times,
        )
        antolini_rows.append(antolini)
        weighted_rows.extend(weighted)
        metrics["splits"][split_name]["ctd_antolini"] = antolini["ctd"]
        metrics["splits"][split_name]["horizon_c_index"] = horizon_c_index_dict(weighted)
        metrics["splits"][split_name]["mean_horizon_c_index"] = mean_horizon_c_index(weighted)
        logger.info("DeepSurv %s Harrell C-index: %.4f", split_name, split_metrics["harrell_c_index"])
        if split_name == "test":
            survival.to_csv(Path(paths["predictions_dir"]) / "deepsurv_test_survival_curves.csv")

    pd.concat(predictions, ignore_index=True).to_parquet(Path(paths["predictions_dir"]) / "deepsurv_predictions.parquet", index=False)
    pd.DataFrame(weighted_rows).to_csv(metrics_dir / "deepsurv_weighted_c_index_by_horizon.csv", index=False)
    pd.DataFrame(antolini_rows).to_csv(metrics_dir / "deepsurv_antolini_ctd.csv", index=False)
    save_json(metrics, metrics_dir / "deepsurv_metrics.json")
    return metrics
