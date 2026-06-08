"""
TFG adaptation of the original DeepHit model.

The original TensorFlow implementation is stored in:
src/models_references/DeepHit/

This PyTorch adaptation keeps the static DeepHit structure needed for the TFG:
shared network, cause-specific network, output with shape
num_Event x num_Category, log-likelihood loss, ranking loss and calibration loss.
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

from src.evaluation.metrics import evaluate_predictions
from src.evaluation.time_dependent_survival import horizon_c_index_dict, mean_horizon_c_index, survival_time_dependent_metrics
from src.models.static_common import (
    cap_survival_targets,
    configured_split_names,
    configured_split_frames,
    get_device,
    load_static_splits,
    make_time_grid,
    model_metrics_dir,
    save_json,
    should_save_predictions,
    should_save_test_survival_curves,
    split_xy,
)

EPS = 1e-8


def deephit_output_categories(num_categories, include_tail_category=False):
    return int(num_categories) + int(bool(include_tail_category))


def discretize_time(time, max_horizon_days, num_categories):
    clipped = np.clip(np.asarray(time, dtype=float), 0.0, float(max_horizon_days))
    bins = np.ceil(clipped / float(max_horizon_days) * num_categories).astype(int) - 1
    return np.clip(bins, 0, num_categories - 1)


def deephit_event_probability_and_survival(pred, num_categories, include_tail_category=False, event_idx=0):
    pred = np.asarray(pred, dtype=float)
    event_prob = pred[:, event_idx, :num_categories]
    survival = 1.0 - np.cumsum(event_prob, axis=1)
    tail_probability = None
    if include_tail_category:
        tail_probability = pred[:, event_idx, num_categories:].sum(axis=1)
        survival[:, -1] = tail_probability
    return event_prob, np.clip(survival, 0.0, 1.0), tail_probability


def build_deephit_masks(time_bins, event, num_events, num_categories, include_tail_category=False):
    """
    Build DeepHit masks following the original static implementation.

    mask1 selects the observed event bin for uncensored stays and the tail
    probability after censoring for censored stays. mask2 selects bins up to
    the observed time for ranking and calibration terms.
    """
    time_bins = np.asarray(time_bins, dtype=int)
    event = np.asarray(event, dtype=int)
    output_categories = deephit_output_categories(num_categories, include_tail_category)
    mask1 = np.zeros((len(time_bins), num_events, output_categories), dtype="float32")
    mask2 = np.zeros((len(time_bins), output_categories), dtype="float32")
    for i, (t, e) in enumerate(zip(time_bins, event)):
        t = int(np.clip(t, 0, num_categories - 1))
        if e > 0:
            mask1[i, e - 1, t] = 1.0
        else:
            mask1[i, :, (t + 1) :] = 1.0
        mask2[i, : (t + 1)] = 1.0
    return mask1, mask2


class DeepHitNet(nn.Module if nn is not None else object):
    def __init__(self, in_features, num_events, num_categories, shared_layers, cause_layers, dropout):
        if nn is None:
            raise ImportError("torch is required for DeepHit")
        super().__init__()
        self.num_events = num_events
        self.num_categories = num_categories
        self.shared = self._mlp(in_features, shared_layers, dropout)
        shared_out = shared_layers[-1] if shared_layers else in_features
        cause_in = in_features + shared_out
        self.cause_nets = nn.ModuleList([self._mlp(cause_in, cause_layers, dropout) for _ in range(num_events)])
        cause_out = cause_layers[-1] if cause_layers else cause_in
        self.output = nn.Linear(num_events * cause_out, num_events * num_categories)

    @staticmethod
    def _mlp(in_features, layers, dropout):
        modules = []
        previous = in_features
        for hidden in layers:
            modules.extend([nn.Linear(previous, hidden), nn.ReLU()])
            if dropout:
                modules.append(nn.Dropout(dropout))
            previous = hidden
        return nn.Sequential(*modules) if modules else nn.Identity()

    def forward(self, x):
        shared = self.shared(x)
        combined = torch.cat([x, shared], dim=1)
        cause_outputs = [net(combined) for net in self.cause_nets]
        out = torch.cat(cause_outputs, dim=1)
        out = self.output(out)
        out = torch.softmax(out, dim=1)
        return out.view(-1, self.num_events, self.num_categories)


def deephit_loss(pred, time_bins, event, mask1, mask2, alpha, beta, gamma, ranking_sigma=0.1):
    event_indicator = (event > 0).float()
    selected = (pred * mask1).sum(dim=(1, 2)).clamp_min(EPS)
    loss_event = -event_indicator * torch.log(selected)
    loss_censor = -(1.0 - event_indicator) * torch.log(selected)
    loglik = (loss_event + loss_censor).mean()

    event_times = time_bins.view(-1, 1)
    ranking_terms = []
    for event_idx in range(pred.shape[1]):
        event_label = event_idx + 1
        event_specific_pmf = pred[:, event_idx, :]
        risk_at_subject_times = torch.matmul(event_specific_pmf, mask2.T)
        own_risk = torch.diag(risk_at_subject_times).view(-1, 1)
        risk_diff = own_risk - risk_at_subject_times.T
        comparable = ((event.view(-1, 1) == event_label) & (event_times < event_times.T)).float()
        ranking_terms.append((comparable * torch.exp(-risk_diff / float(ranking_sigma))).mean())
    ranking = torch.stack(ranking_terms).sum()

    observed_at_bin = ((event == 1).float().view(-1, 1) * mask2).mean(dim=0)
    predicted_at_bin = (pred[:, 0, :] * mask2).mean(dim=0)
    calibration = torch.mean((predicted_at_bin - observed_at_bin) ** 2)
    return alpha * loglik + beta * ranking + gamma * calibration


def _make_loader(x, time_bins, event, masks, batch_size, shuffle):
    mask1, mask2 = masks
    dataset = TensorDataset(
        torch.tensor(x.values, dtype=torch.float32),
        torch.tensor(time_bins, dtype=torch.long),
        torch.tensor(event.values, dtype=torch.long),
        torch.tensor(mask1, dtype=torch.float32),
        torch.tensor(mask2, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _default_grid(max_horizon_days):
    upper = int(np.floor(float(max_horizon_days)))
    if upper > 1:
        return list(range(1, upper))
    return [float(max_horizon_days)]


def _predict(
    model,
    df,
    config,
    device,
    split_name,
    evaluation_time_grid,
    censoring_time,
    censoring_event,
):
    x, time, event, ids = split_xy(df)
    time, event = cap_survival_targets(time, event, config["model"]["max_horizon_days"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x.values, dtype=torch.float32, device=device)).cpu().numpy()
    num_categories = config["model"]["num_Category"]
    include_tail_category = config["model"].get("include_tail_category", False)
    event_prob, survival, tail_probability = deephit_event_probability_and_survival(
        pred,
        num_categories,
        include_tail_category,
    )
    time_grid = make_time_grid(config["model"]["max_horizon_days"], num_categories)
    surv_df = pd.DataFrame(survival.T, index=time_grid)
    risk = 1.0 - survival[:, -1]
    preds = ids.copy()
    preds["risk_score"] = risk
    preds["model"] = "deephit"
    for idx, horizon in enumerate(time_grid):
        preds[f"event_probability_bin_{idx + 1}"] = event_prob[:, idx]
        preds[f"survival_at_{horizon:.2f}d"] = survival[:, idx]
    if tail_probability is not None:
        preds["tail_probability_beyond_horizon"] = tail_probability
    return (
        preds,
        surv_df,
        evaluate_predictions(
            time,
            event,
            risk_scores=risk,
            survival=surv_df,
            risk_metric_name="harrell_c_index_final_risk",
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
    logger.info("Resuming DeepHit from %s at epoch %d", checkpoint_path, start_epoch)
    return start_epoch, best_loss


def train_deephit(config, logger):
    if torch is None:
        raise ImportError("torch is required for DeepHit")
    paths = config["paths"]
    model_cfg = config["model"]
    num_events = model_cfg.get("num_Event", 1)
    num_categories = model_cfg.get("num_Category", 10)
    include_tail_category = model_cfg.get("include_tail_category", False)
    output_categories = deephit_output_categories(num_categories, include_tail_category)
    max_horizon_days = model_cfg.get("max_horizon_days", 10)
    train, val, test = load_static_splits(paths, include_test="test" in configured_split_names(config))
    x_train, time_train, event_train, _ = split_xy(train)
    x_val, time_val, event_val, _ = split_xy(val)
    time_train, event_train = cap_survival_targets(time_train, event_train, max_horizon_days)
    time_val, event_val = cap_survival_targets(time_val, event_val, max_horizon_days)
    train_bins = discretize_time(time_train, max_horizon_days, num_categories)
    val_bins = discretize_time(time_val, max_horizon_days, num_categories)
    train_masks = build_deephit_masks(train_bins, event_train, num_events, num_categories, include_tail_category)
    val_masks = build_deephit_masks(val_bins, event_val, num_events, num_categories, include_tail_category)

    device = get_device(model_cfg.get("device", "auto"))
    logger.info("DeepHit device: %s", device)
    model = DeepHitNet(
        in_features=x_train.shape[1],
        num_events=num_events,
        num_categories=output_categories,
        shared_layers=model_cfg.get("shared_layers", [128, 64]),
        cause_layers=model_cfg.get("cause_layers", [64]),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.get("learning_rate", 1e-3))
    loader = _make_loader(x_train, train_bins, event_train, train_masks, model_cfg.get("batch_size", 256), True)
    val_loader = _make_loader(x_val, val_bins, event_val, val_masks, len(x_val), False)

    checkpoints_dir = Path(paths["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_loss = np.inf
    best_state = None
    patience = model_cfg.get("early_stopping_patience", 10)
    waiting = 0
    log_rows = []
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
        losses = []
        for batch in loader:
            bx, bt, be, bm1, bm2 = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad()
            loss = deephit_loss(
                model(bx),
                bt,
                be,
                bm1,
                bm2,
                model_cfg.get("alpha", 1.0),
                model_cfg.get("beta", 1.0),
                model_cfg.get("gamma", 0.0),
                model_cfg.get("ranking_sigma", 0.1),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            vb = next(iter(val_loader))
            vx, vt, ve, vm1, vm2 = [tensor.to(device) for tensor in vb]
            val_loss = float(
                deephit_loss(
                    model(vx),
                    vt,
                    ve,
                    vm1,
                    vm2,
                    model_cfg.get("alpha", 1.0),
                    model_cfg.get("beta", 1.0),
                    model_cfg.get("gamma", 0.0),
                    model_cfg.get("ranking_sigma", 0.1),
                ).cpu()
            )
        log_rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            waiting = 0
            if model_cfg.get("save_best_checkpoint", True):
                _save_checkpoint(
                    checkpoints_dir / "deephit_best_model.pt",
                    model,
                    optimizer,
                    epoch,
                    best_loss,
                    x_train,
                    model_cfg,
                )
        else:
            waiting += 1
        if model_cfg.get("save_last_checkpoint", True):
            _save_checkpoint(
                checkpoints_dir / "deephit_last_model.pt",
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
                checkpoints_dir / f"deephit_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                best_loss,
                x_train,
                model_cfg,
            )
        if epoch % model_cfg.get("log_every", 10) == 0:
            logger.info("DeepHit epoch %d val_loss %.5f", epoch, val_loss)
        if waiting >= patience:
            logger.info("DeepHit early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_state is not None and model_cfg.get("save_best_checkpoint", True):
        _save_checkpoint(checkpoints_dir / "deephit_best_model.pt", model, optimizer, last_epoch, best_loss, x_train, model_cfg)
    metrics_dir = model_metrics_dir(paths, "deephit")
    pd.DataFrame(log_rows).to_csv(metrics_dir / "deephit_train_log.csv", index=False)

    eval_cfg = config.get("evaluation", {})
    evaluation_time_grid = eval_cfg.get("evaluation_time_grid", model_cfg.get("evaluation_time_grid", _default_grid(max_horizon_days)))
    horizon_times = eval_cfg.get("horizon_times", model_cfg.get("horizon_times", _default_grid(max_horizon_days)))
    metrics = {
        "model": "deephit",
        "best_val_loss": best_loss,
        "evaluation_time_grid": evaluation_time_grid,
        "horizon_times": horizon_times,
        "splits": {},
    }
    save_predictions_flag = should_save_predictions(config)
    save_test_survival_flag = should_save_test_survival_curves(config)
    predictions = []
    antolini_rows = []
    weighted_rows = []
    train_time_values = time_train.to_numpy(dtype=float)
    train_event_values = event_train.to_numpy(dtype=int)
    for split_name, split_df in configured_split_frames(config, train, val, test).items():
        preds, surv, split_metrics, time_values, event_values = _predict(
            model,
            split_df,
            config,
            device,
            split_name,
            evaluation_time_grid,
            train_time_values,
            train_event_values,
        )
        preds["split"] = split_name
        if save_predictions_flag:
            predictions.append(preds)
        metrics["splits"][split_name] = split_metrics
        antolini, weighted = survival_time_dependent_metrics(
            split_name,
            time_values,
            event_values,
            surv,
            train_time_values,
            train_event_values,
            horizon_times,
        )
        antolini_rows.append(antolini)
        weighted_rows.extend(weighted)
        metrics["splits"][split_name]["ctd_antolini"] = antolini["ctd"]
        metrics["splits"][split_name]["horizon_c_index"] = horizon_c_index_dict(weighted)
        metrics["splits"][split_name]["mean_horizon_c_index"] = mean_horizon_c_index(weighted)
        logger.info(
            "DeepHit %s final-risk C-index: %.4f",
            split_name,
            split_metrics["harrell_c_index_final_risk"],
        )
        if split_name == "test" and save_test_survival_flag:
            surv.to_csv(Path(paths["predictions_dir"]) / "deephit_test_survival_curves.csv")

    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(Path(paths["predictions_dir"]) / "deephit_predictions.parquet", index=False)
    weighted_path = Path(config.get("evaluation", {}).get("weighted_c_index_path", metrics_dir / "deephit_weighted_c_index_by_horizon.csv"))
    antolini_path = Path(config.get("evaluation", {}).get("antolini_ctd_path", metrics_dir / "deephit_antolini_ctd.csv"))
    weighted_path.parent.mkdir(parents=True, exist_ok=True)
    antolini_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weighted_rows).to_csv(weighted_path, index=False)
    pd.DataFrame(antolini_rows).to_csv(antolini_path, index=False)
    save_json(metrics, metrics_dir / "deephit_metrics.json")
    return metrics
