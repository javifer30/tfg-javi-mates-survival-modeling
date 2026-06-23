"""Losses for dynamic_landmark survival models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logistic_hazard_nll(logits, t_idx, event):
    hazards = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    log_h = torch.log(hazards)
    log_surv_step = torch.log1p(-hazards)
    event = event.float()
    idx = t_idx.view(-1, 1)
    log_h_t = log_h.gather(1, idx).view(-1)
    surv_before = torch.cat([torch.zeros_like(log_surv_step[:, :1]), log_surv_step.cumsum(1)[:, :-1]], dim=1)
    log_surv_before_t = surv_before.gather(1, idx).view(-1)
    log_surv_through_t = log_surv_step.cumsum(1).gather(1, idx).view(-1)
    log_lik = event * (log_surv_before_t + log_h_t) + (1.0 - event) * log_surv_through_t
    return -log_lik.mean()


def hazards_to_survival(logits):
    hazards = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    return torch.cumprod(1.0 - hazards, dim=1)


def pmf_nll(pmf, t_idx, event):
    pmf = pmf.clamp(1e-8, 1.0)
    event = event.float()
    idx = t_idx.view(-1, 1)
    p_event = pmf.gather(1, idx).view(-1)
    cif = pmf.cumsum(1)
    p_survive = (1.0 - cif.gather(1, idx).view(-1)).clamp(1e-8, 1.0)
    return -(event * torch.log(p_event) + (1.0 - event) * torch.log(p_survive)).mean()


def deephit_ranking_loss(pmf, t_idx, event, sigma=0.1):
    cif = pmf.cumsum(1)
    event_idx = torch.where(event == 1)[0]
    if event_idx.numel() == 0:
        return pmf.sum() * 0.0
    losses = []
    for i in event_idx:
        comparable = t_idx > t_idx[i]
        if comparable.any():
            risk_i = cif[i, t_idx[i]]
            risk_j = cif[comparable, t_idx[i]]
            losses.append(torch.exp((risk_j - risk_i) / float(sigma)).mean())
    if not losses:
        return pmf.sum() * 0.0
    return torch.stack(losses).mean()


def mse_observed(pred, target, mask=None):
    if mask is None:
        return F.mse_loss(pred, target)
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom

