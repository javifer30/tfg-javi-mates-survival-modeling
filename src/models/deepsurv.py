import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


class CoxDataset(Dataset):
    """Dataset for Cox proportional hazards model."""
    
    def __init__(self, X, time, event):
        self.X = torch.from_numpy(X.astype("float32"))
        self.time = torch.from_numpy(time.astype("float32"))
        self.event = torch.from_numpy(event.astype("float32"))
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]


class DeepSurvNet(nn.Module):
    """
    Deep Survival neural network for Cox proportional hazards model.
    
    Args:
        in_features: Number of input features
        hidden: List of hidden layer sizes
        dropout: Dropout rate
    """
    
    def __init__(self, in_features, hidden=[64, 32], dropout=0.1):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def cox_ph_loss(log_hazard, time, event):
    """
    Cox proportional hazards loss function.
    
    Args:
        log_hazard: Log hazard predictions from the model
        time: Survival/censoring times
        event: Event indicators (1=event, 0=censored)
        
    Returns:
        Negative partial log-likelihood
    """
    order = torch.argsort(time, descending=True)
    time_ord = time[order]
    event_ord = event[order]
    loghaz_ord = log_hazard[order]
    hazard_exp = torch.exp(loghaz_ord)
    cum_risk = torch.cumsum(hazard_exp, dim=0)
    log_lik = loghaz_ord - torch.log(cum_risk)
    log_lik_event = log_lik * event_ord
    loss = -torch.sum(log_lik_event) / torch.sum(event_ord)
    return loss


def concordance_index(time, event, risk_scores):
    """
    Calculate concordance index (C-index) for survival predictions.
    
    Args:
        time: Survival/censoring times
        event: Event indicators
        risk_scores: Risk scores from the model
        
    Returns:
        C-index value
    """
    n = len(time)
    num, den = 0.0, 0.0
    for i in range(n):
        for j in range(n):
            if time[i] < time[j] and event[i] == 1:
                den += 1.0
                if risk_scores[i] > risk_scores[j]:
                    num += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    num += 0.5
    return num / den if den > 0 else np.nan
