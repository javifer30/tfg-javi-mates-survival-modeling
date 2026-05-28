import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PWEDataset(Dataset):
    """
    Dataset for Piecewise Exponential (PWE) model with Poisson regression.
    
    Args:
        df_pwe: DataFrame from PWEDataExpander with columns 'y', 'd', 'k', 'id', 'x0', 'x1', ...
        p: Number of covariates
    """
    
    def __init__(self, df_pwe, p):
        # Store tensors
        self.y = torch.tensor(df_pwe["y"].values, dtype=torch.float32)
        self.d = torch.tensor(df_pwe["d"].values, dtype=torch.float32)
        self.k = torch.tensor(df_pwe["k"].values, dtype=torch.long)  # interval index
        self.id = torch.tensor(df_pwe["id"].values, dtype=torch.long)
        
        # Extract covariate columns
        X_cols = [f"x{j}" for j in range(p)]
        self.X = torch.tensor(df_pwe[X_cols].values, dtype=torch.float32)
        
        # Precompute log(y) for offset
        eps = 1e-8
        self.log_y = torch.log(self.y + eps)
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.k[idx],
            self.log_y[idx],
            self.d[idx],
            self.id[idx],
        )


class PWEPoisson(nn.Module):
    """
    Piecewise Exponential model using Poisson regression.
    
    Models the hazard function as piecewise constant across time intervals:
        log(μ_ik) = log(y_ik) + α_k + β^T x_i
    
    where:
        - μ_ik is the expected number of events for subject i in interval k
        - y_ik is the exposure time
        - α_k is the baseline log-hazard for interval k
        - β are the covariate effects
    
    Args:
        p: Number of covariates
        K: Number of time intervals
    """
    
    def __init__(self, p, K):
        super().__init__()
        # β: coefficient vector (p x 1)
        self.beta_layer = nn.Linear(p, 1, bias=False)
        # α_k: interval-specific baseline hazard (K parameters)
        self.alpha = nn.Parameter(torch.zeros(K))
    
    def forward(self, X, k, log_y):
        """
        Forward pass.
        
        Args:
            X: (N, p) - Covariates
            k: (N,) - Interval indices [0, ..., K-1]
            log_y: (N,) - Log of time at risk
            
        Returns:
            log_mu: (N,) - Log of expected event count
        """
        lin = self.beta_layer(X).squeeze(-1)  # β^T x
        alpha_k = self.alpha[k]               # select α_k for each row
        log_mu = log_y + alpha_k + lin        # log μ_ik
        return log_mu


def train_pwe_epoch(model, loader, optimizer, criterion, device):
    """
    Train PWE model for one epoch.
    
    Args:
        model: PWEPoisson model
        loader: DataLoader with PWEDataset
        optimizer: PyTorch optimizer
        criterion: Loss function (PoissonNLLLoss)
        device: torch.device
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X, k, log_y, d, _ in loader:
        X = X.to(device)
        k = k.to(device)
        log_y = log_y.to(device)
        d = d.to(device)
        
        optimizer.zero_grad()
        log_mu = model(X, k, log_y)
        loss = criterion(log_mu, d)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def eval_pwe_epoch(model, loader, criterion, device):
    """
    Evaluate PWE model on validation/test set.
    
    Args:
        model: PWEPoisson model
        loader: DataLoader with PWEDataset
        criterion: Loss function (PoissonNLLLoss)
        device: torch.device
        
    Returns:
        Average loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for X, k, log_y, d, _ in loader:
            X = X.to(device)
            k = k.to(device)
            log_y = log_y.to(device)
            d = d.to(device)
            
            log_mu = model(X, k, log_y)
            loss = criterion(log_mu, d)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def get_pwe_risk_scores(model, X_np, device):
    """
    Get risk scores from PWE model (β^T x).
    
    Args:
        model: Trained PWEPoisson model
        X_np: numpy array (n, p) of covariates
        device: torch.device
        
    Returns:
        risk_scores: numpy array (n,) of risk scores
    """
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        risk = model.beta_layer(X_t).squeeze(-1)  # (n,)
    return risk.cpu().numpy()
