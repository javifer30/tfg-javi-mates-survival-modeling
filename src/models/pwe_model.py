import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseSurvivalModel
from src.features.pwe_transformer import PWEDataExpander
from src.models.pwe_poisson import PWEPoisson, PWEDataset, train_pwe_epoch, get_pwe_risk_scores

class PWEPoissonModel(BaseSurvivalModel):
    """
    Wrapper model for Piecewise Exponential (PWE) Poisson implementation.
    """
    def __init__(self, breaks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                 learning_rate=0.001, epochs=20, batch_size_train=1024, 
                 device=None, **kwargs):
        self.breaks = breaks
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size_train = batch_size_train
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y, event_col="actualhospitalmortality", duration_col="actualiculos", **kwargs):
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        if isinstance(y, pd.DataFrame):
            time_np = y[duration_col].values
            event_np = y[event_col].values
        else:
            time_np = y[:, 0]
            event_np = y[:, 1]
            
        # 1. Expand standard wide data locally using PWEDataExpander
        expander = PWEDataExpander(breaks=self.breaks)
        df_pwe = expander.transform(X_np, time_np, event_np)
        
        # 2. Configure PyTorch Model
        p_features = X_np.shape[1]
        K_intervals = len(self.breaks) - 1
        self.model = PWEPoisson(p=p_features, K=K_intervals).to(self.device)
        
        # 3. Prepare Dataset and DataLoader
        dataset = PWEDataset(df_pwe, p=p_features)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_train, shuffle=True)
        
        # 4. Optimizer and Loss Function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.PoissonNLLLoss(log_input=True)
        
        # 5. Train Bucle (invoking pre-built functions)
        for epoch in range(self.epochs):
            train_pwe_epoch(self.model, dataloader, optimizer, criterion, self.device)
            
        return self

    def predict_risk(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained. Please call fit() first.")
            
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        # Call the existing helper function
        return get_pwe_risk_scores(self.model, X_np, self.device)

    def predict_survival_function(self, X):
        raise NotImplementedError("Baseline survival function is not needed yet and not implemented.")
