import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.base import BaseSurvivalModel
from src.models.deepsurv import DeepSurvNet, CoxDataset, cox_ph_loss

class DeepSurvModel(BaseSurvivalModel):
    """
    Wrapper model for DeepSurv PyTorch implementation to match BaseSurvivalModel.
    """
    def __init__(self, in_features=None, hidden_layers=[64, 32], dropout=0.1, 
                 learning_rate=0.001, epochs=50, batch_size=256, device=None, **kwargs):
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y, event_col="actualhospitalmortality", duration_col="actualiculos", **kwargs):
        # Ensure NumPy data
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        if isinstance(y, pd.DataFrame):
            # Extract target vectors safely
            time_np = y[duration_col].values
            event_np = y[event_col].values
        else:
            # Fallback assumption if tuples or 2D numpy arrays
            time_np = y[:, 0]
            event_np = y[:, 1]
            
        self.in_features = X_np.shape[1]
        
        # Instantiate base NN from deepsurv.py
        self.model = DeepSurvNet(in_features=self.in_features, 
                                 hidden=self.hidden_layers, 
                                 dropout=self.dropout).to(self.device)
        
        dataset = CoxDataset(X_np, time_np, event_np)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_time, batch_event in dataloader:
                batch_X = batch_X.to(self.device)
                batch_time = batch_time.to(self.device)
                batch_event = batch_event.to(self.device)
                
                optimizer.zero_grad()
                risk_pred = self.model(batch_X)
                loss = cox_ph_loss(risk_pred, batch_time, batch_event)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_risk(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained. Please call fit() first.")
            
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
            
        self.model.eval()
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            risk = self.model(X_tensor)
        return risk.cpu().numpy()

    def predict_survival_function(self, X):
        raise NotImplementedError("Baseline survival function (e.g., Breslow) is not needed yet and not implemented.")
