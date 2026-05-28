import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest

from src.models.base import BaseSurvivalModel

class RSFModel(BaseSurvivalModel):
    """
    Wrapper for RandomSurvivalForest using scikit-survival.
    Automatically handles the structured array format required by sksurv for targets.
    """
    def __init__(self, n_estimators=100, min_samples_split=10, min_samples_leaf=3,
                 max_depth=None, n_jobs=-1, random_state=42, **kwargs):
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        
    def _prepare_sksurv_y(self, y, event_col, duration_col):
        """
        Converts a Pandas DataFrame or Numpy Array to the nested structured array
        format: [(event_bool, time_float), ...] required by scikit-survival.
        """
        if isinstance(y, pd.DataFrame):
            # Extract lists
            events = y[event_col].values.astype(bool)
            times = y[duration_col].values.astype(float)
        else:
            # Assume numpy array with shape (N, 2) where col 0 is time, col 1 is event
            events = y[:, 1].astype(bool)
            times = y[:, 0].astype(float)
            
        # Create numpy structured array
        y_structured = np.array(list(zip(events, times)), 
                                dtype=[('Event', '?'), ('Time', '<f8')])
        return y_structured

    def fit(self, X, y, event_col="actualhospitalmortality", duration_col="actualiculos", **kwargs):
        # Enforce Pandas DataFrame just in case for features
        if isinstance(X, pd.DataFrame):
            X_data = X
        else:
            X_data = pd.DataFrame(X)
            
        # Transform y to scikit-survival format
        y_structured = self._prepare_sksurv_y(y, event_col, duration_col)
        
        self.model.fit(X_data, y_structured)
        return self

    def predict_risk(self, X):
        """
        Returns the risk sum (cumulative hazard). Higher means higher relative risk.
        """
        if isinstance(X, pd.DataFrame):
            X_data = X
        else:
            X_data = pd.DataFrame(X)
            
        return self.model.predict(X_data)

    def predict_survival_function(self, X):
        """
        scikit-survival does support returning a matrix of survival probabilities.
        To maintain current compatibility guidelines across PyTorch models, we raise
        NotImplementedError or we could bypass it and actually return it.
        We stick to the guide for now.
        """
        raise NotImplementedError("Baseline survival function is bypassed as per guidelines.")
