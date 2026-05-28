import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PWEDataExpander(BaseEstimator, TransformerMixin):
    """
    Piecewise Exponential (PWE) data transformer for survival analysis.
    
    Expands survival data into interval format suitable for Poisson regression.
    Each subject contributes one row per time interval they are at risk.
    
    Args:
        breaks: Array of time breakpoints defining intervals (e.g., [0, 1, 2, ..., 10])
    """
    
    def __init__(self, breaks):
        self.breaks = np.array(breaks)
    
    def fit(self, X, y=None):
        """Fit method (does nothing, included for sklearn compatibility)."""
        return self
    
    def transform(self, X, time, event):
        """
        Transform survival data to PWE format.
        
        Args:
            X: np.array (n, p) - Matrix of preprocessed covariates
            time: np.array (n,) - Survival/censoring times in days
            event: np.array (n,) - Event indicators (1=event, 0=censored)
            
        Returns:
            df_pwe: DataFrame with columns:
                - 'id': Patient ID
                - 'y': Exposure time in interval
                - 'd': Event indicator in interval
                - 'k': Interval index (0, ..., K-1)
                - 'x0', 'x1', ... 'x_{p-1}': Covariates
        """
        return self._expand_to_pwe(X, time, event)
    
    def _expand_to_pwe(self, X, time, event):
        """Internal method to perform the PWE expansion."""
        n, p = X.shape
        K = len(self.breaks) - 1
        rows = []
        
        for i in range(n):
            t_i = time[i]
            e_i = event[i]
            x_i = X[i, :]
            
            for k in range(K):
                start = self.breaks[k]
                end = self.breaks[k + 1]
                
                # If patient exits before this interval, stop
                if t_i <= start:
                    break
                
                # Time at risk in this interval
                y_ik = min(t_i, end) - start
                if y_ik <= 0:
                    continue
                
                # Event occurs in this interval only if:
                # - event happened (e_i == 1)
                # - time falls within (start, end]
                d_ik = 1 if (e_i == 1 and start < t_i <= end) else 0
                
                row = {
                    "id": i,
                    "y": y_ik,
                    "d": d_ik,
                    "k": k,
                }
                
                # Add covariates as x0, x1, ...
                for j in range(p):
                    row[f"x{j}"] = x_i[j]
                
                rows.append(row)
        
        df_pwe = pd.DataFrame(rows)
        return df_pwe
