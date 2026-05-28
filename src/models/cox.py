import pandas as pd
from lifelines import CoxPHFitter
from .base import BaseSurvivalModel

class CoxPHModel(BaseSurvivalModel):
    """
    Implementación concreta (Wrapper) del modelo de Riesgos Proporcionales de Cox
    utilizando la librería 'lifelines', acatando la interfaz BaseSurvivalModel.
    """
    def __init__(self, penalizer=0.0):
        self.penalizer = penalizer
        # Inicializamos el modelo de lifelines
        self.model = CoxPHFitter(penalizer=self.penalizer)
        
    def fit(self, X, y, event_col="actualhospitalmortality", duration_col="actualiculos", **kwargs):
        """
        Ajusta el CoxPHFitter. 
        Lifelines requiere que las features y los targets vivan en el mismo DataFrame.
        """
        # Aseguramos que X es DataFrame para concatenar limpiamente
        if not isinstance(X, pd.DataFrame):
            # Si fuesen arrays de numpy, los metemos a un DataFrame simple
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=[duration_col, event_col])
            
        df_train = pd.concat([X, y], axis=1)
        
        # Ajustar cuidando no re-nombrar variables dummy por error etc.
        self.model.fit(df_train, duration_col=duration_col, event_col=event_col, **kwargs)
        return self
        
    def predict_risk(self, X):
        """
        Devuelve el log-hazard ratio predictivo. Valores más altos implican mayor riesgo.
        """
        return self.model.predict_partial_hazard(X)
        
    def predict_survival_function(self, X):
        """
        Devuelve P(T > t).
        Retorna por defecto un DataFrame de shape (tiempo x pacientes).
        """
        return self.model.predict_survival_function(X)
