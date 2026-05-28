from abc import ABC, abstractmethod

class BaseSurvivalModel(ABC):
    """
    Clase abstracta base para todos los modelos de supervivencia.
    Asegura que cualquier modelo (Cox, ML, Deep Learning) cumpla con
    una interfaz común ("Contrato" de Liskov en principios SOLID).
    """
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Ajusta el modelo a los datos de entrenamiento.
        
        Args:
            X (pd.DataFrame o np.ndarray): Matriz de características (features).
            y (pd.DataFrame o estructurado): Targets (evento y tiempo).
            **kwargs: Parámetros adicionales (e.g., pesos de muestra).
        """
        pass
        
    @abstractmethod
    def predict_risk(self, X):
        """
        Predice el riesgo relativo para cada instancia.
        """
        pass
        
    @abstractmethod
    def predict_survival_function(self, X):
        """
        Predice las probabilidades de supervivencia en el tiempo P(T > t).
        """
        pass
