"""
Métricas útiles para modelos de supervivencia.
Puntos de entrada recomendados: concordance_index_censored de lifelines o scikit-survival.
"""

import pandas as pd
from lifelines.utils import concordance_index

def calculate_concordance_index(y_time, y_event, risk_scores):
    """
    Calcula el Índice de Concordancia (C-index) de Harrell.
    
    Args:
        y_time (array-like): Tiempos de seguimiento reales.
        y_event (array-like): Indicadores de evento (1=ocurrió, 0=censurado).
        risk_scores (array-like): Predicciones del modelo (mayor valor = mayor riesgo).
        
    Returns:
        float: Valor del Concordance Index.
    """
    # Lifelines asume que risk_scores es "tiempo esperado de vida" para C-index.
    # Dado que nuestros modelos (como Cox) devuelven el log-hazard (mayor riesgo),
    # debemos invertir la polaridad del riesgo (-risk) para usar esta métrica matemáticamente.
    return concordance_index(y_time, -risk_scores, y_event)
