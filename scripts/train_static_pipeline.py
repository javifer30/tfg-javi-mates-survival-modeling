import itertools
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importaciones locales
from src.utils.logger import get_logger
from src.models.cox import CoxPHModel
from src.evaluation.metrics import calculate_concordance_index
from src.models.deepsurv_model import DeepSurvModel
from src.models.pwe_model import PWEPoissonModel
from src.models.rsf_model import RSFModel

# Instanciar logger global
logger = get_logger("train_static_pipeline")

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def evaluate_pipeline(model, df_train, df_val, df_test, duration_col, event_col):
    """
    Realiza la evaluación del pipeline completo (Train, Val, Test)
    """
    logger.info("==========================================================")
    logger.info(" EVALUACIÓN DEL RENDIMIENTO (C-INDEX)")
    logger.info("==========================================================")
    
    metrics = {}
    
    for df, name in zip([df_train, df_val, df_test], ["TRAINING", "VALIDATION", "TEST"]):
        logger.info(f"Evaluando sobre conjunto de {name}...")
        T = df[duration_col]
        E = df[event_col]
        
        cols_to_drop = [duration_col, event_col, "patientunitstayid"]
        X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Predicción de riesgo
        risk_scores = model.predict_risk(X)
        
        # Calcular C-index (Desacoplado vía src/evaluation/metrics.py)
        c_index = calculate_concordance_index(T, E, risk_scores)
        
        logger.info(f"==> {name} C-index: {c_index:.4f}")
        metrics[name] = c_index
        
    return metrics

def run_grid_search(df_train, df_val, event_col, duration_col, model_cfg):
    """
    Aplica Grid Search manual para combinaciones de hiperparámetros pasados como listas.
    """
    logger.info("Iniciando Búsqueda de Hiperparámetros (Grid Search)...")
    
    # Manejar nulos para CoxPH en este baseline estático
    df_train_clean = df_train.dropna()
    df_val_clean = df_val.dropna()
    
    cols_to_drop = [event_col, duration_col, "patientunitstayid"]
    X_train = df_train_clean.drop(columns=[c for c in cols_to_drop if c in df_train_clean.columns])
    y_train = df_train_clean[[duration_col, event_col]]
    
    X_val = df_val_clean.drop(columns=[c for c in cols_to_drop if c in df_val_clean.columns])
    
    # Extraer posibles combinaciones del YAML
    hyperparams = model_cfg.get('hyperparameters', {})
    
    # Aseguramos que todos sean listas para el itertools
    for k, v in hyperparams.items():
        if not isinstance(v, list):
            hyperparams[k] = [v]
            
    keys, values = zip(*hyperparams.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"Se evaluarán {len(combinations)} combinaciones de hiperparámetros.")
    
    best_c_index = 0
    best_params = None
    best_model = None
    
    for params in combinations:
        model_name = model_cfg.get('name', 'CoxPH')
        logger.info(f"Probando configuración para {model_name}: {params}")
        
        if model_name == 'CoxPH':
            model = CoxPHModel(**params)
        elif model_name == 'DeepSurv':
            model = DeepSurvModel(**params)
        elif model_name == 'PWEPoisson':
            model = PWEPoissonModel(**params)
        elif model_name == 'RandomSurvivalForest':
            model = RSFModel(**params)
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
            

        try:
            model.fit(X_train, y_train, event_col=event_col, duration_col=duration_col)
            
            # Evaluar en validación
            risk_val = model.predict_risk(X_val)
            c_index_val = calculate_concordance_index(df_val_clean[duration_col], df_val_clean[event_col], risk_val)
            
            logger.info(f" -> Val C-index: {c_index_val:.4f}")
            
            if best_model is None or c_index_val > best_c_index:
                best_c_index = c_index_val
                best_params = params
                best_model = model
        except Exception as e:
            logger.warning(f"Configuración {params} falló por: {str(e)}")
            
    if best_model is None:
        raise RuntimeError(f"Todas las configuraciones fallaron para el modelo {model_name}.")
        
    logger.info(f"==> Mejor Modelo Encontrado con Val C-index: {best_c_index:.4f} usando {best_params}")
    return best_model, best_params

def main():
    logger.info("==========================================================")
    logger.info(" ORQUESTADOR CENTRAL DE FLUJO DE TRABAJO ESTÁTICO ")
    logger.info("==========================================================")
    
    # 0. Configuración
    config_path = PROJECT_ROOT / "configs" / "train.yaml"
    
    config = load_config(config_path)
    flow_cfg = config['pipeline_flow']
    data_cfg = config['data']
    target_cfg = config['target']
    models_list = config.get('models', [])
    
    experiment_name = flow_cfg['experiment_name']
    logger.info(f"Experimento en curso: {experiment_name}")
    
    # Ruteo Relativo Base
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]
    interim_dir = PROJECT_ROOT / "data" / "interim"
    processed_dir = PROJECT_ROOT / "data" / "processed" / "model_input"
    
    # 1. FASE 1: Data Split (Opcional)
    if flow_cfg['runtimes']['run_data_split']:
        from src.utils.split_train_test import prepare_and_split_data
        split_kwargs = data_cfg['split_config']
        
        prepare_and_split_data(
            raw_data_dir=raw_dir,
            base_interim_dir=interim_dir,
            experiment_name=experiment_name,
            **split_kwargs
        )
        
    # 2. FASE 2: Feature Engineering (Opcional)
    if flow_cfg['runtimes']['run_feature_engineering']:
        from src.features.build_features import run_feature_engineering
        run_feature_engineering(PROJECT_ROOT, interim_dir, processed_dir, experiment_name)
        
    # 3. FASE 3 & 4: Entrenar y Evaluar Modelo
    exp_processed_dir = processed_dir / experiment_name
    logger.info(f"Cargando matrices matemáticas desde: {exp_processed_dir}")
    
    df_train = pd.read_csv(exp_processed_dir / "X_train_processed.csv")
    df_val = pd.read_csv(exp_processed_dir / "X_val_processed.csv")
    df_test = pd.read_csv(exp_processed_dir / "X_test_processed.csv")
    
    event_col = target_cfg['event_col']
    duration_col = target_cfg['duration_col']
    
    # 3.1. Iterar sobre todos los modelos definidos en YAML
    for model_cfg in models_list:
        model_name = model_cfg.get('name', 'UnknownModel')
        logger.info(f"--- Iniciando entrenamiento para el modelo: {model_name} ---")
        
        if flow_cfg['runtimes'].get('run_hyperparameter_tuning', False):
            trained_model, best_params = run_grid_search(df_train, df_val, event_col, duration_col, model_cfg)
        else:
            logger.info(f"Entrenamiento estático directo (Sin GridSearch) para {model_name}...")
            # Preparamos hiperparámetros asumiendo escalares directos, no listas.
            hyperparams = model_cfg.get('hyperparameters', {})
            param_kwargs = {k: (v[0] if isinstance(v, list) else v) for k, v in hyperparams.items()}
            
            if model_name == 'CoxPH':
                trained_model = CoxPHModel(**param_kwargs)
            elif model_name == 'DeepSurv':
                trained_model = DeepSurvModel(**param_kwargs)
            elif model_name == 'PWEPoisson':
                trained_model = PWEPoissonModel(**param_kwargs)
            elif model_name == 'RandomSurvivalForest':
                trained_model = RSFModel(**param_kwargs)
            else:
                raise ValueError(f"Modelo no soportado: {model_name}")
            
            df_train_clean = df_train.dropna()
            cols_to_drop = [event_col, duration_col, "patientunitstayid"]
            X_train = df_train_clean.drop(columns=[c for c in cols_to_drop if c in df_train_clean.columns])
            y_train = df_train_clean[[duration_col, event_col]]
            
            trained_model.fit(X_train, y_train, event_col=event_col, duration_col=duration_col)
            
        # 4. Evaluación Activa del Modelo y Serialización
        df_train_clean = df_train.dropna()
        df_val_clean = df_val.dropna()
        df_test_clean = df_test.dropna()
        
        metrics = evaluate_pipeline(trained_model, df_train_clean, df_val_clean, df_test_clean, duration_col, event_col)
        logger.info(f"C-Index de Test para {model_name}: {metrics['TEST']:.4f}")
        
        # 5. Persistencia del Modelo Cacheado
        model_dir = PROJECT_ROOT / config.get("outputs", {}).get("models_dir", "outputs/models") / experiment_name
        model_dir.mkdir(parents=True, exist_ok=True)
        dump_path = model_dir / f"{model_name}.pkl"
        joblib.dump(trained_model, dump_path)
        logger.info(f"Modelo guardado en disco: {dump_path}\n")
        
    logger.info("Pipeline Finalizado con Éxito. Todos los modelos han sido ajustados iterativamente.")

if __name__ == "__main__":
    main()
