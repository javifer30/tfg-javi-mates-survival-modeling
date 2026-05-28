import glob
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
import warnings

# Para evitar prints masivos por librerías al predecir
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.evaluation.metrics import calculate_concordance_index

logger = get_logger("evaluate_models")

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    logger.info("==========================================================")
    logger.info("          EVALUADOR COMPARATIVO DE MODELOS (TEST)         ")
    logger.info("==========================================================")
    
    # 1. Configuración de Directorios
    config_path = PROJECT_ROOT / "configs" / "train.yaml"
    
    config = load_config(config_path)
    experiment_name = config['pipeline_flow']['experiment_name']
    target_cfg = config['target']
    
    duration_col = target_cfg['duration_col']
    event_col = target_cfg['event_col']
    
    model_dir = PROJECT_ROOT / config.get("outputs", {}).get("models_dir", "outputs/models") / experiment_name
    processed_dir = PROJECT_ROOT / "data" / "processed" / "model_input" / experiment_name
    test_csv_path = processed_dir / "X_test_processed.csv"
    
    # Comprobaciones pre-vuelo
    if not test_csv_path.exists():
        logger.error(f"No se encontró el conjunto de Test en: {test_csv_path}")
        return
        
    if not model_dir.exists():
        logger.error(f"No se encontró la carpeta de modelos guardados para el experimento: {model_dir}")
        return
        
    # 2. Cargar Dataset de Test
    logger.info(f"Cargando conjunto de Test: {test_csv_path}")
    df_test = pd.read_csv(test_csv_path).dropna()
    
    cols_to_drop = [event_col, duration_col, "patientunitstayid"]
    X_test = df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns])
    T_test = df_test[duration_col]
    E_test = df_test[event_col]
    
    # 3. Detectar todos los modelos cacheados para el experimento
    pkl_files = glob.glob(str(model_dir / "*.pkl"))
    if not pkl_files:
        logger.error("No se detectó ningún modelo `.pkl` en el directorio.")
        return
        
    logger.info(f"Se han detectado {len(pkl_files)} modelos pre-entrenados.")
    
    results = []
    
    # 4. Bucle de Ingesta, Predicción y Benchmark
    for pkl_file in pkl_files:
        model_name = Path(pkl_file).stem
        logger.info(f" -> Evaluando {model_name}...")
        
        try:
            model = joblib.load(pkl_file)
            risk_scores = model.predict_risk(X_test)
            c_index = calculate_concordance_index(T_test, E_test, risk_scores)
            
            results.append({"model": model_name, "test_c_index": round(c_index, 4)})
            
        except Exception as e:
            logger.error(f"Fallo al evaluar el modelo {model_name}: {e}")
            
    # 5. Generar y mostrar Tabla Ranking
    if results:
        df_results = pd.DataFrame(results)
        # Ordenar de mejor a peor (C-Index más alto es mejor)
        df_results = df_results.sort_values(by="test_c_index", ascending=False).reset_index(drop=True)
        
        # Rankings terminal
        logger.info("\n" + "="*50)
        logger.info("            RANKING FINAL DEL EXPERIMENTO")
        logger.info("="*50)
        for idx, row in df_results.iterrows():
            logger.info(f"#{idx+1} | {row['model']:<22} | C-index: {row['test_c_index']:.4f}")
        logger.info("="*50)
        
        # Persistencia del reporte
        reports_dir = PROJECT_ROOT / config.get("outputs", {}).get("metrics_dir", "outputs/metrics")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{experiment_name}_benchmark.csv"
        df_results.to_csv(report_path, index=False)
        logger.info(f"\nReporte tabulado guardado en: {report_path}")

if __name__ == "__main__":
    main()
