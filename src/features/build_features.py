import os
import pandas as pd
import yaml
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_config(config_path):
    """Carga configuración YAML externa."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def build_preprocessor_pipeline(num_cols, cat_cols):
    """
    Construye el ColumnTransformer según las reglas de Phase 2.
    """
    # 1. Pipeline Numérico: Imputación con mediana + StandardScaler
    # Usamos la mediana en lugar de 0 o la media para ser más robustos
    # frente a las distribuciones asimétricas (típicas en biometría como MIMIC).
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline Categórico: Imputación con constante 'missing' + OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Ensamblaje en ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop' # Ignoramos cualquier columna no especificada explícitamente en config
    )
    
    return preprocessor

def run_feature_engineering(base_dir, interim_dir, processed_dir, experiment_name):
    print(f"==> Iniciando Fase 2: Feature Engineering (Experimento: {experiment_name})...")
    
    # Rutas dinámicas aisladas por experimento
    exp_interim_dir = os.path.join(interim_dir, experiment_name)
    exp_processed_dir = os.path.join(processed_dir, experiment_name)
    config_path = os.path.join(base_dir, 'configs', 'features.yaml')
    
    os.makedirs(exp_processed_dir, exist_ok=True)
    
    # 1. Cargar Configuración
    print(f"Cargando configuración de características desde {config_path}")
    config = load_config(config_path)['features']
    num_cols = config['numerical']
    cat_cols = config['categorical']
    ignore_cols = config['ignore_cols']
    
    # 2. Cargar Datos (Fase 1 outputs)
    print(f"Cargando conjuntos de datos desde {exp_interim_dir}...")
    df_train = pd.read_csv(os.path.join(exp_interim_dir, 'data_train.csv'))
    df_val = pd.read_csv(os.path.join(exp_interim_dir, 'data_val.csv'))
    df_test = pd.read_csv(os.path.join(exp_interim_dir, 'data_test.csv'))
    
    # 3. Construir Pipeline
    print("Construyendo Pipeline Maestra...")
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    
    # 4. Ajustar (.fit) SOLO en Train
    print("Ajustando pipeline en Train (.fit_transform)...")
    X_train_processed = preprocessor.fit_transform(df_train)
    
    # Recuperar nombres de columnas resultantes (especialmente las dummy del OHE)
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(cat_feature_names)
    
    # Convertir a DataFrames de Pandas puramente numéricos
    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=df_train.index)
    
    # 5. Transformar Val y Test (.transform)
    print("Transformando Validation y Test (.transform)...")
    X_val_processed = preprocessor.transform(df_val)
    X_test_processed = preprocessor.transform(df_test)
    
    X_val_df = pd.DataFrame(X_val_processed, columns=all_feature_names, index=df_val.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=df_test.index)
    
    # Extraer variables target e identificadores para no perder el cruce
    # Queremos guardar las features X junto con el patientunitstayid y las labels para el pipeline de modelado final
    print("Guardando matrices procesadas y modelo preprocesador...")
    
    def merge_and_save(X_processed_df, original_df, filename):
        ret_df = X_processed_df.copy()
        # Añadir targets y IDs para la capa de Modelos
        ret_df['patientunitstayid'] = original_df['patientunitstayid']
        ret_df['actualhospitalmortality'] = original_df['actualhospitalmortality']
        ret_df['actualiculos'] = original_df['actualiculos']
        
        # Mover IDs y Targets al principio por limpieza visual
        cols = ['patientunitstayid', 'actualhospitalmortality', 'actualiculos'] + all_feature_names
        ret_df = ret_df[cols]
        
        ret_df.to_csv(os.path.join(exp_processed_dir, filename), index=False)
    
    merge_and_save(X_train_df, df_train, 'X_train_processed.csv')
    merge_and_save(X_val_df, df_val, 'X_val_processed.csv')
    merge_and_save(X_test_df, df_test, 'X_test_processed.csv')
    
    # Opcional: Guardar el objeto transformador para el futuro (despliegue)
    models_dir = os.path.join(base_dir, 'outputs', 'models', 'preprocessors', experiment_name)
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(models_dir, 'pipeline_preprocesamiento.pkl'))
    
    print(f"Archivos guardados en: {exp_processed_dir}")
    print(f"Dimensiones finales -> Train: {X_train_df.shape}, Val: {X_val_df.shape}, Test: {X_test_df.shape}")
    print("==> Fase 2 completada con éxito.")

if __name__ == "__main__":
    # Test local rápido fallback
    bd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_feature_engineering(bd, os.path.join(bd, 'data', 'interim'), os.path.join(bd, 'data', 'processed', 'model_input'), 'manual_baseline')
