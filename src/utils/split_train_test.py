import pandas as pd
import os
from sklearn.model_selection import train_test_split

def prepare_and_split_data(raw_data_dir, base_interim_dir, experiment_name, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_seed=42):
    print(f"==> Iniciando Fase 1: Split de Datos (Experimento: {experiment_name})...")
    
    # Directorio aislado para este experimento
    output_dir = os.path.join(base_interim_dir, experiment_name)
    
    # 1. Leer Datos
    features_path = os.path.join(raw_data_dir, 'flat_features.csv')
    labels_path = os.path.join(raw_data_dir, 'labels.csv')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"No se encontraron los archivos en {raw_data_dir}")
        
    print("Leyendo features y labels...")
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    
    # 2. Merge de Datos (Inner Join por patientunitstayid)
    print("Fusionando datos por 'patientunitstayid'...")
    df_merged = pd.merge(features, labels, on='patientunitstayid', how='inner')
    
    print(f"Total pacientes tras fusión: {len(df_merged)}")
    
    # Asegurar que no hay duplicados de patientunitstayid
    df_merged = df_merged.drop_duplicates(subset=['patientunitstayid'])
    
    # 3. Stratified Split
    print("Realizando train/val/test split estratificado...")
    # Primero separamos Train y (Val + Test)
    val_test_frac = val_frac + test_frac
    
    train_df, val_test_df = train_test_split(
        df_merged,
        test_size=val_test_frac,
        random_state=random_seed,
        stratify=df_merged['actualhospitalmortality']
    )
    
    # Ahora separamos Val y Test
    # test_size relativo al conjunto (Val + Test)
    rel_test_frac = test_frac / val_test_frac
    
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=rel_test_frac,
        random_state=random_seed,
        stratify=val_test_df['actualhospitalmortality']
    )
    
    print(f"Tamaños -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("Proporciones de evento (actualhospitalmortality == 1):")
    print(f"  Train: {train_df['actualhospitalmortality'].mean():.4f}")
    print(f"  Val:   {val_df['actualhospitalmortality'].mean():.4f}")
    print(f"  Test:  {test_df['actualhospitalmortality'].mean():.4f}")
    
    # Verificando intersección vacía de paciences
    train_pats = set(train_df['patientunitstayid'])
    val_pats = set(val_df['patientunitstayid'])
    test_pats = set(test_df['patientunitstayid'])
    
    assert len(train_pats.intersection(val_pats)) == 0, "Fuga de datos: Superposición entre Train y Val"
    assert len(train_pats.intersection(test_pats)) == 0, "Fuga de datos: Superposición entre Train y Test"
    assert len(val_pats.intersection(test_pats)) == 0, "Fuga de datos: Superposición entre Val y Test"
    
    # 4. Guardar Resultados
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Guardando particiones en {output_dir}...")
    train_df.to_csv(os.path.join(output_dir, 'data_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'data_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'data_test.csv'), index=False)
    
    print("==> Split finalizado con éxito.")

if __name__ == "__main__":
    # Rutas relativas asumiendo ejecución desde Tree Root
    RAW_DIR = os.path.join("data", "processed", "mimic_extraction") 
    INTERIM_DIR = os.path.join("data", "interim")
    EXPERIMENT = "manual_baseline"
    
    prepare_and_split_data(
        raw_data_dir=RAW_DIR,
        base_interim_dir=INTERIM_DIR,
        experiment_name=EXPERIMENT,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        random_seed=42
    )
