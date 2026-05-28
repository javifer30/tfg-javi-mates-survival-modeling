import pandas as pd
import numpy as np
import os
import gc
import sys
import json
from itertools import islice

# --- CONFIGURACIÓN DE RUTAS (Igual que antes para importar eICU) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# -------------------------------------------------------------------

# Constantes del Paper
DROP_COLS_EXACT = [
    'WBC', 'HCO3 (serum)', 'Lactic Acid', 'PH (Arterial)', 'Arterial O2 pressure', 
    'Arterial CO2 Pressure', 'Arterial Base Excess', 'TCO2 (calc) Arterial', 
    'Ionized Calcium', 'BUN', 'Calcium non-ionized', 'Anion gap',
    '18 Gauge Dressing Occlusive', '18 Gauge placed in outside facility', '18 Gauge placed in the field',
    '20 Gauge Dressing Occlusive', '20 Gauge placed in outside facility', '20 Gauge placed in the field',
    'Alarms On', 'Ambulatory aid', 'CAM-ICU MS Change', 'Eye Care', 'High risk (>51) interventions',
    'History of falling (within 3 mnths)', 'IV/Saline lock', 'Mental status', 'Parameters Checked',
    'ST Segment Monitoring On', 'Secondary diagnosis', 'Acuity Workload Question 1',
    'Acuity Workload Question 2', 'Arterial Line Dressing Occlusive', 'Arterial Line Zero/Calibrate',
    'Arterial Line placed in outside facility', 'Back Care', 'Cough/Deep Breath', 'Cuff Pressure',
    'Gait/Transferring', 'Glucose (whole blood)', 'Goal Richmond-RAS Scale', 'Inspiratory Ratio',
    'Inspiratory Time', 'Impaired Skin Odor #1', 'Multi Lumen placed in outside facility',
    'O2 Saturation Pulseoxymetry Alarm - High', 'Orientation', 'Orientation to Person',
    'Orientation to Place', 'Orientation to Time', 'Potassium (whole blood)', 'Skin Care',
    'SpO2 Desat Limit', 'Subglottal Suctioning', 'Ventilator Tank #1', 'Ventilator Tank #2', 'Ventilator Type'
]

BRADEN_COLS = ['Braden Activity', 'Braden Friction/Shear', 'Braden Mobility',
               'Braden Moisture', 'Braden Nutrition', 'Braden Sensory Perception']

# --- FUNCIONES DEL PAPER (Copiadas tal cual) ---

def reconfigure_timeseries(timeseries, offset_column, feature_column=None, test=False):
    if test:
        timeseries = timeseries.iloc[300000:5000000]
    timeseries.set_index(['patientunitstayid', pd.to_timedelta(timeseries[offset_column], unit='T')], inplace=True)
    timeseries.drop(columns=offset_column, inplace=True)
    if feature_column is not None:
        timeseries = timeseries.pivot_table(columns=feature_column, index=timeseries.index)
    # convert index to multi-index with both patients and timedelta stamp
    timeseries.index = pd.MultiIndex.from_tuples(timeseries.index, names=['patient', 'time'])
    return timeseries

def resample_and_mask(timeseries, eICU_path, header, mask_decay=False, decay_rate=4/3, test=False,
                       verbose=False, length_limit=24*14):
    if test:
        mask_decay = False
        verbose = True
    if verbose:
        print('Resampling to 1 hour intervals...')
    # take the mean of any duplicate index entries for unstacking
    timeseries = timeseries.groupby(level=[0, 1]).mean()

    timeseries.reset_index(level=1, inplace=True)
    timeseries.time = timeseries.time.dt.ceil(freq='H')
    timeseries.set_index('time', append=True, inplace=True)
    timeseries.reset_index(level=0, inplace=True)
    resampled = timeseries.groupby('patient').resample('H', closed='right', label='right').mean().drop(columns='patient')
    del (timeseries)

    def apply_mask_decay(mask_bool):
        mask = mask_bool.astype(int)
        mask.replace({0: np.nan}, inplace=True)  # so that forward fill works
        inv_mask_bool = ~mask_bool
        count_non_measurements = inv_mask_bool.cumsum() - \
                                 inv_mask_bool.cumsum().where(mask_bool).ffill().fillna(0)
        decay_mask = mask.ffill().fillna(0) / (count_non_measurements * decay_rate).replace(0, 1)
        return decay_mask

    # store which values had to be imputed
    if mask_decay:
        if verbose:
            print('Calculating mask decay features...')
        mask_bool = resampled.notnull()
        mask = mask_bool.groupby('patient').transform(apply_mask_decay)
        del (mask_bool)
    else:
        if verbose:
            print('Calculating binary mask features...')
        mask = resampled.notnull()
        mask = mask.astype(int)

    if verbose:
        print('Filling missing data forwards...')
    # carry forward missing values (note they will still be 0 in the nulls table)
    resampled = resampled.fillna(method='ffill')

    # simplify the indexes of both tables
    mask = mask.rename(index=dict(zip(mask.index.levels[1],
                                      mask.index.levels[1].days*24 + mask.index.levels[1].seconds//3600)))
    resampled = resampled.rename(index=dict(zip(resampled.index.levels[1],
                                                resampled.index.levels[1].days*24 +
                                                resampled.index.levels[1].seconds//3600)))

    # clip to length_limit
    if length_limit is not None:
        within_length_limit = resampled.index.get_level_values(1) < length_limit
        resampled = resampled.loc[within_length_limit]
        mask = mask.loc[within_length_limit]

    if verbose:
        print('Filling in remaining values with zeros...')
    resampled.fillna(0, inplace=True)

    # rename the columns in pandas for the mask so it doesn't complain
    mask.columns = [str(col) + '_mask' for col in mask.columns]

    # merge the mask with the features
    final = pd.concat([resampled, mask], axis=1)
    final.reset_index(level=1, inplace=True)
    final = final.loc[final.time > 0]

    if verbose:
        print('Saving progress...')
    # save to csv
    if test is False:
        final.to_csv(eICU_path + 'preprocessed_timeseries.csv', mode='a', header=header)
    return

def gen_patient_chunk(patients, size=1000):
    it = iter(patients)
    chunk = list(islice(it, size))
    while chunk:
        yield chunk
        chunk = list(islice(it, size))


r""" def add_time_of_day(processed_timeseries, flat_features):

    print('==> Adding time of day features...')
    processed_timeseries = processed_timeseries.join(flat_features[['hour']], how='inner', on='patient')
    processed_timeseries['hour'] = processed_timeseries['time'] + processed_timeseries['hour']
    hour_list = np.linspace(0, 1, 24)  # make sure it's still scaled well
    processed_timeseries['hour'] = processed_timeseries['hour'].apply(lambda x: hour_list[x%24 - 24])
    return processed_timeseries  """ 


def further_processing(eICU_path, test=False):

    if test:
        processed_timeseries = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv', nrows=999999)
    else:
        processed_timeseries = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv')
    processed_timeseries.rename(columns={'Unnamed: 1': 'time'}, inplace=True)
    processed_timeseries.set_index('patient', inplace=True)
    flat_features = pd.read_csv(eICU_path + 'flat_features.csv')
    flat_features.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    processed_timeseries.sort_values(['patient', 'time'], inplace=True)
    flat_features.set_index('patient', inplace=True)

    processed_timeseries = add_time_of_day(processed_timeseries, flat_features)

    if test is False:
        print('==> Saving finalised preprocessed timeseries...')
        # this will replace old one that was updated earlier in the script
        processed_timeseries.to_csv(eICU_path + 'preprocessed_timeseries.csv')

    return

def add_time_of_day(processed_timeseries, flat_features):
    # Unimos la hora de ingreso (estática) a la serie temporal
    # Nota: flat_features debe tener la columna 'hour' con valores 0-23
    processed_timeseries = processed_timeseries.join(flat_features[['hour']], how='inner', on='patient', rsuffix='_entry')
    
    # Calculamos la hora actual: horas transcurridas + hora de ingreso
    processed_timeseries['current_hour'] = processed_timeseries['time'] + processed_timeseries['hour_entry']
    
    # Creamos una lista de 24 valores normalizados entre 0 y 1. Normalizar la hora hace que su importancia sea parecido al resto del vars
    hour_scale = np.linspace(0, 1, 24)
    
    # Aplicamos la operación módulo 24 para mantenernos en el rango de un día
    # y mapeamos al valor escalado
    processed_timeseries['time_of_day'] = processed_timeseries['current_hour'].apply(lambda x: hour_scale[int(x % 24)])
    
    # Limpiamos columnas auxiliares si es necesario
    return processed_timeseries.drop(columns=['hour_entry', 'current_hour'])

def preprocess_flat(flat):
    """
    Logic adapted from src/preprocessing/MIMIC_IV-preprocessing/flat_and_labels.py
    """
    print('    Preprocessing flat features...')
    # make naming consistent with the other tables
    flat.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    flat.set_index('patient', inplace=True)

    flat['gender'].replace({'M': 1, 'F': 0}, inplace=True)

    cat_features = ['ethnicity', 'first_careunit', 'admission_location', 'insurance']
    # get rid of any really uncommon values
    for f in cat_features:
        # iteritems() is deprecated/removed in newer pandas, using items()
        too_rare = [value for value, count in flat[f].value_counts().items() if count < 1000]
        flat.loc[flat[f].isin(too_rare), f] = 'misc'

    # convert the categorical features to one-hot
    flat = pd.get_dummies(flat, columns=cat_features)

    # note that the features imported from the time series have already been normalised
    # standardisation is for features that are probably normally distributed
    features_for_standardisation = 'height'
    # Calculate stats only on existing data to avoid NaN propagation if not handled yet
    means = flat[features_for_standardisation].mean(axis=0)
    stds = flat[features_for_standardisation].std(axis=0)
    flat[features_for_standardisation] = (flat[features_for_standardisation] - means) / stds

    # probably not normally distributed
    features_for_min_max = ['weight', 'age', 'hour', 'eyes', 'motor', 'verbal']

    def scale_min_max(flat_df):
        quantiles = flat_df.quantile([0.05, 0.95])
        maxs = quantiles.loc[0.95]
        mins = quantiles.loc[0.05]
        # Avoid division by zero
        denoms = (maxs - mins)
        return 2 * (flat_df - mins) / denoms - 1

    flat[features_for_min_max] = flat[features_for_min_max].apply(scale_min_max)

    # we then need to make sure that ridiculous outliers are clipped to something sensible
    flat[features_for_standardisation] = flat[features_for_standardisation].clip(lower=-4, upper=4)  # room for +- 3 on each side of the normal range, as variables are scaled roughly between -1 and 1
    flat[features_for_min_max] = flat[features_for_min_max].clip(lower=-4, upper=4)

    # fill in the NaNs
    # these are mainly found in height
    # so we create another variable to tell the model when this has been imputed
    flat['nullheight'] = flat['height'].isnull().astype(int)
    flat['weight'].fillna(0, inplace=True)  # null in only 83 patients
    flat['height'].fillna(0, inplace=True)  # null in 38217 patients
    flat['eyes'].fillna(0, inplace=True)  # null in 192 patients
    flat['motor'].fillna(0, inplace=True)  # null in 270 patients
    flat['verbal'].fillna(0, inplace=True)  # null in 6240 patients

    return flat

def preprocess_labels(labels):
    """
    Logic adapted from src/preprocessing/MIMIC_IV-preprocessing/flat_and_labels.py
    """
    print('    Preprocessing labels...')
    # make naming consistent with the other tables
    labels.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    labels.set_index('patient', inplace=True)

    return labels

# --- FIN FUNCIONES DEL PAPER, INICIO DE NUEVAS FUNCIONES PARA SPARSE + MERGE ---
r"""
def load_raw_data(MIMIC_path, test=False):
    # Carga y limpieza inicial (igual que antes)
    print('==> 1. Cargando datos crudos (Lab + Chart)...')
    nrows = 500000 if test else None
    
    # Lab
    df_lab = pd.read_csv(os.path.join(MIMIC_path, 'timeserieslab.csv'), 
                         usecols=['patientunitstayid', 'labresultoffset', 'labname', 'labresult'],
                         nrows=nrows)
    df_lab.rename(columns={'labresultoffset': 'offset', 'labname': 'feature', 'labresult': 'value'}, inplace=True)
    
    # Chart
    df_chart = pd.read_csv(os.path.join(MIMIC_path, 'timeseries.csv'), 
                           usecols=['patientunitstayid', 'chartoffset', 'chartvaluelabel', 'chartvalue'],
                           nrows=nrows)
    df_chart.rename(columns={'chartoffset': 'offset', 'chartvaluelabel': 'feature', 'chartvalue': 'value'}, inplace=True)
    
    # Unir
    df = pd.concat([df_lab, df_chart], ignore_index=True)
    del df_lab, df_chart
    gc.collect()

    # Limpieza Columnas
    df = df[~df['feature'].isin(DROP_COLS_EXACT)]
    
    # Braden Score
    mask_braden = df['feature'].isin(BRADEN_COLS)
    if mask_braden.any():
        df_braden = df[mask_braden].groupby(['patientunitstayid', 'offset']).sum(numeric_only=True)['value'].reset_index()
        df_braden['feature'] = 'Braden Score'
        df_braden['value'].replace(0, np.nan, inplace=True)
        df = pd.concat([df[~mask_braden], df_braden], ignore_index=True)

    # Discretizar a Horas
    df['hour'] = np.floor(df['offset'] / 60).astype(int)

        # Transformar horas negativas a 0 (puede haber valores negativos por errores de sincronización, pero no queremos perder esos datos)
    df['hour'] = np.where(df['hour'] < 0, 0, df['hour'])
    
    
    # Promediar duplicados en la misma hora
    df = df.groupby(['patientunitstayid', 'hour', 'feature'])['value'].mean().reset_index()
    
    return df

def calculate_normalization_stats(df, MIMIC_path):
    #Calcula y GUARDA las estadísticas para usarlas luego en el entrenamiento
    print('==> 2. Calculando estadísticas globales (Min/Max)...')
    # Usamos percentiles robustos como el paper
    stats = df.groupby('feature')['value'].quantile([0.05, 0.95]).unstack()
    stats.columns = ['min', 'max']
    
    # Guardamos esto en un CSV pequeño. Lo necesitaremos al entrenar para normalizar al vuelo.
    stats_path = os.path.join(MIMIC_path, 'feature_stats.csv')
    stats.to_csv(stats_path)
    print(f'    Stats guardados en {stats_path}')
    return stats

def process_and_save_sparse(df, stats, df_flat, df_labels, MIMIC_path):
    
    #Versión V3: Vectorización total + Merge con Flat/Labels
    output_dir = os.path.join(MIMIC_path, 'timeseries_parquet_complete')
    os.makedirs(output_dir, exist_ok=True)
    
    unique_patients = df['patientunitstayid'].unique()
    CHUNK_SIZE = 2000
    
    print(f'==> 4. Procesando {len(unique_patients)} pacientes en formato COMPLETAMENTE UNIDO (Parquet)...')
    print(f'       Guardando en: {output_dir}')
    
    # Bucle EXTERNO: Indispensable para no saturar la RAM (Batch Processing)
    for i in range(0, len(unique_patients), CHUNK_SIZE):
        batch_ids = unique_patients[i : i + CHUNK_SIZE]
        
        # 1. Filtrado rápido
        df_chunk = df[df['patientunitstayid'].isin(batch_ids)].copy()
        
        # 2. Pivot (Wide Format)
        df_wide = df_chunk.pivot_table(index=['patientunitstayid', 'hour'], columns='feature', values='value')
        
        # 3. Relleno de huecos (Vectorizado sobre índices)
        df_wide = df_wide.groupby(level=0).ffill()
        df_wide = df_wide.groupby(level=0).bfill()
        
        # 4. Normalización Time Series (Broadcasting)
        
        # A. Alineamos las estadísticas con las columnas que existen en este chunk
        cols_to_norm = df_wide.columns.intersection(stats.index)
        
        if len(cols_to_norm) > 0:
            # Seleccionamos solo las stats relevantes
            chunk_stats = stats.loc[cols_to_norm]
            mins = chunk_stats['min']
            maxs = chunk_stats['max']
            denoms = (maxs - mins).replace(0, 1) # Evitar división por cero
            
            # B. Operación Matricial Masiva
            df_subset = df_wide[cols_to_norm]
            df_normalized = 2 * (df_subset - mins) / denoms - 1
            
            # C. Aplicamos Clip y guardamos de vuelta
            df_wide[cols_to_norm] = df_normalized.clip(-4, 4)
        
        df_wide = df_wide.fillna(0)
        
        # 5. MERGE CON FLAT y LABELS
        # Reseteamos index para tener 'patientunitstayid' como columna y poder hacer merge
        df_merged = df_wide.reset_index()
        
        # Merge Flat Features
        # df_flat index name is 'patient', df_merged col is 'patientunitstayid'
        df_merged = df_merged.merge(df_flat, left_on='patientunitstayid', right_index=True, how='left')
        
        # Merge Labels
        # df_labels index name is 'patient'
        df_merged = df_merged.merge(df_labels, left_on='patientunitstayid', right_index=True, how='left')
        
        # 6. IMPORTANTE: Guardar en Parquet (mucho más eficiente que CSV para datos grandes)
        fname = os.path.join(output_dir, f'batch_{i}.parquet')
        df_merged.to_parquet(fname, index=False, engine='pyarrow', compression='snappy')
        
        print(f'    Batch {i} guardado. ({len(batch_ids)} pacientes)')
        gc.collect()

def main_sparse(MIMIC_path):
    # 1. Cargar todo en memoria (formato largo es ligero)
    df = load_raw_data(MIMIC_path)
    
    # 2. Calcular stats
    stats = calculate_normalization_stats(df, MIMIC_path)
    
    # 3. Cargar y Preprocesar Flat y Labels
    print('==> 3. Cargando y preprocesando Flat Features y Labels...')
    
    # Cargar CSVs
    flat = pd.read_csv(os.path.join(MIMIC_path, 'flat_features.csv'))
    labels = pd.read_csv(os.path.join(MIMIC_path, 'labels.csv'))
    
    # Preprocesar
    flat = preprocess_flat(flat)
    labels = preprocess_labels(labels)
    
    # 4. Guardar en Parquet eficiente con TODO unido
    process_and_save_sparse(df, stats, flat, labels, MIMIC_path)
    
    print("==> ¡Hecho! Datos guardados en carpeta 'timeseries_parquet_complete'.")

if __name__ == '__main__':
    # Determine project root dynamically (assumes this script is in src/data/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Target path
    MIMIC_PATH = os.path.join(project_root, "data", "processed", "mimic_extraction")
    
    # Ensure directory exists before executing manually
    if not os.path.exists(MIMIC_PATH):
        print(f"Directory {MIMIC_PATH} does not exist. Please run mimic_direct_extraction first.")
    else:
        main_sparse(MIMIC_PATH)

"""
def process_and_save_complete(df_lab, df_chart, df_flat, df_labels, output_path, test=False):
    output_dir = os.path.join(output_path, 'timeseries_parquet_complete')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Reconfiguración inicial (Lógica Paper)
    print('==> Reconfiguring timeseries...')
    ts_lab = reconfigure_timeseries(df_lab, 'labresultoffset', 'labname', test=test)
    ts_lab.columns = ts_lab.columns.droplevel()
    
    ts_chart = reconfigure_timeseries(df_chart, 'chartoffset', 'chartvaluelabel', test=test)
    ts_chart.columns = ts_chart.columns.droplevel()

    # Limpieza de duplicados entre lab y chart (Lógica Paper)
    cols_to_drop = DROP_COLS_EXACT
    ts_chart.drop(columns=cols_to_drop, inplace=True)

    # Braden Score (Lógica Paper)
    braden_cols = ['Braden Activity', 'Braden Friction/Shear', 'Braden Mobility', 
                   'Braden Moisture', 'Braden Nutrition', 'Braden Sensory Perception']
    existing_braden = [c for c in braden_cols if c in ts_chart.columns]
    if existing_braden:
        ts_chart['Braden Score'] = ts_chart[existing_braden].sum(axis=1)
        ts_chart['Braden Score'].replace(0, np.nan, inplace=True)
        ts_chart.drop(columns=existing_braden, inplace=True)

    # 2. Procesamiento por Chunks (Eficiencia CPU/RAM)
    all_patients = ts_chart.index.unique(level=0)
    size = 2000
    print(f'==> Processing {len(all_patients)} patients in chunks of {size}...')

    header_stats = True
    mins, maxs = None, None

    for i, patient_chunk in enumerate(gen_patient_chunk(all_patients, size)):
        # Merge temporal de las dos fuentes
        merged = pd.concat([ts_lab.loc[patient_chunk], ts_chart.loc[patient_chunk]], sort=False)
        
        # 3. Normalización y Clipping (Lógica Paper)
        if i == 0:
            quantiles = merged.quantile([0.05, 0.95])
            maxs = quantiles.loc[0.95]
            mins = quantiles.loc[0.05]
            # Guardar stats para el modelo
            pd.concat([mins, maxs], axis=1, keys=['min', 'max']).to_csv(os.path.join(output_path, 'feature_stats.csv'))

        merged = 2 * (merged - mins) / (maxs - mins) - 1
        merged.clip(lower=-4, upper=4, inplace=True)

        # 4. Resampling, Masking e Imputación (Lógica Paper)
        # Nota: Aquí es donde ocurre el ffill y el cálculo de la máscara
        chunk_final = resample_and_mask(merged, mask_decay=False)

        # 5. Aplicar la función de hora del día
        chunk_final = add_time_of_day(chunk_final, df_flat)

        # 6. Merge con Flat y Labels (Tu estructura original)
        chunk_final = chunk_final.merge(df_flat, left_on='patient', right_index=True, how='left')
        chunk_final = chunk_final.merge(df_labels, left_on='patient', right_index=True, how='left')

        # 7. Guardado en Parquet
        fname = os.path.join(output_dir, f'batch_{i}.parquet')
        chunk_final.to_parquet(fname, index=False, engine='pyarrow', compression='snappy')
        
        print(f'    Batch {i} procesado ({len(patient_chunk)} pacientes).')
        gc.collect()

def main(path, test=True):
    # Carga de datos crudos
    nrows = 500000 if test else None
    df_lab = pd.read_csv(os.path.join(path, 'timeserieslab.csv'), nrows=nrows)
    df_chart = pd.read_csv(os.path.join(path, 'timeseries.csv'), nrows=nrows)
    flat = pd.read_csv(os.path.join(path, 'flat_features.csv'))
    labels = pd.read_csv(os.path.join(path, 'labels.csv'))

    flat = preprocess_flat(flat)
    labels = preprocess_labels(labels)

    # Procesamiento principal
    process_and_save_complete(df_lab, df_chart, flat, labels, path, test=test)

if __name__ == '__main__':
    # Cargar ruta desde paths.json o manual
    MIMIC_PATH = r"C:\Users\Javi\Desktop\A 5º Mat-Info\TFG Mates\Mathematics-Dissertation-Survival-Modeling-Javier\data\processed\mimic_extraction"
    main(MIMIC_PATH, test=False)