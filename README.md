# Landmark Survival Modeling TFG

Repositorio del TFG de Matemáticas sobre modelos de supervivencia en pacientes
adultos de UCI usando MIMIC-IV.

La versión actual del proyecto se centra en un único pipeline parametrizable por
landmark:

```text
landmark_hours = 24, 48, 72
```

Para cada landmark se construye una cohorte de pacientes todavía en riesgo, se
usan solo variables disponibles hasta ese instante y se evalúa supervivencia en
los 10 días posteriores.

## Estructura

```text
configs/      Configs base para datos, tuning y final seeds landmark.
scripts/      Entrypoints ejecutables desde terminal.
src/data/     Construcción de datasets landmark estáticos, dinámicos y faithful.
src/models/   Modelos estáticos pycox/lifelines y modelos dinámicos faithful.
src/evaluation/ Métricas comunes de supervivencia.
src/utils/    Configuración, logging, reproducibilidad y resolución landmark.
tests/        Tests ligeros del pipeline landmark.
Imagenes/     Figuras seleccionadas para la memoria.
```

No se versionan datos MIMIC-IV, datasets derivados, modelos entrenados,
checkpoints ni outputs.

## Instalación

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

En este proyecto se ha usado el entorno local `tfg-survival` para pruebas y
validación.

## Datos Esperados

El pipeline landmark parte de artefactos MIMIC-IV procesados localmente:

```text
data/processed/mimic_extraction/flat_features_with_time_since_admission.csv
data/processed/mimic_extraction/labels.csv
data/processed/mimic_extraction/timeseries.csv
data/processed/mimic_extraction/timeserieslab.csv
```

Si solo existe `flat_features.csv`, generar primero la versión enriquecida:

```bash
python scripts/add_time_since_admission_to_flat_features.py
```

## Pipeline Landmark

El argumento `--landmark-hours` es la fuente operativa principal y acepta:

```text
24, 48, 72
```

### 1. Dataset estático

```bash
python scripts/build_landmark_static_data.py ^
  --config configs/landmark_static_data.yaml ^
  --landmark-hours 72 ^
  --force
```

### 2. Dataset dinámico base

```bash
python scripts/build_landmark_dynamic_data.py ^
  --config configs/landmark_dynamic_data.yaml ^
  --landmark-hours 72 ^
  --force
```

### 3. Subconjunto temporal DySurv

```bash
python scripts/filter_landmark_dysurv_features.py ^
  --landmark-hours 72 ^
  --force
```

### 4. Dataset faithful común

```bash
python scripts/prepare_landmark_faithful_dataset.py ^
  --config configs/landmark_dysurv_faithful.yaml ^
  --landmark-hours 72 ^
  --force
```

## Modelos

Modelos estáticos:

- Kaplan-Meier descriptivo.
- CoxPH.
- Random Survival Forest.
- DeepSurv-style CoxPH.
- LogisticHazard.
- PCHazard.
- DeepHitSingle.

Modelos dinámicos faithful:

- DySurv faithful temporal.
- Dynamic-DeepHit faithful.
- DySurv static-only faithful.

Todos los modelos de un mismo landmark comparten cohortes, splits, targets y
horizonte de evaluación.

## Tuning

Planificar sin entrenar:

```bash
python scripts/tune_landmark_static_models.py ^
  --config configs/landmark_static_tuning.yaml ^
  --landmark-hours 72 ^
  --models coxph random_survival_forest deepsurv pchazard deephit_single ^
  --dry-run
```

Tuning dinámico faithful:

```bash
python scripts/tune_landmark_dysurv_faithful.py ^
  --config configs/landmark_dysurv_faithful.yaml ^
  --landmark-hours 72 ^
  --device cuda ^
  --resume
```

```bash
python scripts/tune_landmark_dynamic_deephit_faithful.py ^
  --config configs/landmark_dynamic_deephit_faithful.yaml ^
  --landmark-hours 72 ^
  --device cuda ^
  --resume
```

```bash
python scripts/tune_landmark_dysurv_static_faithful.py ^
  --config configs/landmark_dysurv_static_faithful.yaml ^
  --landmark-hours 72 ^
  --device cuda ^
  --resume
```

## Final Seeds

Tras seleccionar hiperparámetros por validación:

```bash
python scripts/run_final_landmark_static_seeds.py ^
  --config configs/landmark_static_tuning.yaml ^
  --landmark-hours 72 ^
  --models coxph random_survival_forest deepsurv logistic_hazard pchazard deephit_single
```

```bash
python scripts/run_final_landmark_dynamic_deephit_faithful_seeds.py ^
  --config configs/landmark_dynamic_deephit_faithful.yaml ^
  --landmark-hours 72 ^
  --device cuda
```

Los seeds finales son:

```text
42, 123, 2026
```

## Outputs

Los resultados se separan por landmark:

```text
data/processed/landmark_24h/
data/processed/landmark_48h/
data/processed/landmark_72h/

outputs/landmark_24h/
outputs/landmark_48h/
outputs/landmark_72h/
```

Cada ejecución guarda una configuración resuelta, por ejemplo:

```text
outputs/landmark_72h/static/config_used.yaml
outputs/landmark_72h/dysurv_faithful/config_used.yaml
outputs/landmark_72h/dynamic_deephit_faithful/config_used.yaml
outputs/landmark_72h/dysurv_static_faithful/config_used.yaml
```

## Auditoría y Figuras Finales

Las tablas de revisión final y las figuras de la memoria se generan desde los
artefactos finales ya existentes, sin entrenar modelos ni modificar métricas:

```bash
python scripts/audit_landmark_results_artifacts.py
```

El script escribe tablas de auditoría en:

```text
outputs/results_audit/
```

y figuras completas en:

```text
outputs/figures/landmark_final/
```

## Figuras

La carpeta `Imagenes/Bitmap/` contiene únicamente las figuras seleccionadas
para la versión final de la memoria:

```text
Imagenes/Bitmap/DySurv.png
Imagenes/Bitmap/ctd_landmark_main.png
Imagenes/Bitmap/cindex_horizon_main_24_48_72.png
Imagenes/Bitmap/km_risk_groups_dynamic_models_24_48_72.png
Imagenes/Bitmap/km_vs_predicted_survival_dynamic_dysurv_24_72.png
Imagenes/Bitmap/ctd_landmark_appendix_all_models.png
Imagenes/Bitmap/cindex_horizon_appendix_all_models_24_48_72.png
Imagenes/Bitmap/km_risk_groups_faithful_models_24_48_72.png
Imagenes/Bitmap/km_vs_predicted_survival_dynamic_dysurv_24_48_72.png
Imagenes/Bitmap/km_vs_predicted_survival_dysurv_static_vs_temporal_24_48_72.png
```

Los datos MIMIC-IV, datasets derivados completos, outputs de entrenamiento,
checkpoints y pesos de modelos se mantienen fuera del repositorio.
