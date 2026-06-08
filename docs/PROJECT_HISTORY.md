# Project History Governance

## Purpose

This file is the consolidated chronological history of the project. It summarizes
major phases, artifacts, methodological shifts and current status. It should
link outward to operational documentation rather than duplicate every run-level
detail.

## Maintenance Rules

- Only the Project Manager/Historian consolidates this file, unless the user
  explicitly instructs another agent to do so.
- Technical agents should record new decisions in [DECISIONS.md](DECISIONS.md),
  experiments in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), open work in
  [TODO.md](TODO.md), and reproducibility changes in
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- The Project Manager/Historian periodically incorporates stable information
  from those files into this history.
- Do not rewrite the historical narrative during routine technical work.
- Preserve existing history unless a correction is explicitly required and
  documented.

# Historia cronologica del proyecto

Este documento reconstruye la evolucion del repositorio a partir de archivos
locales, notebooks, scripts, configuraciones, datasets y artefactos de salida. El
historial Git disponible solo contiene el commit inicial `596482a` del
28/05/2026, asi que las fechas previas y posteriores se infieren por marcas
temporales locales y por el contenido de los artefactos.

## Resumen ejecutivo

El proyecto empezo como una exploracion en notebooks de supervivencia en UCI con
DeepSurv y PWE Poisson sobre una tabla maestra. Despues se construyo una
extraccion propia de MIMIC-IV v3.1, con labels, variables estaticas y series
temporales clinicas. Esa extraccion genero una cohorte de 93.502 estancias UCI
adultas con mortalidad hospitalaria como evento y estancia UCI en dias como
tiempo observado.

La primera fase experimental uso datasets `interim` con split 70/15/15
aproximado y un preprocesado `sklearn` que produjo matrices de 79 columnas.
Sobre esa base se entrenaron DeepSurv, CoxPH y PWE Poisson. El benchmark local
`baseline_cox_70_split_benchmark.csv` dejo:

| Modelo | C-index test |
| --- | ---: |
| DeepSurv | 0.7599 |
| CoxPH | 0.7378 |
| PWEPoisson | 0.5061 |

La fase final refactorizo el trabajo hacia un pipeline estatico reproducible en
`scripts/`, `configs/` y `src/`. Se rehizo el split como 60/20/20 estratificado,
se ajusto el preprocesamiento solo con train y se implementaron cinco modelos:
Kaplan-Meier, CoxPH, DeepSurv, PCHazard y DeepHit.

Resultados finales principales:

- DeepSurv obtuvo el mejor C-index estatico en test: 0.7606.
- PCHazard obtuvo las mejores metricas de curva en test: IBS 0.1134 y NBLL
  0.3642, con C-index 0.7508.
- CoxPH fue competitivo y estable: C-index test 0.7404.
- DeepHit estatico fallo como discriminador global: C-index test 0.4879, aunque
  sus metricas dependientes del tiempo fueron informativas.
- Kaplan-Meier quedo como analisis descriptivo, no como predictor.

La adaptacion dinamica completa de DySurv/Dynamic-DeepHit aparece como objetivo
metodologico y hay codigo de referencia en `src/models_references/`, pero no hay
en el pipeline final un entrenamiento dinamico propio con landmarks diarios.

## 1. Orientacion inicial del TFG

Los documentos de `TFG/` fijan el objetivo: evaluar modelos de supervivencia
estaticos y dinamicos en pacientes adultos de UCI usando MIMIC-IV. La pregunta
central es si los modelos dinamicos, especialmente DySurv, aportan mejora real
frente a modelos estaticos al incorporar trayectorias temporales del paciente.

La guia academica insiste en que el proyecto debe ser una evaluacion experimental
y no una validacion clinica definitiva. Tambien establece una regla metodologica:
replicar correctamente modelos existentes antes de adaptarlos.

Modelos previstos por la guia:

- Kaplan-Meier como baseline descriptivo.
- Cox Proportional Hazards como referencia clasica.
- DeepSurv como extension neuronal de Cox.
- PWE/PCHazard como familia piecewise.
- DeepHit como control estatico frente a Dynamic-DeepHit.
- Dynamic-DeepHit y DySurv como objetivos dinamicos principales.

El repositorio conserva implementaciones originales o de consulta en
`src/models_references/`: DeepSurv, DeepHit, Dynamic-DeepHit, DySurv, XMI-ICU y
preprocesamientos MIMIC-IV/eICU del paper. Las adaptaciones propias estan en
`src/models/`, `src/data/`, `src/evaluation/`, `src/features/` y `scripts/`.

## 2. Primera fase: notebooks y prototipos

### DeepSurv en notebook

`notebooks/icu_train cross-validation DeepSurv.ipynb`, fechado a principios de
enero, parte de `data/processed/icu_master.csv`. Implementa seleccion de
variables estaticas, splits 60/20/20 por semilla, preprocesado con
`ColumnTransformer`, red DeepSurv en PyTorch, perdida Cox partial likelihood,
C-index, supervivencia con riesgo acumulado basal de Breslow, Kaplan-Meier de
censura, IBS/IBLL experimentales y visualizaciones.

El notebook registra C-index de validacion alrededor de 0.78 en las primeras
epocas y una supervivencia Kaplan-Meier aproximada a 10 dias de 0.878 en test.

### PWE Poisson en notebook

`notebooks/train_modelo_PWE.ipynb` implementa un modelo piecewise-exponential
como Poisson. Cada paciente se expande a intervalos con:

- `y_ik`: tiempo en riesgo en el intervalo;
- `d_ik`: indicador de evento en el intervalo;
- covariables estaticas repetidas;
- offset `log(y_ik)`;
- efectos de intervalo `alpha_k`;
- coeficientes `beta`.

Resultados registrados:

- PWE test C-index: 0.7543;
- curvas de supervivencia test con forma `(8670, 100)`;
- PWE test IBS: 0.7929;
- PWE test IBLL: -0.2196.

Estos resultados pertenecen al prototipo de notebook. En el benchmark modular
posterior PWE Poisson obtuvo 0.5061 de C-index test.

### Primeros modelos guardados

La carpeta `models/` contiene artefactos del 03/01/2026:

- `best_deepsurv_model.pth`;
- `deepsurv_final.pth`;
- `best_pwe_model.pth`;
- `pwe_poisson_final.pth`.

Esto corresponde a una primera etapa de entrenamiento manual fuera del pipeline
final.

## 3. Extraccion propia de MIMIC-IV

El archivo `src/data/mimic_direct_extraction.py` construye la cohorte desde los
CSV comprimidos de MIMIC-IV v3.1 en `data/raw/mimic-iv-3.1/`.

### Labels

`generate_labels()`:

- carga `patients`, `admissions` e `icustays`;
- adapta `race` a `ethnicity`;
- calcula edad con `year(intime) - anchor_year + anchor_age`;
- filtra estancias con `los > 5/24` y edad mayor que 17;
- define `patientunitstayid`, `patienthealthsystemstayid`, `uniquepid`,
  `actualhospitalmortality` y `actualiculos`;
- guarda `labels.csv`.

Dataset generado:

| Archivo | Filas | Columnas |
| --- | ---: | ---: |
| `data/processed/mimic_extraction/labels.csv` | 93.502 | 5 |

### Variables estaticas

`generate_flat_features(labels_df)` extrae de `chartevents` peso, altura, GCS
eyes, GCS motor y GCS verbal en una ventana de -24h a +5h respecto al ingreso en
UCI. Promedia duplicados y anade genero, edad, etnia, unidad UCI inicial, origen
de admision, seguro y hora de ingreso.

Dataset generado:

| Archivo | Filas | Columnas |
| --- | ---: | ---: |
| `data/processed/mimic_extraction/flat_features.csv` | 93.502 | 13 |

### Series temporales

`generate_timeseries(labels_df)`:

- identifica labs comunes en `labevents` con cobertura mayor del 25% y mas de 3
  observaciones medias por estancia;
- identifica chart events comunes en `chartevents` con cobertura mayor del 25%
  y mas de 5 observaciones medias por estancia;
- filtra valores no nulos;
- calcula offsets en minutos desde ingreso UCI;
- usa ventana desde -1 dia hasta el final de la estancia UCI;
- guarda `timeserieslab.csv` y `timeseries.csv`;
- re-filtra labels y flat features para conservar estancias con datos
  temporales.

Datasets generados:

| Archivo | Filas aproximadas | Columnas | Tamano local |
| --- | ---: | ---: | ---: |
| `timeserieslab.csv` | 1.048.575 | 4 | 30 MB |
| `timeseries.csv` | 136.319.088 | 4 | 6.0 GB |

## 4. Preprocesamiento temporal y optimizacion

El objetivo de la parte temporal fue convertir labs y chart events irregulares en
representaciones por hora para modelos dinamicos: pivotar a formato ancho,
resamplear, imputar, normalizar, crear mascaras y guardar en Parquet.

`src/data/mimic_timeseries_sparse.py` conserva dos lineas de trabajo:

- funciones inspiradas en el paper: `reconfigure_timeseries`,
  `resample_and_mask`, `further_processing`, `add_time_of_day`,
  `preprocess_flat` y `preprocess_labels`;
- una version sparse/vectorizada que elimina variables listadas en
  `DROP_COLS_EXACT`, agrega componentes Braden, discretiza a horas, calcula
  cuantiles 0.05/0.95 por feature, normaliza a `[-1, 1]`, aplica clipping
  `[-4, 4]`, rellena huecos y guarda batches parquet de 2.000 pacientes.

Artefactos temporales encontrados:

- `data/processed/mimic_extraction/feature_stats.csv`;
- `data/processed/mimic_extraction/preprocessed_timeseries.csv` (13.99 GB);
- `data/processed/mimic_extraction/timeseries_parquet/` (47 batches parquet);
- `data/processed/mimic_extraction/timeseries_parquet_complete/`.

`basura/mimic_timeseries_sparse_optimization_plan.md` documenta cuellos de
botella: carga completa de CSV, operaciones de indice, `groupby().transform()` y
mask decay no vectorizado. Propone `usecols`, dtypes, vectorizacion, `numba` y
procesamiento por chunks.

Estado final: los datos temporales y referencias existen, pero no hay pipeline
final propio que entrene DySurv con landmarks diarios sobre estos parquet.

## 5. Dataset interim y feature engineering previo

La primera estructura modular genero datasets CSV en `data/interim/`:

| Split | Filas | Columnas |
| --- | ---: | ---: |
| `data_train.csv` | 65.451 | 17 |
| `data_val.csv` | 14.025 | 17 |
| `data_test.csv` | 14.026 | 17 |

`notebooks/01_initial_eda_and_baseline.ipynb` confirma que el EDA se hizo solo
sobre train para evitar data leakage. Comprobo overlap cero de `patientunitstayid`
entre splits, trazo Kaplan-Meier global y por genero, analizo nulos,
categoricas, numericas y relaciones con mortalidad y LOS, y preparo una tabla de
landmarks por horas con pacientes en riesgo y eventos en los siguientes 10 dias.

`src/features/build_features.py` implementa el preprocesamiento previo:

- numericas: imputacion por mediana y `StandardScaler`;
- categoricas: imputacion `"missing"` y `OneHotEncoder(handle_unknown="ignore")`;
- ajuste solo en train;
- transformacion de validation y test;
- guardado de matrices procesadas con ids y targets;
- guardado del preprocesador con `joblib`.

Datasets generados:

| Archivo | Filas | Columnas |
| --- | ---: | ---: |
| `X_train_processed.csv` | 65.451 | 79 |
| `X_val_processed.csv` | 14.025 | 79 |
| `X_test_processed.csv` | 14.026 | 79 |

Tambien existe una copia bajo `baseline_cox_70_split/`, coherente con el
benchmark previo.

## 6. Benchmark previo `baseline_cox_70_split`

La carpeta `models/baseline_cox_70_split/` contiene:

- `CoxPH.pkl`;
- `DeepSurv.pkl`;
- `PWEPoisson.pkl`.

El archivo `outputs/metrics/baseline_cox_70_split_benchmark.csv` resume:

| Modelo | C-index test |
| --- | ---: |
| DeepSurv | 0.7599 |
| CoxPH | 0.7378 |
| PWEPoisson | 0.5061 |

Este experimento parece ser el puente entre notebooks y pipeline final: mantiene
DeepSurv y CoxPH con buen rendimiento, pero muestra que PWE Poisson no quedo
competitivo bajo ese protocolo.

## 7. Refactorizacion final del repositorio

El commit Git disponible `596482a` del 28/05/2026 creo la estructura inicial del
TFG: `configs/`, `data/`, `notebooks/`, `outputs/`, `references/`, `scripts/`,
`src/`, `tests/`, `README.md` e instrucciones academicas.

El estado local actual sustituye scripts antiguos como `train_static_pipeline.py`,
`evaluate.py` y `run_mimic_pipeline.py` por:

- `scripts/build_static_data.py`;
- `scripts/train_static_model.py`;
- `scripts/evaluate_static_model.py`;
- `scripts/run_static_pipeline.py`;
- `scripts/evaluate_deephit_time_dependent.py`;
- `scripts/evaluate_pchazard_time_dependent.py`.

La configuracion final esta en:

- `configs/static_data.yaml`;
- `configs/static_pipeline.yaml`;
- `configs/static_evaluation.yaml`;
- `configs/kaplan_meier.yaml`;
- `configs/coxph.yaml`;
- `configs/deepsurv.yaml`;
- `configs/pchazard.yaml`;
- `configs/deephit.yaml`.

`configs/static_pipeline.yaml` orquesta construccion del dataset estatico,
entrenamiento de modelos y consolidacion de metricas.

## 8. Dataset estatico final

`src/data/static_dataset.py` define el builder final. La decision metodologica
central es hacer el split antes de ajustar el preprocesamiento, y ajustar todos
los parametros solo en train.

Se unen `flat_features.csv` y `labels.csv`, se eliminan ids/eventos/tiempos
invalidos y se exige `time_to_event > 0`.

Columnas target finales:

- `patientunitstayid`;
- `time_to_event`;
- `observed_event`;
- `split`.

`configs/static_data.yaml` fija seed 42, split 60/20/20 y estratificacion por
evento. `outputs/metrics/static_dataset_summary.json` registra:

| Split | Pacientes | Eventos | Censurados | Event rate |
| --- | ---: | ---: | ---: | ---: |
| Train | 56.101 | 6.684 | 49.417 | 0.11914 |
| Validation | 18.700 | 2.228 | 16.472 | 0.11914 |
| Test | 18.701 | 2.228 | 16.473 | 0.11914 |

Preprocesamiento final:

- `gender`: mapa binario, desconocidos a 0.5;
- `height`: estandarizacion z-score;
- `weight`, `age`, `hour`, `eyes`, `motor`, `verbal`: escalado robusto por
  cuantiles 0.05/0.95;
- clipping `[-4, 4]`;
- `nullheight`: indicador de altura imputada;
- categoricas `ethnicity`, `first_careunit`, `admission_location`, `insurance`;
- categorias raras con menos de 1000 apariciones agrupadas como `misc`;
- one-hot encoding con categorias aprendidas solo en train;
- validacion de no overlap, mismas columnas, evento binario, tiempos positivos y
  ausencia de nulos inesperados.

Artefactos:

| Archivo | Contenido |
| --- | --- |
| `data/processed/static/train_static.parquet` | train final |
| `data/processed/static/val_static.parquet` | validation final |
| `data/processed/static/test_static.parquet` | test final |
| `data/processed/static/split_assignments.parquet` | asignacion paciente-split |
| `outputs/preprocessors/static_preprocessor.pkl` | preprocesador ajustado |
| `outputs/metrics/static_dataset_summary.json` | resumen dataset |

El dataset final tiene 35 features.

## 9. Modelos finales implementados

### Kaplan-Meier

`src/models/kaplan_meier_tfg.py` usa `lifelines.KaplanMeierFitter` como analisis
descriptivo. Guarda curva de supervivencia, metricas descriptivas y figura si
`matplotlib` esta disponible. No se trata como predictor principal.

### CoxPH

`src/models/coxph_tfg.py` usa `lifelines.CoxPHFitter` con `penalizer: 0.1` y
`l1_ratio: 0.0`. Entrena en train, predice partial hazard como `risk_score`,
genera curvas de supervivencia en un grid de 10 duraciones hasta 10 dias y
guarda modelo, predicciones, curvas test, metricas y coeficientes.

### DeepSurv

`src/models/deepsurv_tfg.py` es una adaptacion propia en PyTorch: MLP `[64, 32]`,
dropout 0.1, salida escalar de log-riesgo, Cox partial likelihood, Adam,
weight decay 0.0001, batch size 256, maximo 50 epocas, early stopping con
paciencia 10 y checkpoints best/last/cada 5 epocas.

### PCHazard

`src/models/pchazard_tfg.py` sustituye al PWE Poisson previo usando
`pycox.models.PCHazard` y `torchtuples`. Usa MLP `[128, 64]`, batch normalization,
dropout 0.1, horizonte maximo 10 dias, 10 duraciones discretas y targets capados
al horizonte. Evalua C-index, IBS, NBLL y metricas dependientes del tiempo.

### DeepHit

`src/models/deephit_tfg.py` es una adaptacion estatica propia en PyTorch:
`num_Event: 1`, `num_Category: 10`, red compartida `[128, 64]`, red por causa
`[64]`, salida `num_Event x num_Category`, mascaras `mask1`/`mask2`, perdida con
log-likelihood, ranking y calibracion opcional. La configuracion final usa
`alpha: 1.0`, `beta: 1.0`, `gamma: 0.0`.

## 10. Evaluacion implementada

`src/evaluation/metrics.py` centraliza:

- Harrell C-index con convencion riesgo alto = peor supervivencia;
- fallback manual si falta `lifelines`;
- Integrated Brier Score con `pycox.EvalSurv`;
- Integrated Binomial Log-Likelihood con `pycox.EvalSurv`;
- `NaN` en metricas de curva si no hay curvas o dependencias.

`src/evaluation/time_dependent_survival.py` y
`src/evaluation/deephit_time_dependent.py` implementan:

- supervivencia a incidencia acumulada `F(t) = 1 - S(t)`;
- aproximacion de Antolini Ctd evaluando riesgo en el tiempo de evento;
- C-index ponderado por censura en horizontes fijos;
- Kaplan-Meier de censura estimado en train;
- horizontes 1, 3, 5, 7 y 9 dias.

Los tests cubren splits sin overlap, eventos binarios, tiempos positivos,
columnas consistentes, ausencia de nulos, formas de mascaras DeepHit, logica de
Antolini Ctd, C-index ponderado por horizonte, conversion supervivencia a
incidencia acumulada e importabilidad de modelos.

## 11. Experimentos finales y resultados

### CoxPH

| Split | C-index | IBS | NBLL |
| --- | ---: | ---: | ---: |
| Train | 0.7462 | NaN | NaN |
| Validation | 0.7407 | 0.1246 | 0.3975 |
| Test | 0.7404 | 0.1246 | 0.3973 |

CoxPH queda como baseline clasico competitivo y estable.

### DeepSurv

| Split | C-index |
| --- | ---: |
| Train | 0.7867 |
| Validation | 0.7650 |
| Test | 0.7606 |

Mejor validation loss: 8.1157. El log mejora hasta la epoca 16 y para tras 26
epocas. Es el mejor modelo final por C-index test. Como no produce curvas en el
pipeline final, IBS/NBLL quedan como `NaN`.

### PCHazard

| Split | C-index | IBS | NBLL |
| --- | ---: | ---: | ---: |
| Train | 0.7972 | NaN | NaN |
| Validation | 0.7621 | 0.1130 | 0.3631 |
| Test | 0.7508 | 0.1134 | 0.3642 |

Antolini Ctd:

| Split | Ctd |
| --- | ---: |
| Train | 0.8078 |
| Validation | 0.7685 |
| Test | 0.7642 |

C-index ponderado por horizonte en test:

| Horizonte dias | Weighted C-index |
| ---: | ---: |
| 1 | 0.7873 |
| 3 | 0.7550 |
| 5 | 0.7429 |
| 7 | 0.7243 |
| 9 | 0.7200 |

PCHazard es el mejor modelo de curva y conserva buena discriminacion.

### DeepHit

| Split | C-index | IBS | NBLL |
| --- | ---: | ---: | ---: |
| Train | 0.4925 | NaN | NaN |
| Validation | 0.4956 | 0.4076 | 1.0505 |
| Test | 0.4879 | 0.4044 | 1.0431 |

Antolini Ctd:

| Split | Ctd |
| --- | ---: |
| Train | 0.7809 |
| Validation | 0.7523 |
| Test | 0.7509 |

C-index ponderado por horizonte en test:

| Horizonte dias | Weighted C-index |
| ---: | ---: |
| 1 | 0.7860 |
| 3 | 0.7427 |
| 5 | 0.7256 |
| 7 | 0.7077 |
| 9 | 0.7002 |

El `risk_score` global final no discrimina bien, pero las metricas dependientes
del tiempo sugieren que la salida por bins contiene informacion. DeepHit necesita
revision antes de presentarse como modelo estatico competitivo.

## 12. Conclusiones encontradas

Sobre datos:

- la cohorte final MIMIC-IV contiene 93.502 estancias UCI adultas;
- la tasa de evento final queda cerca del 11.9% en todos los splits;
- el preprocesamiento estatico final produce 35 features limpias y sin fuga;
- la informacion temporal existe, pero es grande y costosa de procesar.

Sobre modelos:

- DeepSurv gana en C-index test;
- PCHazard gana en IBS/NBLL y Antolini Ctd;
- CoxPH es un baseline robusto;
- PWE Poisson no quedo competitivo en el benchmark modular;
- DeepHit requiere revision de objetivo, resumen de riesgo o configuracion.

Sobre dinamica:

- hay referencias y datasets temporales preparados;
- no hay resultados finales de DySurv dinamico entrenado de extremo a extremo;
- la siguiente fase natural seria construir el pipeline de landmarks dinamicos y
  compararlo contra los resultados estaticos consolidados.

## 13. Linea temporal reconstruida

### Enero 2026

- Notebooks de DeepSurv y PWE Poisson.
- Primeros modelos `.pth` en `models/`.
- Extraccion MIMIC-IV a `labels.csv`, `flat_features.csv`, `timeserieslab.csv`
  y `timeseries.csv`.
- Primer preprocesamiento temporal pesado y batches parquet.

### Febrero 2026

- Optimizacion sparse/vectorizada de series temporales.
- Generacion de `timeseries_parquet_complete/`.
- Consolidacion academica del enfoque DySurv y comparacion estatico/dinamico.

### Marzo 2026

- Refactor de notebooks hacia estructura modular.
- Datasets `data/interim/`.
- Matrices `data/processed/model_input/`.
- Preprocesadores `joblib`.
- Benchmark `baseline_cox_70_split`.

### Mayo 2026

- Commit inicial Git el 28/05/2026.
- Reorganizacion de instrucciones academicas hacia `TFG/`.
- Pipeline estatico final con configs y scripts.
- Splits estaticos finales 60/20/20.
- Entrenamiento de Kaplan-Meier, CoxPH, DeepSurv, PCHazard y DeepHit.
- Evaluaciones dependientes del tiempo para DeepHit y PCHazard el 29/05/2026.

### Junio 2026

- Reconstruccion de la historia del proyecto en este documento.
- Inicializacion del rol permanente Project Manager/Historian para mantener la
  continuidad documental, consolidar `SESSION_NOTES.md` en historia estable y
  vigilar la coherencia entre `PROJECT_HISTORY.md`, `DECISIONS.md`,
  `EXPERIMENT_LOG.md`, `TODO.md` y `REPRODUCIBILITY.md`.
- Revision documental del primer benchmark estatico de DeepHit: DeepSurv y
  PCHazard quedan como las referencias estaticas mas fuertes por ahora, mientras
  DeepHit se marca como pendiente de revision de implementacion/calibracion
  antes de iniciar tuning.

## 14. Estado actual

Consolidado:

- extraccion MIMIC-IV directa;
- dataset estatico final reproducible;
- preprocesamiento estatico sin data leakage;
- pipeline de entrenamiento por configs;
- modelos Kaplan-Meier, CoxPH, DeepSurv, PCHazard y DeepHit;
- evaluacion C-index, IBS, NBLL y metricas dependientes del tiempo;
- tests de datasets, arquitectura y metricas temporales.

Pendiente o parcial:

- pipeline dinamico con landmarks diarios;
- entrenamiento propio de DySurv sobre MIMIC-IV;
- comparacion final estatica vs dinamica;
- consolidacion de `static_model_comparison.csv`, cuya configuracion existe pero
  cuyo archivo no aparece actualmente en `outputs/metrics/`;
- limpieza de `basura/`, `models/` historica y artefactos duplicados antes de
  publicar el repositorio.

## 15. Fuentes locales revisadas

- `README.md`;
- `TFG/Decisiones_clave_y_guia_enfoque_TFG.md`;
- `TFG/CODEX_TFG_MATES_JAVI.md`;
- `TFG/indice_definitivo_TFG.md`;
- `data/mimic_extraction_explanation.md`;
- `src/data/mimic_direct_extraction.py`;
- `src/data/mimic_timeseries_sparse.py`;
- `src/data/static_dataset.py`;
- `src/features/build_features.py`;
- `src/features/pwe_transformer.py`;
- `scripts/*.py`;
- `src/models/*_tfg.py`;
- `src/evaluation/*.py`;
- `configs/*.yaml`;
- `notebooks/*.ipynb`;
- `outputs/metrics/*.json`;
- `outputs/metrics/*.csv`;
- `tests/*.py`;
- `basura/instrucciones_refactorizacion.txt`;
- `basura/mimic_timeseries_sparse_optimization_plan.md`.
