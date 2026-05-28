# Documentación del Pipeline de Supervivencia y Arquitectura Base

Este documento registra el progreso, las decisiones arquitectónicas y las buenas prácticas (SOLID, Clean Code, MLOps) implementadas durante la construcción del proyecto de Modelado de Supervivencia y Análisis de Series Temporales (MIMIC-IV). Además, es un **Manual Vivo** (Guidelines) con tips constructivos para futuros desarrollos.

## Visión General
El objetivo de las Fases 1 a 4 ha sido construir una base sólida, escalable y 100% reproducible sobre la cual entrenar nuestros modelos predictivos puramente estáticos, previniendo fugas de datos (Data Leakage) y sentando las bases de la trazabilidad en Data Science.

---

## Fase 1: Data Split & EDA Inicial (`src/utils/split_train_test.py`)
En esta fase nos enfocamos en dividir nuestra cohorte clínica cruda (flat features y labels) en conjuntos garantizados para validación y prueba inesgada.

### Lógica y Buenas Prácticas Implementadas:
1. **Merge Seguro:** Se ha cruzado la tabla de características extraídas con las etiquetas biológicas (muerte u horas en UCI) utilizando siempre la clave primaria `patientunitstayid` (Inner Join) para evitar pérdida artificial de registros.
2. **Stratified Splitting:** Dado el desbalance en la mortalidad hospitalaria, el split a Train/Val/Test (70/15/15) no es aleatorio simple, sino **estratificado** sobre la columna target `actualhospitalmortality`.
3. **Comprobación de Fugacidad (Assertions):** Prevención activa de que un paciente (ID) colocado en Train acabe en Validation o Test.

---

## Fase 2: Feature Engineering (`src/features/build_features.py`)
Procesamiento algebraico de valores crudos para convertirlos a matrices consumibles por los engranajes matemáticos.

### Lógica y Buenas Prácticas Implementadas:
1. **ColumnTransformer & Pipeline (sklearn):** Las transformaciones se encapsulan en Pipelines puros para evitar la "dispersión de lógica".
2. **StandardScaler (Numéricas):** Variables continuas son centradas (media=0) y re-escaladas (dev_est=1), garantizando la óptima convergencia en algoritmos analíticos.
3. **Imputación Numérica Robusta:** Las NaNs numéricas se imputan con la **mediana ('median')** en lugar de con ceros o la media. 
   - *¿Por qué no Ceros?* En variables fisiológicas (ej. Frecuencia Cardíaca), inyectar un cero destruye la varianza real creando un pico anómalo, y arrastra artificialmente la media global hacia abajo engañando gravísimamente al `StandardScaler`.
   - *¿Por qué Mediana y no Media?* En datos clínicos (MIMIC-IV), variables como el recuento de leucocitos o tiempos en UCI tienen distribuciones fuertemente asimétricas y *outliers*. La media es sensible a estos valores extremos, mientras que la mediana (el percentil 50) es matemáticamente robusta y representa mejor al "paciente típico" sin desplazar la distribución real geométrica antes del escalado.
4. **Imputación Categórica Fuerte:** Una nueva categoría `'missing'` ha sido declarada para los campos nulos en texto, transformados finalmente por un `OneHotEncoder` (ignorando categorías desconocidas en test).
5. **Zero Data Leakage:** El transformador se entrena (`.fit()`) obligatoriamente solo sobre el `Train`. Luego se aplica matemática congelada ciega (`.transform()`) a Validacion y Test.

---

## Fase 3: Arquitectura Base de Modelos (`src/models/`, SOLID)
Aplicación estricta de POO (Programación Orientada a Objetos) para estandarización científica.

### Lógica y Buenas Prácticas Implementadas:
1. **Liskov Substitution Principle:** Creada `BaseSurvivalModel` (`base.py`). Contrato que obliga a todos los modelos futuros a instanciar metodos obligatorios (`fit()`, `predict_risk()`, `predict_survival_function()`).
2. **Wrapper Predictivo:** `CoxPHModel` (`cox.py`) oculta las demandas peculiares de librerías en favor de una interfaz limpia de Scikit-Learn.

---

## Fase 4: Orquestación y Refactorización End-to-End (`train_static_pipeline.py`)

### Lógica y Buenas Prácticas Implementadas:
1. **Panel de Control (train.yaml):** Configuraciones completas como JSON/YAML para controlar flujos sin abrir un solo `.py`.
2. **Experiment Isolation**: Guardado condicionado a un string de `experiment_name` para la trazabilidad completa y poder sobreescribir con seguridad en un pipeline MLOps.
3. **Hyperparameter Grid Search:** Las listas de parámetros YAML mutan hacia permutaciones automáticas evaluadas a nivel sub-pipeline maximizando siempre el C-Index (Validación estricta desacoplada en `metrics.py`).

---

# 🚀 GUÍA DE FUTURO (Guidelines & Tips)
A partir de este punto, expandir el proyecto con *Nuevos Modelos*, *Nuevas Features Dinámicas (Series Temporales)* u *Optimizaciones*, debe realizarse obedeciendo las siguientes reglas cardinales:

### 1. Extensión de Nuevos Modelos Algorítmicos
* **Cumple el Contrato SOLID:** Si vas a programar Random Survival Forests (RSF), Gradient Boosters, o Redes Densas (DeepSurv), crea su respectivo archivo (ej. `src/models/rsf.py`). 
* Ese archivo DEBE heredar de `BaseSurvivalModel` en `base.py`. Si tu modelo requiere una forma distinta de consumir las "X" y las "Y", es **tu responsabilidad** hacer ese parseo dentro del método `fit()` para que, por fuera, el orquestador lo llame exactamente igual.
* Añade tu modelo al orquestador `train_static_pipeline.py` en el paso de instanciación con un triste bloque condicional (`elif model_cfg['name'] == 'XGBoostSurv': ...`).

### 2. Adición de Características Estáticas y Extracción
* **Regla de Cero Alteración Estructural:** Modificar extracciones crudas afectará a `Fase 1` solo si los nombres/nulos difieren gravemente. 
* Si agregas una variable biológica, declárala como categórica, numérica o ID obligatoriamente dentro de `config/features.yaml`. El pipeline se encargará de imputar y escalar por ti.
* *El pipeline confía ciegamente en dictados de Config YAML.* 

### 3. Modelado Dinámico y Variables Time-Series
* Los modelos avanzados de supervivencia longitudinal van a pedir un formato completamente distinto de matriz al estático (ej. formato *Long* de Start/Stop vs Formato *Wide* clásico). 
* **Tip Vital**: No corrompas el `train_static_pipeline.py`. Ese script ya hace su cometido a la perfección. Es preferible crear un nuevo orquestador `train_dynamic_pipeline.py` desde cero, que consuma su propio bloque `config/train_dynamic.yaml`, utilizando la misma filosofía (Métricas desacopladas e Híper-tuneadas).

### 4. Mantén la Trazabilidad MLOps (Experimentos)
* Antes de lanzar un entrenamiento que se prevea de larga duración o que tenga una configuración sutilmente distinta, cambia `experiment_name` en el yaml (ej. `cox_penalizacion_l1_05` en vez del clásico `baseline_cox`).
* Si sobre-escribes, asumes pérdida, nunca rompas *runs* de resultados competitivos.

### 5. Añadiendo Nuevas Métricas
* Todas las funciones matemáticas (Traducción de riesgo, Log-Likelihood, Brier Scores en varios Tiempos) deben depositarse en `src/evaluation/metrics.py`. 
* Su interfaz debe ser de matrices puras: Pide `y_true_time`, `y_true_event` y `y_pred_metric`, para que puedan invocarse ciegamente por el Orquestador general. No incluyas objetos de Modelos ("model") dentro de los argumentos por dependencia cruzada.

Siguiendo esta Guía, el código permanecerá inmaculado y presentable de cara al TFG (y a un ambiente pre-productivo en empresa real).
