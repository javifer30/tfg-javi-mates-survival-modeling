# Fase 1: Data Split & Exploratory Data Analysis (EDA) Inicial

## Objetivo
El objetivo de esta fase es preparar los datos brutos (`raw` o resultantes de la primera extracción `processed`) dividiéndolos correcta y rigurosamente en conjuntos de Entrenamiento, Validación y Prueba, asegurando que no haya fuga de información (data leakage). Adicionalmente, se construirá una curva *baseline* de supervivencia.

## Tareas a Realizar

1.  **Lectura y Fusión de Datos:**
    *   Cargar `data/processed/mimic_extraction/flat_features.csv` y `data/processed/mimic_extraction/labels.csv`.
    *   Hacer un `merge` (JOIN) seguro basándose en la clave única de paciente/estancia (ej. `patientunitstayid`).

2.  **Partición de Datos (Data Split):**
    *   Modificar/crear el script `src/utils/split_train_test.py`.
    *   Separar el dataset resultante en tres conjuntos: `Train` (e.g., 70%), `Validation` (e.g., 15%) y `Test` (e.g., 15%).
    *   **Estratificación:** Es obligatorio estratificar por la variable indicadora del evento (`actualhospitalmortality`) para mantener la misma proporción de eventos/censuras en los tres conjuntos.
    *   Guardar los tres conjuntos en la carpeta correspondiente (ej. `data/interim/` o `data/processed/` dependiendo de si se consideran ya finales pre-entrenamiento) con sufijos claros: `_train.csv`, `_val.csv`, `_test.csv`.

3.  **EDA Inicial y Baseline (Kaplan-Meier):**
    *   Crear un notebook exploratorio en `notebooks/` (ej. `01_initial_eda_and_baseline.ipynb`).
    *   **Regla de Oro:** Este EDA debe realizarse **ÚNICAMENTE sobre el conjunto de `Train`**. Tocar `Val` o `Test` aquí invalida el rigor del proyecto.
    *   Ajustar un estimador de Kaplan-Meier global usando la librería `lifelines` o `scikit-survival` para visualizar la función de supervivencia $S(t)$ de la cohorte.
    *   *(Opcional pero recomendado)*: Hacer curvas de Kaplan-Meier estratificadas por algunas variables categóricas clave (ej. género o tipo de unidad de admisión) para ver si existen diferencias intuitivas en la supervivencia.

## Validación de la Fase (Cómo comprobar que está bien)

*   [ ] Existen tres archivos resultantes del split (Train, Val, Test) en disco.
*   [ ] La suma de filas de los tres archivos es exactamente igual a la cantidad de filas del dataset original cruzado.
*   [ ] Ningún `patientunitstayid` está presente en más de un conjunto a la vez (Intersección vacía).
*   [ ] La proporción de `actualhospitalmortality == 1` es prácticamente idéntica (varía menos de un 1-2%) entre `Train`, `Val` y `Test`.
*   [ ] El Notebook de EDA importa datos **exclusivamente** del archivo `_train.csv` (o dataframe de train) y genera al menos un gráfico Kaplan-Meier visible.
