# Fase 2: Feature Engineering & Preprocessing Pipeline

## Objetivo
Construir una tubería de preprocesamiento (Pipeline) robusta, orientada a objetos (compatible con la API de `scikit-learn` o similar), que transforme los datos crudos/divididos en una matriz matemática $(X)$ lista para ser consumida por cualquier modelo matemático o de Deep Learning.

## Dependencias Requeridas
⛔ **No iniciar esta fase hasta cumplir:**
*   **Fase 1 completada:** Deben existir archivos físicos o estructuras de datos separadas rigurosamente en `Train`, `Val` y `Test`.
*   **Validación:** Se ha confirmado que no existe superposición de pacientes entre particiones y que la proporción del evento está balanceada.

## Tareas a Realizar

1.  **Diseño de Pipelines por Tipo de Variable (`src/features/`):**
    *   **Variables Numéricas:** Crear un pipeline que incluya:
        1.  **Manejo de Nulos:** Por decisión técnica actualizada, **imputaremos los nulos con ceros ('0')**. Esto permite a modelos lineales estrictos (como CoxPH) ejecutarse sobre toda la cohorte sin descartar filas.
        2.  **Escalado:** Transformación de la escala utilizando `StandardScaler`.
    *   **Variables Categóricas:** Crear un pipeline que incluya:
        1.  **Imputación:** Estrategia para nulos usando una constante `'missing'` (`SimpleImputer(strategy='constant', fill_value='missing')`).
        2.  **Codificación:** Transformación a variables dummy utilizando `OneHotEncoder` (configurado para ignorar categorías desconocidas o `handle_unknown='ignore'`).

2.  **Integración de la Tubería Maestra:**
    *   Unir los pipelines anteriores mediante un `ColumnTransformer` (si se usa `sklearn`), mapeando cada columna a su pipeline correspondiente.
    *   Encapsular esta lógica en una clase o función constructora limpia dentro de `src/features/build_features.py` (ej. `def build_preprocessor_pipeline(num_cols, cat_cols):`).

3.  **Prevención Estricta de Data Leakage:**
    *   El preprocesador completo (`ColumnTransformer`) se debe ser ajustado (`.fit()`) **SOLO al conjunto de entrenamiento (`Train`)**.
    *   Los conjuntos `Validation` y `Test` se deben transformar (`.transform()`) utilizando los parámetros (medias, varianzas, categorías aprendidas) del ajuste sobre el `Train`.

## Validación de la Fase (Cómo comprobar que está bien)

*   [ ] Existe un script modular en `src/features/` responsable de construir la pipeline de preprocesamiento, sin "hardcodear" listas enteras de columnas (las columnas deben pasarse como argumento o leerse de `config/`).
*   [ ] Las columnas categóricas se han transformado en variables numéricas (dummies u otros) y la matriz es 100% numérica.
*   [ ] Al cargar `X_val` o `X_test`, se ejecuta únicamente `.transform()` (nunca `.fit()`) y el proceso no falla por nulos o categorías nuevas no vistas.
*   [ ] (Prueba de Cordura) La varianza intrínseca de `X_test` escalado NO es exactamente 1, ya que se escaló con la desviación estándar calculada sobre `X_train`. Si es 1, hay Data Leakage.
