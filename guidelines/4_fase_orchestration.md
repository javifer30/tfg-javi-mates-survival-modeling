# Fase 4: Orquestación, Entrenamiento y Pipeline Principal

## Objetivo
Esta fase tiene como objetivo unir todos los módulos construidos previamente (carga de datos, preprocesamiento, y la arquitectura de modelos) en un flujo de ejecución centralizado (Controlador) que cargue configuraciones dinámicas desde YAML, entrene algoritmos y evalúe su rendimiento matemático.

## Dependencias Requeridas
⛔ **No iniciar esta fase hasta cumplir:**
*   **Fase 3 completada:** Existe una interfaz probada (`BaseSurvivalModel`) y al menos un algoritmo instanciable (ej. `CoxPHModel`).
*   **Infraestructura de Datos Completa:** Se disponen de rutas o flujos listos desde la carga original (Fase 1) hasta el preprocesamiento (Fase 2).

## Tareas a Realizar

1.  **Configuración Global Externa (`config/`):**
    *   Diseñar un archivo YAML (ej. `config/experiment_01.yaml` o general `config/main.yaml`).
    *   Remover toda ruta quemada (*hardcoded paths*) del código en Python y declararlas aquí: rutas de raw data, de processed data, diccionarios de hiperparámetros del modelo (ej: `learning_rate` si fuera DeepSurv o `penalizer` para un CoxPH regularizado).
    *   Escribir un helper dentro de `src/utils/` (si no existe) capaz de leer y parsear este archivo a un diccionario inyectable durante la ejecución.

2.  **Scripts de Orquestación o Controlador (Ej: `train.py` o `run_mimic_pipeline.py`):**
    *   Este archivo, localizado en el directorio raíz (`Tree Root`), servirá como punto de entrada (debe usar el bloque `if __name__ == "__main__":`).
    *   Secuencia de ejecución:
        1.  Inyectar dependencias y leer el archivo `.yaml`.
        2.  LLamar a las lógicas de la Fase 1 o cargar los csv previamente spliteados de `data/interim`.
        3.  LLamar al constructor del pipeline o clase transformadora de la Fase 2, ejecutar `.fit_transform()` en Train y `.transform()` en Validation.
        4.  Instanciar el modelo dinámico desde `src/models/` suministrando los diccionarios de hiperparámetros externos correspondientes.
        5.  Realizar el llamado a `model.fit(X_train, y_train)` y predecir sobre validación.

3.  **Evaluación Matemática y Feedback (Opcional Inicial, Mandatorio Final):**
    *   Implementar un calculador de Indice de Concordancia de Harrell (`C-index`) o Brier Score.
    *   Reportar esa métrica hacia la consola (stdout) usando el sistema de *Logging* nativo o hacia un experiment tracker (WandB/MLflow, si se determinase más adelante).

## Validación de la Fase (Cómo comprobar que está bien)

*   [ ] Al ejecutar `python run_mimic_pipeline.py` (o similar) por consola, la ejecución termina de principio a fin (End-to-End) con código de salida `0` (Sin stacktraces).
*   [ ] No explotan dependencias ni problemas de importaciones cruzadas relativas.
*   [ ] Cualquier cambio en un parámetro numérico (ej. `test_size`,  peso de regularización) no exige modificar código `*.py`, solo actualizar el archivo externo en `config/*.yaml`.
*   [ ] Existe una métrica reportada final (ej. `C-index: 0.72`) que comprueba que el predict del modelo compila con las etiquetas reales de supervivencia del conjunto de evaluación.
