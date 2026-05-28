# Fase 3: Arquitectura Base de Modelos (SOLID)

## Objetivo
El objetivo de esta fase es estructurar el código de los algoritmos de supervivencia (`src/models/`) utilizando programación orientada a objetos (POO) y cumpliendo estrictamente con el principio Open/Closed de SOLID. Se diseñará una interfaz común abstracta que todos los modelos desarrollados deberán heredar e implementar.

## Dependencias Requeridas
⛔ **No iniciar esta fase hasta cumplir:**
*   **Fase 2 completada:** Debe existir una función o clase constructora del pipeline (`src/features/build_features.py`) validada en el conjunto `Train` que retorne matrices numéricas $X_{train}, X_{val}, X_{test}$ sin nulos.

## Tareas a Realizar

1.  **Definición de la Interfaz Base (`src/models/base.py`):**
    *   Crear una clase abstracta `BaseSurvivalModel` heredando de `abc.ABC`.
    *   Definir el método abstracto `fit(self, X, y, **kwargs)` que deba ser implementado por subclases para el ajuste de los pesos matemáticos del modelo.
    *   Definir el método abstracto `predict_risk(self, X)` que debe devolver un vector unidimensional de riesgos relativos (el análogo a la salida de un modelo de Cox Proporcional).
    *   Definir el método abstracto `predict_survival_function(self, X)` que debe devolver las probabilidades de supervivencia para un tiempo $t$ o una matriz de probabilidades de supervivencia evaluada en múltiples tiempos.

2.  **Implementación del Baseline Model (Ej: CoxPH):**
    *   Heredar de `BaseSurvivalModel` para crear una clase `CoxPHModel` (posiblemente sirviendo como wrapper o envoltura de la implementación de `lifelines.CoxPHFitter` o `sksurv.linear_model.CoxPHSurvivalAnalysis`).
    *   Implementar obligatoriamente los métodos `.fit()`, `.predict_risk()` y `.predict_survival_function()` en concordancia con las salidas matemáticas y firmas exigidas por la clase abstracta.

## Validación de la Fase (Cómo comprobar que está bien)

*   [ ] Existe el archivo `src/models/base.py` con una clase hija de `abc.ABC` y sus métodos usan el decorador `@abstractmethod`.
*   [ ] Intentar instanciar la clase `BaseSurvivalModel()` directamente arroja un error (`TypeError: Can't instantiate abstract class...`).
*   [ ] Existe un primer modelo (ej. `CoxPHModel` en `src/models/cox.py`) que hereda de la base y funciona correctamente sin errores de tipado.
*   [ ] El uso o instanciación del objeto no revela rutas de archivos (`No hardcoded paths`); toda inicialización paramétrica ocurre mediante el `__init__(self, **kwargs)` del modelo, respetando que sea inyectado desde la configuración.
