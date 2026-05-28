# Guía de Arquitectura del Proyecto (Cookiecutter Data Science)

Este documento define la estructura estándar de directorios y los principios arquitectónicos que rigen el proyecto. Su objetivo es asegurar que cualquier agente de IA o desarrollador humano entienda la separación de responsabilidades a la hora de crear, modificar o leer código.

---

## 📁 Estructura del Directorio Raíz (`Tree Root`)

```text
Project_Root/
│
├── data/                    <- ⚠️ Datos fuente y derivados.
│   ├── raw/                 <- Datos originales descargados. Inmutables.
│   ├── interim/             <- Datos intermedios transitorios.
│   └── processed/           <- Datos finales, listos para entrenar modelos.
│
├── notebooks/               <- 📓 Solo para exploración (EDA) y prototipado. NO para código de producción.
│
├── config/                  <- ⚙️ Archivos YAML. Adiós al "hardcoding" (hiperparámetros, rutas).
│
├── models/                  <- 💾 Modelos entrenados y serializados (.pkl, .pt) o pesos de redes neuronales.
│
├── reports/                 <- 📊 Documentos, análisis y visualizaciones generadas.
│   └── figures/             <- Gráficos guardados por scripts de evaluación (ej. Curvas ROC, K-M).
│
├── references/              <- 📚 Papers, manuales de terceros, o diccionarios de datos externos.
│
├── guidelines/              <- 📜 Reglas del proyecto y convenciones para agentes de IA u otros desarrolladores.
│
├── src/                     <- 🧠 El corazón del software. Librerías y Clases PURAS.
│   ├── __init__.py
│   │
│   ├── data/                <- Scripts modulares para limpieza, extracción y curación de datos.
│   │
│   ├── features/            <- Scripts modulares para Feature Engineering (transformaciones para ML).
│   │
│   ├── models/              <- Definición matemática y arquitectura orientada a objetos de los modelos.
│   │
│   ├── evaluation/          <- Funciones puras para calcular métricas (C-Index, Brier Score, etc.).
│   │
│   └── utils/               <- Herramientas transversales o helpers (ej. Logging, carga de YAMLs).
│
├── .gitignore               <- Archivos locales y temporales excluidos de control de versiones.
├── requirements.txt         <- Dependencias congeladas para total reproducibilidad.
│
└── *.py (en la raíz)        <- 🚀 SCRIPTS ORQUESTADORES (ej. train.py, evaluate.py, data_pipeline.py).
```

---

## 🧠 Lógica y Directrices Arquitectónicas

La estructura anterior está fundamentada en dos pilares del desarrollo de software moderno aplicado a la Ciencia de Datos:

### 1. El estándar *Cookiecutter Data Science*
Es el marco de trabajo estándar de la industria para proyectos de Machine Learning. Establece que un proyecto debe ser "Un conjunto de dependencias en una sola dirección":

*   **Los Datos son Inmutables (`data/raw/`):** Una vez que un archivo CSV o base de datos entra en `raw/`, NUNCA debe ser modificado por un script. Toda modificación genera un nuevo archivo (en `interim/` o `processed/`). Esto garantiza la reproducibilidad.
*   **Separación entre Experimentación y Producción:** Un Notebook de Jupyter (`notebooks/`) se usa para investigar, ver datos rápido y hacer gráficos. Una vez que el código del notebook se considera válido, debe refactorizarse en clases/funciones y trasladarse a `.py` dentro de `src/`.
*   **Separación de las fases de los Datos:** "Procesar datos" es un término ambiguo. Por ello, el estándar divide la canalización en dos:
    *   **Extracción y Limpieza (`src/data/`):** El proceso de tomar las tablas en bruto, cruzar IDs, rellenar NaNs básicos, etc.
    *   **Creación de Variables (`src/features/`):** El proceso donde se aplica el conocimiento de Machine Learning (ej. One-Hot Encoding, creación de interacciones matemáticas, expansiones polinómicas o tipo Poisson).

### 2. Principios de Diseño de Software (Influencia SOLID)

Para asegurar que los modelos sean mantenibles y escalables, la carpeta `src/` obedece estrictamente a principios de diseño de software (particularmente *Single Responsibility* y *Open/Closed* de SOLID):

*   **Responsabilidad Única (Single Responsibility Principle):** Cada carpeta y archivo en `src/` debe tener una sola razón para cambiar.
    *   Si cambia cómo se extrae la información de la base de datos de la UCI, solo se modifica `src/data/`.
    *   Si cambia la métrica de evaluación (de métrica C-Index a Brier Score), solo se modifica `src/evaluation/`.
    *   Si se añade una nueva capa a la red neuronal, solo se modifica `src/models/`.
*   **Abierto a la extensión, Cerrado a la modificación (Open/Closed Principle):**
    *   **Abstracción de Modelos:** En `src/models/`, es obligatorio definir Clases base genéricas o interfaces (ej. usando `abc.ABC` de Python). Todos los algoritmos (desde un modelo Cox simple hasta una arquitectura profunda DeepSurv) deben implementar métodos estándar como `.fit(X,y)` y `.predict_risk(X)`. 
    *   Gracias a esto, el archivo maestro que evalúa los modelos (`evaluate.py`) no necesita saber *qué* modelo está evaluando; simplemente aplica un bucle sobre las clases instanciadas de `src/models/`, volviendo el sistema fácilmente extendible para incluir nuevos algoritmos sin reescribir la pipeline de evaluación.
*   **Inyección de Dependencias (Configuraciones Externas):**
    *   Ningún valor crítico (rutas, learning rates, epochs, listas de features a dropear) debe estar codificado "a fuego" (hardcoded) dentro de las clases en `src/`.
    *   Todo hiperparámetro debe habitar en los archivos de la carpeta `config/`. Los scripts orquestadores leen esta información y la inyectan a las clases durante la inicialización.
*   **Controladores vs. Servicios:**
    *   **Servicios (La Lógica):** Residen internamente en `src/` en forma de clases de Python y funciones puras. No ejecutan nada al ser importadas.
    *   **Controladores (Los Botones de Ejecución):** Residen en el directorio raíz (ej. `train_models.py`, `run_pipeline.py`). Estos archivos son diseñados para ejecutarse por terminal (con el bloque `if __name__ == "__main__":`). Importan las herramientas de `src/`, importan las rutas de `config/` y orquestan el proceso desde arriba. Haciendo esto, las importaciones relativas dentro del proyecto nunca colisionan y siempre nacen del entorno global (`Project_Root`).
