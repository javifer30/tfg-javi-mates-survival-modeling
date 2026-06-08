# CODEX Instructions — TFG Survival Models

Este archivo define las instrucciones que debe seguir el agente de Codex al escribir, modificar o revisar el código del TFG.

El objetivo principal del proyecto es construir un pipeline claro, eficiente y reproducible para preparar datos clínicos, replicar modelos de supervivencia existentes, entrenar modelos adaptados al experimento del TFG y evaluar sus resultados de forma rigurosa.

Prioridades del código:

1. **Eficiencia**: evitar cálculos innecesarios, duplicación de datos en memoria y procesos lentos que puedan simplificarse.
2. **Simplicidad**: escribir código fácil de leer, mantener y explicar en la memoria del TFG.
3. **Fidelidad metodológica**: cuando se replique un modelo de la literatura, respetar la estructura original del modelo.
4. **Reproducibilidad**: cualquier experimento debe poder repetirse con la misma configuración.
5. **Claridad académica**: el código debe poder defenderse y explicarse en un contexto universitario.

La regla general del proyecto es:

> Primero replicar correctamente. Después adaptar al experimento. No reescribir desde cero un modelo si ya existe una implementación original disponible.

---

## 0. Entorno del proyecto

-Usar siempre el entorno tfg-survival para pruebas, validaciones y test. 
-Mantener los requirements del entorno siempre actualizados, si añades algo de codigo donde incluyas nuevas dependencias tienes que añadirlas a requirements.txt

## 1. Principios generales

### 1.1. Escribir código simple antes que código sofisticado

- No introducir abstracciones complejas si una función clara resuelve el problema.
- Evitar clases salvo que realmente organicen mejor el código.
- No crear frameworks internos innecesarios.
- No añadir patrones avanzados si no aportan una mejora clara.
- Preferir funciones pequeñas, bien nombradas y con una responsabilidad concreta.

Ejemplo preferido:

```python
def load_static_covariates(path: str) -> pd.DataFrame:
    """Load the static covariates table from disk."""
    return pd.read_parquet(path)
```

Evitar:

```python
class StaticCovariateManagerFactory:
    ...
```

si no hay una razón real para esa complejidad.

---

### 1.2. Eficiencia antes que comodidad

Cuando haya varias formas razonables de implementar algo, elegir la opción más eficiente siempre que no haga el código mucho más difícil de entender.

Priorizar:

- `parquet` frente a `csv` para ficheros grandes.
- Procesamiento por chunks cuando los datos no quepan cómodamente en memoria.
- Operaciones vectorizadas de `pandas` o `numpy` frente a bucles fila a fila.
- Guardar resultados intermedios costosos si se reutilizan varias veces.
- Evitar recalcular features, splits o tensores si ya existen en disco y son válidos.
- Liberar objetos grandes cuando ya no se usen.
- Usar GPU solo cuando aporte mejora real de tiempo.

Evitar:

- Cargar todos los CSV grandes si solo se necesitan algunas columnas.
- Copiar DataFrames grandes sin necesidad.
- Repetir el mismo preprocesamiento dentro de cada entrenamiento.
- Guardar el mismo dataset en muchos formatos diferentes sin motivo.
- Crear pipelines excesivamente generales para un único experimento concreto.

---

### 1.3. Comentarios sencillos y útiles

El código debe tener comentarios explicativos, pero no ruido.

Usar comentarios para explicar:

- Por qué se hace una decisión metodológica.
- Qué representa una variable importante.
- Cómo se evita data leakage.
- Qué forma tiene un tensor o matriz.
- Qué parte del pipeline corresponde a datos, modelo o evaluación.
- Qué parte procede del modelo original y qué parte es adaptación al TFG.

No comentar lo obvio.

Ejemplo bueno:

```python
# Use only the first 24 hours to avoid using information after the prediction time.
features_24h = events[events["hours_from_icu_admit"] <= 24]
```

Ejemplo innecesario:

```python
# Add one to counter.
counter += 1
```

---

## 2. Uso de modelos originales en `src/models_references/`

### 2.1. Papel de la carpeta `src/models_references/`

La carpeta:

```text
src/models_references/
```

debe contener los proyectos, scripts o implementaciones originales de los modelos de referencia utilizados en el TFG.

Ejemplos:

```text
src/models_references/
├── DeepHit/
├── DeepSurv/
├── Dynamic-DeepHit-master/
├── DySurv/
└── XMI-ICU/
```

Esta carpeta debe tratarse como **código de referencia metodológica**.

Su objetivo es conservar la implementación original de cada modelo para poder:

- entender la arquitectura real propuesta por los autores;
- revisar la función de pérdida original;
- mantener la lógica de censura, eventos y discretización temporal;
- comprobar cómo se preparan las entradas del modelo;
- adaptar el modelo al experimento del TFG sin alterar su idea principal.

---

### 2.2. Regla principal para replicar modelos

Cuando se implemente un modelo basado en una carpeta de `src/models_references/`, Codex debe seguir esta regla:

> Usar los scripts originales como base metodológica y adaptar únicamente lo necesario para integrarlos en nuestro experimento.

Esto significa que Codex debe intentar conservar:

- la arquitectura del modelo;
- la estructura de capas principales;
- la función de pérdida;
- el tratamiento de censura;
- el formato temporal usado por el modelo;
- la estrategia de entrenamiento;
- la forma de generar predicciones;
- la interpretación de las salidas.

Codex no debe rediseñar el modelo desde cero salvo que sea estrictamente necesario.

---

### 2.3. Qué se puede adaptar

Se permite adaptar el código original en los siguientes puntos:

- rutas de entrada y salida;
- carga de datos;
- nombres de columnas;
- conversión al formato de datos del TFG;
- integración con nuestros splits de train, validation y test;
- guardado de métricas;
- logging;
- configuración externa mediante YAML o JSON;
- uso de GPU;
- compatibilidad con nuestro pipeline;
- pequeñas limpiezas de código que no cambien la metodología.

Ejemplo:

```python
# Original model logic is preserved.
# This wrapper only adapts the TFG dataset format to the input expected by DySurv.
```

---

### 2.4. Qué no se debe cambiar sin justificación

No cambiar sin una razón clara:

- número o tipo de salidas del modelo;
- definición de la función de pérdida;
- tratamiento de pacientes censurados;
- definición del horizonte temporal;
- discretización temporal;
- forma de calcular el riesgo o supervivencia;
- criterio principal de optimización;
- lógica interna de atención, recurrencia, autoencoder o componente generativo;
- interpretación de las predicciones.

Si algún cambio es necesario, debe quedar documentado en un comentario breve y en la configuración del experimento.

Ejemplo:

```python
# Adaptation for the TFG:
# The original model uses 72-hour windows. Here we keep the same temporal logic,
# but align the final prediction horizon with the 10-day survival task.
```

---

### 2.5. Separar código original y código adaptado

No modificar de forma agresiva los archivos originales dentro de `src/models_references/`.

La estructura recomendada es:

```text
src/
├── models_references/
│   ├── DeepSurv/
│   ├── DeepHit/
│   ├── Dynamic-DeepHit-master/
│   ├── DySurv/
│   └── XMI-ICU/
│
├── models/
│   ├── deepsurv_tfg.py
│   ├── deephit_tfg.py
│   ├── dynamic_deephit_tfg.py
│   ├── dysurv_tfg.py
│   └── xmi_icu_tfg.py
```

La carpeta `models_references` conserva la referencia original.

La carpeta `models` contiene las versiones limpias, adaptadas y ejecutables dentro del pipeline del TFG.

---

### 2.6. Wrappers antes que reescrituras

Cuando sea posible, crear un wrapper alrededor del modelo original.

Ejemplo:

```python
class DySurvTFGWrapper:
    """
    Wrapper that adapts the original DySurv implementation
    to the data format and training pipeline of the TFG.
    """

    def __init__(self, config):
        self.config = config
        self.model = build_original_dysurv_model(config)

    def fit(self, train_loader, val_loader):
        ...

    def predict(self, test_loader):
        ...
```

El wrapper puede encargarse de:

- transformar datos;
- lanzar entrenamiento;
- recoger métricas;
- guardar resultados;
- adaptar predicciones al formato común del TFG.

Pero no debe alterar la lógica central del modelo sin justificación.

---

### 2.7. Documentar la relación con el modelo original

Cada modelo adaptado debe incluir al inicio del archivo un comentario breve:

```python
"""
TFG adaptation of the original DySurv model.

The original implementation is stored in:
src/models_references/DySurv/

This file keeps the main model structure and adapts:
- data loading,
- train/validation/test splits,
- output paths,
- metric logging,
- configuration handling.
"""
```

Esto es importante para poder explicar en la memoria qué se ha replicado y qué se ha adaptado.

---

## 3. Uso de scripts originales para el preprocesamiento de datos

### 3.1. Preprocesamiento basado en el repositorio de DySurv

En la carpeta `src/models_references/` se encuentran archivos procedentes del repositorio original de DySurv, incluyendo scripts usados para el preprocesamiento de datos.

Cuando se construya el pipeline de datos del TFG, Codex debe revisar primero esos scripts antes de crear una implementación nueva.

La regla principal es:

> Para el preprocesamiento de datos, usar todo lo que se pueda del código original de DySurv y adaptarlo lo mínimo necesario cuando no siga exactamente la lógica del proyecto.

Esto aplica especialmente a:

- extracción de cohortes;
- construcción de estancias de UCI;
- definición de tiempos de supervivencia;
- definición de evento y censura;
- tratamiento del horizonte temporal;
- selección de variables clínicas;
- construcción de covariables estáticas;
- construcción de covariables dinámicas;
- discretización temporal;
- agregación por ventanas temporales;
- imputación;
- normalización;
- preparación final de tensores o tablas para el modelo.

---

### 3.2. Qué debe conservarse del preprocesamiento original

Si los scripts originales de DySurv ya implementan una parte del preprocesamiento, Codex debe intentar conservar:

- la lógica de selección de pacientes;
- la definición original de la tarea de supervivencia;
- el tratamiento de censura;
- la definición del horizonte máximo;
- el formato temporal usado para las secuencias;
- la forma de resumir o imputar variables;
- los criterios de inclusión y exclusión;
- la estructura general de los datos de entrada al modelo.

No se debe reescribir desde cero el procesamiento si el repositorio original ya lo resuelve de forma razonable.

---

### 3.3. Qué se puede adaptar en el preprocesamiento

Se permite adaptar el preprocesamiento original cuando sea necesario por diferencias entre el repositorio original y nuestro proyecto.

Adaptaciones permitidas:

- cambiar rutas locales por rutas relativas del proyecto;
- adaptar nombres de columnas;
- convertir salidas de CSV a parquet para mejorar eficiencia;
- separar outputs intermedios y finales;
- adaptar el formato para que encaje con nuestro pipeline común;
- añadir logs;
- añadir validaciones de forma, nulos, eventos y censura;
- hacer el código ejecutable desde scripts del proyecto;
- dividir scripts demasiado largos en funciones simples;
- añadir configuración externa;
- hacer el procesamiento compatible con Lightning AI;
- evitar cargar datos demasiado grandes en memoria de forma innecesaria.

Ejemplo:

```python
# Based on the original DySurv preprocessing script.
# Adaptation for the TFG:
# - use relative paths,
# - save intermediate tables as parquet,
# - keep the original event/censoring logic unchanged.
```

---

### 3.4. Qué no se debe cambiar en el preprocesamiento sin justificación

No cambiar sin justificación:

- la definición de evento;
- la definición de censura;
- el horizonte temporal del experimento;
- la ventana temporal usada para construir covariables;
- los criterios de exclusión;
- la discretización temporal;
- la relación entre estancia, paciente y observaciones temporales;
- la forma en la que el modelo original espera recibir las entradas.

Si hay que modificar algo porque la lógica exacta del TFG lo exige, debe quedar indicado claramente.

Ejemplo:

```python
# Methodological adaptation:
# The original DySurv preprocessing uses a different prediction window.
# Here we keep the same preprocessing structure but align the horizon with the TFG task.
```

---

### 3.5. Separar preprocesamiento original y preprocesamiento adaptado

No modificar agresivamente los scripts originales dentro de `src/models_references/`.

La estructura recomendada es:

```text
src/
├── models_references/
│   └── DySurv/
│       └── original preprocessing scripts
│
├── data/
│   ├── dysurv_preprocessing_tfg.py
│   ├── static_dataset.py
│   ├── dynamic_dataset.py
│   └── survival_targets.py
```

La carpeta de referencia conserva el código original.

La carpeta `src/data/` contiene la versión adaptada al pipeline del TFG.

---

### 3.6. Validar que la adaptación no cambia la lógica clínica

Después de adaptar scripts de preprocesamiento, Codex debe comprobar:

- número de pacientes o estancias resultantes;
- distribución de tiempos observados;
- porcentaje de eventos;
- porcentaje de censura;
- número de variables;
- número de timesteps;
- forma final de los tensores;
- ausencia de información posterior al instante de predicción.

Añadir validaciones simples:

```python
assert targets["observed_event"].isin([0, 1]).all()
assert targets["time_to_event"].min() >= 0
assert dynamic_tensor.shape[0] == len(targets)
```

Si los resultados cambian mucho respecto al procesamiento original, dejar una nota en logs o documentación.

---

## 4. Organización recomendada del proyecto

La estructura debe ser sencilla y fácil de navegar.

```text
project/
│
├── configs/
│   ├── paths.yaml
│   ├── data_preprocessing.yaml
│   ├── deepsurv.yaml
│   ├── deephit.yaml
│   ├── dynamic_deephit.yaml
│   ├── dysurv.yaml
│   └── xmi_icu.yaml
│
├── data/
│   ├── raw/                 # Datos originales, no modificar
│   ├── interim/             # Datos intermedios
│   └── processed/           # Datos listos para entrenar
│
├── notebooks/               # Solo exploración, no pipeline principal
│
├── outputs/
│   ├── models/              # Pesos entrenados
│   ├── checkpoints/         # Checkpoints para reanudar entrenamientos
│   ├── metrics/             # Métricas en CSV/JSON
│   ├── predictions/         # Predicciones guardadas
│   ├── figures/             # Gráficos para la memoria
│   └── logs/                # Logs de ejecución
│
├── scripts/
│   ├── build_static_data.py
│   ├── build_dynamic_data.py
│   ├── run_dysurv_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── run_experiment.py
│
├── src/
│   ├── data/
│   │   ├── loading.py
│   │   ├── preprocessing.py
│   │   ├── dysurv_preprocessing_tfg.py
│   │   ├── static_dataset.py
│   │   ├── dynamic_dataset.py
│   │   └── survival_targets.py
│   │
│   ├── models_references/
│   │   ├── DeepSurv/
│   │   ├── DeepHit/
│   │   ├── Dynamic-DeepHit-master/
│   │   ├── DySurv/
│   │   └── XMI-ICU/
│   │
│   ├── models/
│   │   ├── deepsurv_tfg.py
│   │   ├── deephit_tfg.py
│   │   ├── dynamic_deephit_tfg.py
│   │   ├── dysurv_tfg.py
│   │   └── xmi_icu_tfg.py
│   │
│   ├── training/
│   │   ├── losses.py
│   │   ├── train_loop.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── comparison.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       ├── reproducibility.py
│       └── paths.py
│
├── tests/
│   ├── test_data_shapes.py
│   ├── test_no_leakage.py
│   ├── test_targets.py
│   └── test_metrics.py
│
├── requirements.txt
├── README.md
└── CODEX_INSTRUCTIONS.md
```

No crear más carpetas si no son necesarias.

---

## 5. Instrucciones específicas para el TFG

### 5.1. Contexto del trabajo

El código está orientado a un TFG sobre modelos de supervivencia en UCI, con datos tipo MIMIC-IV y comparación entre modelos clásicos, modelos de deep learning y extensiones más avanzadas.

El pipeline debe permitir trabajar, como mínimo, con:

- covariables estáticas;
- covariables dinámicas o temporales;
- tiempos hasta evento;
- indicadores de censura;
- horizontes temporales definidos;
- modelos de supervivencia entrenables y evaluables.

---

### 5.2. Evitar data leakage

Esta es una regla crítica.

No usar información posterior al instante de predicción para construir features.

Ejemplos:

- Si el modelo usa las primeras 24 horas de UCI, ninguna feature puede usar datos posteriores a esas 24 horas.
- Si se predice supervivencia o riesgo a varios días, las variables de entrada deben estar disponibles antes del horizonte evaluado.
- Los splits de train, validation y test deben hacerse por paciente o estancia, no por filas temporales sueltas que puedan mezclar información del mismo paciente.
- Las transformaciones como normalización, imputación o selección de variables deben ajustarse solo con train y aplicarse después a validation/test.

Añadir comentarios cuando una línea de código esté evitando leakage.

---

### 5.3. Censura y eventos

Mantener siempre separadas estas variables:

```python
time_to_event      # duración observada hasta evento o censura
observed_event     # 1 si ocurre el evento, 0 si está censurado
```

No confundir:

- alta de UCI;
- muerte;
- censura;
- pérdida de seguimiento;
- final del horizonte de observación.

Cuando se aplique un horizonte máximo, documentar explícitamente la regla.

Ejemplo:

```python
# Cap observed times at the maximum prediction horizon.
# Patients without event before this horizon are treated as censored at max_horizon_days.
time_to_event = np.minimum(raw_time_to_event, max_horizon_days)
observed_event = (raw_event == 1) & (raw_time_to_event <= max_horizon_days)
```

---

### 5.4. Datos estáticos y dinámicos

Separar claramente:

- **datos estáticos**: variables medidas una vez o resumidas en una ventana inicial;
- **datos dinámicos**: trayectorias temporales, secuencias o tensores.

No mezclar ambos formatos dentro de una misma función si no es necesario.

Funciones recomendadas:

```python
def build_static_features(...):
    ...

def build_dynamic_features(...):
    ...

def merge_features_with_survival_targets(...):
    ...
```

Para datos dinámicos, documentar siempre la forma del tensor.

Ejemplo:

```python
# Shape: (n_patients, n_timesteps, n_features)
dynamic_tensor = ...
```

---

### 5.5. Replicación de modelos originales

Antes de implementar o adaptar un modelo, Codex debe revisar su carpeta correspondiente en:

```text
src/models_references/
```

Ejemplos:

```text
src/models_references/DeepSurv/
src/models_references/DeepHit/
src/models_references/Dynamic-DeepHit-master/
src/models_references/DySurv/
src/models_references/XMI-ICU/
```

La implementación del TFG debe intentar responder a esta pregunta:

> ¿Estamos ejecutando una adaptación fiel del modelo original o estamos creando un modelo nuevo inspirado en él?

Para el TFG, se debe priorizar la primera opción.

Por tanto:

- no cambiar la arquitectura principal salvo que sea imprescindible;
- no cambiar la pérdida salvo que el experimento lo exija;
- no cambiar la definición de salida del modelo sin documentarlo;
- no cambiar cómo el modelo trata la censura sin documentarlo;
- no simplificar el modelo hasta convertirlo en otro distinto;
- no añadir componentes nuevos solo porque parezcan mejorar el rendimiento.

La adaptación debe centrarse en integrar el modelo con nuestro dataset, no en inventar una versión nueva.

---

### 5.6. Implementación de modelos estáticos finales

Los modelos estáticos principales del TFG son:

```text
Kaplan-Meier
CoxPH
DeepSurv
PCHazard
DeepHit
````

`Kaplan-Meier` se utiliza como análisis descriptivo de la supervivencia observada de la cohorte, no como modelo predictivo principal. Los modelos predictivos estáticos deben compartir el mismo dataset base:

```text
X_static
time_to_event
observed_event
```

donde `X_static` contiene las covariables estáticas ya imputadas, codificadas y normalizadas, `time_to_event` representa la duración observada hasta evento o censura, y `observed_event` vale 1 si se observa la muerte y 0 si el paciente está censurado.

---

#### 5.6.1. CoxPH con `lifelines`

`CoxPH` debe implementarse con:

```python
from lifelines import CoxPHFitter
```

Codex no debe reimplementar desde cero la verosimilitud parcial de Cox. Debe usar `lifelines.CoxPHFitter` como implementación estándar y crear un wrapper simple:

```text
src/models/coxph_tfg.py
configs/coxph.yaml
```

Entrada esperada:

```text
DataFrame con:
- covariables estáticas
- duration_col = "time_to_event"
- event_col = "observed_event"
```

Ejemplo mínimo:

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter(penalizer=config.get("penalizer", 0.0))
cph.fit(
    train_df,
    duration_col="time_to_event",
    event_col="observed_event",
)
```

El wrapper debe encargarse de:

* cargar `train`, `validation` y `test`;
* ajustar el modelo solo con `train`;
* guardar coeficientes, hazard ratios y supervivencia base;
* generar predicciones de riesgo o supervivencia en `validation` y `test`;
* guardar métricas en el formato común del proyecto.

No hacer selección manual de variables dentro del wrapper salvo que esté definida en configuración.

---

#### 5.6.2. DeepSurv basado en la implementación original

`DeepSurv` debe implementarse a partir de la referencia original guardada en:

```text
src/models_references/DeepSurv/
```

La clase original recibe:

```text
x = covariables del paciente
t = tiempo observado
e = indicador de evento
```

Por tanto, para nuestro proyecto:

```text
X_static       → x
time_to_event  → t
observed_event → e
```

Codex debe conservar la lógica principal de DeepSurv:

* red neuronal feed-forward sobre covariables estáticas;
* salida escalar de riesgo;
* pérdida basada en la verosimilitud parcial de Cox;
* evaluación mediante C-index u otras métricas comunes del proyecto.

La adaptación debe estar en:

```text
src/models/deepsurv_tfg.py
configs/deepsurv.yaml
```

El objetivo de `deepsurv_tfg.py` no es rediseñar DeepSurv, sino adaptar el dataset del TFG al formato esperado por la implementación original.

---

#### 5.6.3. Sustituir PWE Poisson por PCHazard

No implementar `PWE Poisson` como modelo principal.

El modelo por intervalos del TFG será:

```text
PCHazard
```

La razón es que `PCHazard` está más alineado con los benchmarks de supervivencia neuronal y representa directamente un modelo de hazard constante por tramos. Codex debe implementarlo con `pycox`, no mediante una regresión Poisson manual por intervalos.

Implementación recomendada:

```python
from pycox.models import PCHazard
```

Archivos esperados:

```text
src/models/pchazard_tfg.py
configs/pchazard.yaml
```

Entrada esperada:

```text
X_static
time_to_event
observed_event
```

La discretización temporal debe estar definida en configuración:

```yaml
num_durations: 10
max_horizon_days: 10
```

Codex debe usar las herramientas de transformación de etiquetas de `pycox` cuando sea posible, en lugar de crear una discretización manual distinta para este modelo.

La arquitectura de red puede ser una MLP sencilla mediante `torchtuples`, manteniendo la configuración en YAML:

```yaml
hidden_layers: [128, 64]
dropout: 0.1
batch_norm: true
learning_rate: 0.001
batch_size: 256
```

El wrapper debe:

* transformar etiquetas con la lógica de `PCHazard`;
* entrenar el modelo solo con `train`;
* usar `validation` para early stopping y selección de hiperparámetros;
* generar curvas de supervivencia en `test`;
* evaluar con las mismas métricas que el resto de modelos cuando sea posible.

No debe mezclarse `PCHazard` con el antiguo `PWE Poisson`. Si queda código anterior de `PWE Poisson`, debe moverse a una carpeta exploratoria o dejarse fuera del pipeline principal.

---

#### 5.6.4. DeepHit basado en la implementación original

`DeepHit` debe implementarse a partir de la referencia original guardada en:

```text
src/models_references/DeepHit/
```

La clase original `Model_DeepHit` espera:

```text
x_dim        = número de covariables
num_Event    = número de eventos, sin contar censura
num_Category = número de intervalos temporales de salida
x            = covariables
k            = etiqueta de evento/censura
t            = tiempo hasta evento/censura
mask1        = máscara para log-likelihood
mask2        = máscara para ranking/calibración
```

Para nuestro proyecto:

```text
X_static      → x
muerte        → evento 1
censura       → evento 0
num_Event     → 1
num_Category  → 10
```

La adaptación debe estar en:

```text
src/models/deephit_tfg.py
configs/deephit.yaml
```

Codex debe conservar la lógica principal de DeepHit:

* subred compartida feed-forward;
* subred específica por evento;
* salida `num_Event × num_Category`;
* pérdida de log-likelihood;
* ranking loss;
* calibration loss si se mantiene la versión del código original usada como referencia;
* tratamiento correcto de censura mediante máscaras.

La creación de `mask1` y `mask2` debe implementarse en una función separada y testeable:

```python
def build_deephit_masks(time_bins, event, num_events, num_categories):
    ...
```

No hardcodear `num_Event` ni `num_Category` dentro del modelo. Deben venir de configuración.

---

#### 5.6.5. Comparación justa entre modelos estáticos

Para comparar `CoxPH`, `DeepSurv`, `PCHazard` y `DeepHit`, Codex debe mantener:

* mismo split de train, validation y test;
* mismas covariables estáticas;
* misma definición de evento y censura;
* mismo horizonte máximo cuando aplique;
* misma normalización e imputación ajustada solo con train;
* mismas métricas comunes.

Diferencias permitidas:

* `CoxPH` usa una implementación estadística de `lifelines`;
* `DeepSurv` usa la estructura original del repo de DeepSurv;
* `PCHazard` usa `pycox`;
* `DeepHit` usa la estructura original del repo de DeepHit.

Las diferencias de rendimiento deben atribuirse al modelo, no a cambios innecesarios en el preprocesamiento.

````

Además, en la estructura del proyecto cambia estos bloques:

```markdown
configs/
│   ├── paths.yaml
│   ├── data_preprocessing.yaml
│   ├── coxph.yaml
│   ├── deepsurv.yaml
│   ├── pchazard.yaml
│   ├── deephit.yaml
│   ├── dynamic_deephit.yaml
│   ├── dysurv.yaml
│   └── xmi_icu.yaml
````

y:

```markdown
│   ├── models/
│   │   ├── coxph_tfg.py
│   │   ├── deepsurv_tfg.py
│   │   ├── pchazard_tfg.py
│   │   ├── deephit_tfg.py
│   │   ├── dynamic_deephit_tfg.py
│   │   ├── dysurv_tfg.py
│   │   └── xmi_icu_tfg.py
```

Y en la sección de dependencias añade:

````markdown
Para los modelos estáticos finales, las dependencias mínimas deben incluir:

```text
lifelines
pycox
torchtuples
````

`lifelines` se usará para `CoxPH`. `pycox` y `torchtuples` se usarán para `PCHazard`.

```
::contentReference[oaicite:1]{index=1}
```

[1]: https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html "CoxPHFitter — lifelines 0.30.3 documentation"

---

### 5.7. Modelos deep learning

Para PyTorch:

- Usar `Dataset` y `DataLoader`.
- Mover tensores a GPU solo cuando sea necesario.
- Usar `torch.no_grad()` durante validación y test.
- Usar `model.train()` y `model.eval()` correctamente.
- Guardar el mejor modelo según la métrica de validación definida.
- Evitar redes más grandes de lo necesario.
- No duplicar training loops si varios modelos pueden compartir una lógica común.

Ejemplo básico:

```python
model.train()
for batch in train_loader:
    optimizer.zero_grad()
    predictions = model(batch["x"].to(device))
    loss = loss_fn(predictions, batch)
    loss.backward()
    optimizer.step()
```

---

### 5.8. Autoencoder + survival model

Si se combina un autoencoder con un modelo de supervivencia:

- Mantener separada la pérdida de reconstrucción y la pérdida de supervivencia.
- Nombrar claramente cada componente.
- Permitir ajustar los pesos de cada pérdida desde configuración.
- Guardar métricas separadas para cada parte.
- Documentar si el autoencoder procede de un modelo original o es una extensión propia del TFG.

Ejemplo:

```python
reconstruction_loss = mse_loss(reconstructed_x, x)
survival_loss = survival_loss_fn(risk_scores, time_to_event, observed_event)

loss = (
    config.reconstruction_weight * reconstruction_loss
    + config.survival_weight * survival_loss
)
```

Añadir comentario breve explicando la combinación.

---

## 6. Uso en Lightning AI con GPU

### 6.1. Objetivo

El proyecto debe poder ejecutarse en un workspace de Lightning AI sin depender del entorno local del ordenador personal.

Esto implica que el código debe funcionar al subirlo al workspace y cambiar únicamente la configuración necesaria.

---

### 6.2. Rutas relativas

No usar rutas absolutas locales.

Evitar:

```python
path = "C:/Users/Javier/Desktop/TFG/data/..."
```

Usar:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_path = PROJECT_ROOT / "data" / "processed" / "dataset.parquet"
```

Todas las rutas importantes deben estar en configuración:

```yaml
data_dir: data
processed_dir: data/processed
outputs_dir: outputs
checkpoints_dir: outputs/checkpoints
```

---

### 6.3. Código agnóstico al dispositivo

El código debe detectar automáticamente si hay GPU disponible.

Ejemplo:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

No hardcodear `"cuda"` sin comprobar disponibilidad.

También debe ser posible forzar el dispositivo desde configuración:

```yaml
device: auto
```

Valores permitidos:

```text
auto
cpu
cuda
```

---

### 6.4. Mover a GPU solo lo necesario

No cargar el dataset completo en GPU.

Correcto:

```python
for batch in train_loader:
    x = batch["x"].to(device)
    time = batch["time_to_event"].to(device)
    event = batch["observed_event"].to(device)
```

Evitar:

```python
full_dataset = full_dataset.to("cuda")
```

La GPU debe usarse para:

- el modelo;
- los batches actuales;
- la función de pérdida;
- validación y test con `torch.no_grad()`.

Los datos grandes deben permanecer en CPU/disco y cargarse por batches.

---

### 6.5. Checkpoints para reanudar entrenamiento

Como los entrenamientos en Lightning AI pueden depender del tiempo disponible en el workspace, cada entrenamiento debe guardar checkpoints.

Guardar como mínimo:

```text
outputs/checkpoints/best_model.pt
outputs/checkpoints/last_model.pt
```

El entrenamiento debe poder reanudarse desde `last_model.pt` si existe y la configuración lo indica.

Ejemplo de configuración:

```yaml
resume_from_checkpoint: null
save_best_checkpoint: true
save_last_checkpoint: true
checkpoint_metric: val_c_index
```

---

### 6.6. Dependencias claras

Mantener actualizado:

```text
requirements.txt
```

No instalar dependencias manualmente sin añadirlas al archivo.

Si se añade una librería nueva, debe estar justificada por una necesidad clara.

---

### 6.7. Configuración ajustable para GPU

Los parámetros que dependen de la GPU deben estar en YAML o JSON:

```yaml
batch_size: 256
num_workers: 4
pin_memory: true
learning_rate: 0.001
num_epochs: 100
device: auto
mixed_precision: false
```

No fijar estos valores dentro del código.

---

### 6.8. Formatos eficientes para datos grandes

Para entrenar en Lightning AI con datos grandes:

- usar `parquet` para tablas;
- usar `npy`, `npz` o `pt` para arrays/tensores;
- evitar CSV enormes en entrenamiento;
- guardar datasets procesados para no repetir el preprocesamiento;
- evitar duplicar versiones grandes de los datos sin necesidad.

---

### 6.9. Ejecución desde terminal

El proyecto debe poder ejecutarse desde terminal dentro del workspace.

Ejemplo:

```bash
python scripts/run_dysurv_preprocessing.py --config configs/data_preprocessing.yaml
python scripts/train_model.py --config configs/dysurv.yaml
python scripts/evaluate_model.py --run-dir outputs/runs/2026-05-28_dysurv_seed42
```

El objetivo es no depender de notebooks para ejecutar el pipeline principal.

---

## 7. Configuración y reproducibilidad

### 7.1. Configuración externa

No hardcodear hiperparámetros importantes dentro del código.

Usar archivos YAML o JSON para:

- rutas;
- tamaño de batch;
- learning rate;
- número de épocas;
- horizonte temporal;
- semillas;
- columnas utilizadas;
- modelo seleccionado;
- parámetros específicos del modelo;
- configuración de GPU;
- rutas a modelos originales si son necesarias.

Ejemplo:

```yaml
seed: 42
max_horizon_days: 10
batch_size: 256
learning_rate: 0.001
num_epochs: 100
early_stopping_patience: 10
model_name: dysurv
device: auto
```

---

### 7.2. Semillas

Todo experimento debe fijar semilla.

Implementar una función única:

```python
def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Llamarla al inicio de cada script principal.

---

### 7.3. Guardado de resultados

Cada experimento debe guardar:

- configuración usada;
- métricas;
- predicciones principales;
- pesos del modelo si aplica;
- logs de entrenamiento;
- figuras generadas.

Usar nombres de carpeta claros:

```text
outputs/runs/2026-05-28_dysurv_seed42/
```

Dentro de cada run:

```text
config.yaml
metrics.json
train_log.csv
predictions.parquet
model.pt
```

---

## 8. Manejo de datos grandes

### 8.1. Formato de almacenamiento

Para datasets medianos o grandes, preferir:

- `parquet` para tablas;
- `npz` o `npy` para arrays grandes;
- `pt` para tensores de PyTorch.

Evitar CSV grandes salvo para inspección rápida o resultados finales pequeños.

---

### 8.2. Lectura eficiente

Cuando se lean datos grandes:

- seleccionar solo columnas necesarias;
- usar tipos adecuados;
- evitar conversiones repetidas;
- cachear datos procesados.

Ejemplo:

```python
columns = ["stay_id", "charttime", "heart_rate", "systolic_bp"]
events = pd.read_parquet(path, columns=columns)
```

---

### 8.3. Validaciones de tamaño y forma

Después de construir datasets importantes, comprobar:

- número de pacientes;
- número de features;
- porcentaje de eventos;
- porcentaje de censura;
- existencia de valores nulos;
- forma de tensores.

Ejemplo:

```python
assert dynamic_tensor.ndim == 3
assert dynamic_tensor.shape[0] == len(targets)
assert set(targets["observed_event"].unique()).issubset({0, 1})
```

---

## 9. Evaluación

### 9.1. Métricas

Las métricas deben estar centralizadas en:

```text
src/evaluation/metrics.py
```

No duplicar implementaciones de métricas en notebooks o scripts.

Guardar siempre las métricas en formato estructurado.

Ejemplo:

```json
{
  "model": "dysurv",
  "c_index": 0.71,
  "integrated_brier_score": 0.18,
  "seed": 42
}
```

---

### 9.2. Comparación justa entre modelos

Para comparar modelos:

- usar los mismos splits;
- usar el mismo horizonte temporal;
- usar las mismas variables disponibles cuando sea posible;
- guardar la configuración exacta;
- no seleccionar resultados manualmente;
- diferenciar claramente si un modelo usa datos estáticos y otro datos dinámicos;
- indicar si el modelo es una réplica directa, una adaptación o una extensión propia.

Si un modelo usa datos dinámicos y otro estáticos, dejarlo claro en el nombre del experimento y en las métricas.

---

### 9.3. Figuras

Las figuras deben generarse desde scripts reproducibles, no manualmente.

Guardar figuras en:

```text
outputs/figures/
```

Usar nombres claros:

```text
c_index_comparison.png
calibration_curve_dysurv.png
survival_curves_example_patients.png
```

---

## 10. Scripts principales

Los scripts deben ser simples y ejecutables desde terminal.

Ejemplo:

```bash
python scripts/run_dysurv_preprocessing.py --config configs/data_preprocessing.yaml
python scripts/build_static_data.py --config configs/deepsurv.yaml
python scripts/build_dynamic_data.py --config configs/dysurv.yaml
python scripts/train_model.py --config configs/dysurv.yaml
python scripts/evaluate_model.py --run-dir outputs/runs/2026-05-28_dysurv_seed42
```

Cada script debe:

1. cargar configuración;
2. fijar semilla;
3. ejecutar una tarea principal;
4. guardar salidas;
5. mostrar logs claros.

No hacer demasiadas cosas en un único script.

---

## 11. Logging

Usar `logging`, no `print`, salvo para pruebas rápidas.

Formato recomendado:

```python
logger.info("Loading processed dataset from %s", dataset_path)
logger.info("Training samples: %d", len(train_dataset))
logger.info("Validation C-index: %.4f", val_c_index)
```

Los logs deben ayudar a entender:

- qué datos se han cargado;
- qué configuración se usa;
- cuántas muestras hay;
- en qué época va el entrenamiento;
- qué métricas se obtienen;
- qué modelo original se está adaptando;
- qué parte del preprocesamiento procede de DySurv.

---

## 12. Testing mínimo

Añadir tests sencillos para evitar errores silenciosos.

Priorizar tests de:

- formas de datos;
- ausencia de leakage evidente;
- consistencia de targets;
- métricas;
- carga de configuración;
- compatibilidad entre el formato del dataset y la entrada esperada por cada modelo;
- compatibilidad con CPU/GPU.

Ejemplo:

```python
def test_event_indicator_is_binary(targets):
    assert set(targets["observed_event"].unique()).issubset({0, 1})
```

No intentar cubrir todo con tests complejos. Mejor pocos tests útiles y mantenibles.

---

## 13. Qué debe hacer Codex antes de modificar código

Antes de editar o crear archivos, Codex debe:

1. Identificar qué parte del pipeline está tocando: datos, modelo, entrenamiento, evaluación o utilidades.
2. Revisar si ya existe una función parecida para no duplicar lógica.
3. Mantener compatibilidad con la estructura existente.
4. Evitar cambiar nombres de columnas, rutas o formatos sin necesidad.
5. Si va a tocar un modelo replicado, revisar antes el proyecto original en `src/models_references/`.
6. Si va a tocar preprocesamiento de datos, revisar antes los scripts originales de DySurv en `src/models_references/`.
7. Usar todo lo posible del preprocesamiento original antes de crear lógica nueva.
8. Comprobar que el código puede ejecutarse en Lightning AI con rutas relativas y detección automática de GPU.
9. Indicar si el cambio es:
   - réplica del modelo original;
   - adaptación al formato de datos del TFG;
   - adaptación del preprocesamiento original;
   - mejora técnica;
   - cambio metodológico.
10. Explicar brevemente qué cambia y por qué.

---

## 14. Qué debe evitar Codex

Codex no debe:

- introducir dependencias nuevas sin necesidad clara;
- reescribir todo un módulo si basta con cambiar una función;
- crear clases o abstracciones innecesarias;
- mezclar notebooks con código de producción;
- usar variables globales para configuración importante;
- hardcodear rutas absolutas locales;
- cargar datos grandes varias veces;
- ignorar censura o eventos en métricas de supervivencia;
- usar información futura para crear features;
- borrar resultados o datos sin confirmación explícita;
- hacer cambios metodológicos sin dejar comentario o configuración;
- reimplementar desde cero un modelo que ya está disponible en `src/models_references/`;
- reescribir desde cero el preprocesamiento si puede adaptarse desde DySurv;
- cambiar la arquitectura principal de un modelo original sin justificación;
- cambiar la función de pérdida original sin explicarlo;
- convertir una réplica en una versión nueva del modelo sin documentarlo;
- asumir que siempre habrá GPU disponible;
- cargar todo el dataset en GPU;
- depender de rutas locales del ordenador personal.

---

## 15. Formato de respuestas de Codex

Cuando Codex proponga cambios, debe responder de forma breve y estructurada:

```text
Cambios realizados:
- Añadida adaptación del preprocesamiento original de DySurv al formato del TFG.
- Conservada la lógica original de evento y censura.
- Añadido guardado intermedio en parquet para mejorar eficiencia.
- Añadida detección automática de GPU para entrenamiento en Lightning AI.
- Añadido wrapper del modelo DySurv para integrarlo con el pipeline del TFG.

Archivos modificados:
- src/data/dysurv_preprocessing_tfg.py
- src/models/dysurv_tfg.py
- scripts/run_dysurv_preprocessing.py
- scripts/train_model.py
- configs/dysurv.yaml

Notas:
- No se ha cambiado la definición de evento.
- No se ha cambiado el horizonte temporal.
- No se ha modificado la función de pérdida original.
- El código usa rutas relativas y puede ejecutarse en CPU o GPU.
```

No dar explicaciones largas si no son necesarias.

---

## 16. Criterio final de calidad

Antes de considerar una tarea terminada, el código debe cumplir:

- Se entiende leyendo los nombres y comentarios principales.
- No hay complejidad añadida sin justificación.
- No hay duplicación importante.
- No hay data leakage evidente.
- Los datos grandes se manejan de forma razonablemente eficiente.
- Las salidas se guardan de manera reproducible.
- El experimento puede ejecutarse desde scripts.
- Las decisiones importantes están en configuración o documentadas.
- Si el modelo procede de la literatura, la adaptación respeta la estructura original.
- Si el preprocesamiento procede de DySurv, se usa todo lo posible del código original.
- La relación entre `src/models_references/`, `src/models/` y `src/data/` queda clara.
- El proyecto puede ejecutarse en Lightning AI sin depender de rutas locales.
- El código detecta GPU automáticamente y permite entrenar por batches.
- Existen checkpoints para reanudar entrenamientos largos.

La regla final es:

> Si una solución más simple produce el mismo resultado con eficiencia suficiente, usar la solución más simple. Si una solución más eficiente reduce claramente tiempo o memoria sin complicar demasiado el código, usar la más eficiente. Si se replica un modelo original, preservar primero su estructura metodológica y adaptar después solo lo necesario para el experimento del TFG. Si se adapta preprocesamiento de DySurv, reutilizar todo lo posible y cambiar únicamente lo necesario para ajustarlo a la lógica exacta del proyecto.