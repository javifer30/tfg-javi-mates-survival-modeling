Sí. Con lo que tienes en Notion sobre DySurv, yo dejaría un índice **compacto pero suficientemente detallado**. La memoria tiene que explicar bien supervivencia, DySurv y la adaptación dinámica, pero sin convertirse en una tesis larga. Además, la guía del TFG insiste en que el trabajo debe centrarse en replicar/evaluar DySurv en MIMIC-IV, comparando modelos estáticos y dinámicos sin plantearlo como validación clínica definitiva. 

# Índice final propuesto

## Resumen

## Abstract

## 1. Introducción

Antes de entrar en secciones, este capítulo debe tener un párrafo inicial breve que presente el problema: predicción de riesgo en UCI, análisis de supervivencia, censura y uso de MIMIC-IV.

### 1.1. Contexto y motivación

Predicción de mortalidad en UCI, limitaciones de una clasificación muerto/vivo y utilidad del análisis de supervivencia.

### 1.2. Pregunta del trabajo

Pregunta central:

> ¿Hasta qué punto los modelos dinámicos de supervivencia, y en particular DySurv, aportan mejora frente a modelos estáticos al incorporar trayectorias temporales del paciente?

### 1.3. Objetivos

Separaría:

```text
Objetivo general
Objetivos específicos
```

Aquí incluiría construir la cohorte, definir evento/censura, preparar datos estáticos y temporales, implementar modelos, adaptar DySurv y comparar resultados.

### 1.4. Alcance del trabajo

Dejar claro que es una evaluación académica sobre MIMIC-IV, no una herramienta clínica validada.

### 1.5. Estructura de la memoria

Párrafo breve explicando los capítulos siguientes.

---

## 2. Preliminares matemáticos y estadísticos

Este capítulo debe introducir las herramientas necesarias antes de hablar de modelos concretos. Las instrucciones del TFG piden separar preliminares, estado de la cuestión, metodología, resultados y discusión, y evitar introducir definiciones básicas por primera vez en resultados. 

### 2.1. Tiempo hasta evento y censura

Definir:

[
T,\quad C,\quad Y=\min(T,C),\quad \delta
]

Explicar muerte, alta/censura y tiempo observado.

### 2.2. Función de supervivencia

[
S(t)=P(T>t)
]

Interpretación en el contexto UCI.

### 2.3. Función de riesgo y riesgo acumulado

[
h(t), \quad H(t)
]

Explicar la relación intuitiva entre riesgo instantáneo, riesgo acumulado y supervivencia.

### 2.4. Covariables estáticas y covariables temporales

Distinguir:

```text
X_i = variables estáticas
X_i(t) = trayectoria temporal
X_{i,0:t} = historia observada hasta t
```

Esta notación será clave para explicar los landmarks.

### 2.5. Predicción dinámica mediante landmarks

Definir qué es un landmark (t_l), qué significa predecir desde ese momento y por qué solo se usan pacientes en riesgo.

### 2.6. Métricas de evaluación en supervivencia

Incluir solo las que realmente usarás:

```text
C-index / Ctd-index
Integrated Brier Score
Integrated Binomial Log-Likelihood
AUROC/AUPRC a 24h si usáis el primer bin
```

---

## 3. Estado de la cuestión y modelos considerados

Este capítulo debe explicar la evolución desde modelos clásicos hasta modelos dinámicos. No haría una lista larga de papers; lo estructuraría por familias.

### 3.1. Modelos clásicos de supervivencia

#### 3.1.1. Kaplan-Meier

Como estimador descriptivo de la supervivencia de la cohorte.

#### 3.1.2. Cox Proportional Hazards

Modelo semiparamétrico clásico y baseline académico.

### 3.2. Modelos neuronales estáticos

#### 3.2.1. DeepSurv

Explicar que sustituye el predictor lineal de Cox por una red neuronal.

### 3.3. Modelos por intervalos y supervivencia discreta

#### 3.3.1. PWE Poisson

Modelo piecewise-exponential, útil como puente entre tiempo continuo y discretización.

#### 3.3.2. DeepHit

Modelo estático discreto, antecedente directo de Dynamic-DeepHit.

### 3.4. Modelos dinámicos de supervivencia

#### 3.4.1. Dynamic-DeepHit

Explicar que extiende DeepHit incorporando información longitudinal.

#### 3.4.2. Enfoques con landmarks

Conectar con vuestra adaptación diaria.

### 3.5. DySurv

Esta debe ser una sección central, pero controlada. La página de Notion ya resume bien la estructura: DySurv combina datos clínicos temporales y estáticos, un encoder, una distribución latente, un vector (z), un predictor de supervivencia y una rama de reconstrucción tipo autoencoder para regularizar el espacio latente. 

#### 3.5.1. Idea general de DySurv

Explicar el flujo:

```text
datos estáticos + temporales
→ encoder
→ distribución latente
→ z
→ predictor de supervivencia
```

#### 3.5.2. Encoder temporal y representación latente

Explicar LSTM, (\mu_z), (\log\sigma_z^2), reparametrización y espacio latente.

No hace falta entrar en cada capa con exceso, pero sí indicar la lógica.

#### 3.5.3. Decoder y regularización variacional

Explicar que la reconstrucción no predice mortalidad directamente, sino que ayuda a aprender una representación latente más estable.

#### 3.5.4. Módulo de supervivencia

Explicar los 10 intervalos de salida y la relación con Logistic Hazard como wrapper/loss de supervivencia discreta, no como sustituto de DySurv.

#### 3.5.5. Interpretación del preprocesamiento temporal original

Aquí incluiría la aclaración importante de Notion:

```text
72-hour timesteps = 72 timesteps horarios
```

y no “un timestep de 72 horas”. También indicaría que el notebook original usa una ventana temporal fija de 72 horas, mientras que vuestra metodología con landmarks diarios es una adaptación explícitamente dinámica. 

### 3.6. Modelos no incluidos en el análisis principal

Mención breve:

```text
CoxTime
Logistic Hazard como modelo independiente
Random Survival Forest
MTLR
PMF
BCESurv
CoxCC
```

Solo para justificar que se dejan fuera por alcance.

---

## 4. Datos y construcción de los datasets

Este capítulo debe ser metodológico, no de resultados. Explica cómo se construyen los datos.

### 4.1. Fuente de datos: MIMIC-IV

Descripción breve de la base y del entorno UCI.

### 4.2. Definición de la cohorte

Pacientes adultos de UCI, criterios de inclusión/exclusión y unidad de análisis.

### 4.3. Evento de interés, censura e inicio del seguimiento

Definir con precisión:

```text
inicio = ingreso en UCI
evento = muerte
censura = alta / pérdida de observación según vuestra definición
```

### 4.4. Dataset estático

Variables estáticas, limpieza, imputación, normalización/codificación y partición train/validation/test.

### 4.5. Dataset temporal

Variables horarias, laboratorios, constantes vitales, imputación, normalización, máscaras y formato tensorial.

### 4.6. Construcción del dataset dinámico mediante landmarks diarios

Esta sección es clave.

Explicar:

```text
landmarks = 24h, 48h, ..., 240h
input = datos horarios disponibles hasta el landmark
output = riesgo diario durante los próximos 10 días
```

### 4.7. Discretización del tiempo futuro en 10 intervalos

Explicar los 10 bins diarios:

```text
bin 1 = muerte entre τ y τ+24h
bin 2 = muerte entre τ+24h y τ+48h
...
bin 10 = muerte entre τ+216h y τ+240h
```

### 4.8. Evitar filtraciones de información futura

Muy importante. Explicar que el split se hace por paciente antes de generar landmarks.

### 4.9. Pacientes en riesgo y censura por landmark

Aquí no pondría todavía resultados completos, pero sí la regla:

[
Y_i > t_l
]

para incluir al paciente en el landmark (t_l).

---

## 5. Metodología experimental

Este capítulo explica qué modelos se entrenan y cómo se comparan.

### 5.1. Diseño general de los experimentos

Presentar el flujo:

```text
cohorte MIMIC-IV
→ dataset estático
→ dataset dinámico
→ modelos estáticos
→ modelos dinámicos
→ comparación
```

### 5.2. Modelos estáticos implementados

#### 5.2.1. Kaplan-Meier

Uso descriptivo.

#### 5.2.2. CoxPH

Baseline clásico.

#### 5.2.3. DeepSurv

Baseline neuronal basado en Cox.

#### 5.2.4. PWE Poisson

Modelo por intervalos.

#### 5.2.5. DeepHit

Modelo discreto estático.

### 5.3. Modelos dinámicos implementados

#### 5.3.1. Dynamic-DeepHit

Benchmark dinámico.

#### 5.3.2. DySurv static

Control para aislar el efecto de la arquitectura DySurv.

#### 5.3.3. DySurv con series temporales

Modelo central.

Explicar que usa:

```text
X_static
X_ts hasta el landmark
X_mask
salida de 10 intervalos diarios
```

### 5.4. Adaptación propia de DySurv

Esta sección debe ser muy clara. Diría:

```text
DySurv original:
ventana temporal fija de 72 timesteps horarios y salida discreta de 10 intervalos.

Adaptación del TFG:
landmarks diarios durante los 10 primeros días de UCI, usando historia acumulada hasta cada landmark y predicción diaria para los 10 días siguientes.
```

Esto evita venderlo como réplica exacta cuando no lo es.

### 5.5. Hiperparámetros y entrenamiento

Breve en cuerpo principal. Tabla resumida. Detalles completos a anexo.

### 5.6. Métricas de evaluación

Separar:

```text
métricas de supervivencia global
métricas por landmark
métricas de riesgo a 24h usando el primer bin
```

### 5.7. Estrategia de comparación

Comparaciones clave:

```text
CoxPH vs DeepSurv
DeepSurv vs DeepHit
DeepHit vs Dynamic-DeepHit
DySurv static vs DySurv time-series
Dynamic-DeepHit vs DySurv time-series
Estáticos vs dinámicos
```

---

## 6. Resultados

Este capítulo debe empezar con resultados descriptivos y después pasar a modelos.

### 6.1. Descripción de la cohorte

Número de pacientes, eventos, censurados, distribución de tiempos.

### 6.2. Supervivencia observada de la cohorte

Curva Kaplan-Meier y pacientes en riesgo.

### 6.3. Análisis del conjunto dinámico

Tabla por landmark:

```text
landmark
pacientes en riesgo
eventos próximos 10 días
censurados próximos 10 días
porcentaje de eventos
```

Esto es importante para justificar la viabilidad del enfoque.

### 6.4. Resultados de modelos estáticos

Tabla con CoxPH, DeepSurv, PWE Poisson y DeepHit.

### 6.5. Resultados de modelos dinámicos

Tabla con Dynamic-DeepHit, DySurv static y DySurv time-series.

### 6.6. Comparación estático-dinámica

Interpretar si las trayectorias temporales aportan mejora.

### 6.7. Resultados por landmark

Ver si el rendimiento cambia en:

```text
24h, 48h, 72h, ..., 240h
```

### 6.8. Curvas de supervivencia predichas

Ejemplos de curvas desde distintos landmarks.

### 6.9. Riesgo a 24h a partir del primer intervalo

Análisis complementario usando (p_1).

---

## 7. Discusión

Aquí no repetiría todas las tablas. Debe interpretar.

### 7.1. Respuesta a la pregunta principal

¿Aportan valor los modelos dinámicos frente a los estáticos?

### 7.2. Interpretación de DySurv frente a sus controles

Comparar:

```text
DySurv static
DySurv time-series
Dynamic-DeepHit
```

### 7.3. Diferencias respecto al paper original de DySurv

Muy importante.

Explicar:

```text
mismo objetivo general
misma idea de supervivencia dinámica
misma salida discreta de 10 intervalos
diferente construcción temporal
diferente protocolo de landmarks
posibles diferencias de variables/cohorte/preprocesamiento
```

### 7.4. Impacto de la censura y caída de pacientes en riesgo

Interpretar por qué puede afectar más a landmarks tardíos.

### 7.5. Limitaciones metodológicas

Incluir:

```text
censura posiblemente informativa
ausencia de validación externa
sensibilidad a hiperparámetros
diferencias frente a DySurv original
coste computacional
dependencia de MIMIC-IV
```

### 7.6. Posibles extensiones

Por ejemplo:

```text
validación externa en eICU
más landmarks o predicción horaria
ventanas rolling de 72h
análisis de calibración más profundo
incertidumbre
```

---

## 8. Conclusiones

Breve y directo.

### 8.1. Conclusiones principales

Responder a objetivos.

### 8.2. Aportaciones del trabajo

Construcción de cohorte, adaptación dinámica, comparación ordenada.

### 8.3. Trabajo futuro

Extensiones realistas.

---

## Bibliografía

---

## Anexos

### Anexo A. Variables utilizadas

Tabla de variables estáticas y temporales.

### Anexo B. Detalles de preprocesamiento

Imputación, normalización, máscaras, landmarks.

### Anexo C. Hiperparámetros

Configuración de cada modelo.

### Anexo D. Resultados complementarios

Tablas largas por landmark, curvas adicionales, sensibilidad.

### Anexo E. Fragmentos relevantes de implementación

Solo si aporta valor. No meter código largo en el cuerpo principal.

---

# Versión compacta del índice

```text
Resumen
Abstract

1. Introducción
2. Preliminares matemáticos y estadísticos
3. Estado de la cuestión y modelos considerados
4. Datos y construcción de los datasets
5. Metodología experimental
6. Resultados
7. Discusión
8. Conclusiones

Bibliografía
Anexos
```

La parte de DySurv la pondría principalmente en **3.5**, con la adaptación propia en **5.4**. Así separas bien teoría/modelo original de metodología propia. Esa separación es clave para que el tribunal vea que entiendes DySurv, pero también que tu implementación es una adaptación defendible y no una copia confusa del paper.
