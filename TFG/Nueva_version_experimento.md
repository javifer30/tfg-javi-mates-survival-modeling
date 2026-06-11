# Documento de decisión metodológica: nueva versión del experimento de supervivencia dinámica

## 1. Contexto del cambio metodológico

Durante la revisión del planteamiento del TFG se ha identificado una cuestión metodológica central: cómo construir de forma limpia la entrada temporal de los modelos dinámicos y cómo definir el instante desde el que se realiza la predicción.

El trabajo partía de la intención de replicar y evaluar modelos de supervivencia estáticos y dinámicos inspirados en DySurv sobre MIMIC-IV. Inicialmente se consideró una estrategia basada en landmarks diarios, donde cada paciente podía generar varias muestras temporales a distintos instantes de predicción. Sin embargo, esta estrategia introduce una complejidad computacional considerable, multiplica el número de ejemplos y complica la comparación directa entre modelos estáticos y dinámicos.

A partir de la revisión del paper de DySurv, del notebook asociado y de literatura relacionada en UCI, se ha decidido reformular el experimento dinámico hacia una estrategia más sencilla, reproducible y defendible: una única predicción por paciente a partir de una ventana temporal inicial fija.

La nueva pregunta experimental queda formulada así:

> Entre los pacientes que siguen en riesgo tras las primeras 72 horas de ingreso en UCI, ¿aporta valor predictivo utilizar la trayectoria temporal inicial del paciente frente a utilizar únicamente una representación estática?

Esta formulación conserva la idea principal de DySurv y Dynamic-DeepHit, es decir, utilizar modelos secuenciales para resumir la evolución clínica del paciente, pero evita el problema de definir ventanas retrospectivas en función del evento observado.

---

## 2. Problema detectado: ventana retrospectiva y posible fuga temporal

La principal duda surgió al analizar la lógica del notebook de DySurv. En dicho notebook, la serie temporal parece filtrarse para conservar únicamente información hasta 24 horas antes del tiempo observado de evento o censura. Después se construye una ventana temporal de longitud fija, por ejemplo 72 timesteps horarios.

Esta construcción puede interpretarse de dos formas distintas:

1. Si la predicción se entiende como realizada desde el final de la ventana observada, entonces no hay fuga temporal, siempre que el objetivo sea futuro respecto a ese instante.
2. Si la predicción se interpreta como realizada desde el inicio de la estancia en UCI, entonces usar información cercana al evento o censura posterior implicaría fuga temporal.

El problema es que, en el notebook, la ventana temporal se define usando el tiempo observado final del paciente, pero el target parece seguir asociado al tiempo total desde el ingreso. Esto crea una ambigüedad metodológica importante.

Ejemplo:

* Un paciente muere en el día 10.
* Si se toman las 72 horas anteriores a las 24 horas previas al evento, la ventana puede contener información aproximadamente de los días 6 a 9.
* Si después se interpreta que el modelo predice desde el ingreso, se estaría usando información futura para predecir desde el día 0.

La conclusión metodológica es clara:

> Usar información de días posteriores al ingreso para predecir desde el ingreso sería fuga temporal. Solo es válido si la predicción se define desde el momento en el que esa información ya está disponible.

Por eso se abandona la idea de utilizar ventanas ancladas al evento o censura para el experimento principal del TFG.

---

## 3. Reinterpretación de DySurv

DySurv como arquitectura sigue siendo una referencia central del TFG. Su idea principal es combinar variables estáticas y trayectorias longitudinales mediante un encoder temporal, una representación latente y un módulo de supervivencia. Esta idea es plenamente relevante para MIMIC-IV.

Sin embargo, se distingue entre:

### DySurv como arquitectura

Es válido conservar:

* uso de variables estáticas y temporales;
* entrada secuencial de longitud fija;
* encoder tipo LSTM/RNN;
* representación latente del paciente;
* módulo de supervivencia discreta;
* salida temporal por intervalos;
* evaluación con métricas de supervivencia.

### DySurv como protocolo exacto de notebook

Es más delicado replicar sin cambios:

* selección de ventana temporal en función del tiempo observado final;
* eliminación de pacientes fuera del horizonte;
* interpretación de la salida temporal si el target no se redefine desde el final de la ventana;
* posible dependencia retrospectiva del evento o censura para construir la entrada.

Por tanto, la nueva metodología no rechaza DySurv, sino que adapta su lógica para construir un experimento temporal más prospectivo y limpio.

La formulación final será:

> Se conserva la idea arquitectónica de DySurv, basada en resumir una trayectoria temporal clínica mediante un modelo secuencial, pero se redefine la ventana de entrada como las primeras 72 horas de UCI para evitar dependencia del tiempo final observado.

---

## 4. Literatura utilizada para justificar la nueva estrategia

La decisión metodológica se apoya en varias referencias.

### DySurv

DySurv propone un modelo dinámico de supervivencia que utiliza información estática y longitudinal del paciente para estimar riesgo individual. Es la referencia principal del TFG y justifica el uso de trayectorias temporales y modelos secuenciales en supervivencia clínica.

La adaptación del TFG conserva la idea central de DySurv, pero modifica la construcción temporal para evitar ambigüedad sobre el instante de predicción.

### Dynamic-DeepHit

Dynamic-DeepHit extiende DeepHit al caso longitudinal. La idea relevante para el TFG es que una historia temporal del paciente puede alimentar una red recurrente o módulo de atención para estimar una distribución discreta de tiempo hasta evento.

Esta referencia justifica incluir un modelo dinámico alternativo a DySurv, con salida tipo DeepHit, para comparar dos formas de utilizar series temporales en supervivencia.

### Thorsen-Meyer et al., “Discrete-time survival analysis in the critically ill”

Este paper es especialmente importante para la nueva estrategia porque utiliza puntos de predicción fijos en UCI, como 0, 24, 48 y 72 horas, y evalúa supervivencia futura a distintos horizontes.

La lección metodológica es que tiene sentido definir un baseline temporal fijo, usar solo la información disponible hasta ese baseline y predecir riesgo futuro desde ahí.

Esta lógica encaja directamente con el nuevo diseño:

* baseline: 72 horas desde ingreso UCI;
* entrada: primeras 72 horas;
* población: pacientes vivos y no censurados a las 72 horas;
* objetivo: evento futuro desde las 72 horas.

### Deasy et al., “Dynamic survival prediction in intensive care units from heterogeneous time series”

Este trabajo usa datos temporales heterogéneos de UCI y actualiza la predicción durante las primeras horas de estancia. Aunque no usa exactamente una salida de supervivencia discreta como DeepHit, refuerza la idea de usar datos horarios iniciales de UCI y separar claramente información observada y predicción futura.

La lección para el TFG es que los modelos temporales en UCI deben construirse respetando el orden temporal: datos disponibles hasta un instante de predicción, y objetivo posterior a ese instante.

### pycox

La librería pycox proporciona implementaciones estándar de modelos de supervivencia neuronal y herramientas de evaluación como C-index dependiente del tiempo, IBS e IBLL/NBLL.

Se usará para aumentar reproducibilidad y reducir riesgo de errores derivados de implementaciones propias.

---

## 5. Nueva definición del experimento principal

La nueva metodología abandona los landmarks diarios como experimento principal y adopta una única predicción por paciente.

### Instante de predicción

Se fija:

```text
t_pred = 72 horas desde el ingreso en UCI
```

Este instante es suficientemente tardío para disponer de una ventana temporal informativa y suficientemente temprano para seguir siendo clínicamente razonable.

### Población incluida

Solo se incluyen pacientes que siguen en riesgo en la hora 72:

```text
Y_i > 72h
```

donde (Y_i) es el tiempo observado hasta evento o censura.

Se excluyen:

```text
pacientes con evento en Y_i <= 72h
pacientes censurados en Y_i <= 72h
```

La pregunta experimental queda condicionada a supervivencia hasta la hora 72:

> Dado que el paciente ha sobrevivido y sigue observable a las 72 horas, ¿qué riesgo tiene durante los siguientes 10 días?

### Entrada temporal

La entrada temporal de los modelos dinámicos será:

```text
primeras 72 horas de UCI
```

organizadas como:

```text
72 timesteps horarios × variables temporales
```

Si un paciente no tiene medición en una hora concreta, se usará imputación, máscaras de missing o padding según el modelo.

### Entrada estática

Los modelos estáticos usarán variables estáticas o agregadas disponibles hasta la hora 72. Para que la comparación sea justa, no deben usar información posterior a la hora 72.

La comparación fuerte no será contra el benchmark estático antiguo sobre toda la cohorte, sino contra nuevos modelos estáticos entrenados y evaluados sobre la misma cohorte viva a 72 horas.

### Target relativo

El tiempo objetivo debe redefinirse desde la hora 72:

[
Y_i^{rel} = Y_i - 72h
]

El evento relativo será:

[
\delta_i^{rel} = 1
]

si el paciente experimenta el evento después de la hora 72 y dentro del horizonte de predicción.

### Horizonte de predicción

Se define un horizonte de 10 días después de la hora 72:

```text
horizonte = 240 horas
```

La salida se organizará en 10 bins diarios:

```text
bin 1: (0h, 24h] después de t_pred
bin 2: (24h, 48h]
bin 3: (48h, 72h]
...
bin 10: (216h, 240h]
```

Esto permite mantener coherencia con DySurv y DeepHit, que trabajan con salidas discretas de supervivencia/riesgo.

### Pacientes sin evento dentro del horizonte

No se deben eliminar del experimento principal los pacientes que no tienen evento en los 10 días posteriores a la hora 72.

La regla será:

```text
si evento ocurre dentro de 240h después de t_pred:
    evento en el bin correspondiente

si no ocurre evento dentro de 240h:
    censura administrativa en 240h
    o categoría de cola si el modelo predice masa de probabilidad discreta
```

Formalmente:

[
Y_i^{rel,h} = \min(Y_i - 72h, 240h)
]

[
\delta_i^{rel,h} = 1{\delta_i = 1 \ \text{y} \ Y_i - 72h \leq 240h}
]

Esta decisión evita sesgar la cohorte eliminando supervivientes largos.

---

## 6. Implicaciones para los modelos estáticos

Se implementará una nueva capa de benchmarks estáticos siguiendo la lógica del notebook de DySurv y usando librerías estándar siempre que sea posible.

El objetivo es mejorar seriedad, reproducibilidad y comparabilidad.

### Modelos estáticos previstos

Los modelos estáticos se entrenarán sobre la cohorte viva a 72 horas y con targets relativos desde la hora 72.

Modelos:

```text
Kaplan-Meier
CoxPH
DeepSurv / CoxPH neuronal con pycox
PCHazard o LogisticHazard con pycox
DeepHitSingle con pycox
```

La decisión concreta entre PCHazard y LogisticHazard debe depender de qué se quiera replicar exactamente del notebook y de qué modelo se quiera conservar como benchmark principal por intervalos.

### Justificación

Usar librerías estándar reduce:

* errores de implementación propia;
* diferencias accidentales respecto al benchmark original;
* dificultad para reproducir resultados;
* carga de justificación técnica.

Además, permite centrar el TFG en el diseño experimental y la interpretación de resultados, no en demostrar que cada implementación propia es correcta.

### Relación con los resultados anteriores

Los resultados estáticos ya obtenidos no se eliminan. Se conservarán como:

```text
experimento previo
implementación propia auditada
baseline histórico
material de comparación secundaria o apéndice
```

La nueva versión será la principal si se completa correctamente.

---

## 7. Implicaciones para los modelos dinámicos

Los modelos dinámicos dejarán de formularse con landmarks diarios en la versión principal.

La nueva formulación será:

```text
una muestra temporal por paciente
entrada = primeras 72 horas
predicción = próximos 10 días desde hora 72
```

### DySurv adaptado

DySurv usará:

```text
variables estáticas
serie temporal de 72 timesteps horarios
encoder temporal
representación latente
módulo de supervivencia discreta
```

La adaptación se justificará como una versión prospectiva del planteamiento de DySurv.

### Dynamic-DeepHit adaptado

Dynamic-DeepHit usará:

```text
serie temporal hasta t_pred
máscaras de missing
RNN/LSTM o atención temporal
salida discreta tipo DeepHit
```

Para un único evento, la salida será una distribución discreta sobre los 10 bins futuros.

### Comparación principal

La comparación fuerte será:

```text
DeepHit estático vs Dynamic-DeepHit temporal
DySurv static vs DySurv temporal
modelos estáticos vs modelos temporales
```

Todas estas comparaciones deben hacerse sobre:

```text
misma cohorte viva a 72h
mismo split
mismo target relativo
mismo horizonte
mismas métricas
```

---

## 8. Métricas de evaluación

La nueva versión debe replicar las métricas principales usadas en DySurv y en los notebooks asociados.

### Métricas principales

Se usarán:

```text
C-index dependiente del tiempo de Antolini
IBS
IBLL / NBLL
```

Estas métricas se calcularán preferentemente con librerías estándar, especialmente pycox EvalSurv, para mantener reproducibilidad.

### C-index por horizonte como ampliación propia

Además de las métricas principales, se calculará el C-index por día futuro:

```text
C-index día 1
C-index día 2
...
C-index día 9 o día 10
```

Esta ampliación es metodológicamente interesante porque permite analizar cómo cambia la discriminación a lo largo del horizonte.

Debe presentarse como una extensión propia, no como una métrica original del paper DySurv.

### Curvas individuales de supervivencia

Se incluirán ejemplos de curvas de supervivencia para pacientes concretos.

Su función será cualitativa:

* comprobar monotonicidad;
* comparar perfiles de riesgo;
* observar separación entre pacientes;
* detectar comportamientos extraños;
* ilustrar diferencias entre modelos.

No deben sustituir a las métricas agregadas.

---

## 9. Decisiones metodológicas finales adoptadas

### Decisión 1: abandonar landmarks diarios en el experimento principal

Justificación:

* reducen reproducibilidad y aumentan complejidad;
* multiplican muestras por paciente;
* complican la comparación con modelos estáticos;
* no son necesarios para estudiar si la trayectoria temporal inicial aporta información.

Nueva regla:

```text
una predicción por paciente a las 72h
```

### Decisión 2: usar una ventana fija prospectiva

Justificación:

* evita ventanas definidas en función del evento;
* elimina ambigüedad temporal;
* se alinea con literatura UCI que usa baselines fijos;
* permite una interpretación clara del instante de predicción.

Nueva regla:

```text
input temporal = primeras 72h de UCI
```

### Decisión 3: incluir solo pacientes en riesgo a las 72h

Justificación:

* no se puede predecir desde 72h para pacientes que ya han tenido evento o han sido censurados;
* define correctamente la población objetivo.

Nueva regla:

```text
incluir solo Y_i > 72h
```

### Decisión 4: redefinir el target desde la hora 72

Justificación:

* si la predicción se realiza en la hora 72, el tiempo hasta evento debe ser futuro respecto a la hora 72;
* evita mezclar información de entrada y tiempo total desde ingreso.

Nueva regla:

```text
Y_rel = Y - 72h
```

### Decisión 5: usar horizonte de 10 días con bins diarios

Justificación:

* mantiene coherencia con DySurv y modelos discretos;
* es interpretable clínicamente;
* permite comparar curvas y métricas por horizonte.

Nueva regla:

```text
10 bins diarios desde la hora 72
```

### Decisión 6: no eliminar supervivientes más allá del horizonte

Justificación:

* eliminarlos sesgaría la cohorte;
* la censura administrativa es una opción estándar;
* para DeepHit/Dynamic-DeepHit puede usarse una categoría de cola.

Nueva regla:

```text
si no hay evento antes de 10 días, censurar en 10 días o usar tail category
```

### Decisión 7: usar librerías para la réplica estática principal

Justificación:

* mayor reproducibilidad;
* menor riesgo de errores propios;
* mayor proximidad a notebooks de referencia;
* más fácil defensa académica.

Nueva regla:

```text
benchmarks estáticos principales implementados con librerías siempre que sea posible
```

### Decisión 8: conservar resultados anteriores

Justificación:

* el trabajo previo puede servir como comparación secundaria;
* puede ser útil para discusión;
* no conviene perder evidencia ni experimentos ya auditados.

Nueva regla:

```text
no borrar ni sobrescribir la versión anterior; crear una nueva versión experimental
```

---

## 10. Qué cambia respecto a la metodología anterior

Antes, el proyecto tenía una capa estática completa sobre toda la cohorte y se planteaba una extensión dinámica con landmarks diarios.

Ahora, el experimento principal se reorganiza así:

```text
Cohorte original MIMIC-IV
        ↓
Filtrar pacientes con Y > 72h
        ↓
Construir input estático hasta 72h
Construir input temporal primeras 72h
        ↓
Definir target relativo desde 72h
        ↓
Entrenar modelos estáticos y temporales sobre la misma cohorte
        ↓
Evaluar con Antolini C-index, IBS, IBLL/NBLL
        ↓
Analizar C-index por horizonte y curvas individuales
```

Esto convierte el TFG en un experimento más compacto, reproducible y defendible.

---

## 11. Riesgos y limitaciones que deben reconocerse

La nueva metodología también tiene limitaciones.

### Sesgo de supervivencia a 72h

Al incluir solo pacientes vivos y observables a 72h, la cohorte excluye eventos tempranos.

Esto debe explicarse así:

> El experimento temporal evalúa riesgo futuro condicionado a supervivencia hasta las 72 horas, no mortalidad desde el ingreso para toda la cohorte.

### Menor comparabilidad con resultados antiguos

Los resultados anteriores sobre toda la cohorte no son directamente comparables con los nuevos resultados sobre pacientes vivos a 72h.

La comparación justa exige reentrenar o reevaluar modelos estáticos sobre la nueva cohorte.

### Adaptación respecto a DySurv original

La estrategia no replica literalmente todos los detalles del notebook de DySurv, porque modifica la construcción de la ventana temporal.

Debe explicarse como:

> adaptación prospectiva de la lógica de DySurv para evitar dependencia de la ventana respecto al tiempo observado final.

### Horizonte limitado

El horizonte de 10 días condiciona las conclusiones. No se podrán extrapolar los resultados a mortalidad a largo plazo sin nuevos experimentos.

### Dependencia del preprocesamiento temporal

Los resultados dependerán de cómo se imputen, agreguen y normalicen las variables horarias de MIMIC-IV.

---

## 12. Frase recomendada para la memoria

Una formulación adecuada sería:

> Para evitar ambigüedades temporales y asegurar una comparación prospectiva entre modelos, el experimento dinámico se reformuló utilizando un instante de predicción fijo situado a las 72 horas del ingreso en UCI. Solo se incluyeron pacientes que seguían en riesgo en ese instante. Los modelos temporales recibieron como entrada la trayectoria clínica de las primeras 72 horas, mientras que el objetivo se definió como el tiempo restante hasta evento o censura desde ese punto. De este modo, la comparación entre modelos estáticos y temporales se realiza sobre la misma cohorte, con el mismo horizonte de predicción y sin utilizar información posterior al instante de predicción.

Otra frase útil:

> La adaptación conserva la motivación principal de DySurv, consistente en aprovechar trayectorias longitudinales del paciente para estimar supervivencia individual, pero modifica la construcción de la ventana temporal para ajustarla a un protocolo prospectivo más claro sobre MIMIC-IV.

---

## 13. Fuentes utilizadas

Las fuentes principales que justifican esta nueva metodología son:

1. DySurv: dynamic deep learning model for survival analysis with conditional variational inference.
2. Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis With Competing Risks.
3. Thorsen-Meyer et al., Discrete-time survival analysis in the critically ill.
4. Deasy et al., Dynamic survival prediction in intensive care units from heterogeneous time series.
5. pycox: Survival analysis with PyTorch.
6. DeepHit: A Deep Learning Approach to Survival Analysis With Competing Risks.
7. DeepSurv: Personalized Treatment Recommender System Using a Cox Proportional Hazards Deep Neural Network.

---

## 14. Conclusión metodológica

La nueva metodología prioriza seriedad, reproducibilidad y claridad temporal.

El TFG deja de depender de una construcción compleja por landmarks diarios y pasa a evaluar una pregunta concreta y defendible:

> ¿Mejora la predicción de supervivencia futura incorporar la trayectoria clínica de las primeras 72 horas frente a usar solo información estática?

Esta pregunta es suficientemente interesante para un TFG de Matemáticas, está alineada con DySurv y Dynamic-DeepHit, y se puede evaluar con métricas estándar de supervivencia usando librerías reproducibles.
