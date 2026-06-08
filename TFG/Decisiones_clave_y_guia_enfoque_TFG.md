# Decisiones clave y guía de enfoque del TFG

## Tema.

Replicación y evaluación de modelos de supervivencia estáticos y dinámicos en pacientes adultos de UCI usando MIMIC-IV.

El trabajo se centra en validar empíricamente el enfoque DySurv y analizar si la incorporación de trayectorias temporales mejora la predicción del riesgo frente a modelos basados en una representación inicial del paciente.

## Pregunta principal.

Hasta qué punto los modelos dinámicos de supervivencia, y en particular DySurv, aportan una mejora real frente a modelos estáticos cuando se aplican a datos clínicos de UCI.

La comparación debe centrarse en el valor añadido de la información temporal, no en hacer una competición desordenada entre muchos modelos.

## Objetivo principal.

Replicar y contrastar los resultados de DySurv en el entorno de MIMIC-IV, evaluando su comportamiento frente a modelos clásicos, neuronales y dinámicos de supervivencia.

El objetivo debe formularse como evaluación académica y experimental, no como validación clínica definitiva.

## Objetivos específicos.

Construir una cohorte de pacientes adultos de UCI a partir de MIMIC-IV.

Definir correctamente evento, censura, horizonte temporal y tiempo de seguimiento.

Preparar variables estáticas y trayectorias temporales del paciente.

Implementar modelos estáticos de referencia.

Implementar modelos dinámicos basados en landmarks diarios.

Adaptar la lógica experimental de DySurv al contexto de MIMIC-IV.

Comparar modelos mediante métricas adecuadas de supervivencia.

Analizar si la información temporal mejora la predicción del riesgo.

Discutir similitudes y diferencias respecto a los resultados del artículo original de DySurv.

Reconocer limitaciones metodológicas, clínicas y computacionales.

## Alcance.

El TFG es un trabajo académico de Matemáticas aplicado al análisis de supervivencia.

No pretende desarrollar una herramienta clínica lista para uso hospitalario.

No pretende demostrar que DySurv sea universalmente superior.

No pretende validar clínicamente el modelo fuera de MIMIC-IV.

Las conclusiones dependerán de la cohorte construida, las variables disponibles, el horizonte temporal, los modelos implementados y el protocolo de evaluación utilizado.

## Enfoque general.

El TFG debe plantearse como una comparación estructurada entre modelos estáticos y dinámicos de supervivencia.

Los modelos estáticos sirven para establecer una referencia basada en una única representación inicial del paciente.

Los modelos dinámicos permiten estudiar el efecto de incorporar la evolución temporal del paciente.

DySurv debe aparecer como el modelo central del trabajo, pero dentro de una comparación más amplia y defendible.

## Metodología general.

Primero se construye la cohorte de pacientes adultos de UCI a partir de MIMIC-IV.

Después se define el evento de interés, la censura y el horizonte de predicción.

A continuación se genera una representación estática del paciente, basada en información inicial.

En paralelo, se construyen trayectorias temporales para los modelos dinámicos.

Los modelos estáticos se entrenan con una única representación por paciente.

Los modelos dinámicos se entrenan usando landmarks diarios e historia acumulada hasta cada landmark.

La evaluación compara rendimiento predictivo, estabilidad, interpretación y limitaciones de cada familia de modelos.

## Modelos descriptivos.

Kaplan-Meier.

Debe usarse como baseline descriptivo para entender la supervivencia observada de la cohorte.

No debe presentarse como modelo predictivo principal.

Sirve para analizar supervivencia global, censura, eventos y número de pacientes en riesgo.

## Modelos estáticos principales.

Cox Proportional Hazards.

Debe mantenerse como modelo clásico de referencia.

DeepSurv.

Debe mantenerse porque conecta Cox con redes neuronales y supervivencia moderna.

PWE Poisson o piecewise-exponential.

Debe mantenerse como modelo por intervalos, especialmente si ya está implementado.

DeepHit.

Tiene sentido mantenerlo como control estático para comparar después con Dynamic-DeepHit.

## Modelos dinámicos principales.

Dynamic-DeepHit.

Debe usarse como benchmark dinámico principal frente a DySurv.

DySurv static.

Debe usarse como control para separar el efecto de la arquitectura DySurv del efecto de añadir series temporales.

DySurv con series temporales.

Debe ser el modelo central del TFG.

Usará landmarks diarios, historia acumulada hasta cada landmark y predicción diaria para los próximos 10 días.

## Modelos no principales.

CoxTime.

Puede mencionarse como alternativa que relaja riesgos proporcionales, pero no lo incluiría como modelo principal salvo que esté ya implementado sin coste adicional.

Logistic Hazard.

Puede mencionarse como familia relacionada de riesgo discreto, pero no lo incluiría como modelo principal si ya se usan PWE, DeepHit y Dynamic-DeepHit.

Random Survival Forest, PMF, MTLR, BCESurv y CoxCC.

Los dejaría fuera del análisis principal.

Pueden aparecer brevemente en estado de la cuestión si ayudan a contextualizar.

## Comparaciones clave.

CoxPH frente a DeepSurv.

Permite comparar el modelo clásico de Cox con su extensión neuronal.

DeepSurv frente a DeepHit.

Permite comparar un enfoque basado en Cox neuronal con un enfoque discreto de supervivencia.

DeepHit frente a Dynamic-DeepHit.

Permite estudiar el efecto de introducir dinámica temporal dentro de una misma familia de modelos.

DySurv static frente a DySurv time-series.

Permite aislar el efecto de añadir trayectorias temporales dentro de DySurv.

Dynamic-DeepHit frente a DySurv time-series.

Permite comparar dos modelos dinámicos avanzados.

Modelos estáticos frente a modelos dinámicos.

Permite responder a la pregunta principal del TFG.

## Decisiones temporales.

El enfoque dinámico se basará en landmarks diarios.

En cada landmark, el modelo usará la información acumulada hasta ese momento.

La predicción se hará para los próximos 10 días.

Esta decisión debe justificarse por coherencia con DySurv y por la disponibilidad de pacientes en riesgo.

Debe explicarse claramente que no se predice una única mortalidad global, sino riesgo condicionado a la información disponible en cada instante.

## Datos.

La base de datos principal será MIMIC-IV.

La cohorte debe limitarse a pacientes adultos de UCI.

Debe definirse con precisión el inicio del seguimiento.

Debe definirse con precisión el evento de interés.

Debe explicarse cómo se trata la censura.

Debe indicarse qué variables son estáticas y cuáles forman parte de la serie temporal.

Debe evitarse introducir variables que filtren información futura.

## Evaluación.

Usar métricas adecuadas para supervivencia.

Distinguir discriminación, calibración y error predictivo.

No depender de una sola métrica.

Analizar el rendimiento global y, si es posible, el rendimiento por horizonte temporal.

Interpretar los resultados teniendo en cuenta censura y pacientes en riesgo.

Evitar conclusiones fuertes cuando el número de eventos sea bajo.

## Limitaciones.

Puede existir censura informativa.

El número de pacientes en riesgo puede caer rápidamente con el tiempo.

Puede haber muchos más pacientes censurados que eventos observados.

Los resultados dependen del horizonte temporal elegido.

Los modelos dinámicos pueden ser más difíciles de entrenar y ajustar.

La comparación entre modelos con estructuras distintas no siempre es directa.

La ausencia de validación externa limita la generalización.

El uso de MIMIC-IV condiciona las conclusiones al contexto de esa base de datos.

## Contribución personal.

Construcción y depuración de la cohorte.

Definición del evento, censura y horizonte temporal.

Preparación de variables estáticas y temporales.

Adaptación de la metodología de DySurv a MIMIC-IV.

Implementación de modelos estáticos y dinámicos.

Diseño del protocolo experimental.

Cálculo e interpretación de métricas.

Comparación crítica entre resultados propios y resultados reportados en DySurv.

Discusión de limitaciones y posibles extensiones.

## Estructura de la memoria.

Introducción.

Debe presentar el problema, la motivación, el objetivo y la estructura del TFG.

Preliminares.

Deben explicar análisis de supervivencia, censura, función de supervivencia, función de riesgo y métricas básicas.

Estado de la cuestión.

Debe revisar Cox, DeepSurv, DeepHit, Dynamic-DeepHit, DySurv y modelos relacionados.

Metodología.

Debe explicar datos, cohorte, variables, modelos, landmarks, horizonte temporal y evaluación.

Resultados.

Deben presentar primero la cohorte y después la comparación de modelos.

Discusión.

Debe interpretar el efecto de la información temporal y la replicación de DySurv.

Conclusiones.

Deben responder a los objetivos sin introducir resultados nuevos.

## Redacción.

Evitar presentar el TFG como una investigación clínica definitiva.

Evitar estructuras repetitivas tipo “no solo X, sino Y”.

Evitar frases demasiado largas.

Definir todas las siglas y términos técnicos en su primera aparición.

Explicar los anglicismos cuando aparezcan por primera vez.

Mantener un tono académico claro, natural y preciso.

No añadir modelos, tablas o métricas que no aporten a la pregunta principal.

## Idea central que debe guiar todo el TFG.

El trabajo replica y evalúa DySurv en MIMIC-IV, usando una comparación ordenada entre modelos estáticos y dinámicos para estudiar si las trayectorias temporales del paciente aportan información predictiva adicional en el análisis de supervivencia en UCI.
