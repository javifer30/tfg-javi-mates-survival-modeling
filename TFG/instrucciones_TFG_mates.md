Sí. He adaptado tus reglas anteriores al nuevo TFG de Matemáticas sobre modelos de supervivencia en UCI. He añadido además las normas específicas de la Facultad: memoria recomendada de 20-40 páginas, castellano, resumen en inglés, defensa de 15 minutos, fórmula de calificación (T = 0.35U + 0.35M + 0.30P), carácter académico del TFG y necesidad de que todo lo evaluable esté incluido en la memoria.  

# Instrucciones de revisión del TFG de Matemáticas según normativa, rúbrica, comentarios del tutor y especificaciones del autor

## 0. Reglas globales obligatorias

1. Respetar la naturaleza académica del TFG.

   * El TFG no debe presentarse como una tesis doctoral ni como una investigación médica original.
   * Debe plantearse como un trabajo académico de grado en el que se integran conocimientos matemáticos, estadísticos y computacionales.
   * El objetivo central debe ser demostrar comprensión, aplicación y análisis crítico de modelos de supervivencia.
   * Evitar afirmaciones grandilocuentes como “se propone un nuevo modelo”, “se demuestra la superioridad” o “se resuelve el problema clínico”.
   * Formular el trabajo como una replicación, implementación, comparación y evaluación razonada de modelos de supervivencia sobre datos clínicos.

2. Ajustarse a la extensión y contenido formal exigido.

   * La memoria debe mantenerse, salvo indicación distinta del tutor, entre 20 y 40 páginas más anexos.
   * Debe estar escrita en castellano.
   * Debe incluir portada normalizada, introducción sobre antecedentes, objetivos y plan de trabajo, desarrollo del trabajo y breve resumen en inglés. 
   * Los anexos deben reservarse para material secundario: tablas largas, detalles de implementación, hiperparámetros, código relevante o resultados complementarios.
   * No trasladar al anexo contenido necesario para entender la metodología o los resultados principales.

3. Introducir todos los capítulos antes de la primera sección.

   * No juntar nunca dos títulos seguidos.
   * Después de cada título de capítulo debe haber un párrafo breve de introducción.
   * No crear una subsección llamada “Introducción” dentro de cada capítulo salvo que sea estrictamente necesario.
   * La introducción de capítulo debe explicar qué se hace, cómo se conecta con lo anterior y qué estructura seguirá.
   * En capítulos técnicos, la introducción debe orientar al lector antes de entrar en definiciones, ecuaciones o modelos.

4. Separar claramente preliminares, estado de la cuestión, metodología, resultados y discusión.

   * Los preliminares deben explicar los conceptos necesarios para entender el trabajo.
   * El estado de la cuestión debe revisar literatura y enfoques existentes.
   * La metodología debe explicar qué se ha hecho en este TFG.
   * Los resultados deben mostrar qué se ha obtenido.
   * La discusión debe interpretar los resultados, reconocer limitaciones y conectar con los objetivos.
   * No mezclar resultados propios dentro del estado de la cuestión.
   * No introducir definiciones básicas por primera vez dentro de los capítulos de resultados.

5. Definir siglas, anglicismos y términos clave.

   * Toda sigla debe definirse la primera vez que aparece: UCI, MIMIC-IV, CoxPH, PEH, Brier Score, IBS, C-index, etc.
   * Todo anglicismo debe explicarse la primera vez que aparece.
   * Términos como dataset, benchmark, baseline, pipeline, split, train, test, censoring, hazard, survival, deep learning o calibration deben traducirse o introducirse con su equivalente en español.
   * Si se mantiene el término inglés por ser habitual en la literatura, debe explicarse brevemente.
   * Una vez definido el término, usarlo de forma consistente en todo el trabajo.

6. Cuidar la presentación matemática.

   * Toda variable debe definirse antes o inmediatamente después de aparecer.
   * No introducir ecuaciones largas sin explicación previa.
   * Cada ecuación importante debe tener una interpretación en texto.
   * Evitar símbolos distintos para el mismo concepto.
   * Mantener una notación homogénea para tiempo, evento, censura, función de supervivencia, función de riesgo y covariables.
   * No abusar de demostraciones si no aportan directamente al objetivo del TFG.
   * En un TFG de Matemáticas, los modelos deben tener formulación rigurosa, pero la memoria no debe convertirse en una acumulación de fórmulas sin interpretación.

7. No usar negrita fuera de usos permitidos.

   * No utilizar negrita en el cuerpo del texto.
   * La negrita solo se permite en bullet points, títulos de tablas, títulos de secciones o títulos de subsecciones.
   * En el cuerpo del texto, sustituir la negrita por redacción natural.
   * No destacar palabras artificialmente para guiar al lector; la estructura del párrafo debe hacerlo.

8. No usar formatos de letra innecesarios.

   * No usar `\texttt` en el cuerpo de la memoria salvo para nombres de archivos, funciones o fragmentos de código cuando sea imprescindible.
   * No usar formatos especiales para enfatizar palabras.
   * La cursiva mediante `\textit{}` debe reservarse para usos justificados.
   * Evitar que el documento parezca un manual de código o una presentación técnica.

9. Usar cursiva solo en situaciones justificadas.

   * Títulos de publicaciones independientes, bases de datos o recursos: por ejemplo, `\textit{MIMIC-IV}` si se trata como base de datos.
   * Palabras extranjeras no adaptadas al español en su primera aparición.
   * Terminología latina o científica.
   * Primera aparición de términos clave cuando se definan explícitamente.
   * Variables matemáticas y símbolos estadísticos.
   * No abusar de la cursiva para enfatizar ideas.

10. Evitar estructuras repetitivas asociadas a redacción de IA.

* Revisar y reducir estructuras del tipo:

  * “no solo X, sino Y”
  * “no es X, sino Y”
  * “no tanto X como Y”
  * “más que X, Y”
  * “no pretende X, sino Y”
  * “por un lado..., por otro lado...” repetido muchas veces
  * “en este sentido...” usado como cierre automático
  * “cabe destacar que...” usado sin necesidad
* Sustituir por frases directas.
* No negar una idea para afirmar otra salvo que el contraste sea realmente necesario.
* Evitar párrafos que terminen con frases genéricas.
* Priorizar naturalidad, precisión y variedad sintáctica.

11. No extender más de lo necesario.

* Cada párrafo debe aportar información nueva.
* Eliminar frases de cierre que solo repiten la idea anterior.
* No repetir el objetivo general del TFG al final de cada capítulo.
* Evitar recapitulaciones largas si el capítulo ya es claro.
* Si una definición ya aparece en preliminares, no volver a definirla en resultados.
* En la discusión, interpretar; no volver a describir todas las tablas.

---

## 1. Estructura recomendada de la memoria

1. Portada.

   * Debe incluir título exacto, titulación, curso académico, nombre del estudiante, tutores, fecha y logo de la UCM.
   * No incluir otros logos en la portada. 

2. Resumen y abstract.

   * Incluir un resumen en castellano.
   * Incluir un breve resumen en inglés, obligatorio según la normativa de la Facultad.
   * El resumen debe mencionar problema, datos, modelos comparados, metodología general y conclusión principal.
   * No incluir resultados demasiado detallados ni ecuaciones.

3. Introducción.

   * Presentar el contexto clínico y matemático.
   * Explicar por qué el análisis de supervivencia es adecuado para estudiar mortalidad o riesgo en UCI.
   * Introducir la motivación de comparar modelos clásicos, modelos de aprendizaje profundo y modelos bayesianos piecewise-exponential.
   * Formular claramente la pregunta del trabajo.
   * Cerrar con la estructura de la memoria.

4. Objetivos y plan de trabajo.

   * Separar objetivo general y objetivos específicos.
   * Los objetivos deben ser evaluables: implementar, comparar, analizar, discutir, reproducir, adaptar.
   * Evitar objetivos imposibles de demostrar con el alcance del TFG.
   * Incluir un plan de trabajo breve: revisión, preparación de datos, implementación, evaluación y análisis.

5. Preliminares matemáticos y estadísticos.

   * Definir análisis de supervivencia.
   * Explicar censura, función de supervivencia, función de riesgo, riesgo acumulado y datos censurados.
   * Presentar Kaplan-Meier, Cox, modelos piecewise-exponential y métricas de supervivencia.
   * Mantener esta parte clara y didáctica, porque será la base para el resto del TFG.

6. Estado de la cuestión.

   * Revisar literatura relacionada.
   * Agrupar trabajos por familias: modelos clásicos, modelos de machine learning para supervivencia, deep survival models, modelos dinámicos, modelos bayesianos.
   * No limitarse a una tabla de papers.
   * Explicar qué aporta cada grupo de trabajos y cómo se conecta con tu TFG.
   * Cerrar identificando el hueco práctico que justifica tu comparación.

7. Metodología común.

   * Añadir un capítulo o sección metodológica antes de los resultados.
   * Debe explicar el flujo general del trabajo.
   * Incluir dataset, criterios de selección, variable evento, censura, covariables, particiones, modelos, métricas y estrategia de comparación.
   * Debe servir de marco común para todos los experimentos.
   * Los capítulos posteriores deben centrarse en resultados e interpretación.

8. Resultados.

   * Separar resultados descriptivos, resultados de modelos y comparación final.
   * Cada figura o tabla debe tener un propósito claro.
   * No presentar tablas sin explicación.
   * No convertir los resultados en una sucesión de números; explicar qué significan.

9. Discusión.

   * Interpretar los resultados con cautela.
   * Explicar por qué un modelo funciona mejor o peor.
   * Relacionar los resultados con las propiedades del dataset: censura, desbalance de eventos, pocos pacientes en riesgo a largo plazo, covariables estáticas, horizonte temporal.
   * Reconocer limitaciones.
   * No vender conclusiones que los datos no sostienen.

10. Conclusiones.

* Responder de forma directa a los objetivos.
* Indicar qué se ha aprendido.
* Mencionar extensiones futuras razonables.
* No introducir resultados nuevos.
* No hacer una discusión larga encubierta.

---

## 2. Instrucciones específicas para la introducción

1. La introducción debe ser progresiva.

   * Empezar por el problema general: predicción de riesgo y supervivencia en UCI.
   * Pasar al enfoque matemático: análisis de supervivencia.
   * Explicar la dificultad: censura, evolución temporal, datos clínicos, interpretabilidad.
   * Presentar la comparación de modelos como respuesta natural al problema.
   * Terminar con objetivos y estructura.

2. No cargar la introducción con demasiada teoría.

   * Kaplan-Meier, Cox, DeepSurv o modelos bayesianos deben mencionarse solo de forma contextual.
   * Las definiciones formales deben ir en preliminares.
   * La literatura detallada debe ir en estado de la cuestión.

3. Evitar una introducción médica excesiva.

   * El TFG es de Matemáticas.
   * El contexto clínico debe justificar el problema, no dominar el trabajo.
   * No hacer afirmaciones clínicas fuertes si no están apoyadas en fuentes.

4. Dejar claro el alcance.

   * El trabajo evalúa modelos sobre un conjunto concreto de datos.
   * No busca construir una herramienta clínica lista para uso médico.
   * No afirma validez externa sin validación externa.
   * No sustituye juicio clínico.

---

## 3. Instrucciones específicas para preliminares matemáticos

1. Definir primero los objetos básicos.

   * Tiempo hasta evento.
   * Evento de interés.
   * Censura.
   * Tiempo observado.
   * Indicador de evento.
   * Función de supervivencia.
   * Función de riesgo.
   * Riesgo acumulado.
   * Covariables estáticas.
   * Covariables temporales.
   * Landmark o instante de actualización de la predicción.
   * Horizonte de predicción.
   * Trayectoria temporal del paciente.

2. Introducir la notación con orden.

   * Usar (T) para el tiempo real hasta el evento.
   * Usar (C) para el tiempo de censura.
   * Usar (Y = \min(T,C)) para el tiempo observado.
   * Usar (\delta) para el indicador de evento.
   * Usar (X) para covariables estáticas.
   * Usar (X(t)) o (X_{0:t}) para representar la información temporal acumulada hasta el instante (t).
   * Usar (t_l) para los landmarks diarios.
   * Usar (h) para el horizonte de predicción.
   * Mantener esta notación durante todo el trabajo.
   * Distinguir claramente entre el tiempo de seguimiento del paciente y el horizonte futuro que se predice desde cada landmark.

3. Explicar cada modelo con el nivel justo.

   * Kaplan-Meier: estimación no paramétrica de la supervivencia observada de la cohorte. Debe usarse como análisis descriptivo, no como modelo predictivo principal.
   * CoxPH: modelo clásico semiparamétrico basado en la hipótesis de riesgos proporcionales. Debe actuar como referencia académica.
   * DeepSurv: extensión neuronal de Cox que sustituye el predictor lineal por una red neuronal. Sirve para conectar supervivencia clásica y aprendizaje profundo.
   * PWE Poisson o piecewise-exponential: modelo que discretiza el tiempo en intervalos y estima el riesgo por tramos. Debe presentarse como puente entre modelos continuos y formulaciones discretas del riesgo.
   * DeepHit: modelo estático de supervivencia discreta. Debe explicarse como antecedente directo de Dynamic-DeepHit.
   * Dynamic-DeepHit: extensión dinámica de DeepHit que incorpora información longitudinal del paciente. Debe funcionar como benchmark dinámico frente a DySurv.
   * DySurv static: versión de DySurv sin series temporales. Debe explicarse como control para aislar el efecto de la arquitectura.
   * DySurv con series temporales: modelo central del TFG. Debe explicarse como enfoque dinámico que actualiza la predicción usando la historia acumulada del paciente en landmarks diarios.
   * Logistic Hazard, CoxTime, Random Survival Forest, PMF, MTLR, BCESurv y CoxCC pueden mencionarse como modelos relacionados, pero no deben desarrollarse al mismo nivel si no forman parte del análisis principal.

4. Evitar demostraciones demasiado largas.

   * Incluir demostraciones solo si ayudan a entender la lógica del modelo.
   * No desarrollar en exceso la verosimilitud parcial de Cox si rompe el ritmo del capítulo.
   * No convertir el capítulo de preliminares en una revisión matemática exhaustiva de todos los modelos.
   * Para modelos complejos como DeepHit, Dynamic-DeepHit o DySurv, priorizar la intuición matemática, la función objetivo y el tipo de entrada que usan.
   * Llevar a anexos derivaciones secundarias, detalles de implementación o formulaciones demasiado largas.
   * El lector debe acabar entendiendo qué estima cada modelo, qué información usa y por qué es relevante para la comparación estático-dinámica.

5. Cuidar el lenguaje matemático.

   * Evitar frases infladas o artificiales.
   * No usar términos como “asintótico”, “paramétrico”, “estructural”, “bayesiano” o “probabilístico” si no tienen una función clara en la frase.
   * Cada término técnico debe aparecer definido o contextualizado.
   * Explicar las ecuaciones con texto antes o después de presentarlas.
   * No abusar de fórmulas si la idea puede explicarse de forma clara.
   * Mantener un equilibrio entre rigor matemático y lectura fluida.
   * Recordar que el foco del TFG es replicar y evaluar DySurv en MIMIC-IV, no demostrar formalmente todos los modelos desde cero.

---

## 4. Instrucciones específicas para el estado de la cuestión

1. No hacer una lista de papers.

   * Cada bloque de literatura debe tener una explicación.
   * La tabla de literatura puede complementar, pero no sustituir al texto.
   * El lector debe entender la evolución desde modelos clásicos hacia modelos dinámicos.
   * No limitarse a decir qué modelo existe; explicar qué problema intenta resolver.
   * Conectar cada familia de modelos con la pregunta del TFG: si las trayectorias temporales mejoran la predicción de supervivencia en UCI.

2. Orden recomendado.

   * Modelos clásicos de supervivencia: Kaplan-Meier y CoxPH.
   * Modelos neuronales estáticos: DeepSurv y extensiones de Cox mediante redes neuronales.
   * Modelos de supervivencia por tiempo discreto o por intervalos: PWE Poisson, Logistic Hazard y DeepHit.
   * Modelos dinámicos o longitudinales: Dynamic-DeepHit y enfoques basados en landmarks.
   * DySurv: explicar su propuesta, su motivación, su uso de información temporal y por qué es el modelo central del TFG.
   * Estudios recientes de comparación o benchmark, como SurvBench, si ayudan a justificar el protocolo experimental o la selección de modelos.
   * Modelos secundarios no incluidos, como CoxTime, RSF, MTLR, PMF, BCESurv o CoxCC, solo como contexto breve si aportan algo.

3. Conectar literatura con tu trabajo.

   * Explicar por qué se eligen CoxPH, DeepSurv, PWE Poisson, DeepHit, Dynamic-DeepHit y DySurv.
   * Justificar que Kaplan-Meier se usa de forma descriptiva.
   * Explicar que DeepHit se mantiene porque permite comparar con Dynamic-DeepHit.
   * Explicar que DySurv static se incluye para aislar el efecto de la arquitectura frente al efecto de las series temporales.
   * Explicar que DySurv con series temporales es el modelo central porque el TFG busca replicar y evaluar su planteamiento en MIMIC-IV.
   * Justificar qué modelos quedan fuera para evitar una comparación excesivamente amplia.
   * No mencionar con demasiado detalle modelos que luego no aparecen en la metodología.

4. Evitar afirmaciones absolutas.

   * No decir que los modelos dinámicos son mejores en general.
   * No afirmar que DySurv supera siempre a otros enfoques.
   * Usar expresiones como “en el marco de este trabajo”, “bajo esta configuración experimental”, “sobre la cohorte construida” o “con las variables disponibles”.
   * Distinguir rendimiento predictivo, calibración, interpretabilidad, coste computacional y facilidad de implementación.
   * Presentar la replicación de DySurv como validación empírica en un contexto concreto, no como confirmación universal de sus resultados.
   * Si los resultados no reproducen completamente los del artículo original, explicarlo como parte natural del análisis.

---

## 5. Instrucciones específicas para la metodología

1. La metodología debe ser reproducible.

   * Explicar la fuente de datos: MIMIC-IV.
   * Explicar criterios de inclusión y exclusión.
   * Indicar que la cohorte se centra en pacientes adultos de UCI.
   * Definir el inicio del seguimiento.
   * Definir el evento de interés.
   * Definir la censura.
   * Definir el horizonte temporal global.
   * Definir los landmarks diarios.
   * Explicar cómo se construyen las predicciones para los próximos 10 días.
   * Explicar qué variables son estáticas.
   * Explicar qué variables forman parte de la serie temporal.
   * Explicar cómo se agregan, imputan o normalizan las variables.
   * Explicar cómo se evitan filtraciones de información futura.
   * Explicar la partición de datos en entrenamiento, validación y test.
   * Explicar los modelos implementados.
   * Explicar las métricas utilizadas.
   * Explicar software, librerías y recursos computacionales relevantes.

2. Separar decisiones metodológicas de resultados.

   * La metodología explica qué se hace.
   * Los resultados muestran qué se obtiene.
   * La discusión interpreta por qué ocurre.
   * No adelantar conclusiones dentro de la metodología.
   * No justificar un modelo por sus resultados antes de presentarlos.
   * No mezclar explicación de métricas con interpretación de valores concretos.

3. Justificar decisiones importantes.

   * Por qué se utiliza MIMIC-IV.
   * Por qué se estudian pacientes adultos de UCI.
   * Por qué se construyen modelos estáticos y dinámicos.
   * Por qué los modelos estáticos usan una representación inicial del paciente.
   * Por qué los modelos dinámicos usan landmarks diarios.
   * Por qué se predice el riesgo diario para los próximos 10 días.
   * Por qué se mantiene Kaplan-Meier como análisis descriptivo.
   * Por qué CoxPH funciona como baseline clásico.
   * Por qué DeepSurv es el baseline neuronal estático principal.
   * Por qué PWE Poisson representa la familia piecewise-exponential.
   * Por qué DeepHit se incluye como control estático de Dynamic-DeepHit.
   * Por qué Dynamic-DeepHit se usa como benchmark dinámico.
   * Por qué DySurv static permite aislar el efecto de la arquitectura.
   * Por qué DySurv con series temporales es el modelo central.
   * Por qué se dejan fuera modelos como CoxTime, Logistic Hazard o Random Survival Forest del análisis principal.
   * Por qué se eligen las métricas de evaluación utilizadas.
   * Por qué la partición de datos elegida es adecuada para evitar sesgos.

4. Explicar limitaciones metodológicas desde el principio.

   * Covariables estáticas frente a trayectorias temporales.
   * Posible censura informativa.
   * Desbalance entre eventos y censuras.
   * Reducción del conjunto de pacientes en riesgo con el paso del tiempo.
   * Pérdida de robustez en horizontes con pocos eventos.
   * Sensibilidad a decisiones de preprocesamiento.
   * Sensibilidad a hiperparámetros.
   * Riesgo de sobreajuste, especialmente en modelos profundos.
   * Diferencias entre la implementación propia y la del artículo original de DySurv.
   * Dificultad de replicar exactamente DySurv si no se dispone de los mismos datos, variables o detalles experimentales.
   * Falta de validación externa, si no se realiza.
   * Limitación de las conclusiones al protocolo experimental definido.

5. No ocultar información evaluable.

   * Todo lo necesario para valorar el TFG debe aparecer en la memoria o en anexos.
   * La definición de la cohorte, evento, censura, variables y particiones debe quedar clara.
   * El preprocesamiento debe estar suficientemente documentado.
   * Las diferencias respecto a DySurv deben explicarse con transparencia.
   * Las métricas deben definirse antes de usarse.
   * Los hiperparámetros principales deben aparecer, al menos en tabla o anexo.
   * Si algún detalle técnico no puede incluirse en el cuerpo principal, debe resumirse y remitirse al anexo.
   * No presentar resultados de modelos cuya implementación no puedas explicar en la defensa.

---

## 6. Instrucciones específicas para resultados

1. Empezar con descriptivos.

   * Número de pacientes.
   * Número de eventos.
   * Número de censurados.
   * Distribución de tiempos.
   * Curva Kaplan-Meier.
   * Número de pacientes en riesgo por horizonte.
   * Descripción básica de covariables.

2. Interpretar la censura y el conjunto en riesgo.

   * No presentar la curva de supervivencia sin comentar qué implica.
   * Si el número de pacientes en riesgo cae mucho, indicarlo.
   * Si hay muchos censurados, explicar cómo afecta a la evaluación.
   * Si hay pocos eventos en ciertos intervalos, evitar conclusiones fuertes para esos tiempos.

3. Comparar modelos con varias métricas.

   * No depender de una sola métrica.
   * Diferenciar discriminación y calibración.
   * Explicar qué mide cada métrica antes de usarla.
   * Presentar resultados de forma clara y consistente.

4. Evitar sobreinterpretar pequeñas diferencias.

   * Si las diferencias son pequeñas, decirlo.
   * Si no hay intervalos de confianza o análisis de incertidumbre, no presentar diferencias mínimas como concluyentes.
   * Si el modelo bayesiano aporta incertidumbre, destacarlo como dimensión interpretativa, no solo predictiva.

5. Cuidar tablas y figuras.

   * Toda tabla debe tener título claro.
   * Toda figura debe tener pie explicativo.
   * Toda figura debe ser citada en el texto.
   * No poner figuras decorativas.
   * No saturar la memoria con resultados secundarios.

---

## 7. Instrucciones específicas para discusión

1. La discusión debe responder a los objetivos.

   * Retomar cada objetivo de forma natural.
   * Explicar qué se ha observado.
   * Relacionar resultados con propiedades de los modelos y del dataset.

2. Separar interpretación, limitaciones y futuras líneas.

   * Interpretación: qué significan los resultados.
   * Limitaciones: qué impide generalizar.
   * Futuras líneas: qué se podría mejorar.

3. Reconocer limitaciones importantes.

   * Datos de una base concreta.
   * Posible falta de validación externa.
   * Uso de covariables estáticas.
   * Posible pérdida de información temporal.
   * Censura y reducción del conjunto en riesgo.
   * Sensibilidad a hiperparámetros.
   * Dificultad de comparar modelos con supuestos distintos.

4. No convertir la discusión en una repetición de resultados.

   * No volver a describir todas las tablas.
   * Seleccionar los hallazgos principales.
   * Explicar causas, implicaciones y cautelas.

5. Evitar conclusiones clínicas fuertes.

   * No afirmar aplicabilidad clínica directa.
   * Hablar de potencial utilidad metodológica.
   * Plantear el trabajo como evaluación académica.

---

## 8. Instrucciones específicas para conclusiones

1. Deben ser breves y directas.

   * No repetir la introducción.
   * No introducir teoría nueva.
   * No incluir tablas nuevas.
   * No meter citas extensas.

2. Deben responder a la pregunta del TFG.

   * Qué modelo se ha comportado mejor bajo las condiciones analizadas.
   * Qué aporta cada enfoque.
   * Qué limitaciones condicionan la interpretación.
   * Qué se podría hacer en trabajos futuros.

3. Evitar cierre genérico.

   * No terminar con frases vagas como “queda mucho por investigar”.
   * Terminar con una idea concreta sobre el valor del trabajo.

---

## 9. Reglas de estilo para maximizar claridad y nota

1. Frases más cortas.

   * Evitar frases de más de 4 líneas.
   * Separar definición, explicación e interpretación.
   * Una idea principal por frase.

2. Párrafos con función clara.

   * Primer párrafo: introduce.
   * Párrafos centrales: desarrollan.
   * Último párrafo: conecta o sintetiza, solo si aporta algo.

3. Tono académico natural.

   * Evitar tono publicitario.
   * Evitar lenguaje excesivamente solemne.
   * Evitar frases vacías.
   * Priorizar precisión.

4. Verbos recomendados.

   * analizar
   * comparar
   * evaluar
   * implementar
   * estimar
   * interpretar
   * discutir
   * contrastar
   * justificar

5. Verbos o expresiones a usar con cuidado.

   * demostrar
   * probar
   * garantizar
   * optimizar
   * revolucionar
   * validar clínicamente
   * predecir con precisión
   * superar de forma clara
   * estado del arte, si no está bien justificado

---

## 10. Reglas sobre citas y bibliografía

1. Citar toda afirmación externa relevante.

   * Modelos.
   * Datasets.
   * Métricas.
   * Trabajos relacionados.
   * Afirmaciones clínicas.
   * Afirmaciones metodológicas no triviales.

2. No citar de forma decorativa.

   * Cada cita debe sostener una idea concreta.
   * No poner varias citas al final de un párrafo si no queda claro qué respalda cada una.

3. Diferenciar fuentes.

   * Papers metodológicos para modelos.
   * Documentación oficial para dataset.
   * Papers clínicos para contexto médico.
   * Papers de benchmark para comparación.

4. No abusar de citas en la introducción.

   * La introducción necesita apoyo, pero no debe convertirse en revisión bibliográfica.
   * La revisión detallada va en estado de la cuestión.

---

## 11. Reglas sobre código, implementación y reproducibilidad

1. Explicar el código sin convertir la memoria en código.

   * Describir librerías, flujo y decisiones.
   * No pegar bloques largos de código en el cuerpo principal.
   * Llevar código relevante o pseudocódigo a anexos si aporta valor.

2. Documentar el preprocesamiento.

   * Exclusión de pacientes.
   * Tratamiento de valores faltantes.
   * Normalización o escalado.
   * Codificación de variables.
   * Construcción de tiempo y evento.
   * División train/test.

3. Documentar hiperparámetros importantes.

   * Arquitectura de DeepSurv.
   * Número de capas.
   * Función de pérdida.
   * Optimizador.
   * Learning rate.
   * Número de épocas.
   * Regularización.
   * Número y posición de intervalos en el modelo piecewise-exponential.
   * Priors del modelo bayesiano.

4. Explicar fallos o decisiones descartadas.

   * Si un modelo no converge, indicarlo.
   * Si una estrategia se descarta, explicar por qué.
   * Esto suma elaboración personal y criterio.

---

Mi recomendación es que estas instrucciones las uses como documento fijo de revisión. Cada vez que redactemos una sección, la revisamos contra este checklist, especialmente contra: separación entre teoría/metodología/resultados, tono académico natural, claridad matemática y ausencia de estructuras repetitivas.
