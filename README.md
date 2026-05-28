# TFG Survival Modeling

Repositorio final del TFG de Matemáticas sobre modelos de supervivencia estáticos y dinámicos en pacientes adultos de UCI usando MIMIC-IV.

El objetivo del código es mantener un pipeline simple, reproducible y defendible para:

- construir datos estáticos y temporales;
- replicar modelos de supervivencia de referencia;
- adaptar DySurv al experimento con landmarks diarios;
- comparar modelos con métricas adecuadas de supervivencia.

## Estructura

```text
configs/                 Configuración de datos, features y entrenamiento.
data/                    Datos locales. No se versionan datos MIMIC-IV ni derivados.
guidelines/TFG/          Instrucciones académicas y de código del TFG.
notebooks/               Exploración. No son el pipeline principal.
outputs/                 Modelos, métricas, figuras, predicciones, logs y checkpoints.
scripts/                 Puntos de entrada ejecutables desde terminal.
src/
  data/                  Carga y preprocesamiento de datos.
  evaluation/            Métricas y comparación de modelos.
  features/              Construcción de variables.
  models/                Modelos adaptados al pipeline del TFG.
  models_references/     Repositorios originales consultados localmente.
  utils/                 Utilidades compartidas.
tests/                   Tests mínimos de arquitectura y consistencia.
```

## Reglas de trabajo

- No subir datos de MIMIC-IV ni datasets derivados a GitHub.
- Mantener `src/models_references/` como referencia metodológica local, sin subir repositorios externos a GitHub.
- Implementar adaptaciones propias en `src/models/` y `src/data/`.
- Usar rutas relativas al proyecto.
- Guardar resultados reproducibles en `outputs/`.
- Ejecutar el pipeline principal desde `scripts/`, no desde notebooks.

## Instalación

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Uso

Entrenamiento del baseline estático configurado en `configs/train.yaml`:

```bash
python scripts/train_static_pipeline.py
```

Evaluación de modelos guardados:

```bash
python scripts/evaluate.py
```

Preprocesamiento MIMIC-IV basado en scripts de referencia locales:

```bash
python scripts/run_mimic_pipeline.py
```

## Configuración

El archivo principal actual es:

```text
configs/train.yaml
```

Contiene el nombre del experimento, rutas de datos locales, definición de evento/duración, modelos estáticos iniciales y carpetas de salida.

## Referencias internas

Las instrucciones completas del TFG están en:

```text
guidelines/TFG/
```

El documento operativo para Codex y futuras modificaciones de código es:

```text
guidelines/TFG/CODEX_TFG_MATES_JAVI.md
```

Los repositorios externos en `src/models_references/` no se versionan. Si se necesita reproducir una adaptación concreta, documentar la fuente original y mantener el wrapper o adaptación propia en `src/models/` o `src/data/`.
