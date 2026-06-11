# CODEX Instructions

Antes de modificar este repositorio, leer:

```text
TFG/CODEX_TFG_MATES_JAVI.md
```

Resumen operativo:

- Priorizar cambios simples, eficientes y reproducibles.
- No subir datos MIMIC-IV ni derivados a GitHub.
- Mantener el código original de `src/models_references/` como referencia metodológica local, sin subir repositorios externos.
- Crear adaptaciones del TFG en `src/models/`, `src/data/`, `scripts/` y `configs/`.
- Evitar data leakage: los splits se hacen por paciente o estancia antes de generar landmarks.
- Usar rutas relativas, configuración externa y salidas en `outputs/`.
- No cambiar arquitectura, pérdidas, censura, discretización u horizonte de modelos replicados sin documentarlo.
