# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md)

> Este es el documento en español de laya-browser-agent. Para la versión más actual, ve [README.md](README.md).

**Decisiones de agente de navegador impulsadas por Laya — el modelo System 1 de código abierto. Una alternativa local a TypeSafe Jev: sin nube, sin clave de API, sin capturas de pantalla.**

[![tests](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml)

Un modelo de decisión responde preguntas tipadas sobre un estado y devuelve **probabilidades calibradas** en lugar de texto generado — por lo que no puede alucinar una instrucción. Esta es exactamente la forma correcta para la parte de *decidir* de un agente de navegador: dale una tabla numerada de los controles de una página y te dirá qué operación ejecutar y sobre qué elemento.

## Mediciones (M4, 16 GB)

| Métrica | Valor |
|---|---|
| Decisión de estado corto | 10–30 ms |
| Paso de navegador (20 elementos) | ~333 ms |
| Rendimiento | hasta ~100 decisiones/seg |
| Objetivos multilingües (con grounding) | 8 de 9 aciertos |
| Costo | **$0** |
| Contenido de página enviado a servidores | **cero** |

## Instalación

```bash
pip install 'laya-browser-agent[mlx]'      # Apple Silicon
pip install 'laya-browser-agent[torch]'    # Linux / Windows / Intel Mac
localdecide doctor                  # diagnóstico de hardware + prueba de humo
```

### Verificado contra la API real de Jev

El dialecto `systemone` de este repositorio fue validado de extremo a extremo contra el endpoint de producción de TypeSafe (`api.typesafe.ai/v1/systemone`, modelo `jev-1.13.0`) el 2026-09-22. **Cada tipo de pregunta exige el campo `criteria`** — en `choice` es un mapa de opción → descripción (no un string), en `score` es un array. El mismo payload apuntado al `localdecide serve` local produce la misma forma de respuesta con el modelo Laya local; cambiar de uno a otro es cambiar una sola URL base.

## Relación con Jev y Laya

Si has leído sobre el modelo "System One" de Jev y quieres la misma idea — decisiones tipadas y calibradas en lugar de texto generado — ejecutándose localmente para tus agentes de navegador, este es el proyecto. Ejecuta el checkpoint de Laya afinado para navegadores y añade lo que ninguno de los dos proyectos incluye: observación de tabla de elementos, validación de respuestas, puerta de confianza, guardias de bucle y un servidor compatible con TypeSafe.

## Limitaciones conocidas

- El zero-shot es débil en tu dominio — el checkpoint del navegador funciona porque alguien lo afinó durante ~5 horas de GPU
- El modelo no puede escribir — `TYPE_TEXT` requiere que tú proporciones el texto
- Los elementos en menús colapsados no son observables — abre el menú primero

Detalles completos en el [README en inglés](README.md).

## Licencia

Apache-2.0
