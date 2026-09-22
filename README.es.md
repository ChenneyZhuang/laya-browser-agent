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

> **Nota de instalación**: la versión en PyPI está alcanzando al repo. Por ahora usa `git clone https://github.com/ChenneyZhuang/laya-browser-agent && cd laya-browser-agent && pip install -e '.[all]'` (Apple Silicon) o `.[torch]` (resto).

```bash
pip install 'laya-browser-agent[mlx]'      # Apple Silicon
pip install 'laya-browser-agent[torch]'    # Linux / Windows / Intel Mac
localdecide doctor                  # diagnóstico de hardware + prueba de humo
```

### Enfrentamiento contra el Jev oficial: datos medidos

`examples/diagnostics/jev_head_to_head.py` (un paso) y `jev_flow_h2h.py` (flujos completos) ejecutan las mismas tareas contra el Laya v10s local y el jev-1.13.0 oficial:

| Un paso, sin contexto (12 metas / 6 idiomas) | local v10s | Jev oficial |
|---|---|---|
| Aciertos estrictos de elemento | 4/12 | **8/12** |
| Metas multilingües | 1/6 | **5/6** |
| Latencia mediana | **618ms** | 716ms |
| Confianza media | 0.90 (sobreconfiado) | 0.81 |

Flujos multi-paso: **ningún motor completa hoy el flujo de compra scripted sin ayuda** — el estado difícil es justo después de escribir la búsqueda; v10s local vuelve a pulsar Search (p=0.84) incluso con el producto visible, el Jev oficial responde BLOCKED o elige el elemento correcto con p=0.45. v10s local sí completó el flujo de navegación en chino (帮助中心→DONE).

**Conclusión**: precisión inmediata (especialmente multilingüe) → Jev oficial (~$0.000017 por decisión); privacidad / offline / volumen gratis → versión local, con latencia comparable y multilingüe como su eje más débil.

### Verificado contra la API real de Jev

El dialecto `systemone` de este repositorio fue validado de extremo a extremo contra el endpoint de producción de TypeSafe (`api.typesafe.ai/v1/systemone`, modelo `jev-1.13.0`) el 2026-09-22. **Cada tipo de pregunta exige el campo `criteria`** — en `choice` es un mapa de opción → descripción (no un string), en `score` es un array. El mismo payload apuntado al `localdecide serve` local produce la misma forma de respuesta con el modelo Laya local; cambiar de uno a otro es cambiar una sola URL base.



**Clasificación de texto** (`jev_text_h2h.py`, textos reales, 21 casos):

| tarea | local v10s | Jev oficial |
|---|---|---|
| leads de piscina `relevant` | 3/7 correctos | **7/7** |
| calidad de lead (0-4) | sesgo bajo (1.0-2.0) | **calibrado (2.4-3.7)** |
| SMS chino: transacción/tipo | **6/8** | 6/8 |
| SMS chino: **detección de phishing** | **0/3** | **3/3** |
| robustez (vacío/5k caracteres/adversarial) | 3/3 | 3/3 |
| latencia mediana | **36ms** | 738ms |

La fila de phishing merece atención: v10s local dio p=0.14 a la estafa clásica "mamá, se me rompió el teléfono… transfiere 5000" y p=0.23 a la del sobre rojo — las dejaría pasar. Jev oficial: p=0.96 en ambas. **Para cualquier enrutado sensible a seguridad, el modelo local solo no es confiable hoy.**

**Casos extremos de navegador** (`jev_edge_h2h.py`, 9 casos donde lo correcto es abstenerse): local 4/9, oficial 5/9, y fallan en direcciones opuestas. v10s es de disparar primero: pide borrar el sitio web entero y pulsa "Delete my account" (p=0.93), desmarca lo ya desmarcado y clica botones deshabilitados, siempre con ~0.9 de confianza. El Jev oficial bloquea lo imposible pero también bloquea metas legítimas. Ninguno tiene aún un concepto fiable de "esta meta no se puede lograr"; los guards del harness son lo que salva estos casos hoy.

Conclusión práctica de las cuatro baterías: precisión y calibración de seguridad → Jev oficial; latencia (10-20x más rápido), privacidad o volumen gratuito → local + guards. El grounding chino, el phishing y la contención son exactamente los ejes a afinar con el fine-tuning.
## Relación con Jev y Laya

Si has leído sobre el modelo "System One" de Jev y quieres la misma idea — decisiones tipadas y calibradas en lugar de texto generado — ejecutándose localmente para tus agentes de navegador, este es el proyecto. Ejecuta el checkpoint de Laya afinado para navegadores y añade lo que ninguno de los dos proyectos incluye: observación de tabla de elementos, validación de respuestas, puerta de confianza, guardias de bucle y un servidor compatible con TypeSafe.

## Limitaciones conocidas

- El zero-shot es débil en tu dominio — el checkpoint del navegador funciona porque alguien lo afinó durante ~5 horas de GPU
- El modelo no puede escribir — `TYPE_TEXT` requiere que tú proporciones el texto
- Los elementos en menús colapsados no son observables — abre el menú primero

Detalles completos en el [README en inglés](README.md).

## Licencia

Apache-2.0
