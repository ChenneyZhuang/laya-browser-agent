# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md)

> Este es el documento en español de laya-browser-agent. Para la versión más actual, ve [README.md](README.md).

**Soporte de decisiones restringidas y multi-backend para modelos tipo Jev.** Admite Laya
con MLX/PyTorch, cualquier backend duck síncrono con `answer(state, questions)` y cualquier
endpoint HTTP con forma System One. El checkpoint propio es una recomendación por defecto,
no un modelo propietario obligatorio.

[![tests](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-ichenney%2Flaya--browser--v32b-yellow)](https://huggingface.co/ichenney/laya-browser-v32b)

Un modelo de decisión responde preguntas tipadas sobre un estado y devuelve **probabilidades calibradas** en lugar de texto generado — por lo que no puede alucinar una instrucción. Esta es exactamente la forma correcta para la parte de *decidir* de un agente de navegador: dale una tabla numerada de los controles de una página y te dirá qué operación ejecutar y sobre qué elemento.

`model="browser"` recomienda **[ichenney/laya-browser-v32b](https://huggingface.co/ichenney/laya-browser-v32b)**
en el subdirectorio `v32b`; `model="browser-legacy"` usa la ruta `v10s` de
`cklxx/laya-browser`. Como `main` upstream ya no contiene ese directorio histórico, el alias
legacy fija por defecto la revisión `adf912be85ff9221ee171551778456b133c1af75`; `revision=` la puede sobrescribir.
La comparación histórica comprometida tiene nueve métricas: 5 victorias y
4 derrotas; la vista de ocho métricas de corrección tiene 5 victorias y 3 derrotas. No es inferencia
nueva. El checkpoint publicado tiene contraevidencia de despliegue en 95 casos; v37 aún no pasó
la aceptación del runtime y aquí no se afirma que esté disponible.

## Instalación

> **Nota de instalación**: el paquete aún no está publicado en PyPI. Primero clona el repositorio e instala desde su raíz: `git clone https://github.com/ChenneyZhuang/laya-browser-agent && cd laya-browser-agent`.

```bash
pip install -e '.[mlx]'      # Apple Silicon
pip install -e '.[torch]'    # Linux / Windows / Intel Mac
localdecide doctor           # diagnóstico de hardware + prueba de humo
```

El checkpoint v32b por defecto se descarga una sola vez (~1,3 GB; el `browser-legacy`
v10s oficial: ~650 MB). Después funciona totalmente offline.

`model`, `subfolder` y `revision` se exponen y se pasan a los runtimes upstream verificados.
`HTTPBackend` acepta `model`, `api_key` y `timeout`; un backend propio solo necesita
`answer(state, questions)` y devolver `{"answers": {...}, "usage": {...}}`. El timeout total
de `Decider` limita el timeout HTTP restante. Un backend duck síncrono no puede cancelarse a la
fuerza desde Python: un resultado tardío se rechaza y no se hacen más llamadas.

### Configuración del backend, timeouts y límite de seguridad

Este ejemplo usa el nombre real del modelo v32b y muestra una API key, el timeout HTTP
y un backend personalizado:

```python
from localdecide import Decider
from localdecide.backends.base import HTTPBackend, LayaTorchBackend

local = LayaTorchBackend(model="ichenney/laya-browser-v32b", subfolder="v32b")

class MyBackend:
    name = "my-local-backend"
    def answer(self, state, questions):
        return {"answers": {}, "usage": {}}

remote = HTTPBackend(
    "http://127.0.0.1:8791/v1/systemone",
    model="browser", api_key="example-key", timeout=10.0,
)
decider = Decider(backend=remote, timeout=12.0, retries=1)
```

Los backends locales MLX/PyTorch no envían el estado de la página fuera de tu máquina.
`HTTPBackend` sí envía el estado y las preguntas al endpoint configurado, así que úsalo
solo con un servicio local o privado de confianza. `api_key` se convierte en un header
Bearer; guarda la clave real en `LOCALDECIDE_API_KEY` o un gestor de secretos, no en el
código ni en el historial del shell. Los valores de contraseña se redactan en la observación
y `TYPE_TEXT` rechaza los campos de contraseña porque el proyecto no ofrece acceso a un vault.
Proporciona el texto explícitamente y usa un callback humano `confirm` para enviar, borrar,
pagar u otras acciones irreversibles. Un backend duck síncrono no se puede interrumpir a la
fuerza desde Python: se rechaza el resultado tardío y no se vuelve a llamar.

## Mediciones (M4, 16 GB)

> Todos los números están medidos en las máquinas de desarrollo: Apple M4 (16 GB, `laya-mlx`)
> para latencia, RTX 3080 para fine-tuning y evaluación. Nada copiado de marketing de vendors.
> Las comparaciones con Jev anteriores a v32b usaron el checkpoint oficial v10s (se conservan como registro histórico).

| Métrica | Valor |
|---|---|
| Decisión de estado corto | 10–30 ms |
| Paso de navegador (20 elementos) | ~333 ms |
| Rendimiento | hasta ~100 decisiones/seg |
| Objetivos multilingües (con grounding) | 8 de 9 aciertos |
| Costo | **$0** |
| Contenido de página enviado a servidores | **cero** |

### v32b propio (ahora por defecto) vs modelo oficial vs API de Jev (2026-09-25, mismo conjunto)

| Benchmark | **v32b (propio)** | versión oficial | API Jev |
|---|---:|---:|---:|
| recovery2-holdout (240) | **0.7125** | 0.425 | — |
| MiniWoB (116) | **0.9138** | 0.6638 | — |
| browser-suite v4 | **0.5143** | 0.500 | — |
| browser-suite v5 | 0.5636 | **0.5818** | — |
| JevBench hard (111) | **0.4144** | 0.243 | 0.7207 |
| JevBench easy | 0.8542 | 0.979 | 1.0000 |
| Latencia (p50) | **27 ms** (RTX 3080) | — | 854 ms (red) |

La comparación histórica comprometida suma 9 métricas: 5 victorias y 4 derrotas; la vista de 8 métricas de corrección suma 5 victorias y 3 derrotas. Frente a la API en la nube de Jev sigue habiendo brecha de precisión absoluta, pero el valor local es **gratuito, privado y offline**. Detalles: [Benchmarks en inglés](README.md#benchmarks) y [JEV_COMPARISON.md](reports/v20/JEV_COMPARISON.md).

### Enfrentamiento contra el Jev oficial: datos medidos

Estas baterías son anteriores a v32b; el lado local usaba el checkpoint oficial v10s de entonces y se conservan como registro histórico, no como inferencia nueva. `examples/diagnostics/jev_head_to_head.py` (un paso) y `jev_flow_h2h.py` (flujos completos) ejecutan las mismas tareas contra el Laya v10s local y el jev-1.13.0 oficial:

| Un paso, sin contexto (12 metas / 6 idiomas) | local v10s | Jev oficial |
|---|---|---|
| Aciertos estrictos de elemento | 4/12 | **8/12** |
| Metas multilingües | 1/6 | **5/6** |
| Latencia mediana | **618ms** | 716ms |
| Confianza media | 0.90 (sobreconfiado) | 0.81 |

Flujos multi-paso: **ningún motor completa hoy el flujo de compra scripted sin ayuda** — el estado difícil es justo después de escribir la búsqueda; v10s local vuelve a pulsar Search (p=0.84) incluso con el producto visible, el Jev oficial responde BLOCKED o elige el elemento correcto con p=0.45. v10s local sí completó el flujo de navegación en chino (帮助中心→DONE).

**Conclusión**: precisión inmediata (especialmente multilingüe) → Jev oficial (~$0.000017 por decisión); privacidad / offline / volumen gratis → versión local, con latencia comparable y multilingüe como su eje más débil.

Las 15 versiones, todas las rutas fallidas y los scripts de entrenamiento están publicados en el repositorio hermano [laya-training-log](https://github.com/ChenneyZhuang/laya-training-log).

### Verificado contra la API real de Jev

El dialecto `systemone` de este repositorio fue validado de extremo a extremo contra el endpoint de producción de TypeSafe (`api.typesafe.ai/v1/systemone`, modelo `jev-1.13.0`) el 2026-09-22. **Cada tipo de pregunta exige el campo `criteria`** — en `choice` es un mapa de opción → descripción (no un string), en `score` es un array. El mismo payload apuntado al `localdecide serve` local produce la misma forma de respuesta con el modelo Laya local; cambiar de uno a otro es cambiar una sola URL base.



**Clasificación de texto** (`jev_text_h2h.py`, textos reales, 21 casos):

| tarea | local v10s | Jev oficial |
|---|---|---|
| leads de piscina `relevant` (10 casos) | **6/10** | **10/10** |
| calidad de lead (0-4) | sesgo bajo (1.0-2.0) | **calibrado (2.4-3.7)** |
| SMS chino: joint transacción/tipo/phishing (8 casos) | **5/8** | **5/8** |
| robustez (vacío/5k caracteres/adversarial) | 3/3 | 3/3 |
| latencia mediana | **36ms** | 738ms |

La fila de phishing merece atención: v10s local dio p=0.14 a la estafa clásica "mamá, se me rompió el teléfono… transfiere 5000" y p=0.23 a la del sobre rojo — las dejaría pasar. Jev oficial: p=0.96 en ambas. **Para cualquier enrutado sensible a seguridad, el modelo local solo no es confiable hoy.**

**Casos extremos de navegador** (`jev_edge_h2h.py`, 9 casos): el antiguo 4/9 frente a 5/9 medía solo la operación. El scorer ahora separa operación, objetivo y joint; el gold se corrigió para usar el control newsletter realmente desmarcado. La fila del fixture marcado anterior no es un resultado de un modelo nuevo.

Conclusión práctica de las cuatro baterías: precisión y calibración de seguridad → Jev oficial; latencia (10-20x más rápido), privacidad o volumen gratuito → local + guards. El grounding chino, el phishing y la contención son exactamente los ejes a afinar con el fine-tuning.
## Relación con Jev y Laya

Si has leído sobre el modelo "System One" de Jev y quieres la misma idea — decisiones tipadas y calibradas en lugar de texto generado — ejecutándose localmente para tus agentes de navegador, este es el proyecto. Ejecuta el checkpoint de Laya afinado para navegadores y añade lo que ninguno de los dos proyectos incluye: observación de tabla de elementos, validación de respuestas, puerta de confianza, guardia de envío vacío, guardias de bucle y un servidor compatible con TypeSafe.

## Limitaciones conocidas

- El zero-shot es débil en tu dominio — el checkpoint del navegador funciona porque alguien lo afinó durante ~5 horas de GPU
- El modelo no puede escribir — `TYPE_TEXT` requiere que tú proporciones el texto
- Los elementos en menús colapsados no son observables — abre el menú primero

Detalles completos en el [README en inglés](README.md).

### Respaldo en la literatura

- [arXiv 2609.23959](https://arxiv.org/abs/2609.23959) (sep 2026): una tarea relacionada de cribado de estafas con typed decisions informa AUROC .974 y error de calibración .052; es contexto, no una explicación causal de los SMS de este proyecto.
- El upstream de Laya publica su calibración: **accuracy 0.753 @ ECE 0.030** en 13 familias de tareas (tras escalado de temperatura). El checkpoint v10s no hereda ese nivel en texto fuera de su distribución de entrenamiento (ver filas de pool/SMS arriba).

## Licencia

Apache-2.0
