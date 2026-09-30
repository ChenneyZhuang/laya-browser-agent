# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md)

> Este es el resumen en español. Consulta [README.md](README.md), [model-compatibility.md](docs/model-compatibility.md) y [release-verification.md](docs/release-verification.md) para compatibilidad, verificación de release y limitaciones completas.

Soporte de decisiones restringidas y multi-backend para modelos tipo Jev: Laya con MLX/PyTorch,
cualquier backend síncrono con `answer(state, questions)` y endpoints HTTP con forma System One.

El modelo devuelve estimaciones de probabilidad sobre las opciones ofrecidas; no genera
instrucciones, pero puede equivocarse o mostrar exceso de confianza. La validación, los checks de
resultado, los timeouts y la confirmación humana siguen siendo necesarios.

## Modelos y configuración

El modelo base oficial [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
no es el fine-tune de navegador. La referencia upstream de navegador es
[`cklxx/laya-browser`](https://huggingface.co/cklxx/laya-browser). La recomendación explícita de
este proyecto es [`ichenney/laya-browser-v32b`](https://huggingface.co/ichenney/laya-browser-v32b):
usa `model="browser"` y `subfolder="v32b"`. La model card identifica `v32b-b15`; esta
revisión verificó el commit inmutable del metadata Hub
`161d54d6000913ff279b0afd1ac77faef8685a9b`; pásalo como `revision` cuando necesites una
configuración reproducible.

```python
LayaTorchBackend(model="ichenney/laya-browser-v32b", subfolder="v32b",
                 revision="161d54d6000913ff279b0afd1ac77faef8685a9b")
LayaTorchBackend(model="browser-legacy", subfolder="v10s",
                 revision="adf912be85ff9221ee171551778456b133c1af75")
```

El extra de PyTorch usa `laya>=0.3.21`, que es la versión cuyo loader comprobado admite y
reenvía `revision`; el extra MLX sigue siendo `laya-mlx>=0.1.0`. Esta verificación solo leyó
fuente publicada, metadata de wheel y metadata Hub: no importó runtimes, descargó pesos ni ejecutó inferencia.
Las pruebas con mocks solo demuestran el reenvío de argumentos, no que un checkpoint cargue o rinda bien.

## Instalación y privacidad

La URL del proyecto en PyPI todavía devuelve 404; instala desde un git clone:

```bash
git clone https://github.com/ChenneyZhuang/laya-browser-agent
cd laya-browser-agent
pip install -e '.[mlx]'       # Apple Silicon
pip install -e '.[torch]'     # Linux / Windows / Intel Mac
pip install -e '.[playwright]' && playwright install chromium
```

Los backends locales MLX/PyTorch mantienen el estado en tu máquina. `HTTPBackend` envía el estado
y las preguntas al endpoint configurado, que puede ser externo y tener sus propias condiciones de
privacidad, disponibilidad y precio. “Gratis/privado” solo describe el backend local correspondiente,
no cualquier configuración HTTP.

## Evidencia y límites

Los diagnósticos históricos y sus condiciones/raw records están enlazados en
[`reports/v20/MULTIDIM_COMPARISON.md`](reports/v20/MULTIDIM_COMPARISON.md) y
[`reports/v20/JEV_COMPARISON.md`](reports/v20/JEV_COMPARISON.md). Este README no repite conteos
antiguos, precios estáticos ni rangos de rendimiento sin raw evidence comprometida; la salida de
los ejemplos es ilustrativa.

La accuracy offline de MiniWoB o de fixtures es una señal diagnóstica, no éxito de tareas del
navegador de extremo a extremo. Si un resultado histórico de edge solo midió `operation`, no implica
`target` ni `joint`; solo se pueden reportar los campos que existan en el registro comprometido.
No se afirma ningún resultado nuevo del modelo.

El fine-tune de navegador no es un agente web general: autenticación/campos de contraseña,
formularios largos o dinámicos, menús colapsados, canvas/shadow DOM, sitios que bloquean browsers
headless y desajustes de idioma o vocabulario pueden fallar. Mantén activadas la validación, los
timeouts y las confirmaciones.

## Fuentes y licencia

Los modelos Laya y los checkpoints de navegador conservan sus licencias Apache-2.0, atribución y
cadena de procedencia upstream. Mind2Web, NNetNav, WebChain y otros datasets conservan sus propios
términos aplicables; este proyecto no los relicencia ni implica derechos adicionales.

Para el detalle completo, consulta el [README en inglés](README.md).

## Licencia

Apache-2.0, además de las condiciones upstream aplicables a modelos, código y datos utilizados.
