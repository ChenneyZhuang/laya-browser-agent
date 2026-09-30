# Model compatibility and revision policy

This project forwards `model`, `subfolder`, and `revision` to the selected Laya
runtime. This documents the boundary that was checked from the upstream source;
it is not a claim that a model can be loaded on this machine.

## Which model is which

- [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) is
  the official Laya base model family. It is not the browser fine-tune.
- [`cklxx/laya-browser`](https://huggingface.co/cklxx/laya-browser) is the
  upstream browser fine-tune/reference checkpoint. Its historical `v10s`
  directory is no longer on `main`.
- [`ichenney/laya-browser-v32b`](https://huggingface.co/ichenney/laya-browser-v32b)
  is this project's recommended browser fine-tune. Use `model="browser"` and
  `subfolder="v32b"`.

The recommended configuration is therefore explicit about the model and
subfolder. The current v32b model card identifies the checkpoint as `v32b-b15`,
and the Hub metadata endpoint was independently checked at immutable commit
`161d54d6000913ff279b0afd1ac77faef8685a9b`. Use that revision when the exact
v32b source is intended; the legacy alias has its own known historical pin:

```python
LayaTorchBackend(
    model="ichenney/laya-browser-v32b",
    subfolder="v32b",
    revision="161d54d6000913ff279b0afd1ac77faef8685a9b",
)

LayaTorchBackend(
    model="browser-legacy",
    subfolder="v10s",
    revision="adf912be85ff9221ee171551778456b133c1af75",
)
```

The same explicit `revision` can be passed to `LayaMLXBackend`. The `browser`
alias in the current code continues to resolve the model and subfolder defaults;
callers that need an immutable v32b selection should pass the revision above.
The checked source metadata is recorded at the
[v32b Hub API endpoint](https://huggingface.co/api/models/ichenney/laya-browser-v32b).

## Version and source evidence

The PyTorch extra is pinned to `laya>=0.3.21` because the inspected published
`laya` 0.3.20 source has no `revision` parameter, while the published 0.3.21
source adds `revision` and passes it to the Hub snapshot resolver. The 0.3.21
package metadata declares Python >=3.10 and minimum runtime dependencies
`torch>=2.0`, `transformers>=4.48`, `safetensors>=0.4`,
`huggingface-hub>=0.20`, and `numpy>=1.20`:
[0.3.20 source](https://raw.githubusercontent.com/NandhaKishorM/laya/v0.3.20/laya/agent.py),
[0.3.21 loader](https://raw.githubusercontent.com/NandhaKishorM/laya/v0.3.21/laya/agent.py),
[0.3.21 metadata](https://raw.githubusercontent.com/NandhaKishorM/laya/v0.3.21/pyproject.toml).

The MLX extra remains `laya-mlx>=0.1.0`: the inspected 0.1.0 wheel source
exposes `revision` in `resolve_model` and passes it to `snapshot_download`.
Its wheel metadata requires Python >=3.11, `huggingface-hub>=0.34,<2`,
`mlx>=0.32.2,<0.33` on Apple Silicon, `numpy>=1.26`, and
`tokenizers>=0.21,<1`. The current 0.2.0 PyPI metadata reports the same
runtime floor and dependency bounds:
[laya-mlx 0.1.0](https://pypi.org/project/laya-mlx/0.1.0/),
[laya-mlx 0.2.0 metadata](https://pypi.org/pypi/laya-mlx/json).

These checks read published source and wheel metadata only. They did not import
either runtime, install a model package during this review, download weights,
or run inference. The project's mock backend tests therefore verify argument
forwarding only; they are not proof that a particular checkpoint loads or
performs well.

## Deployment limits

Loading still requires the relevant runtime and checkpoint files, compatible
hardware, and enough memory for the model plus browser. A browser fine-tune is
not a general web agent: long or dynamic forms, authentication/password
fields, collapsed menus, canvas/shadow-DOM content, sites that block headless
browsers, and vocabulary or language mismatches can fail. Offline MiniWoB or
fixture accuracy is a diagnostic signal, not end-to-end browser task success.
Keep the harness's validation, timeout, and human confirmation gates enabled.

Local MLX/PyTorch backends keep state local. `HTTPBackend` sends state and
questions to the configured endpoint, which may be external and may have its
own privacy, availability, and pricing terms.
