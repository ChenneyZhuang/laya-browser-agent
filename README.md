# laya-browser-agent

**Multi-backend constrained decision support for Jev-like models.** This project can run
Laya through MLX or PyTorch, accept any synchronous `answer(state, questions)` duck backend,
or call an arbitrary System One-shaped HTTP endpoint. The recommended browser default is
the project's own open checkpoint, not a forced proprietary model.

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md)

[![tests](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-ichenney%2Flaya--browser--v32b-yellow)](https://huggingface.co/ichenney/laya-browser-v32b)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Runs on](https://img.shields.io/badge/runs%20on-Apple%20Silicon%20%7C%20CUDA%20%7C%20CPU-black)

A local, open-weight alternative to [TypeSafe Jev](https://docs.typesafe.ai) for the
browser-driving use case — running [Laya](https://github.com/NandhaKishorM/laya), the
open-source "System One" decision model, fully on your own machine. Works with
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) via the same
wire format, and speaks TypeSafe's `/v1/systemone` dialect, so existing Jev tooling
points at it by changing one base URL.

A decision model answers typed questions about a state and returns probability
estimates over the supplied options. It does not generate instructions, but it
can still be wrong or confidently wrong; validation, outcome checks, and human
confirmation remain necessary. That makes it a useful *decision* component for
a browser agent: give it a numbered table of controls and constrain which
operation and element it may propose.

This project wires those models into that role, locally, for whatever agent you
already use. The official [Laya base model](https://huggingface.co/convaiinnovations/laya)
is distinct from the upstream browser fine-tune
([`cklxx/laya-browser`](https://huggingface.co/cklxx/laya-browser)). The recommended
browser checkpoint is this project's
[ichenney/laya-browser-v32b](https://huggingface.co/ichenney/laya-browser-v32b):
`model="browser"`, `subfolder="v32b"`. The current v32b model card identifies
`v32b-b15`; the checked Hub metadata is immutable commit
`161d54d6000913ff279b0afd1ac77faef8685a9b`, which is the revision used in the
reproducible configuration example below.
`model="browser-legacy"` selects the upstream `v10s` path and pins the historical
revision `adf912be85ff9221ee171551778456b133c1af75` by default. See
[model compatibility](docs/model-compatibility.md) for the verified source boundary.

The browser fine-tune has known deployment limits: authentication and password
fields, long or dynamic forms, collapsed menus, canvas/shadow-DOM content, sites
that block headless browsers, and language or vocabulary mismatches can fail. No
current model result is asserted by the model-free test suite.

```python
from localdecide import BrowserDecider
from localdecide.drivers import PlaywrightDriver

with PlaywrightDriver(start_url="https://en.wikipedia.org/wiki/Main_Page") as driver:
    run = BrowserDecider().run(driver, "Click the 'Random article' link in the navigation.")
    print(run.stopped, run.summary()["median_decision_ms"], "ms/decision")
# Illustrative output only; status and timing depend on the page and backend.
```

Performance numbers are historical diagnostics only. They are kept in the
committed reports where their raw evidence and conditions can be reviewed; this
README makes no current latency, throughput, cost, calibration, or model-size
guarantee. See [historical benchmark evidence](#historical-benchmark-evidence) and
[release verification](docs/release-verification.md).

---

## How this relates to Jev and Laya

| | [TypeSafe Jev](https://docs.typesafe.ai) | [Laya](https://github.com/NandhaKishorM/laya) | **localdecide** |
|---|---|---|---|
| Weights | closed, API only | open, Apache-2.0 | runs Laya's open weights |
| Where it runs | TypeSafe's cloud | anywhere PyTorch runs | **your machine** — MLX on Apple Silicon, PyTorch elsewhere |
| Wire format | `POST /v1/systemone` | same contract | speaks it too (`POST /v1/systemone`) |
| Cost | provider terms apply | runtime/provider terms apply | runtime/provider terms apply |
| Browser harness | [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) | — | **included**: loop, drivers, guards, skills |
| Page leaves your machine | yes | no | **depends on backend** |

### Historical interoperability note

A prior repository check recorded the `systemone` dialect against TypeSafe's
`api.typesafe.ai/v1/systemone` endpoint on 2026-09-22. This is historical
interoperability evidence, not a current service guarantee. A working request
looks like this — note that **`criteria` is
required for every question type** (the API rejects questions without it), and
for `choice` it is a *map of option → rubric description*, not a string:

```json
{
  "state": "Hi, my pool pump stopped working...",
  "model": "jev-latest",
  "questions": {
    "is_pool_lead": {
      "type": "noul",
      "instructions": "Is this a swimming-pool related service request?",
      "criteria": {
        "true": "Related to pool maintenance, construction, or supplies",
        "false": "Not pool related"
      }
    },
    "urgency": {
      "type": "choice",
      "instructions": "Which urgency level?",
      "criteria": {
        "low": "Routine inquiry",
        "medium": "Wants service soon",
        "high": "Emergency or explicitly time-sensitive"
      }
    }
  }
}
```

Illustrative response shape from that historical check (values are not current
guarantees): `{"model":"jev-1.13.0","answers":{"is_pool_lead":{"noul":0.99},
"urgency":{"choice":"high","confidence":1.0,...}},"usage":{"input_tokens":398,"output_tokens":58}}`

The same payload, with `url` pointed at the bundled `localdecide serve`
(`POST /v1/systemone`), produces the same answer shape from the local Laya
checkpoint — so code written against one works against the other by changing
one base URL. `score` questions take `criteria` as an **array** of level names.

### Historical comparison note

Older Jev/v10s comparisons are retained in the committed diagnostic scripts and
reports for provenance. They are not current acceptance results, and this README
does not repeat their old test counts or pricing.

Historical comparisons are provenance only; provider pricing, latency, and
accuracy vary with the endpoint, checkpoint, hardware, and task.

The historical edge battery covered operation only; it must not be read as
target or joint correctness. A later scorer may expose those fields separately,
but only fields present in a committed record are supported. Offline MiniWoB or
fixture accuracy is a diagnostic signal, not end-to-end browser task success.

For source/metadata compatibility, deployment limits, privacy boundaries, and
release checks, see [model compatibility](docs/model-compatibility.md) and
[release verification](docs/release-verification.md).

If you want typed decisions instead of generated text for a browser agent, this
is the wiring: element-table observation, answer validation, confidence gates,
loop guards, and a TypeSafe-compatible server around an explicitly configured
browser checkpoint.

## Why this exists

The "System One model" idea — a non-autoregressive model that returns typed
decisions instead of prose — is useful here, but quality and calibration remain
task-dependent. TypeSafe's **Jev** made it familiar; [Convai Innovations' **Laya**](https://github.com/NandhaKishorM/laya)
shipped the same architecture as Apache-2.0 open weights; and a remarkable amount of
work went into making these models drive browsers.

What was missing was the boring part: a **neutral, local, agent-agnostic harness**.
Something you can point Claude Code, Codex, Cursor, Hermes, or your own script at —
that loads a local checkpoint, keeps the model's output inside a safe action space,
and speaks the dialects agents already talk.

That is this repo.

### What a decision model may and may not do here

| The model decides | Your code decides |
|---|---|
| which operation (`CLICK`/`TYPE_TEXT`/`SELECT`/`SCROLL`/`WAIT`/`DONE`/`BLOCKED`) | what each operation means |
| which element index to act on | what that index maps to in the DOM |
| how confident it is | whether the confidence is good enough |

The model's output is validated against the option set it was given before anything
acts on it: a key outside the offered set, a probability vector that does not sum to
one, or a choice that is not the argmax is rejected and the decision **fails open**
(your agent takes its own fallback path instead of acting on junk). Model output can
never become a selector, a coordinate, or executable code — it is only ever an index
into a table your code built.

---

## Install

### Step by step

**1. Install the package with the extras for your platform.** The PyPI project URL
still returns 404 in this release check, so clone the repository and install from
its root:

```bash
git clone https://github.com/ChenneyZhuang/laya-browser-agent && cd laya-browser-agent

# Apple Silicon Mac (M1–M4) — MLX runtime, fastest path:
pip install -e '.[all]'

# Linux / Windows / Intel Mac — same checkpoints through PyTorch:
pip install -e '.[torch]'

# Linux + NVIDIA GPU — PyTorch will pick the CUDA wheel if one is present:
pip install -e '.[torch]'
```

`localdecide` is the import and CLI name when installed from the repository.

**2. Install a browser driver (only needed for the browser loop):**

```bash
pip install -e '.[playwright]' && playwright install chromium
# Or attach to a Chrome you already have open and logged in — no download:
pip install -e '.[cdp]'
```

**3. Run the hardware check.** `doctor` detects your chip and memory, picks the right
runtime, and runs a one-decision smoke test — so a broken install shows up here, not in
your agent:

```bash
localdecide doctor
```

Illustrative output shape; values vary by Python version, hardware, and installed extras:

```
python      3.12.13 (arm64, Darwin)
hardware    Apple M4, 16 GB unified memory
laya-mlx    installed
playwright  installed
backend     laya-mlx
smoke test  OK (illustrative output; first call includes model load)
```

**4. From source** (for development):

```bash
git clone https://github.com/ChenneyZhuang/laya-browser-agent
cd laya-browser-agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[playwright,cdp]' pytest
python -m pytest -q     # model-free collection; live classes are skipped by default
```

This model-free development setup deliberately omits `[all]`/`[mlx]` and
`localdecide doctor`: the doctor command loads a runtime for its smoke test.
Install a model extra separately before an explicitly authorized live run.

### What to expect on different devices

The harness is pure Python and portable; what changes by device is which runtime
you install and how much memory and latency your workload can afford. No
performance range is guaranteed here.

| Device | Runtime | Expected experience |
|---|---|---|
| **Apple Silicon M-series, 16 GB+** (M1–M4) | `laya-mlx` | Supported local path; workload and checkpoint determine performance. |
| **Apple Silicon, 8 GB** (M1/M2 base) | `laya-mlx` | Checkpoint plus Chromium can be tight; validate memory, close heavy apps, and keep the subprocess boundary. |
| **Intel Mac** | `laya` (PyTorch) | `laya-mlx` does not run here; use the PyTorch path and validate the installed runtime on the target machine. |
| **Linux server, CPU only** | `laya` (PyTorch) | Suitable for batch deciding and headless browser loops; validate the workload on the target machine. |
| **Linux + NVIDIA GPU** | `laya` (PyTorch, CUDA) | Supported PyTorch path; fine-tuning is outside this release verification. |
| **Windows** | `laya` (PyTorch) | Works; same expectations as Linux CPU. Playwright supports it natively. |
| **Below 8 GB total / Raspberry Pi class** | — | Not a supported local deployment target; use the HTTP backend to reach a separately provisioned machine if appropriate. |
| **Any device, model elsewhere** | `HTTPBackend` | Point `Decider("http://host:8791/v1/...")` at the configured endpoint. It may be another private machine or an external service; verify its trust and privacy terms. |

The pure-Python safety guards are shared across platforms. Accuracy and runtime
parity remain checkpoint-, dependency-, and task-dependent; this repository
makes no universal parity or accuracy guarantee.

### Backend configuration and compatibility

The bundled backends expose `model`, `subfolder`, and `revision` and forward them to the
source-backed runtime APIs. The PyTorch extra requires `laya>=0.3.21` for this revision
support; the MLX extra remains `laya-mlx>=0.1.0`. The recommended browser configuration is:

```python
from localdecide.backends.base import LayaMLXBackend, LayaTorchBackend

mlx = LayaMLXBackend(model="ichenney/laya-browser-v32b", subfolder="v32b",
                     revision="161d54d6000913ff279b0afd1ac77faef8685a9b")
torch = LayaTorchBackend(model="ichenney/laya-browser-v32b", subfolder="v32b",
                         revision="161d54d6000913ff279b0afd1ac77faef8685a9b")
# The explicit legacy path remains available:
legacy = LayaTorchBackend(model="browser-legacy", subfolder="v10s",
                          revision="adf912be85ff9221ee171551778456b133c1af75")
```

The custom backend boundary stays deliberately small:

```python
class MyBackend:
    def answer(self, state, questions):
        return {"answers": {...}, "usage": {}}

decider = Decider(backend=MyBackend())
```

For an arbitrary System One-shaped HTTP service, configure model, key, and transport
timeout explicitly. The `Decider` timeout is the overall deadline; HTTP timeout is
bounded by the remaining deadline:

```python
from localdecide import Decider
from localdecide.backends.base import HTTPBackend

backend = HTTPBackend("http://127.0.0.1:8791/v1/systemone",
                      model="browser", api_key="example-key", timeout=10.0)
decider = Decider(backend=backend, timeout=12.0, retries=1)
```

The legacy synchronous `answer(state, questions)` duck protocol is not forcibly
interruptible from Python; a late result is rejected and no further call is made.

### Security boundary

Local MLX/PyTorch backends keep page state on the local machine. `HTTPBackend` sends
the state and questions to the arbitrary endpoint you configure; it may be local,
private, or external, so verify the destination before use. `api_key` becomes a Bearer header; keep real keys in
`LOCALDECIDE_API_KEY` or a secret manager rather than source files or shell history.
Password values are redacted from observations and `TYPE_TEXT` refuses password fields
because this project does not provide vault access. Supply text explicitly and put a
human `confirm` callback in front of irreversible actions such as sending, deleting,
or paying.

## Five ways to use it

### 1. As a library

```python
from localdecide import Decider, choice, noul, score

decider = Decider()   # auto-detects MLX or PyTorch
state = "Blue Waters Pool Supplies, Newcastle NSW. Pool cleaning, equipment sales and repairs."

result = decider.decide(state, {
    "relevant":  noul("Is this business part of the swimming pool industry?"),
    "category":  choice("Which category fits best?", {
        "service":      "pool cleaning and maintenance",
        "retail":       "sells pool equipment or supplies",
        "construction": "builds or renovates pools",
        "unrelated":    "nothing to do with pools",
    }),
    "lead_score": score("How promising is this as a sales lead?",
                        ["not relevant", "weak", "moderate", "strong", "excellent"]),
})

if result.ok:
    print(result.answers.choice("category"))      # 'service'
    print(result.answers.noul("relevant"))        # illustrative probability
    print(result.answers.score("lead_score"))     # illustrative score
else:
    print("failed open:", result.error)
```

### 2. As a browser agent loop

```python
from localdecide import BrowserDecider, Scope
from localdecide.drivers import PlaywrightDriver

# a real text callback: the decision model cannot WRITE, so supply the writing
def text_for(goal, element):
    return {"Search Wikipedia": "Adelaide"}.get(element.label)

# Scoping is a practical lever on both decision difficulty and runtime; tune it
# for the page and backend rather than assuming a fixed timing.
scope = Scope(max_elements=20, prefer_words=["search", "random", "contents"])

with PlaywrightDriver(headless=False) as driver:
    run = BrowserDecider(text_provider=text_for, max_steps=15, scope=scope).run(
        driver, "Search Wikipedia for 'Adelaide' and open the first result.")
    for step in run.steps:
        print(step.n, step.operation, step.label, f"{step.confidence:.2f}")
```

**Scoping is goal-aware.** An element whose label overlaps the goal is never dropped by
the chrome filter, whatever it looks like — because "Random article" is a navigation link
*and* the thing the user asked for. See [Two failure modes](#two-failure-modes-worth-knowing-before-you-file-a-bug).

### 3. As an MCP server for Claude Desktop, Cursor, and friends

```bash
pip install -e '.[mlx]'   # or [torch]
localdecide-mcp
```

Register once and the agent gets two tools — `decide` (typed questions about anything)
and `page_decide` (page observation + goal → chosen element):

Per client:

```jsonc
// Claude Desktop — claude_desktop_config.json (macOS: ~/Library/Application Support/Claude/)
{ "mcpServers": { "localdecide": { "command": "/opt/homebrew/bin/localdecide-mcp" } } }

// Cursor — ~/.cursor/mcp.json (same shape)
{ "mcpServers": { "localdecide": { "command": "localdecide-mcp" } } }

// Hermes — config.yaml
// mcp_servers: [ { name: localdecide, command: localdecide-mcp, args: [] } ]
```

Find the real path with `which localdecide-mcp`. Zero SDK dependency on either side: the
server speaks MCP over stdio with nothing but stdlib JSON, so it runs anywhere the
package installs. Tools appear in your agent as `decide` and `page_decide`.

### 4. As a service other agents point at

```bash
localdecide serve --port 8791
```

| Endpoint | Shape | Who it is for |
|---|---|---|
| `POST /v1/systemone` | TypeSafe Jev / Laya wire format | anything already written against Jev's HTTP API — change one base URL |
| `POST /v1/decide` | `{state, questions}` | your own code, minimal ceremony |
| `POST /v1/table` | `{goal, observation}` → chosen index | browser tooling that has an observation and wants a decision |
| `GET /v1/models` | served model list (TypeSafe-compatible shape) | tooling that lists models |
| `GET /healthz` | liveness + active backend | ops |

The server answers CORS preflights (`OPTIONS`) and sends
`Access-Control-Allow-Origin: *` on everything, so browser extensions and local
web consoles can call it directly — the same convention Ollama uses for a
localhost-only tool service. It binds to `127.0.0.1` unless told otherwise.

Because the `systemone` dialect is the same contract Jev and Laya speak, projects
that were built for those APIs work against a local model by setting a base URL —
for example [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast)
after applying its local-endpoint patch, or any agent that talks to a
TypeSafe-compatible gateway.

### 5. As an agent skill

`skills/` holds plain `SKILL.md` files — the portable format Claude Code, Codex,
Cursor, and Hermes read. Point your agent at this repo and say *"install the
localdecide skills"*, or copy the folder into your agent's skills directory.

| Skill | What it gives an agent |
|---|---|
| `skills/browser-decide` | how to run the loop, what each operation means, when to stop |
| `skills/decide` | the primitive: typed questions, confidence gating, calibration notes |

**Installing into an agent** (the installer detects which ones you have):

```bash
git clone https://github.com/ChenneyZhuang/laya-browser-agent
cd laya-browser-agent
python3 install_skills.py        # copies skills/ into every agent it finds
python3 install_skills.py --check   # preview only, changes nothing
python3 install_skills.py --uninstall
```

or point your agent at this repo and say *"install the laya-browser-agent skills"* —
the SKILL.md files are plain markdown, so Claude Code, Codex, Cursor, Hermes and
anything that reads the format can follow them without this installer.

---

## How it works

```
observe()  ──►  ElementTable  ──►  questions  ──►  local model  ──►  index
   │                                                                    │
   └──────────────────── your executor resolves the index ◄─────────────┘
```

1. **Observe.** Your driver reads the page and returns controls it found: role, label,
   current value, options. Nothing is decided yet — the driver only reports.
2. **Table.** `build_element_table` numbers the actionable controls and maps each
   operation to the elements that actually support it (`CLICK` only sees clickable
   things, `TYPE_TEXT` only editable fields). An option the executor cannot act on is
   never offered.
3. **Ask.** `table_to_questions` builds one `operation` question plus, speculatively,
   one target question per operation. When a page has several dropdowns, each gets
   its own option question, so the chosen option belongs to the chosen field. They
   are answered in a *single forward pass* — one round trip.
4. **Validate.** Nothing leaves the decision layer until it passes: offered key,
   finite probabilities that sum to one, argmax agreement.
5. **Act.** The loop resolves the chosen index to your handle and calls your executor.
   It also owns the history, the repeat-detection, the step budget, and the
   human-confirmation gate for irreversible-looking actions. A model that loops is
   stopped by the harness, not trusted to notice.

### Design choices worth knowing

**Why scoping is the first knob, not the model.** A decision can get harder and slower as
the option list grows. `Scope` narrows what the model sees and the effect can be larger
than a prompt tweak. Critically it is
**goal-aware**: a label overlapping the goal is protected from the chrome filter, since
nav furniture and legitimate targets are the same elements on many pages.

**Why coarse-to-fine for wide pages.** Laya's decision head has a finite token budget
shared across a question's options, so wide option lists can make labels hard to
distinguish. `localdecide` splits wide observations into interleaved chunks, lets the
chunk winners compete, and recombines the probabilities. Prefer scoping first; chunking
is a safety net, not a performance guarantee.

**Why fail-open is the default.** A decision layer that can *block* an agent is a
liability. A timeout, a malformed answer, or a low-confidence result returns a
`Decision` with `ok=False`; your loop keeps its own control flow.

**Why you supply the text.** Decision models physically cannot write a string, so a
`TYPE_TEXT` step needs a text provider — a small LLM, a lookup table, a regex over the
goal. If you do not supply one, the loop refuses the step rather than guessing. This
split (the decision model chooses *what*, a separate provider writes *what into it*)
keeps text generation outside the typed decision contract.

**Why no screenshots.** The model reads text. Screenshots are for your logs, not for
the loop — and skipping them is a large part of why this is fast and cheap.

---

## Historical benchmark evidence

`model="browser"` recommends
[ichenney/laya-browser-v32b](https://huggingface.co/ichenney/laya-browser-v32b);
`model="browser-legacy"` selects the upstream v10s path. Historical checkpoint
diagnostics, methodology, and raw records are linked in
[`reports/v20/MULTIDIM_COMPARISON.md`](reports/v20/MULTIDIM_COMPARISON.md) and
[`reports/v20/JEV_COMPARISON.md`](reports/v20/JEV_COMPARISON.md). They are not
current acceptance results. This README does not repeat old test counts or
pricing, and it makes no current performance guarantee.

The companion training-log repository is a separate project and is not treated
as evidence here unless a committed report links the exact raw record.

### Performance evidence

Latency depends on the checkpoint, runtime, hardware, page size, network, and
whether the model is already loaded. Historical ranges belong in the committed
reports with their raw inputs and conditions; they are not current guarantees.
Scope the observation and treat any timing printed by examples as illustrative.

### Decision quality

The models are task-specific foundation models, not oracles. A browser fine-tune
may fail on a vocabulary mismatch, an unseen control, a dynamic state, or a
domain outside its training distribution. The committed reports contain
historical spot checks and raw conditions; no current accuracy or confidence
number is asserted here.

### Two failure modes worth knowing before you file a bug

Both of these bit this project during development and are now handled, but they will
shape your experience:

**1. Collapsed menus are invisible.** Wikipedia's "Contents" and "Random article" links
live inside a hamburger menu that is not open, so they are not in the DOM and cannot be
observed. The model cannot be blamed for not clicking a control it was never shown.
`typesafe-computer-use`'s author documented the same limitation for
`jev-ultrafast`'s DOM reader. **Fix: open the menu first** (a `CLICK` on the toggle), or
observe from a URL where the control is already expanded. This is a scoping problem,
not a model problem.

**2. A static "drop the nav bar" filter eats legitimate targets.** "Random article" *is*
a navigation link — and it is also exactly what the user asked the agent to click. A
filter that removes it turns a working agent into one that silently cannot do the task,
and the failure is indistinguishable from a model error. Therefore scoping here is
**goal-aware**: an element whose label overlaps the goal is never dropped, whatever else
it looks like. The `TestScope` regressions cover this, including the one that caught a
quote-handling bug (`'random` / `article'`) that silently disabled the protection.

### Multilingual pages: a grounding layer, not a guarantee

The browser checkpoint may not bridge scripts or vocabulary gaps. `Scope` can
remove obvious other-script distractors for non-Latin goals, but same-script
ambiguity and unseen wording remain model limitations. Treat the grounding layer
as a diagnostic aid, not an accuracy guarantee; the historical reports contain
the committed cases and conditions.

### Known model failure modes and harness responses

The harness guards against repeated actions, low-confidence proposals, empty
submits, toggles already in the requested state, disabled/read-only targets, and
irreversible actions requiring confirmation. These are safety boundaries, not
proof that the model is correct. The option list and page state matter more than
prompt wording; use the committed diagnostics when investigating a failure.

### Live-site evidence

Live-site examples and model-backed diagnostics are separate from the model-free
acceptance suite. They require real browser/network access and must be rerun by
the parent release owner. Any example output, confidence, timing, or success
count is illustrative unless it links to a committed raw record.

---

## Compatibility with jev-ultrafast and other Jev tooling

[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) posts
`{model, state, questions}` to `POST /v1/systemone` and validates replies with
`validate_choice`: choice in the offered ids, probabilities covering exactly those ids,
finite numbers summing to 1±0.02, the chosen id must be the argmax.

**This is verified, not assumed**: `tests/test_jev_compat.py` replays a realistic
jev-ultrafast request through our server and runs their validation verbatim — it passes.
In practice you point jev-ultrafast at this server by setting its TypeSafe base URL to
`http://127.0.0.1:8791/v1/systemone` (their model.py reads `TYPESAFE_BASE_URL` after
applying the community local-endpoint patch).

The key differences from running against hosted Jev:

| | hosted Jev | laya-browser-agent |
|---|---|---|
| Latency per decision | endpoint-dependent | runtime/page/backend-dependent |
| Cost | provider terms apply | local runtime or HTTP provider terms apply |
| Page content | sent to TypeSafe | local backends keep it local; HTTP sends it to the configured endpoint |
| Checkpoint | TypeSafe's, updated server-side | our fine-tuned v32b by default (or upstream v10s via `browser-legacy`) — pass the reviewed v32b revision `161d54d6000913ff279b0afd1ac77faef8685a9b` when reproducibility matters |
| Fine-tuning | not possible | project checkpoint: [v32b](https://huggingface.co/ichenney/laya-browser-v32b); verify any training record separately |

## Reference implementations & sources

This project would not exist without the work below. What was taken from each is
stated explicitly, because attribution matters more than a link dump.

### The models

| Source | License | What it is | How it is used here |
|---|---|---|---|
| [**Convai Innovations — Laya**](https://github.com/NandhaKishorM/laya) ([weights](https://huggingface.co/convaiinnovations/laya)) | Apache-2.0 | The official open-weight Laya base/System 1 decision model family. It is not the browser fine-tune. | Loaded as the decision model when selected. The question/answer contract in `decider.py` follows it. No code copied. |
| [**TypeSafe — Jev**](https://docs.typesafe.ai) | proprietary | The model that defined the "System One model" category and the `/v1/systemone` wire format that agents already speak. | The compatibility dialect in `serve.py` mirrors its public HTTP contract so existing clients work. No code used. |
| [**cklxx/laya-browser**](https://huggingface.co/cklxx/laya-browser) | Apache-2.0 | An upstream browser fine-tune/reference checkpoint and training format; distinct from the official Laya base. | The fine-tuning reference for our [v32b checkpoint](https://huggingface.co/ichenney/laya-browser-v32b) (`model="browser"`); the historical v10s path remains available as `model="browser-legacy"`. No code copied. |
| [**mizorewww/laya-mlx**](https://github.com/mizorewww/laya-mlx) | Apache-2.0 | Independent MLX (Apple Silicon) runtime for Laya, with port-fidelity validation. | Used as the Apple Silicon backend (`LayaMLXBackend`). Dependency, not vendored code. |

### The browser-agent pattern

| Source | License | What it contributed |
|---|---|---|
| [**browser-use/jev-ultrafast**](https://github.com/browser-use/jev-ultrafast) | MIT | The single-request-per-decision-cycle design: operation question + speculative target questions sharing one observation, small LLM only for text entry. The operation vocabulary and the `NEXT_ACTION`/`TARGET` instruction text are adapted from here, with attribution in `page.py`. |
| [**awlevin/typesafe-computer-use**](https://github.com/awlevin/typesafe-computer-use) | MIT | Proved screenshots are unnecessary for GUI control (OCR + accessibility tree + classification). Informed the decision to keep images out of the loop entirely. |
| [**Sac-Y/Jev-cu**](https://github.com/Sac-Y/Jev-cu) | unlicensed | The safety model: default dry-run, a policy gate that stops destructive actions for human confirmation, UI text treated as data rather than instructions, app allow-lists. `RISKY_HINTS` and the confirmation gate follow this. No code used. |
| [**kerpopule/hermes-jev-skills**](https://github.com/kerpopule/hermes-jev-skills) | MIT | The "everything fails open, and here is exactly what leaves the machine" posture, plus proof that this plugs into agent harnesses through public seams only. Read for design; no code used. |

### Research and framing

| Source | What it contributed |
|---|---|
| [Nandakishor Mukkunnoth — *I Built Non-Autoregressive Decision Models with RL a Year Ago*](https://laya.convaiinnovations.com/) | Background/design reference only; its claims are not acceptance evidence for this repository. |
| [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) | Upstream benchmark methodology and limitations. Its values do not transfer automatically to this browser checkpoint. |
| [arXiv 2609.23959 — *Open-Jev Judgments on CallScreenBench*](https://arxiv.org/abs/2609.23959) (Sep 2026) | Independent peer evidence about typed-decision safety screening. It is contextual evidence, not a result for this repository or its browser checkpoint. |
| [jev.guide — Browser Use + Jev](https://jev.guide/en/explore), [madewithjev.com](https://madewithjev.com/categories/agents-and-browsers) | Surveys of independent browser/computer-use projects built on this model class. |

**Nothing here is a fork.** The models are dependencies. The design ideas are
credited above and in code comments at the site where each one is used. Laya and
the browser checkpoints retain their upstream license and attribution chain.
Mind2Web, NNetNav, WebChain, and any other training/evaluation datasets retain
their own applicable terms; this project does not relicense or imply rights to
those datasets.

---

## Limitations — read before trusting it

- **Zero-shot on your own domain may be weak.** These are task-specific,
  fine-tunable foundation models. Authentication-gated flows, oddly-shaped SPAs,
  and canvas apps may not be covered by the browser fine-tune.
- **Text entry is not solved here.** The model picks the field; you supply the string.
- **One decision per cycle.** There is no multi-step lookahead: the model cannot plan.
  It picks the best next action given the current page, which is why the harness
  carries history and rules.
- **Password fields are invisible by design** in the standard observation — login
  flows therefore cannot be automated through this loop and are not intended to be.
- **`SELECT` is two-step** (pick the field, then the option). The drivers verify the
  requested option and read back the final value after change handlers run.
- **Wide pages can degrade.** The harness can chunk large observations, but more
  lookalike options still make a decision harder; prefer scoping the observation.
- **The safety gates are hints, not guarantees.** `RISKY_HINTS` and the toggle-guard
  vocabulary are keyword lists. Do not run unattended against anything that can spend
  money, send messages, or delete data without a `confirm` callback that actually checks.
- **It may propose an already-selected toggle.** The toggle guard refuses a control
  already in the requested state, but ambiguous goal wording still needs testing.
- **Confidence is not correctness.** The gate refuses low-confidence actions, which stops
  the worst case, but a confident wrong answer is not caught by anything here. Verify
  outcomes in your own code when the stakes are real.
- **A clean page state matters more than a clever prompt.** If behaviour looks wrong,
  inspect and scope the observed state before changing prompt wording.

## Project layout

```
localdecide/
  decider.py       questions, answer validation, fail-open, coarse-to-fine
  page.py          the element table contract + question construction
  scope.py         goal-aware observation scoping (the biggest perf lever)
  loop.py          observe → decide → act, loop guards, confirmation gate
  drivers.py       Playwright and CDP drivers
  serve.py         the HTTP dialects (/v1/systemone, /v1/decide, /v1/table)
  cli.py           localdecide doctor|decide|table|serve
  mcp_server.py    MCP over stdio for Claude Desktop / Cursor
  backends/        MLX, PyTorch, and HTTP backends behind one protocol
skills/            portable SKILL.md files for agent harnesses
tests/             model-free contract, wire, MCP, packaging, and safety regressions
examples/          runnable examples
examples/diagnostics/   the measurement scripts behind the benchmark tables
```

## Development

```bash
git clone https://github.com/ChenneyZhuang/laya-browser-agent
cd laya-browser-agent
python3.12 -m venv .venv && .venv/bin/pip install -e '.[playwright,cdp]' pytest
.venv/bin/python -m pytest -q   # model-free collection; live classes are skipped by default
```

### Testing philosophy

Two suites, deliberately separated:

| Suite | What it proves | Needs |
|---|---|---|
| `tests/test_contract.py` | the **harness** rules: answer validation, fail-open, index resolution, loop guards, confidence gate, toggle guard, and scoping against fake backends. | nothing |
| `tests/test_live.py` | the **real checkpoint in a real Chromium** against fixture pages: element families, multilingual labels, multi-step flows, the safety gates end to end. | a model runtime + `playwright install chromium` |

The model-free release path does not install a model extra, and ordinary pytest
collection cannot load a checkpoint because live tests require the explicit
`LOCALDECIDE_RUN_LIVE=1` opt-in. `LOCALDECIDE_SKIP_LIVE=1` still overrides it.
A live run requires the model runtime, Chromium, and an explicitly authorized
environment.

**Run the live suite in batches, not all at once:**

```bash
./scripts/run_live_batched.sh
# The launcher exports LOCALDECIDE_RUN_LIVE=1 after checking that this is a source checkout.
```

That script exists for a real reason. A full live run puts a model checkpoint and a
Chromium on the machine at once, and on a 16 GB laptop that was enough to trigger a
kernel watchdog panic and reboot it. Batching keeps the peak low, the script aborts if
free memory drops below 25%, and the browser work runs in a subprocess
(`tests/_browser_child.py`) that asks this process for decisions over a pipe. The
model-free lifecycle tests verify lazy model construction and driver cleanup; the
subprocess timeout is exercised by the live helper when that suite is authorized.

### Fixture pages

`tests/fixtures/` holds pages built to break an observation reader, not to look nice:

- `element_gym.html` — every interactive element family, plus hidden/zero-size/disabled
  controls that must **not** be offered.
- `multilingual.html` — labels in Chinese, Japanese, Korean, Arabic (RTL), Russian,
  Greek, Thai, Hindi, Vietnamese, Turkish, and emoji-prefixed English.
- `flow_shop.html` — a 4-step flow (search → basket → payment → done) with a destructive
  action and a payment step to exercise the confirmation gate.

### Diagnostics

`examples/diagnostics/` contains historical diagnostic scripts. They are runnable
when their optional runtime and data are available, but their outputs are not
current release acceptance results:

| Script | Question it answers |
|---|---|
| `observation_profile.py` | how many elements does the reader see, and where does latency go |
| `scope_effect.py` | what scoping actually buys |
| `instruction_ablation.py` | does instruction wording change the answer (mostly no) |
| `state_ablation.py` | does page text change the answer (yes, a lot) |
| `position_bias.py` | is the answer position-dependent (English: no; CJK: unstable) |
| `multilingual_accuracy.py` | per-script accuracy on both checkpoints (raw numbers) |
| `grounding_effect.py` | the grounding filter's before/after on the same cases |
| `text_priming.py` | which specific words in the page text cause the wrong answer |
| `checkbox_probe.py` | the checkbox-undoing behaviour, in isolation |
| `check_hidden.py` | hidden elements are excluded from the observation |

## Troubleshooting

**`localdecide doctor` says "No local decision runtime yet"**
Install the extras for your platform: `pip install -e '.[mlx]'` on Apple Silicon,
`pip install -e '.[torch]'` everywhere else. Then run doctor again — it now runs a
one-decision smoke test, so "OK" means the checkpoint loaded and answered.

**The first decision may be slower**
The first use may download and load the selected checkpoint; time and storage
depend on the checkpoint, cache, hardware, and runtime. Pre-warm by running
`localdecide doctor` after installing a model extra.

**The model picks the wrong element**
Look at what it was offered before tuning anything. Diagnose with:

```bash
localdecide table --observation page.json --goal '...' --verbose
```

Common causes include too many options (scope to a small task-relevant set), page text priming
(see [known model failure modes](#known-model-failure-modes-and-harness-responses)),
or a goal phrased in a different language from the labels (see [Multilingual](#multilingual-pages-a-grounding-layer-not-a-guarantee)).

**My run errored with "model is not confident enough to act"**
The confidence gate refused twice — that is the harness protecting you from a near-coin-flip
action. Either the page is genuinely ambiguous (narrow the observation), or your goal does
not match what is on the page.

**Playwright raises "It looks like you are using Playwright Sync API inside the asyncio loop"**
The model runtime created an event loop. Do not mix them in one process — run browser work
in a subprocess (see `tests/_browser_child.py` for the pattern). On a 16 GB machine this is
not optional: the two together can exhaust memory and hard-reboot an Apple Silicon Mac.

**Windows: `UnicodeDecodeError` reading files**
Always pass `encoding="utf-8"` when your code reads this repo's text. (This bit our own CI;
the fix is applied everywhere in-repo.)

**Chinese/Japanese goals pick the wrong control**
Script grounding filters to same-script labels automatically. Remaining misses come from
Han-family overlap (Japanese labels on a Chinese page) — disambiguate the goal, or fine-tune
on your domain.

---

## License

Apache-2.0 — matching the license of the Laya models it runs.

## Contributing

Most useful contributions, roughly in order:

1. **More drivers** — a Selenium one, a remote-CDP one, an Android one. The `Driver`
   protocol is three methods.
2. **A text provider** that turns a goal into a field value well enough to drop in.
3. **A fine-tuning recipe** for a new domain, in the spirit of `laya-browser`.
4. **More dialects** — MCP server, OpenAI-compatible tool-calling shim, whatever your
   stack speaks.

If you build something with this, an issue with a link is very welcome.
