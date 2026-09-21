# laya-browser-agent

**Browser agent decisions powered by Laya — the open-source System 1 model. A local alternative to TypeSafe Jev: no cloud, no API key, no screenshots.**

[![tests](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![GitHub release](https://img.shields.io/github/v/tag/ChenneyZhuang/laya-browser-agent)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Runs on](https://img.shields.io/badge/runs%20on-Apple%20Silicon%20%7C%20CUDA%20%7C%20CPU-black)

A local, open-weight alternative to [TypeSafe Jev](https://docs.typesafe.ai) for the
browser-driving use case — running [Laya](https://github.com/NandhaKishorM/laya), the
open-source "System One" decision model, fully on your own machine. Works with
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) via the same
wire format, and speaks TypeSafe's `/v1/systemone` dialect, so existing Jev tooling
points at it by changing one base URL.

A decision model answers typed questions about a state and returns calibrated
probabilities. It never writes text, so it cannot hallucinate an instruction. That
makes it exactly the right shape for the *deciding* half of a browser agent: hand it
a numbered table of the controls on a page, and it tells you which operation to run
and which element to act on.

This project wires those models into that role, locally, for whatever agent you
already use.

```python
from localdecide import BrowserDecider
from localdecide.drivers import PlaywrightDriver

with PlaywrightDriver(start_url="https://en.wikipedia.org/wiki/Main_Page") as driver:
    run = BrowserDecider().run(driver, "Click the 'Random article' link in the navigation.")
    print(run.stopped, run.summary()["median_decision_ms"], "ms/decision")
# -> done 142 ms/decision
```

Measured on an M4 MacBook Air, 16 GB (see [Benchmarks](#benchmarks)):

| | |
|---|---|
| Decision latency | **10–30 ms** steady state, **~150 ms** with a 60-element page |
| Throughput | **up to ~100 decisions/second** |
| Cost | **$0.00** — no API, no metering |
| Model size | 322 M params (~644 MB) on disk |
| Page content sent to a server | **none** |

---

## How this relates to Jev and Laya

| | [TypeSafe Jev](https://docs.typesafe.ai) | [Laya](https://github.com/NandhaKishorM/laya) | **localdecide** |
|---|---|---|---|
| Weights | closed, API only | open, Apache-2.0 | runs Laya's open weights |
| Where it runs | TypeSafe's cloud | anywhere PyTorch runs | **your machine** — MLX on Apple Silicon, PyTorch elsewhere |
| Wire format | `POST /v1/systemone` | same contract | speaks it too (`POST /v1/systemone`) |
| Cost | $0.042/M input tokens | free | free |
| Browser harness | [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (12.6k★) | — | **included**: loop, drivers, guards, skills |
| Page leaves your machine | yes | no | **no** |

If you have read about Jev's "System One" model and want the same idea — typed,
calibrated decisions instead of generated text — running locally for your browser
agents, this is the wiring for it. It uses the browser-tuned Laya checkpoint
(`cklxx/laya-browser`, which itself documents 0% → 62% task success after fine-tuning)
and adds the parts neither project ships: element-table observation, answer validation,
confidence gating, loop guards, and a TypeSafe-compatible server.

## Why this exists

The "System One model" idea — a non-autoregressive model that returns typed,
calibrated decisions instead of prose — went from research to production-worthy in
2026. TypeSafe's **Jev** made it famous; [Convai Innovations' **Laya**](https://github.com/NandhaKishorM/laya)
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

```bash
# Apple Silicon (fastest path — MLX)
pip install 'localdecide[mlx]'

# Linux / Windows / Intel Mac (same checkpoints via PyTorch)
pip install 'localdecide[torch]'

# plus a browser driver
pip install 'localdecide[playwright]' && playwright install chromium
# or attach to a Chrome you already have open and logged in
pip install 'localdecide[cdp]'
```

Check what you have — `doctor` detects your hardware and tells you exactly what to do:

```bash
localdecide doctor
```

On an M4 it prints the chip, memory, runtime, and runs a one-decision smoke test so a
broken install shows up here instead of in your agent:

```
python      3.12.13 (arm64, Darwin)
hardware    Apple M4, 16 GB unified memory
laya-mlx    installed
playwright  installed
backend     laya-mlx
smoke test  OK (85 ms, first call includes model load)
```

### What to expect on different devices

Everything here is measured on real hardware or stated as a limit. The harness is
identical everywhere (pure Python, verified by CI on 6 platform/Python combinations); what
changes by device is which runtime you install and how big a decision you can afford.

| Device | Runtime | Expected experience |
|---|---|---|
| **Apple Silicon M-series, 16 GB+** (M1–M4) | `laya-mlx` | The reference experience: 10–30 ms short decisions, ~330 ms scoped browser steps, everything local. This is what the benchmarks above measure. |
| **Apple Silicon, 8 GB** (M1/M2 base) | `laya-mlx` | Works, but 650 MB checkpoint + Chromium is tight. The subprocess design keeps one model OR one browser resident; close heavy apps. Expect swap pressure on big pages. |
| **Intel Mac** | `laya` (PyTorch) | `laya-mlx` does not run here. Model works; expect ~2–4× the Apple Silicon latency on CPU. Browser loop fine. (Note: GitHub retired the macos-13 runner image in Dec 2025; Intel macOS CI now runs on macos-15-intel, which GitHub itself plans to retire in 2027 — Intel macOS support has a countdown.) |
| **Linux server, CPU only** | `laya` (PyTorch) | Good for batch deciding (no browser needed for classification). Browser loops work headless. Latency similar to Intel Mac CPU. |
| **Linux + NVIDIA GPU** | `laya` (PyTorch, CUDA) | Best PyTorch path — GPU inference cuts latency well below CPU. Also the only place you can *fine-tune* (the laya-browser recipe needs CUDA). |
| **Windows** | `laya` (PyTorch) | Works; same expectations as Linux CPU. Playwright supports it natively. |
| **Below 8 GB total / Raspberry Pi class** | — | Not supported. The checkpoint alone is 650 MB and the decision heads want ~1 GB resident. Use the HTTP backend to reach another machine instead. |
| **Any device, model elsewhere** | `HTTPBackend` | Point `Decider("http://host:8791/v1/...")` at a machine that has the model. Your page content goes to *your* other machine, not to a cloud. |

Two things that do **not** change by device:

- **Accuracy.** The same checkpoint makes the same decision everywhere; the runtimes are
  verified to agree to four decimal places (the MLX port publishes 378/378 parity checks).
- **The safety guards.** Validation, fail-open, confidence gate, toggle guard, loop guard —
  all pure Python, all identical, all CI-tested on every platform in the matrix.

The model downloads once on first use (~644 MB) and is cached.

## Three ways to use it

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
    print(result.answers.noul("relevant"))        # 0.805
    print(result.answers.score("lead_score"))     # 2.72
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

# Scoping is the biggest lever on both speed and accuracy - 20 elements decide in
# ~330 ms, 120 elements take ~1.2 s and make more mistakes.
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
pip install 'localdecide[mlx]'   # or [torch]
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
| `GET /healthz` | liveness + active backend | ops |

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
   one target question per operation. They are all answered in a *single forward pass*
   — two decisions, one round trip.
4. **Validate.** Nothing leaves the decision layer until it passes: offered key,
   finite probabilities that sum to one, argmax agreement.
5. **Act.** The loop resolves the chosen index to your handle and calls your executor.
   It also owns the history, the repeat-detection, the step budget, and the
   human-confirmation gate for irreversible-looking actions. A model that loops is
   stopped by the harness, not trusted to notice.

### Design choices worth knowing

**Why scoping is the first knob, not the model.** A decision gets worse and slower as
the option list grows (measured: 183 ms at 10 options → 1204 ms at 120). `Scope` narrows
what the model sees and the effect is larger than any prompt tweak. Critically it is
**goal-aware**: a label overlapping the goal is protected from the chrome filter, since
nav furniture and legitimate targets are the same elements on many pages.

**Why coarse-to-fine for wide pages.** Laya's decision head has a fixed token budget
shared across a question's options (~192–256 tokens), so beyond roughly 20 options
each label gets ~3 tokens and becomes indistinguishable. Rather than let accuracy
quietly collapse, `localdecide` splits anything wider into interleaved chunks, runs
them in the same pass, then has the chunk winners compete in one extra pass and
recombines the probabilities exactly (`p(o) = p_final(winner) · p_chunk(o)`). This is
directly informed by the failure the `laya-browser` authors documented on Banking77.
Prefer scoping first; chunking is the safety net, not the strategy.

**Why fail-open is the default.** A decision layer that can *block* an agent is a
liability. A timeout, a malformed answer, or a low-confidence result returns a
`Decision` with `ok=False`; your loop keeps its own control flow.

**Why you supply the text.** Decision models physically cannot write a string, so a
`TYPE_TEXT` step needs a text provider — a small LLM, a lookup table, a regex over the
goal. If you do not supply one, the loop refuses the step rather than guessing. This
split (the cheap model chooses *what*, a small model writes *what into it*) is the
same division that made the fastest published browser agents fast.

**Why no screenshots.** The model reads text. Screenshots are for your logs, not for
the loop — and skipping them is a large part of why this is fast and cheap.

---

## Benchmarks

Every number here was measured on this project's development machine — Apple M4,
16 GB, `laya-mlx` 0.1.0, `cklxx/laya-browser` **v10s** checkpoint — against real pages
in a real Chromium. Nothing is copied from a vendor's marketing.

### Latency is dominated by how much you show the model

This is the single most important operational fact in this repo. Same page, same
question, only the number of offered elements changes:

| Elements offered | Latency | Passes |
|---|---|---|
| 10 | **183 ms** | 1 |
| 20 | **333 ms** | 1 |
| 30 | 487 ms | 2 (auto-chunked) |
| 60 | 678 ms | 2 (auto-chunked) |
| 120 | **1204 ms** | 2 (auto-chunked) |

So: **scope your observation to ~20 elements and decisions cost a third of a second.**
Feeding it the whole page costs four times that and makes more mistakes, because every
extra lookalike candidate is a chance to pick the wrong one. `Scope` exists for exactly
this, and `BrowserDecider` applies it by default.

For comparison, the same checkpoint on a short non-browser state (3 questions, a
paragraph of text) decides in **10–30 ms**, and sustains **42–100 decisions/second**.
Browser work is slow because a page is big, not because the model is slow.

| Workload | Latency |
|---|---|
| Short state, 3 questions, steady state | 10–30 ms |
| Two-step flow (fill field → submit) | 143 ms + 155 ms |
| Chinese (multilingual checkpoint), 3 questions | 29–202 ms |
| Model cold load | 0.9–1.3 s cached / ~12 s first download |

### Decision quality

Correctness spot-checks, browser-tuned checkpoint, real pages:

| Task | Result |
|---|---|
| Wikipedia: "Click the Random article link" (8 candidates, hand-built table) | `CLICK` → *Random article*, **0.867** |
| Wikipedia: "Search for Adelaide" step 1 | `TYPE_TEXT` → searchbox, **0.917** |
| Wikipedia: "Search for Adelaide" step 2 (field filled, submit visible) | `CLICK` → *Search* button, **0.897** |
| Hacker News: "Open the newest submissions page" | `CLICK` → *new*, correct |
| Live Chromium, 120-element page → scoped → loop | clicked through, `CLICK` on the correct control |

Bank/finance messages in Chinese (multilingual checkpoint, zero-shot):

| Input | Result |
|---|---|
| 招商银行 "消费128.50元 盒马鲜生" | `expense` **0.997**, is-transaction **0.998** |
| 支付宝 "收益0.85元已到账" | `income` **1.000** |
| "恭喜中奖！点链接领iPhone" | is-scam **1.000** (caught) |
| 中国移动 "验证码837291" | classified as a transaction — **wrong** |

The last row is the honest caveat that applies to every zero-shot use of these
models: they are **foundation models for a task, not oracles**. The browser checkpoints
are fine-tuned for browser decisions and perform well there; a base checkpoint is
documented at ~0.10 top-1 on the same task. For your own domain, expect to fine-tune.

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
it looks like. Ten tests cover this (`TestScope`), including the one that caught a
quote-handling bug (`'random` / `article'`) that silently disabled the protection.

### Multilingual pages: a grounding layer, not a bigger model

Tested against labels in Chinese, Japanese, Korean, Arabic, Russian, Greek, Thai, Hindi,
Vietnamese and Turkish, the browser-tuned checkpoint **cannot bridge scripts**: a Chinese
goal picked a Hindi button (p≈0.06, i.e. no signal at all), and the multilingual base
checkpoint returned nothing usable. Only goals whose own characters appear in a label
worked.

The fix is not a different model — it is letting code do what code is good at. Before the
decision, `Scope` detects the goal's script, scores every label for overlap, and (for
non-Latin goals) **removes the other-script distractors from the option list entirely**:

| | offered | correct |
|---|---|---|
| without grounding | 25 mixed-script labels | 1/9 |
| with script grounding | 1-10 same-script labels | **5-8/9** |

Reordering alone did nothing (measured 1/9 → 1/9); removing the distractors is what
worked. The remaining misses are honest and instructive: on a page mixing Chinese and
Japanese labels, both scripts are Han-family, so a Japanese distractor survives the filter
and can still pull the answer (measured: `カートに追加` beating the intended `加入购物车`).
Per-script precision beyond "same script" needs the model to understand the language —
that is a fine-tuning job, and this harness now makes its failure *visible* instead of
silent. Latin-script goals are untouched — they work fine and the fallback risk is not
worth it. A goal whose script matches no label gets an explicit diagnosis; translating
that goal is your job, and now you know you need to.

### Three things the model gets wrong, and what the harness does about it

These were found by testing the real checkpoint against real pages, and each one is now
either guarded or documented — not hidden.

| Finding | Evidence | Harness response |
|---|---|---|
| **It will undo a checkbox.** Asked to tick an already-ticked box, it answers `CLICK` on that box at p=0.90. The option text reads `checked=true` and the instructions say not to re-toggle — neither helps. | `[1] Terms accepted (checked)` → `CLICK` target 1, p=0.904 | **Toggle guard**: a click on a control already in the requested state is refused and the model is asked again. If the goal explicitly asks to *uncheck*, it goes through. |
| **It fires submits at near-zero confidence.** On an ambiguous page it proposed `CLICK` on a submit button with **p=0.06**, which would start a flow the user never asked for. | live flow run: `CLICK Search, confidence 0.0609` | **Confidence gate**: any action below `min_confidence` (default 0.15) is refused, twice in a row ends the run. `DONE`/`BLOCKED` are exempt — refusing to *stop* would be the worse failure. |
| **Page text steers it more than instructions do.** With identical options, a state whose text echoes the field and button names pushed `CLICK` to p=0.976; a neutral state gave `TYPE_TEXT` at p≈0.93. The *instruction wording* made almost no difference across four variants. | `instruction_ablation.py`, `state_ablation.py`, `text_priming.py` | **Scope the state, not the prompt.** `text_chars` and element scoping are the real knobs; the diagnostics in `examples/diagnostics/` reproduce every number above. |

The third one is the most useful lesson for anyone tuning this: **the option list and the
page text do the work; prompts do not.**

### What the numbers mean in context

The upstream projects publish their own measurements, which are worth reading
alongside these:

- `laya-browser` reports browser-decision success going from **0% → 62%** (16 real
  tasks × 3 runs) after fine-tuning, element top-1 **0.10 → 0.66**, at **17–23 ms**
  per step on a 322 M model.
- `jev-ultrafast` reports a complete Google Flights search in **7.1 s** at **$0.0039**
  using the hosted Jev API.
- `typesafe-computer-use` reports macOS control at about **$0.0002 per step** using
  screenshots-free OCR + classification.

This project's contribution is the local harness, not the models. Where the harness
adds its own measured value: decisions at **10–30 ms with zero marginal cost**, and
the page never leaving the machine.

---

## Reference implementations & sources

This project would not exist without the work below. What was taken from each is
stated explicitly, because attribution matters more than a link dump.

### The models

| Source | License | What it is | How it is used here |
|---|---|---|---|
| [**Convai Innovations — Laya**](https://github.com/NandhaKishorM/laya) ([weights](https://huggingface.co/convaiinnovations/laya)) | Apache-2.0 | The open-weight System 1 decision model family this project runs. `choice`/`score`/`noul` primitives, the `systemone` request contract. | Loaded as the decision model. The question/answer contract in `decider.py` follows it. No code copied. |
| [**TypeSafe — Jev**](https://docs.typesafe.ai) | proprietary | The model that defined the "System One model" category and the `/v1/systemone` wire format that agents already speak. | The compatibility dialect in `serve.py` mirrors its public HTTP contract so existing clients work. No code used. |
| [**cklxx/laya-browser**](https://huggingface.co/cklxx/laya-browser) | Apache-2.0 | **The decisive piece.** Laya fine-tuned into a browser-agent decision head: the training recipe, the v3 input format, and the published v10/v10s/v11s checkpoints that actually work for picking page elements. | Used as the **default checkpoint** (`v10s`). The format insight (elements in the option list, not the state; page text capped ~1.2k) is implemented in `page.state(layout="v3")`. The coarse-to-fine chunking follows its `systemone_server.py` approach. No code copied. |
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
| [Nandakishor Mukkunnoth — *I Built Non-Autoregressive Decision Models with RL a Year Ago*](https://laya.convaiinnovations.com/) | The RLCD framing, the three primitives, and the honest limitations (options beyond ~20 degrade; zero-shot is weak; temperature calibration needed). |
| [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) | The exact per-language and per-task numbers quoted in the caveats, including the Khmer/Armenian/Hebrew confidence failures that motivate confidence-gating being unreliable on its own. |
| [jev.guide — Browser Use + Jev](https://jev.guide/en/explore), [madewithjev.com](https://madewithjev.com/categories/agents-and-browsers) | The survey of the ~60 independent browser/computer-use projects built on this model class — the evidence that this is a real pattern and not a single demo. |

**Nothing here is a fork.** The models are dependencies. The design ideas are
credited above and in code comments at the site where each one is used.

---

## Limitations — read before trusting it

- **Zero-shot on your own domain is weak.** These are fine-tunable foundation models.
  Browser decisions work because someone already spent ~5 GPU-hours on the
  fine-tune. Authentication-gated flows, oddly-shaped SPAs, and canvas apps are not
  covered by that training.
- **Text entry is not solved here.** The model picks the field; you supply the string.
- **One decision per cycle.** There is no multi-step lookahead: the model cannot plan.
  It picks the best next action given the current page, which is why the harness
  carries history and rules.
- **Password fields are invisible by design** in the standard observation — login
  flows therefore cannot be automated through this loop and are not intended to be.
- **`SELECT` is two-step** (pick the field, then the option). It works, but it is the
  least-tested operation.
- **Wide pages degrade.** Beyond ~20 options the harness chunks automatically, but a
  200-element page is still a harder decision than an 8-element one; prefer scoping
  the observation.
- **The safety gates are hints, not guarantees.** `RISKY_HINTS` and the toggle-guard
  vocabulary are keyword lists. Do not run unattended against anything that can spend
  money, send messages, or delete data without a `confirm` callback that actually checks.
- **It will undo a checkbox if you let it.** Proven, measured, p=0.90. The toggle guard
  refuses it, but the guard reads the goal's wording - a goal phrased ambiguously may do
  the opposite of what you meant. Test your phrasings.
- **Confidence is not correctness.** The gate refuses low-confidence actions, which stops
  the worst case, but a confident wrong answer is not caught by anything here. Verify
  outcomes in your own code when the stakes are real.
- **A clean page state matters more than a clever prompt.** Measured four ways: the same
  options produced `TYPE_TEXT` at p≈0.93 in a neutral state and `CLICK` at p≈0.98 when the
  page text echoed the button's name. If behaviour looks wrong, change what you observe
  before you change what you ask.

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
tests/             39 tests covering the contract (no model needed)
examples/          runnable examples
examples/diagnostics/   the measurement scripts behind the benchmark tables
```

## Development

```bash
git clone https://github.com/ChenneyZhuang/laya-browser-agent
cd localdecide
python3.12 -m venv .venv && .venv/bin/pip install -e '.[all]' pytest
.venv/bin/python -m pytest tests/test_contract.py -q   # 48 tests, instant, no model needed
.venv/bin/localdecide doctor
```

### Testing philosophy

Two suites, deliberately separated:

| Suite | What it proves | Needs |
|---|---|---|
| `tests/test_contract.py` | the **harness** rules: answer validation, fail-open, index resolution, loop guards, confidence gate, toggle guard, scoping. Runs in ~0.2 s against fake backends. | nothing |
| `tests/test_live.py` | the **real checkpoint in a real Chromium** against fixture pages: element families, multilingual labels, multi-step flows, the safety gates end to end. | a model runtime + `playwright install chromium` |

The live suite is skipped by default (`LOCALDECIDE_SKIP_LIVE=1` forces it off) so an
ordinary test run cannot load a checkpoint.

**Run the live suite in batches, not all at once:**

```bash
./scripts/run_live_batched.sh
```

That script exists for a real reason. A full live run puts a model checkpoint and a
Chromium on the machine at once, and on a 16 GB laptop that was enough to trigger a
kernel watchdog panic and reboot it. Batching keeps the peak low, the script aborts if
free memory drops below 25%, and the browser work runs in a subprocess
(`tests/_browser_child.py`) that asks this process for decisions over a pipe — one
checkpoint resident per machine, never two. `TestMemorySafety` encodes these rules as
tests so the mistake is not repeated.

### Fixture pages

`tests/fixtures/` holds pages built to break an observation reader, not to look nice:

- `element_gym.html` — every interactive element family, plus hidden/zero-size/disabled
  controls that must **not** be offered.
- `multilingual.html` — labels in Chinese, Japanese, Korean, Arabic (RTL), Russian,
  Greek, Thai, Hindi, Vietnamese, Turkish, and emoji-prefixed English.
- `flow_shop.html` — a 4-step flow (search → basket → payment → done) with a destructive
  action and a payment step to exercise the confirmation gate.

### Diagnostics

`examples/diagnostics/` is where the claims in [Benchmarks](#benchmarks) come from. Every
one is runnable, and each answers a specific question:

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
Install the extras for your platform: `pip install 'localdecide[mlx]'` on Apple Silicon,
`pip install 'localdecide[torch]'` everywhere else. Then run doctor again — it now runs a
one-decision smoke test, so "OK" means the checkpoint loaded and answered.

**First decision is slow (~10 s)**
The checkpoint downloads on first use (~650 MB) and loads once per process. Later
decisions are milliseconds. Pre-warm by running `localdecide doctor` after install.

**The model picks the wrong element**
Look at what it was offered before tuning anything. Diagnose with:

```bash
localdecide table --observation page.json --goal '...' --verbose
```

Common causes, in order of frequency: too many options (scope to ~20), page text priming
(see [Three things](#three-things-the-model-gets-wrong-and-what-the-harness-does-about-it)),
or a goal phrased in a different language from the labels (see [Multilingual](#multilingual-pages-a-grounding-layer-not-a-bigger-model)).

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
