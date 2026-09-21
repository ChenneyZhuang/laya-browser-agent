---
name: decide
description: >-
  Use when you need fast, cheap, calibrated judgments about text — classification,
  routing, scoring, relevance, guardrails — with a local model instead of an LLM
  call. Covers the three question primitives and how to write good ones.
---

# Typed decisions with a local System 1 model

`localdecide` runs an open-weight decision model locally. It answers questions about a
state in one forward pass and returns calibrated probabilities. It **never writes
text** — that is the point, not a limitation: prose costs latency and can be
hallucinated, probabilities cannot.

Reach for this when you have a judgment to make many times, the answer is one of a
few options (or a number, or yes/no), and you do not need an explanation.

## The three primitives

```python
from localdecide import Decider, choice, noul, score
```

| Primitive | Question shape | Returns |
|---|---|---|
| `choice` | pick one of a closed set of named options | the key, probabilities per option, confidence |
| `score` | place the state on an ordered rubric | expected level (float over the levels), the distribution |
| `noul` | is this proposition true? | `P(true)` in 0..1 |

```python
d = Decider()
r = d.decide(ticket_text, {
    "queue":   choice("Which team owns this?", {"billing": "refunds and invoices",
                                                "infra": "outages and deploys",
                                                "support": "how-to questions"}),
    "urgency": score("How urgent?", ["whenever", "this week", "today", "now"]),
    "angry":   noul("Does the writer express anger or threat?"),
})
if r.ok:
    if r.answers.confidence("queue") < 0.7:
        escalate(r)                                  # do not act on a coin flip
    elif r.answers.noul("angry") > 0.8:
        page_a_human(r)
    else:
        route(r.answers.choice("queue"))
```

## Writing questions that work

**Say what each option means, not just its name.** The model reads your criteria text
as its decision context. `{"billing": "refunds, invoices, payment problems"}` beats
`{"billing": "billing"}`.

**Keep choices narrow.** Under ~20 options. Beyond that the per-option token budget
degrades (the harness chunks automatically, but a two-step coarse-to-fine question is
better written by you: pick the category, then the item).

**Prefer `score` over five `noul`s.** One ordered rubric gives you an expected value
and a distribution; five separate yes/no questions give you five numbers you then have
to combine yourself.

**One state, many questions.** Every question in a single `decide()` call is answered
in the same forward pass. Asking three questions costs barely more than asking one, so
batch them rather than looping.

**Put the evidence in the state, not the question.** If the model needs a chunk of text
to judge, that text is the state. Questions should be short and structural.

## Confidence: use it, do not worship it

The returned confidence is calibrated on the model's own training distribution, which
means it is meaningful *in-domain* and meaningless outside it. The upstream Laya
benchmarks show a model reporting 95% confidence while scoring 0% accuracy on a script
it cannot read. So:

- **Do** branch on confidence with a threshold tuned on your own labelled data.
- **Do** route low-confidence items to a human or a bigger model.
- **Do not** treat a high confidence as proof of correctness on a new domain.
- **Do not** use confidence gating as your only safety layer.

## Calibration

Base weights ship with raw temperature logits; fitting one scalar temperature on a
sample of your own data materially improves calibration error (upstream reports ECE
0.466 → 0.081 for Laya with a single fitted scalar per question type). If you are
about to gate on thresholds, fit the temperature first — otherwise your 0.7 threshold
is arbitrary.

## Failure behaviour

Everything fails open. No model, no key, timeout, rate limit, malformed answer, low
confidence: you get `Decision(ok=False)` and your flow continues on its own path.
Inspect `.ok` and `.error` — never assume success.

```python
result = d.decide(state, questions)
if not result.ok:
    return my_rule_based_fallback(state)     # your existing logic stays in charge
```

## When not to use this

- **Anything that needs prose out.** It cannot write. Use it to decide, then let a
  normal model write.
- **Sequential reasoning.** It answers one round of questions about one state. It has
  no scratchpad and no plan. Multi-step logic belongs in your code.
- **High-stakes single decisions with no verification.** It is fast and cheap; when
  the cost of being wrong is high, use it to *triage* and verify the survivors
  properly.
- **A domain it was not trained on, with no fine-tuning budget.** Zero-shot on your
  own data is weak. Budget for a fine-tune (upstream reports ~5 GPU-hours for a
  browser head) or do not adopt it for that task.

## Cost and latency, honestly

On an M4 with the MLX runtime: **10–30 ms** per decision on short states, up to ~100
decisions/second throughput, **zero marginal cost**, and nothing leaves the machine.
A decision via a hosted equivalent costs about $0.00006–$0.0004 depending on payload,
so the local runtime is a latency and privacy win more than a price win at small volume.
