# v37c render-parity probes — how the restraint class was recovered

**Date:** 2026-09-26/27. **Machine:** the 3080 box, probes on **CPU** (ran alongside
training without touching the GPU). **Suite:** the same 95-case wide suite.

## Why these probes exist

Wide-suite scoring (which uses the *deployment* render, `localdecide`) showed the
restraint class at **0/N for every generation**, while training-side probes showed
BLOCKED climbing (0 → 38/40 by epoch 8). Two hypotheses: (a) the class is not
learned; (b) it is learned but the deployment render masks it.

## The renders under test

| tag | state | operation options | texts | rules |
|---|---|---|---|---|
| **A** (deploy) | lean (page + recent_actions) | dynamic set: offered targeted ops + SCROLL_DOWN, SCROLL_UP, WAIT, DONE, BLOCKED (7 on these pages) | `localdecide.OPERATIONS` | `NEXT_ACTION_RULES` |
| **B8 / T8** (training) | lean | fixed 8: CLICK, TYPE_TEXT, SELECT, SCROLL_UP, SCROLL_DOWN, WAIT, DONE, BLOCKED | `build_laya_items.OPERATION_LABELS` | training `RULES` |
| **B7** | lean | B8 minus SELECT | as B8 | as B8 |
| **RICHT** | **rich (v1: elements inside state)** | Training 8-set + texts | as B8 | as B8 |

## Probe 3 — render arms × checkpoints (op-level, 95 cases)

| checkpoint | A pick / rest | B8 pick / rest | B7 pick / rest |
|---|---|---|---|
| v10s (upstream) | **69/79** / 0/16 | 55/79 / 0/16 | 53/79 / 0/16 |
| v32b (published) | 64/79 / 0/16 | 45/79 / 0/16 | 40/79 / 0/16 |
| v37c-e3 | 62/79 / 3/16 | 38/79 / **6/16** | 33/79 / 4/16 |

Reading: upstream and v32b **collapse on pick under the training render** (their
priors expect the deploy render), while v37c already answers restraint under the
training render and none of these three arms shows the full picture.

## Probe 4 — factorized arms (v37c-e5; pick numbers are op-level)

| arm | what changed vs A | pick | restraint |
|---|---|---:|---:|
| A | — (baseline) | 75/79 | 4/16 |
| TX | criterion texts → training texts | 74/79 | 4/16 |
| OR | option order → training order | 74/79 | 3/16 |
| RU | rules text → training RULES | 46/79 | **7/16** |
| T8 | texts + order + rules (lean state) | 50/79 | 6/16 |
| TRO | A-set reordered + training texts/rules | 51/79 | 6/16 |
| RICHA | rich state + A render | — | 6/16 |
| **RICHT** | **rich state + training render** | — | **10/16** |

Reading: the rules text alone unlocks much of the restraint class (RU 7/16) but
wrecks pick; the rich state recovers pick; the **full training-native render on
the rich state (RICHT)** is the only arm where both sides stay healthy.

## Probe 5 — full scoring (op + resolved target label)

| checkpoint / arm | pick | restraint | overall |
|---|---:|---:|---:|
| v10s baseline (deploy, evaluate_wide) | 68/79 | 0/16 | 68/95 |
| v37c-e5 · A | 67/79 | 4/16 | 71/95 |
| v37c-e5 · RICHA | 67/79 | 6/16 | 73/95 |
| v37c-e5 · **RICHT** | 65/79 | **10/16** | **75/95** |
| v37c-e7 · A | 60/79 | 7/16 | 67/95 |
| v37c-e7 · RICHA | 51/79 | 11/16 | 62/95 |
| **v37c-e7 · RICHT** | **64/79** | **12/16** | **76/95** |

The RICHT > RICHA > A ordering repeats on both checkpoints.
(v10s's own RICHT number is pending — its priors are known to prefer A.)

## v37c wide trajectory (deploy render A — the class climbs even there)

| epoch | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pick (sem) | .861 | .772 | .759 | .658 | .835 | .785 | .722 |
| restraint | 0 | 1 | 3 | 5 | 4 | 6 | **7**/16 |

In-training probes (training render, lean state), epoch 8: **DONE 36/40,
BLOCKED 38/40, TYPE_TEXT 33/40, CLICK 60/60**.

## Conclusions

1. **The restraint class was learned; the deployment render masked it.** The gap
   between deploy render (7/16) and training render + rich state (12/16) is a
   render-parity bug, not a data gap.
2. **Target deployment config = RICHT**: rich state (layout `v1`) +
   training-native operation rendering (fixed 8-option set unless the page can't
   support it — see TRO caveat, still to refine against the real runtime).
3. Shipping this needs a **parity mode in `localdecide`**, then re-verification
   through the real runtime (`laya.load(...).predict`) on exports of the new
   render, plus a fair parity table for v10s/v32b.

## Open items at box-off (~01:00 AEST)

- probe5 for e6 and e8 (probe5c for e8 was mid-run), + v10s/v32b under RICHT.
- Regression battery (suite v5 + holdout): e1–e2 complete, e3 mid-run.
- Blend of the winning epoch into v24a-b06 + final candidate selection.
- `localdecide` parity implementation + tests (Mac-side, no GPU needed).
