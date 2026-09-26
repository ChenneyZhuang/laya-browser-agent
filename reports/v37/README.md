# v37 — Outlook / M365 deployment patch

**Why this exists.** The published `ichenney/laya-browser-v32b` checkpoint was
measured against its base (`cklxx/laya-browser` v10s) on a 95-case suite built
from the actual deployment use case — Outlook / Microsoft 365 IT-support browser
work. v32b lost: 60/95 (63.2%) vs 64/95 (67.4%), and it never won a case v10s
lost. The full handoff (method, raw per-case results, failure classes) is in
`handoff/RETRAIN_HANDOFF.md`.

This directory contains the fix attempt and its evidence.

## What was measured (3080 / CUDA, same runner for all checkpoints)

Runner: `evaluate_wide.py` fed by `wide_items.jsonl` (exported from the Mac
harness by `export_wide_items.py`; identical `build_element_table` →
`table_to_questions` pipeline, state layout v3, text 1500 chars — exactly the
payload the Mac bench used).

Scoring is reported under two rules:

* **handoff** — the original bench rule (`op == "CLICK"` and resolved label in
  accepted). Eight fill-target cases (search boxes, compose fields) can never
  pass under it; the rule floor is 71/79 pick + 16 restraint = 87/95.
* **semantic** — the operation the accepted element actually supports
  (`CLICK` or `TYPE_TEXT`). This is the honest measure for fill targets.

### Baselines (3080, 2026-09-26)

| checkpoint | pick handoff | pick semantic | restraint | overall semantic |
|---|---:|---:|---:|---:|
| v10s (cklxx, upstream) | 0.7975 | **0.8608** | 0.0 | **0.7158** |
| v32b (ours, published) | 0.7468 | 0.7975 | 0.0 | 0.6632 |
| v24a-b06 (our base line) | 0.7342 | 0.7848 | 0.0 | 0.6526 |

Reproduce: `evaluate_wide.py <ckpt_dir> wide_items.jsonl --device cuda`.

### Failure classes the patch targets

1. **Operation boundary** — verb-labelled buttons picked as `TYPE_TEXT`
   targets: `New event`, `Today`, `Import contacts`, `Send email to selected`,
   `Add comment`. Six of the "only-v10s-wins" cases are exactly this.
2. **Label confusions** measured across both checkpoints: `Reply` vs
   `Reply all`; `Cc` vs `Show Bcc`; `Sort` vs folder names; `Account manager`
   vs `Move to`; `Pop out` vs `Move to`; `Close` vs `Close ticket`;
   `New mail` for "open the newest email from X"; search field vs `Filter`.
3. **Restraint (0/16 for every checkpoint)** — goals already satisfied must
   answer `DONE`; impossible goals `BLOCKED`. Neither class existed in training.
4. **people / CRM surface** — 20% vs 80–100% on other surfaces.

## The patch data (`generate_v37_wide.py`)

6,762 records in the v20/v29 record contract, seven families:

| family | n | purpose |
|---|---:|---|
| opboundary | 783 | button-CLICK vs field-TYPE_TEXT boundary, with exact-label twins |
| confusions | 1,671 | the measured confusion pairs above, both directions |
| restraint_done | 1,017 | already-satisfied → DONE (goal-states-it / evidence-visible / hard-negative twin) |
| restraint_blocked | 749 | impossible → BLOCKED (same three shapes) |
| people | 382 | CRM surface rebalance |
| folders_focus | 1,117 | folder / search / settings navigation insurance |
| misc | 1,043 | labels that already passed, kept alive |

Counterfactual twins matter: the same wording flips gold when the state
changes, so `DONE`/`BLOCKED` cannot be triggered by keywords.

Build check (3080): 6,762 records → 11,914 items, `candidate_recall` 1.0,
0 target misses, no SELECT golds.

## Training

`v37_refine.py` — the v31 recipe that produced the v32 champions: fine-tune
from `v24a-b06`, encoder frozen, head-only, RLCD loss (noisy-logit PG + soft
CE), mix = v18c + v23 + v31-noul + v37 (38,700 items).

* `v37a` lr 1e-4, 2 epochs — `v37b` lr 5e-5, 1 epoch.

`v37_blend.py` — low-alpha head blends into the v24a-b06 base
(α = 0.03 / 0.08 / 0.15), the step that turned v31 runs into the v32 champions.

`v37_chain.sh` — the full idempotent pipeline: build → train → blend → wide
battery (acceptance gate) → regression battery (suite v5 + holdout).

## Acceptance criteria

1. Wide suite (95): beat v10s — ≥ 64/95 handoff-rule (69/95+ semantic),
   restraint strictly above 0/16 for the first time.
2. No regression on the original batteries: suite v5 ≥ 0.56, holdout ≥ 0.70.
