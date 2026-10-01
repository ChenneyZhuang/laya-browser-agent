# Corrected published-checkpoint acceptance

These are **offline decision regression results**, not live-browser task success and not an independent held-out benchmark. The wide suite has been used in earlier development; do not use it to claim a general win over another model.

Run: `20261001T054646Z-corrected` (2026-10-01 UTC). Aggregated evidence: [`release_acceptance_summary.json`](../examples/diagnostics/release_acceptance_summary.json). Each arm ran the same 95 items and verified 95 unique IDs, zero execution errors, zero nonfinite values, and a real-runtime choice/score/noul contract smoke.

## Checkpoints and protocols

- User model: [`ichenney/laya-browser-v32b`](https://huggingface.co/ichenney/laya-browser-v32b), revision `161d54d6000913ff279b0afd1ac77faef8685a9b`, subfolder `v32b`.
- Upstream multilingual: [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual), revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, subfolder `multilingual`.
- Fixed published snapshots, same CUDA host and runtime, BF16 autocast, encoder/head token limits 4096/1024. Runtime versions are in the JSON evidence.
- **Deployed:** normal Decider, `max_options_per_question=20`. Forty-seven items use coarse-to-fine; 142 backend calls for 95 cases, excluding smoke.
- **All-options:** same Decider/validation/backend but threshold raised to 100000 so no question is split. Exactly one backend call per case; actual backend questions were verified equal to the original wire questions. This is not a validator-free raw scorer.
- Four isolated processes ran serially. The model host returned to approximately 1232 MiB desktop-idle GPU memory with no evaluator remaining. No model loaded on the Mac; no training or checkpoint overwrite occurred.

## Results

Semantic joint requires the appropriate operation and target (when relevant); restraint cases require the appropriate non-action operation. The legacy handoff rule is reported separately because it has different operation semantics.

| Checkpoint / protocol | Semantic joint | Legacy handoff joint | Pick semantic joint | Restraint semantic joint | Warm p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| v32b / deployed | 62/95 | 58/95 | 62/79 | 0/16 | 41.093 / 119.201 |
| v32b / all-options | 63/95 | 59/95 | 63/79 | 0/16 | 39.510 / 67.739 |
| upstream multilingual / deployed | 3/95 | 2/95 | 1/79 | 2/16 | 62.032 / 114.200 |
| upstream multilingual / all-options | 3/95 | 2/95 | 1/79 | 2/16 | 39.687 / 65.492 |

Timing is synchronized decision-call wall time, with the first three cases excluded (92 timed cases); it is not full browser task latency. These are single runs without a variance estimate.

For v32b, operation semantic correctness is 71/95 in both arms. Target correctness is 78/95 deployed versus 79/95 all-options. Nonwide correctness is identical. Three wide cases change semantic correctness: `inbox/inbox_filter`, `inbox/inbox_sent_folder`, and `inbox/inbox_pop_out`; one favors deployed and two favor all-options. Thus the net one-case difference is not evidence that chunking always hurts.

The upstream checkpoint's poor fit to this exact wire/state distribution is a measured result, not proof of generally poor multilingual capability or a diagnosed adapter defect. Both checkpoints pass the typed output contract; protocol compatibility does not imply useful task decisions.

## Invalid historical score and repaired instrumentation

The earlier `20261001T051431Z` runner scored a backend's **last raw call** rather than the final merged `decision.answers.raw`. A second-round target question omitted operation, so 47 cases lost operation in the recorded score. That run's v32b 33/95 result and partial operation calibration are invalid evidence of complete model performance. Do not patch historical rows by inventing missing first-round answers.

The corrected runner preserves the validated merged answers, each backend call's actual questions/raw envelope/error, original wire, transformation flag, and threshold. Failure rows cannot receive success from the last raw envelope. New regressions cover a final call without operation and a merged target different from the last-call winner.

## Remaining boundaries

- v32b still fails all 16 restraint cases in the deployed render. Earlier training-render probes suggest a render-parity investigation is worthwhile; this run does not establish that more training data or a larger encoder is the remedy.
- Operation semantic ECE covers all 95 cases: v32b deployed 0.232045, all-options 0.234227; upstream deployed 0.567581, all-options 0.566882. This is selected-operation-probability calibration on this regression distribution, not universal model calibration.
- Coarse-to-fine target probabilities are framework-reconstructed distributions. They must not be described as the original model's all-options posterior. The reconstruction and final confidence/choice consistency deserve independent production-framework review.
- Joint ECE remains `null`: multiplying marginal operation/target probabilities would invent independence. No calibrated joint probability was returned.
- Do not publish a replacement checkpoint or call the entire application release-ready from these results alone. Render-parity real-runtime acceptance, safety regression, held-out quality checks and relevant live-browser gates remain separate.
