# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/); versioning follows [SemVer](https://semver.org/).

## [Unreleased]

### Fixed
- Password values are redacted at DOM observation, normalization, table serialization,
  and description boundaries; password entry is rejected with an explicit vault-not-provided
  message. Browser execution now resolves stable observation handles and rechecks identity,
  actionability, overlays, disabled/read-only state, and current geometry in both drivers.
- CDP target URL misses fail closed, `Runtime.evaluate` exception details propagate, and
  Playwright constructor failures clean up the runtime and restore the caller's event loop;
  close is idempotent.
- The `browser-legacy` alias pins the historical `cklxx/laya-browser` `v10s` revision that
  upstream removed from `main`; callers can still override it with `revision=`.
- Decision deadlines now cover retries, coarse-to-fine passes, and HTTP transport timeouts;
  late synchronous duck-backend results are rejected without pretending to cancel them.
- Score/noul response schemas, SELECT option confidence, MCP invalid requests/notifications,
  HTTP 413 keep-alive closure, generated question-name collisions, and wheel/sdist contents
  now have model-free regression coverage.
- The live batch launcher honors explicit `PYTHON`, then a repository venv, then an existing
  interpreter, while preserving pipefail and failure propagation.
- `HEAD` responses no longer write a JSON body onto a persistent HTTP connection; a regression test sends `HEAD` and then `GET` on the same socket.
- The Jev wire-compatibility test now starts an ephemeral local server with a deterministic fake backend. It no longer fails with connection-refused when port 8791 is not manually running, and it is suitable for CI.
- `run_live_batched.sh` now fails fast when a pytest batch fails. It previously used `set -u` only, so a failed pytest piped through `tail` could be reported as a successful live run; a regression test now stubs a failing batch and checks the script exits nonzero immediately.
- `localdecide doctor` now prints install steps that work before the package is published to PyPI; it no longer recommends the nonexistent `localdecide[extra]` distribution.
- README install commands in all four languages now match the current Git-clone/editable-install workflow; PyPI availability was checked and the package is not published yet.
- The body-size test now sends an oversized `Content-Length` and asserts the exact 413 response instead of sending a small body and accepting either 413 or 422.
- Removed an identical duplicate `open_kickoff` goal entry in the v37 generator; bound loop data explicitly in its probes.

### Changed
- CI now runs the model-free Jev-wire and CLI install-guidance tests, and pins Ruff to the version already observed on GitHub Actions (`0.16.9`).
- Build metadata uses the SPDX `Apache-2.0` expression (PEP 639) and requires setuptools 77+, removing the license deprecation warnings observed during wheel/sdist builds.

## [0.3.2] - 2026-09-26

### Added
- **Empty-submit guard.** Found by A/B-testing the new default checkpoint: asked to
  "Search products for 'kettle'", v32b answers `CLICK` on the Search button at
  **p=0.74 with the field still empty** — the confidence gate cannot catch a
  confident wrong proposal, and the older v10s default only escaped because its own
  confidence on the same state was 0.06 (safety by accident, not structure). The
  guard refuses a click on a submit-like control (search / submit / send / sign-in /
  checkout / …) while the still-empty field it plainly pairs with; the model is
  asked again, and two refusals end the run with a clear error instead of burning
  the step budget. Filling the field first makes the same click go through.
- 5 contract tests for the guard: refusal, two-strike stop, filled-field
  pass-through, submit-word-without-empty-field stands down, non-submit words
  never blocked.
- Historical live-suite observations remain labelled as checkpoint-specific; current
  acceptance work is model-free unless a live run is explicitly authorized and available.

### Changed
- READMEs (en/zh/ja/es): the empty-submit failure and its guard documented in the
  "things the model gets wrong" table (en) and the key-design guard lists (zh/ja/es).

## [0.3.1] - 2026-09-25

### Changed
- **`model="browser"` now loads the fine-tuned
  [`ichenney/laya-browser-v32b`](https://huggingface.co/ichenney/laya-browser-v32b)
  checkpoint by default** — verified end-to-end on Apple Silicon (load +
  real decision through the MLX path). The upstream checkpoint stays one
  flag away as `model="browser-legacy"`.
- README overhaul in all four languages (en/zh/ja/es): install moved up,
  honest three-way comparison tables (v32b / official / hosted Jev),
  download-size notes updated for the 1.3 GB v32b default, historical
  head-to-head sections labeled as v10s-era provenance.
- Reference-sources table: `cklxx/laya-browser` is documented as the
  fine-tuning base of the default checkpoint, not the default itself.

## [0.3.0] - 2026-09-25

### Added
- Fine-tuned checkpoint release: **v32b-b15** — frozen-encoder head
  fine-tune of `cklxx/laya-browser` published at
  [`ichenney/laya-browser-v32b`](https://huggingface.co/ichenney/laya-browser-v32b).
  Historical final comparison: nine metrics win 5 and lose 4 against the official
  browser-tuned reference; the eight-metric correctness view wins 5 and loses 3.
- `reports/v20/MULTIDIM_COMPARISON.md` — the full v17→v32 recipe,
  per-family breakdowns, and the noul root-cause analysis.
- `reports/v20/JEV_COMPARISON.md` — fresh 231-item head-to-head against
  the hosted Jev API with the user's key (mean 0.8615 vs local 0.5325;
  local wins `score` and `temporal_numeric` families, 31x faster).
- Companion repo [laya-training-log](https://github.com/ChenneyZhuang/laya-training-log) —
  every version, every failed path, all training scripts.

## [0.2.3] - 2026-09-24

### Fixed
- **CORS preflight was broken in 0.2.0–0.2.2**: `do_OPTIONS` was defined twice
  in `serve.py`; the second definition silently overrode the CORS-correct
  first one, so `OPTIONS` responses carried no `Access-Control-*` headers and
  browser-based clients (extensions, local web consoles) failed at preflight.
  Now a single handler routing through `_send`. Found by the new HTTP test
  suite — hand-testing had only checked the 204 status, not the headers.
- Version drift: three different versions lived in the tree (pyproject,
  `mcp_server.SERVER_INFO`, `__init__.__version__`). Both constants now read
  the installed package metadata; one source of truth.
- `goal_tokens` was public API by usage (tests, diagnostics) but missing from
  `__all__`; now exported.
- 32 dead imports removed repo-wide; loop-variable closures in two
  diagnostics bound explicitly (ruff `B023`).

### Added
- `tests/test_serve.py` — 14 tests over the HTTP surface: CORS on every
  response, preflight headers, routing, body limits, error codes. Runs with
  an injected fake backend, no checkpoint needed.
- `tests/test_mcp.py` — 13 tests over the MCP surface: spec version
  negotiation (current, legacy, unknown), tools list/call, error paths.
- `ruff` wired into `pyproject.toml` (correctness rules only: `F`, `E9`,
  `B023`, `RUF100`) and into CI.
- Coverage: `serve.py` 0→83%, `mcp_server.py` 0→81%, project total 50→65%.

## [0.2.2] - 2026-09-24

### Fixed
- `scripts/run_live_batched.sh` no longer hardcodes the author's machine path
  (`/Volumes/SSD/localdecide`); the repo root is resolved from the script
  location, so the `apple-silicon-live` CI job passes on GitHub runners.
- All 17 diagnostic scripts in `examples/diagnostics/` resolve repo paths
  relative to `__file__` instead of hardcoded absolute paths — they now run
  from any clone, on any machine.
- `.gitignore` covers `.env` (previously only `env/`; the file was never
  tracked, but the rule is now explicit).

### Added
- `CHANGELOG.md` (this file).

## [0.2.1] - 2026-09-24

### Added
- Literature section in all four READMEs: the related typed-decision screening result is
  retained as context, without presenting it as a causal explanation of this project's SMS results.
- Upstream calibration baselines in the performance table (p50 38 ms,
  ECE 0.030 across 13 task families).

## [0.2.0] - 2026-09-23

### Added
- Head-to-head batteries vs hosted Jev: single-step (12 cases), multi-step
  flows, text classification (21 real business texts), browser edge cases
  (9 restraint cases) — reproducible from `examples/diagnostics/`.
- MCP: protocol-version negotiation per spec 2025-06-18 (+ legacy
  2024-11-05); `GET /v1/models`; CORS + `OPTIONS` + `HEAD` on the HTTP
  service; fill-target actionability check in both drivers.
- SPA-aware `page_changed` DOM-signature detection in both drivers.

### Fixed
- README benchmark claims are explicitly labelled as selected committed diagnoses;
  fixed star-count claims were removed from the credibility pass.

## [0.1.1] - 2026-09-22

### Added
- Verified against the real TypeSafe production endpoint
  (`api.typesafe.ai/v1/systemone`, `jev-1.13.0`); payload contract documented
  (choice → criteria map, score → criteria array).

## [0.1.0] - 2026-09-21

### Added
- Initial public release: local System-1 decision engine (Laya/laya-mlx
  checkpoint), browser harness (loop, drivers, guards), HTTP service, MCP
  server, skills installer, multilingual docs (en/zh/ja/es).
