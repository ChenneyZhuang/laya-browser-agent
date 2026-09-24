# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/); versioning follows [SemVer](https://semver.org/).

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
- Literature section in all four READMEs: arXiv 2609.23959 (Open-Jev on
  CallScreenBench — AUROC .974, 64.5 ms/decision) and arXiv 2402.09769
  (single-forward-pass lineage).
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
- Every README number traced to a results file; all external links verified;
  star counts and badges current (credibility pass).

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
