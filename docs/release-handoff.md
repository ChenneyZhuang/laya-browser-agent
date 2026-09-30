# Release handoff

## Verified in this checkout

- The model-free test path is portable and does not install a model extra or download
  weights. The live module's skip gate is exercised during import.
- The wheel contains runtime packages and the helper script, but no `tests` namespace.
  The sdist contains the tests, fixtures, subprocess helper, helper script, and release
  docs. CI builds, inspects, installs, and checks both archive types in isolated venvs.
- Live subprocess cleanup is behavior-based: hung `observe` and piped `run` children are
  time-bounded, killed, and reaped. Browser/transport tests remain a separate boundary.
- Playwright and CDP SELECT reject a change handler that does not leave the requested value.
  CDP CLICK rechecks identity and geometry after pressing; on failure it releases at
  `(-1, -1)`. This is fail-closed cleanup, not a transactional guarantee against every
  browser race.
- Published Laya source and wheel metadata were inspected without importing runtimes,
  installing model packages for this review, downloading weights, or running inference.
  The independently checked v32b Hub metadata is pinned by explicit revision
  `161d54d6000913ff279b0afd1ac77faef8685a9b`. Published `laya` 0.3.20 lacks the
  `revision` parameter, 0.3.21 forwards it, and published `laya-mlx` 0.1.0 supports it;
  the configured floors remain `laya>=0.3.21` and `laya-mlx>=0.1.0`.
- Live inference is now an explicit `LOCALDECIDE_RUN_LIVE=1` opt-in, with
  `LOCALDECIDE_SKIP_LIVE=1` taking precedence. The batch launcher sets the opt-in and
  refuses installed standalone layouts that do not contain the source checkout's tests.
- The event-loop compatibility path suppresses only Python 3.14's deprecation warning for
  the legacy policy getter while preserving an existing idle loop for cleanup. CDP's
  press/release recheck remains fail-closed cleanup, not a transactional race guarantee;
  synchronous duck backends can still finish after a deadline and cannot be force-cancelled.

## Remaining release-owner checks

- Rerun the browser-synthetic, HTTP/stdio/wire, and live checkpoint checks outside the
  restricted sandbox. A browser or socket startup failure is still a blocker, not a pass.
- Verify the exact v32b revision/runtime/checkpoint combination on the release target; the
  recorded revision is source/metadata evidence, not proof of checkpoint loading or inference.
- Verify the published package index, archive upload, and the exact runtime/checkpoint
  combination on the release target. The model-free suite does not establish model load,
  quality, calibration, latency, memory fit, or end-to-end browser success.
- Keep real API keys and any private endpoint configuration out of source, examples, and
  release artifacts. `HTTPBackend` sends page state to whatever endpoint is configured;
  “local” is not inferred from an arbitrary URL.
