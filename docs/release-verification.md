# Release verification boundary

This checkout is verified from source. Do not treat it as a published release
until the release owner separately verifies the package index and the live
runtime/browser boundaries.

The portable model-free setup and acceptance boundary is:

```bash
python3 -m venv .release-verify-venv
. .release-verify-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest ruff

LOCALDECIDE_SKIP_LIVE=1 python -m pytest tests -q -p no:cacheprovider
ruff check .
git diff --check
```

On Windows PowerShell, create the venv with `py -3 -m venv
.release-verify-venv`, use `.\.release-verify-venv\Scripts\Activate.ps1`,
and run the same `python -m ...` commands. This installs no model runtime and
does not download weights.

The wheel is runtime-only: it must contain `localdecide/` and the installed
helper script, but no `tests` package. The sdist is the source archive: it must
contain the full tests, fixtures, subprocess helper, live batch script, and
release docs. The package CI job builds both archives, checks these boundaries,
installs each in an isolated environment, and verifies the installed result.

The live browser, stdio/wire, HTTP, wheel/sdist, and model-runtime checks are
separate boundaries. A sandbox failure to bind a socket or start Chromium is a
blocker, not a passing result. The parent release run must rerun those checks
with real browser and transport access. No model weights or inference are part
of the model-free acceptance command. The model-free suite proves harness and
lifecycle behavior only; it is not a model-quality or end-to-end browser result.

Historical reports may contain benchmark numbers, but they are not current
acceptance results. In particular, MiniWoB/fixture accuracy must not be
described as task success, and no new model result should be inferred from
contract or synthetic-browser tests.
