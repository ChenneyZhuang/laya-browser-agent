"""Subprocess entry point for live browser tests.

Why this exists: Playwright's sync API refuses to start inside a running asyncio loop, and
the MLX/PyTorch model runtime creates one. Putting the browser in its own process solves
that cleanly - and it is the right memory design too, because a checkpoint plus a Chromium
in one interpreter is more than a 16 GB laptop should be asked for at once.

Usage (from tests/test_live.py):

    python _browser_child.py observe '<json request>'
    python _browser_child.py run     '<json request>'

Always prints one JSON object as the last stdout line.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))


def _observe(request: Dict[str, Any]) -> Dict[str, Any]:
    from localdecide import Scope
    from localdecide.drivers import PlaywrightDriver

    driver = PlaywrightDriver(headless=True, start_url=request["url"])
    try:
        observation = driver.observe()
    finally:
        driver.close()
    if request.get("max_elements"):
        scope = Scope(
            max_elements=request["max_elements"],
            drop_chrome=request.get("drop_chrome", True),
            include_words=request.get("include_words") or [],
            exclude_words=request.get("exclude_words") or [],
        )
        observation = scope.apply(observation, goal=request.get("goal", ""))
    return observation


def _run(request: Dict[str, Any]) -> Dict[str, Any]:
    """Drive a goal through the loop, talking to the parent for every decision.

    The decisions come from the parent over stdin/stdout rather than loading a second
    copy of the model here - one checkpoint per machine is the whole point.
    """
    from localdecide import BrowserDecider, Decider, Scope
    from localdecide.drivers import PlaywrightDriver

    class ParentDecider:
        """Forwards `decide` calls to the parent process and parses the reply."""

        def __init__(self) -> None:
            self.max_options_per_question = 20
            self._n = 0

        def decide(self, state, questions):
            from localdecide.decider import Answers, Decision

            self._n += 1
            print(json.dumps({"want": "decide", "n": self._n, "state": state,
                              "questions": questions}, ensure_ascii=False), flush=True)
            line = sys.stdin.readline()
            if not line:
                raise RuntimeError("parent closed the decision channel")
            reply = json.loads(line)
            if not reply.get("ok"):
                return Decision(error=reply.get("error", "parent decision failed"), failed_open=True)
            return Decision(answers=Answers(raw=reply["answers"], latency_ms=reply.get("latency_ms", 0),
                                            backend=reply.get("backend", "parent")))

    driver = PlaywrightDriver(headless=True, start_url=request["url"])
    decider = ParentDecider() if request.get("remote_decide") else Decider()
    scope = Scope(max_elements=request.get("max_elements", 25))
    text_provider = None
    if request.get("text_for"):
        # {label_substring: text}
        mapping = request["text_for"]

        def text_provider(goal, element):  # noqa: F811
            for key, value in mapping.items():
                if key.lower() in (element.label or "").lower():
                    return value
            return None

    def confirm(label, element):
        return bool(request.get("confirm"))

    run = BrowserDecider(decider=decider, max_steps=request.get("max_steps", 5),
                         text_provider=text_provider, confirm=confirm, scope=scope).run(
        driver, request["goal"])
    return {"stopped": run.stopped, "error": run.error, "solved": run.solved,
            "steps": [{"n": s.n, "op": s.operation, "target": s.target, "label": s.label,
                       "confidence": s.confidence, "ms": s.latency_ms, "executed": s.executed,
                       "detail": s.detail, "page_changed": s.page_changed} for s in run.steps],
            "summary": run.summary()}


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "observe"
    request = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    try:
        result = _run(request) if action == "run" else _observe(request)
        result["ok"] = True
    except Exception as error:  # noqa: BLE001 - the parent needs the reason, not a traceback
        import traceback
        result = {"ok": False, "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()[-1500:]}
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
