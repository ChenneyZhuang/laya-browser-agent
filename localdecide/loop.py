"""The loop: observe, decide once, act, repeat - with the guardrails that keep it honest.

Two rules make a small decision model safe to run unattended against a real browser:

1. **The model picks an index, your code does everything else.** The decision names
   `operation` and a target index; the executor resolves that index to its own node
   handle. Model output never becomes a selector, a coordinate, or executable code.
2. **The loop, not the model, tracks history.** Repeated actions, unchanged pages and
   step budgets are detected in code, so a model that loops is stopped by the harness
   rather than trusted to notice.

The loop is driver-agnostic on purpose: pass any object with
`observe() -> observation` and `execute(operation, element, text) -> result`.
`localdecide.drivers` ships a Playwright driver; a CDP driver is about 40 lines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .decider import Decision, Decider, choice
from .page import ElementTable, build_element_table, table_to_questions
from .scope import Scope


class Driver(Protocol):
    """What the loop needs from a browser. Implement these three things and you are done."""

    def observe(self) -> Dict[str, Any]:
        """Return the current observation (url, title, text, actions/elements)."""
        ...

    def execute(self, operation: str, element: Optional["ElementRef"], text: Optional[str] = None) -> Dict[str, Any]:
        """Perform the operation. Return {"ok": bool, "detail": str, "page_changed": bool}."""
        ...

    def close(self) -> None:
        ...


@dataclass
class ElementRef:
    """The loop hands your executor the element it must act on."""

    index: str
    label: str
    role: str = ""
    handle: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """One cycle, kept for the history the next decision reads."""

    n: int
    operation: str
    target: Optional[str]
    label: str
    confidence: float
    latency_ms: int
    executed: bool
    detail: str = ""
    page_changed: Optional[bool] = None
    failed_open: bool = False


@dataclass
class Run:
    """The whole attempt: what it did, and why it stopped."""

    goal: str
    steps: List[Step] = field(default_factory=list)
    stopped: str = ""
    error: str = ""

    @property
    def solved(self) -> bool:
        return self.stopped == "done"

    def summary(self) -> Dict[str, Any]:
        decisions = [s for s in self.steps if s.executed]
        latencies = sorted(s.latency_ms for s in self.steps) or [0]
        return {
            "goal": self.goal,
            "stopped": self.stopped,
            "solved": self.solved,
            "steps": len(self.steps),
            "executed": len(decisions),
            "median_decision_ms": latencies[len(latencies) // 2],
            "total_decision_ms": sum(s.latency_ms for s in self.steps),
            "failed_open": sum(1 for s in self.steps if s.failed_open),
            "trace": [
                {"n": s.n, "op": s.operation, "target": s.target, "label": s.label[:60],
                 "conf": round(s.confidence, 3), "ms": s.latency_ms, "detail": s.detail[:80]}
                for s in self.steps
            ],
        }


# Operations the loop will refuse to guess about. If the model asks for one of these
# but names something the harness did not offer, the step is treated as malformed.
_HARNESS_OPERATIONS = {"CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"}

# Safety default: operations that change the world outside the page. A confirmation
# callback gets the last word on these, whatever the model decided.
RISKY_HINTS = ("delete", "remove account", "pay", "purchase", "buy", "checkout", "send",
               "submit order", "confirm transfer", "unsubscribe", "cancel subscription")


class BrowserDecider:
    """Run a goal to completion with a local decision model choosing every step.

    Example:
        from localdecide import BrowserDecider
        from localdecide.drivers import PlaywrightDriver

        with PlaywrightDriver() as driver:
            run = BrowserDecider().run(driver, "Find flights Zurich to London on 2026-09-20")
            print(run.summary())
    """

    def __init__(
        self,
        decider: Optional[Decider] = None,
        *,
        max_steps: int = 30,
        text_provider: Optional[Callable[[str, "ElementRef"], Optional[str]]] = None,
        confirm: Optional[Callable[[str, "ElementRef"], bool]] = None,
        on_step: Optional[Callable[[Step], None]] = None,
        text_chars: int = 1200,
        scope: Optional[Scope] = None,
    ) -> None:
        self.decider = decider or Decider()
        self.max_steps = max_steps
        # TYPE_TEXT needs a string; a decision model cannot write one. Supply a callback
        # (a small LLM, a regex over the goal, a lookup table) or TYPE_TEXT is refused.
        self.text_provider = text_provider
        self.confirm = confirm
        self.on_step = on_step
        self.text_chars = text_chars
        # Scoping is the single biggest lever on both latency and accuracy: 20 elements
        # decide in ~330 ms, 120 elements take ~1.2 s and make more mistakes.
        self.scope = scope if scope is not None else Scope()

    def run(self, driver: Driver, goal: str) -> Run:
        run = Run(goal=goal)
        if not goal.strip():
            run.stopped, run.error = "error", "empty goal"
            return run
        history: List[Dict[str, Any]] = []
        try:
            for number in range(1, self.max_steps + 1):
                observation = driver.observe()
                if self.scope is not None:
                    # The goal goes in: it is what protects a legitimate target ("Random
                    # article" is a nav link AND the thing the user asked for) from being
                    # mistaken for page furniture.
                    observation = self.scope.apply(observation, goal=goal)
                table = build_element_table(observation)
                table.history = list(history)
                questions = table_to_questions(table, goal)
                decision = self.decider.decide(table.state(text_chars=self.text_chars), questions)

                if not decision.ok:
                    step = Step(number, "WAIT", None, "", 0.0, decision.latency_ms, False,
                                detail=f"failed open: {decision.error}", failed_open=True)
                    run.steps.append(step)
                    self._emit(step)
                    # A failed-open decision is not an action. Retry once, then give up.
                    if sum(1 for s in run.steps if s.failed_open) >= 2:
                        run.stopped, run.error = "error", decision.error or "decision failed twice"
                        return run
                    continue

                answers = decision.answers
                assert answers is not None
                operation = answers.choice("operation")
                confidence = answers.confidence("operation")
                element: Optional[ElementRef] = None
                target: Optional[str] = None

                if operation in ("CLICK", "TYPE_TEXT", "SELECT"):
                    question_name = f"{operation.lower()}_target"
                    if question_name in answers.raw:
                        target = answers.choice(question_name)
                        found = table.by_index().get(target)
                        if found is None:
                            step = Step(number, operation, target, "", confidence, decision.latency_ms, False,
                                        detail="model named an index that was not offered")
                            run.steps.append(step)
                            self._emit(step)
                            run.stopped, run.error = "error", "hallucinated target"
                            return run
                        element = ElementRef(found.index, found.label, found.role, found.handle, found.meta)
                        confidence = min(confidence, answers.confidence(question_name))
                    else:
                        step = Step(number, operation, None, "", confidence, decision.latency_ms, False,
                                    detail="operation needs a target this page never offered")
                        run.steps.append(step)
                        self._emit(step)
                        run.stopped, run.error = "error", "no target question"
                        return run

                if operation == "DONE":
                    run.steps.append(Step(number, "DONE", None, "", confidence, decision.latency_ms, False))
                    self._emit(run.steps[-1])
                    run.stopped = "done"
                    return run
                if operation == "BLOCKED":
                    run.steps.append(Step(number, "BLOCKED", None, "", confidence, decision.latency_ms, False))
                    self._emit(run.steps[-1])
                    run.stopped = "blocked"
                    return run
                if operation not in _HARNESS_OPERATIONS:
                    run.steps.append(Step(number, operation, target, "", confidence, decision.latency_ms, False,
                                          detail="unsupported operation"))
                    self._emit(run.steps[-1])
                    run.stopped, run.error = "error", f"unsupported operation {operation!r}"
                    return run

                # Human gate: irreversible-looking actions stop here unless the caller
                # has supplied a confirmation callback that says yes.
                if operation in ("CLICK", "TYPE_TEXT", "SELECT") and self._looks_risky(element, goal):
                    if self.confirm is None or not self.confirm(f"{operation} {element.label if element else ''}", element):  # type: ignore[arg-type]
                        run.steps.append(Step(number, operation, target, element.label if element else "",
                                              confidence, decision.latency_ms, False, detail="needs confirmation"))
                        self._emit(run.steps[-1])
                        run.stopped = "needs_confirmation"
                        return run

                text: Optional[str] = None
                if operation == "TYPE_TEXT":
                    if self.text_provider is None:
                        run.steps.append(Step(number, operation, target, element.label if element else "",
                                              confidence, decision.latency_ms, False, detail="no text provider"))
                        self._emit(run.steps[-1])
                        run.stopped, run.error = "error", "TYPE_TEXT without a text provider"
                        return run
                    text = self.text_provider(goal, element)  # type: ignore[arg-type]
                    if not text:
                        run.steps.append(Step(number, operation, target, element.label if element else "",
                                              confidence, decision.latency_ms, False, detail="text provider returned nothing"))
                        self._emit(run.steps[-1])
                        run.stopped, run.error = "error", "no text for TYPE_TEXT"
                        return run

                # Loop guard: same operation on the same target twice with no page change.
                repeats = sum(1 for item in history[-2:]
                              if item.get("action") == operation and item.get("target") == target
                              and item.get("page_changed") is False)
                if repeats >= 2:
                    run.steps.append(Step(number, operation, target, element.label if element else "",
                                          confidence, decision.latency_ms, False, detail="loop guard: no page change"))
                    self._emit(run.steps[-1])
                    run.stopped, run.error = "error", "stuck: repeated action with no page change"
                    return run

                try:
                    result = driver.execute(operation, element, text) or {}
                except Exception as error:  # a driver failure is not the model's fault
                    step = Step(number, operation, target, element.label if element else "",
                                confidence, decision.latency_ms, False, detail=f"driver error: {error}")
                    run.steps.append(step)
                    self._emit(step)
                    run.stopped, run.error = "error", f"driver error: {error}"
                    return run

                step = Step(number, operation, target, element.label if element else "", confidence,
                            decision.latency_ms, bool(result.get("ok", True)),
                            detail=str(result.get("detail", "")),
                            page_changed=result.get("page_changed"))
                run.steps.append(step)
                self._emit(step)
                history.append({"action": operation, "kind": operation.lower(), "target": target,
                                "text": text, "page_changed": result.get("page_changed")})
            run.stopped = "max_steps"
            return run
        finally:
            try:
                driver.close()
            except Exception:
                pass

    def _looks_risky(self, element: Optional[ElementRef], goal: str) -> bool:
        haystack = f"{element.label if element else ''} {goal}".lower()
        return any(hint in haystack for hint in RISKY_HINTS)

    def _emit(self, step: Step) -> None:
        if self.on_step is not None:
            try:
                self.on_step(step)
            except Exception:
                pass
