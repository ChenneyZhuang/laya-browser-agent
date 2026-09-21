"""The decision layer: typed questions in, calibrated answers out, code keeps control.

The design rule this module enforces is the one that makes a decision model safe to
put in front of a browser: **the model may only choose from what it was shown.**
Every answer is validated against the question that was asked - the chosen key must
be one of the offered keys, the probabilities must be finite and sum to one, and the
chosen key must actually be the argmax. A surprising answer becomes a `DecisionError`
and the caller takes its fail-open path instead of acting on junk.

Three primitives, same as TypeSafe Jev and Laya:

* `choice` - one of a closed set of named options
* `score`  - a position on an ordered rubric
* `noul`   - P(true) for a proposition (yes/no)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

from .backends.base import Backend, BackendError, resolve_backend

State = Union[str, Mapping[str, Any], Sequence[Any]]


class DecisionError(RuntimeError):
    """The answer cannot be trusted or used."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


# ── questions ────────────────────────────────────────────────────────────────

def choice(instructions: Any, criteria: Mapping[str, Any]) -> Dict[str, Any]:
    """Pick one of `criteria`'s keys. At least two options are required."""
    if len(criteria) < 2:
        raise ValueError("a choice needs at least two options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: Any, levels: Sequence[str]) -> Dict[str, Any]:
    """Place the state on an ordered rubric. Levels are ordered worst to best."""
    if len(levels) < 2:
        raise ValueError("a score needs at least two levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def noul(instructions: Any) -> Dict[str, Any]:
    """Probability that the instruction proposition is true."""
    return {"type": "noul", "instructions": instructions}


Question = Dict[str, Any]

# ── answers ──────────────────────────────────────────────────────────────────

@dataclass
class Answers:
    """Validated answers plus the bookkeeping a caller needs to branch safely."""

    raw: Dict[str, Any]
    latency_ms: int
    backend: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)

    def choice(self, name: str) -> str:
        return self.raw[name]["choice"]

    def probabilities(self, name: str) -> Dict[str, float]:
        return self.raw[name]["probabilities"]

    def confidence(self, name: str) -> float:
        return float(self.raw[name].get("confidence", 0.0))

    def score(self, name: str) -> float:
        return float(self.raw[name]["score"])

    def noul(self, name: str) -> float:
        return float(self.raw[name]["noul"])

    def __getitem__(self, name: str) -> Any:
        return self.raw[name]


def _finite_unit(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError("malformed", f"{where} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DecisionError("malformed", f"{where} is outside [0,1]: {value!r}")
    return number


def _check_answer(name: str, question: Mapping[str, Any], answer: Any) -> Dict[str, Any]:
    """Reject anything the caller should not act on. Same contract as the wire format."""
    if not isinstance(answer, dict):
        raise DecisionError("malformed", f"{name}: answer is not an object")
    kind = question.get("type")
    if kind == "choice":
        criteria = question.get("criteria") or {}
        expected = set(criteria) if isinstance(criteria, Mapping) else set(range(len(criteria)))
        chosen = answer.get("choice")
        if chosen not in expected:
            raise DecisionError("malformed", f"{name}: choice {chosen!r} is not an offered option")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != {str(k) for k in expected}:
            raise DecisionError("malformed", f"{name}: probabilities do not cover the options")
        numbers = {str(k): _finite_unit(v, f"{name}.{k}") for k, v in probabilities.items()}
        if abs(sum(numbers.values()) - 1.0) > 0.02:
            raise DecisionError("malformed", f"{name}: probabilities sum to {sum(numbers.values()):.3f}")
        if numbers[str(chosen)] < max(numbers.values()) - 1e-6:
            raise DecisionError("malformed", f"{name}: {chosen!r} is not the most probable option")
        cleaned = dict(answer)
        cleaned["probabilities"] = numbers
        cleaned["confidence"] = _finite_unit(answer.get("confidence", numbers[str(chosen)]), f"{name}.confidence")
        return cleaned
    if kind == "score":
        _finite_unit(0.0, name)  # touch: keeps the numeric contract uniform
        value = answer.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DecisionError("malformed", f"{name}: score is not finite")
        return dict(answer)
    if kind == "noul":
        cleaned = dict(answer)
        cleaned["noul"] = _finite_unit(answer.get("noul"), f"{name}.noul")
        return cleaned
    raise DecisionError("malformed", f"{name}: unknown question type {kind!r}")


# ── the decider ──────────────────────────────────────────────────────────────

@dataclass
class Decision:
    """One decide() call: the answers, what it cost, and what happened if it failed."""

    answers: Optional[Answers] = None
    error: Optional[str] = None
    failed_open: bool = False
    latency_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.answers is not None


class Decider:
    """Ask a local decision model typed questions about a state.

    The interesting knob is `fail_open`: a decision layer that can block a browser
    is a liability, so a backend error, a timeout, or a malformed answer returns
    `Decision(ok=False)` rather than raising into your loop. Inspect `.ok` and take
    your own fallback path.

    Example:
        d = Decider()
        result = d.decide(state, {"next": choice("What now?", {"a": "...", "b": "..."})})
        if result.ok:
            do(result.answers.choice("next"))
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        timeout: float = 30.0,
        retries: int = 1,
        fail_open: bool = True,
        max_options_per_question: int = 20,
        **backend_kwargs: Any,
    ) -> None:
        self.backend: Backend = resolve_backend(backend, **backend_kwargs)
        self.timeout = timeout
        self.retries = retries
        self.fail_open = fail_open
        # Laya's head budget degrades past ~20 flat options; wider choice questions go
        # coarse-to-fine (split, then a final round over the chunk winners) instead.
        self.max_options_per_question = max_options_per_question

    # -- public ---------------------------------------------------------------

    def decide(self, state: State, questions: Mapping[str, Question]) -> Decision:
        """Answer every question in one shot. Never raises unless fail_open=False."""
        if not questions:
            raise ValueError("no questions")
        started = time.monotonic()
        try:
            payload = self._answer_with_retries(state, questions)
        except BackendError as error:
            return self._fail(error.code, str(error), started)
        except DecisionError as error:
            return self._fail(error.code, str(error), started)
        except Exception as error:  # last-resort fail-open
            return self._fail("unexpected", f"{type(error).__name__}: {error}", started)
        return Decision(
            answers=Answers(
                raw=payload["answers"],
                latency_ms=payload.get("latency_ms", int((time.monotonic() - started) * 1000)),
                backend=payload.get("backend", getattr(self.backend, "name", "")),
                usage=payload.get("usage", {}),
            ),
            latency_ms=payload.get("latency_ms", int((time.monotonic() - started) * 1000)),
        )

    # -- internals ------------------------------------------------------------

    def _answer_with_retries(self, state: State, questions: Mapping[str, Question]) -> Dict[str, Any]:
        last: Optional[BackendError] = None
        prepared, plan = self._apply_coarse_to_fine(questions)
        for attempt in range(self.retries + 1):
            try:
                payload = self.backend.answer(state, prepared)
            except BackendError as error:
                last = error
                if attempt < self.retries:
                    time.sleep(min(0.25 * (attempt + 1), 2.0))
                    continue
                raise
            try:
                validated = {name: _check_answer(name, question, payload["answers"].get(name))
                             for name, question in prepared.items()}
            except DecisionError:
                if attempt < self.retries:
                    time.sleep(0.2)
                    continue
                raise
            if plan:
                validated = self._finish_coarse_to_fine(state, validated, plan)
            return {
                "answers": validated,
                "usage": payload.get("usage", {}),
                "backend": payload.get("backend", getattr(self.backend, "name", "")),
                "latency_ms": payload.get("latency_ms", 0),
            }
        raise last or BackendError("backend_failed", "no attempt produced a result")

    def _apply_coarse_to_fine(self, questions: Mapping[str, Question]):
        """Split wide choice questions so every option keeps a readable budget.

        A choice with more than `max_options_per_question` options is turned into
        interleaved chunks; each chunk is asked in the same pass, then the chunk
        winners compete in one extra pass. The combined probability keeps the
        decomposition exact: p(o) = p_final(chunk_of_o) * p_chunk(o).
        """
        prepared: Dict[str, Dict[str, Any]] = {}
        plan: Dict[str, Any] = {}
        limit = max(2, int(self.max_options_per_question))
        for name, question in questions.items():
            if question.get("type") != "choice":
                prepared[name] = dict(question)
                continue
            criteria = question.get("criteria") or {}
            keys: List[Any] = list(criteria) if isinstance(criteria, Mapping) else list(range(len(criteria)))
            if len(keys) <= limit:
                prepared[name] = dict(question)
                continue
            chunks = [keys[i::(-(-len(keys) // limit))] for i in range(-(-len(keys) // limit))]
            plan[name] = (dict(question), keys, chunks)
            for index, chunk in enumerate(chunks):
                sub = dict(question)
                sub["criteria"] = ({k: criteria[k] for k in chunk}
                                   if isinstance(criteria, Mapping) else list(criteria[i] for i in chunk))  # type: ignore[index]
                prepared[f"{name}__chunk{index}"] = sub
        return prepared, plan

    def _finish_coarse_to_fine(self, state: State, answers: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Second pass: the chunk winners compete, then probabilities are recombined.

        p(option) = p_final(its chunk winner) * p_chunk(option), renormalised. This keeps
        a 60-element click_target question inside the model's per-option token budget
        instead of letting 60 labels share ~3 tokens each and blur together.
        """
        finals: Dict[str, Dict[str, Any]] = {}
        chunk_answers: Dict[str, List[Dict[str, Any]]] = {}
        for name, (question, keys, chunks) in plan.items():
            winners = [answers.pop(f"{name}__chunk{index}") for index in range(len(chunks))]
            chunk_answers[name] = winners
            criteria = {(w["choice"] if not isinstance(question["criteria"], Mapping)
                         else w["choice"]): (question["criteria"].get(w["choice"], "")
                                             if isinstance(question["criteria"], Mapping) else "") for w in winners}
            finals[name] = {**question, "criteria": criteria}
        second = self.backend.answer(state, finals)
        second_answers = second.get("answers", {})
        for name, (question, keys, chunks) in plan.items():
            final = _check_answer(name, finals[name], second_answers.get(name))
            probabilities: Dict[Any, float] = {}
            for winner, chunk in zip(chunk_answers[name], chunks):
                weight = final["probabilities"].get(winner["choice"], 0.0)
                for key in chunk:
                    probabilities[key] = weight * winner["probabilities"].get(str(key), 0.0)
            total = sum(probabilities.values()) or 1.0
            probabilities = {k: round(v / total, 6) for k, v in probabilities.items()}
            chosen = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
            answers[name] = {
                "type": "choice",
                "choice": chosen,
                "probabilities": probabilities,
                "confidence": final.get("confidence", probabilities[chosen]),
                "coarse_to_fine": {"chunks": len(chunks), "winners": [w["choice"] for w in chunk_answers[name]]},
            }
        return answers

    def _fail(self, code: str, detail: str, started: float) -> Decision:
        latency = int((time.monotonic() - started) * 1000)
        if not self.fail_open:
            raise DecisionError(code, detail)
        return Decision(error=f"{code}: {detail}" if detail else code, failed_open=True, latency_ms=latency)


def decide_all(decider: Decider, items: Iterable[tuple], questions: Mapping[str, Question]) -> List[Decision]:
    """Convenience: run the same questions over many states (one call per state)."""
    return [decider.decide(state, questions) for state, *_ in (items if isinstance(items, list) else list(items))]
