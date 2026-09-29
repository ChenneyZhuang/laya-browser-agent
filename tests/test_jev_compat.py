"""Compatibility test: verify the server wire format with Jev's client validator.

`jev-ultrafast` (browser-use) validates every choice answer for complete, finite,
normalized probabilities and argmax agreement. This test sends a realistic
`/v1/systemone` request through the actual HTTP handler and applies that validator.
A deterministic backend keeps it model-free, so CI can run it without a checkpoint
or a manually-started process; model quality is covered by the live suite.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

from localdecide.decider import Decider
from localdecide.serve import Handler, _State


class _JevCompatibleBackend:
    name = "jev-wire-test"

    def answer(self, state, questions):
        answers = {}
        expected = {"operation": "CLICK", "click_target": "2"}
        for name, question in questions.items():
            ids = [str(key) for key in question["criteria"]]
            selected = expected.get(name, ids[0])
            if selected not in ids:
                raise AssertionError(f"test answer {selected!r} not offered for {name}")
            probabilities = ({key: 1.0 / len(ids) for key in ids} if len(ids) == 1 else
                             {key: (0.9 if key == selected else 0.1 / (len(ids) - 1)) for key in ids})
            answers[name] = {"type": "choice", "choice": selected,
                             "probabilities": probabilities, "confidence": probabilities[selected],
                             "action": {"act_probability": 1.0}}
        return {"answers": answers, "usage": {}, "latency_ms": 1, "backend": self.name}


@contextmanager
def _local_server():
    previous = (_State.decider, _State.backend_name, _State.started_at, _State.calls, _State.errors)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _State.decider = Decider(backend=_JevCompatibleBackend())
    _State.backend_name = "jev-wire-test"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    try:
        thread.start()
        yield f"http://127.0.0.1:{server.server_address[1]}/v1/systemone"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        (_State.decider, _State.backend_name, _State.started_at,
         _State.calls, _State.errors) = previous


def validate_like_jev_ultrafast(answer: dict, ids: list[str]) -> bool:
    """Ported from browser-use/jev-ultrafast jev_ultrafast/model.py::validate_choice."""
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    return valid


def test_jev_ultrafast_wire_compatibility():
    # A realistic jev-ultrafast request: page state + operation + target questions
    body = {
        "model": "localdecide",
        "state": {
            "page": {"url": "https://en.wikipedia.org/wiki/Main_Page",
                     "title": "Wikipedia, the free encyclopedia",
                     "text": "Main page. Featured article. In the news."},
            "elements": [
                {"index": "1", "label": "Main page", "role": "link"},
                {"index": "2", "label": "Random article", "role": "link"},
                {"index": "3", "label": "Search Wikipedia", "role": "searchbox"},
            ],
            "recent_actions": [],
        },
        "questions": {
            "operation": {
                "type": "choice",
                "criteria": {
                    "CLICK": "Click an element.",
                    "TYPE_TEXT": "Enter text in a field.",
                    "DONE": "Goal is satisfied.",
                },
                "instructions": {"goal": "Click the 'Random article' link.", "rules": "Fill fields before submitting."},
            },
            "click_target": {
                "type": "choice",
                "criteria": {
                    "1": "[1] Main page",
                    "2": "[2] Random article",
                },
                "instructions": {"goal": "Click the 'Random article' link.", "operation": "CLICK"},
            },
        },
    }
    with _local_server() as server_url:
        request = urllib.request.Request(server_url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())

    assert "answers" in payload, f"no answers in reply: {list(payload)}"
    operation_answer = payload["answers"]["operation"]
    click_answer = payload["answers"]["click_target"]

    # jev-ultrafast's own validation, verbatim logic
    assert validate_like_jev_ultrafast(operation_answer, ["CLICK", "TYPE_TEXT", "DONE"]), \
        f"operation answer fails jev-ultrafast validation: {operation_answer}"
    assert validate_like_jev_ultrafast(click_answer, ["1", "2"]), \
        f"click_target answer fails validation: {click_answer}"

    # And the decision is the right one for this goal
    assert operation_answer["choice"] == "CLICK"
    assert click_answer["choice"] == "2"  # Random article
    assert click_answer["probabilities"]["2"] > 0.8


if __name__ == "__main__":
    test_jev_ultrafast_wire_compatibility()
    print("jev-ultrafast wire compatibility: VERIFIED")
