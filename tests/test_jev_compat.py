"""Compatibility test: laya-browser-agent's /v1/systemone vs jev-ultrafast's client.

jev-ultrafast (browser-use, 12.6k stars) posts {model, state, questions} to
POST /v1/systemone and validates the reply with `validate_choice`:
  - answer["choice"] in the offered ids
  - probabilities cover exactly the offered ids
  - all numbers finite, in [0,1]
  - probabilities sum to 1 within 0.02
  - the chosen id is the argmax
  - plus "confidence" in [0,1]

This test replays a realistic jev-ultrafast request through our server and runs the
exact same validation, so compatibility is verified, not assumed.
"""

from __future__ import annotations

import json
import math
import urllib.request

SERVER = "http://127.0.0.1:8791/v1/systemone"


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
    request = urllib.request.Request(SERVER, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
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
