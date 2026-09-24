"""The real cause: state shape decides, not instruction wording.

Same options, same goal, opposite answers:
  - fixture state (real page text) -> CLICK on the submit button, p=0.976
  - synthetic state (short text)   -> TYPE_TEXT into the field,  p=0.946

The difference is what the *state* says about where the flow already is. The fixture page
renders "Cart: 0 item(s) · Step 1", and its field carries a `current_value` of "" - read by
the model as "this step may already be under way". This script isolates the variable.
"""

from localdecide import Decider, Scope, build_element_table
from localdecide.drivers import PlaywrightDriver

import pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parents[2]
def _fx(name: str) -> str:
    return (_REPO / "tests" / "fixtures" / name).as_uri()

URL = _fx("flow_shop.html")
GOAL = "Search products for 'kettle' and then show the results."

OPERATIONS = {
    "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
    "TYPE_TEXT": "Enter or replace text in an editable field. A text model will supply the value.",
    "SCROLL_DOWN": "Scroll the page down to reveal more content.",
    "SCROLL_UP": "Scroll the page up.",
    "WAIT": "The needed control is absent or disabled, or submitted results are still loading.",
    "DONE": "Every requirement in the goal is visibly satisfied.",
    "BLOCKED": "No supported operation can make progress.",
}


def ask(state, fill_options, click_options, goal=GOAL):
    questions = {
        "operation": {"type": "choice", "criteria": dict(OPERATIONS),
                      "instructions": {"goal": goal}},
        "type_text_target": {"type": "choice", "criteria": fill_options,
                             "instructions": {"goal": goal, "operation": "TYPE_TEXT"}},
        "click_target": {"type": "choice", "criteria": click_options,
                         "instructions": {"goal": goal, "operation": "CLICK"}},
    }
    result = Decider().decide(state, questions)
    if not result.ok:
        return f"FAILED: {result.error}", 0.0, "-"
    assert result.answers is not None
    operation = result.answers.choice("operation")
    probability = result.answers.probabilities("operation")[operation]
    target = result.answers.choice(f"{operation.lower()}_target") if f"{operation.lower()}_target" in result.answers.raw else "-"
    return operation, probability, target


driver = PlaywrightDriver(headless=True, start_url=URL)
observation = driver.observe()
driver.close()
table = build_element_table(observation)
fill_options = {i: e.describe() for i, e in table.targets_for("TYPE_TEXT").items()}
click_options = {i: e.describe() for i, e in table.targets_for("CLICK").items()}

print(f"fill options : {fill_options}")
print(f"click options: {click_options}\n")
print(f"{'state variant':<46} {'op':<11} {'p':<7} target")
print("-" * 82)

variants = {
    "full fixture state (baseline)":
        table.state(text_chars=1200),
    "state without recent_actions":
        {k: v for k, v in table.state(text_chars=1200).items() if k != "recent_actions"},
    "page text only":
        {"page": table.state(text_chars=1200)["page"]},
    "page without text":
        {"page": {"url": table.state()["page"]["url"], "title": table.state()["page"]["title"], "text": ""},
         "recent_actions": []},
    "page text truncated to 60 chars":
        {"page": {**table.state()["page"], "text": table.state()["page"]["text"][:60]},
         "recent_actions": []},
}

for name, state in variants.items():
    operation, probability, target = ask(state, fill_options, click_options)
    print(f"{name:<46} {operation:<11} {probability:<7.3f} {target}")

print("\n--- and with the field pre-filled (simulating a partial flow) ---")
prefilled = {i: e.describe() for i, e in table.targets_for("TYPE_TEXT").items()}
for key, value in prefilled.items():
    value = value.replace("= ''", "= 'kettle'")
    prefilled[key] = value
operation, probability, target = ask(table.state(text_chars=1200), prefilled, click_options)
print(f"{'field already contains the query':<46} {operation:<11} {probability:<7.3f} {target}")
