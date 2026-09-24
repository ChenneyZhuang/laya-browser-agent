"""Experiment: which instruction shape makes the checkpoint type before it submits?

The observed failure: on a search form, the goal "Search products for 'kettle'" produced
`CLICK` on the Search button (p=0.976) instead of `TYPE_TEXT` into the empty field. The
rules text IS reaching the model (243 tokens, inside the 432-token budget), so the
question is whether the *shape* of the instruction matches what the checkpoint was
fine-tuned on. This runs the same page and goal through several instruction variants and
prints what each one chooses.

Run:  .venv/bin/python examples/diagnostics/instruction_ablation.py
"""

from localdecide import Decider, Scope, build_element_table
from localdecide.drivers import PlaywrightDriver
from localdecide.page import NEXT_ACTION_RULES, TARGET_RULES

import pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parents[2]
def _fx(name: str) -> str:
    return (_REPO / "tests" / "fixtures" / name).as_uri()

URL = _fx("flow_shop.html")
GOAL = "Search products for 'kettle' and then show the results."

driver = PlaywrightDriver(headless=True, start_url=URL)
observation = driver.observe()
driver.close()
scoped = Scope(max_elements=25).apply(observation, goal=GOAL)
table = build_element_table(scoped)
state = table.state(text_chars=1200)

# All the elements that support each operation, as option dicts.
clicks = {index: element.describe() for index, element in table.targets_for("CLICK").items()}
fills = {index: element.describe() for index, element in table.targets_for("TYPE_TEXT").items()}

OPERATIONS = {
    "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
    "TYPE_TEXT": "Enter or replace text in an editable field. A text model will supply the value.",
    "SCROLL_DOWN": "Scroll the page down to reveal more content.",
    "SCROLL_UP": "Scroll the page up.",
    "WAIT": "The needed control is absent or disabled, or submitted results are still loading.",
    "DONE": "Every requirement in the goal is visibly satisfied.",
    "BLOCKED": "No supported operation can make progress.",
}

variants = {
    "A_jev_dict_rules": {
        "operation": {"type": "choice", "criteria": dict(OPERATIONS),
                      "instructions": {"goal": GOAL, "rules": NEXT_ACTION_RULES}},
        "type_text_target": {"type": "choice", "criteria": fills,
                             "instructions": {"goal": GOAL, "operation": "TYPE_TEXT",
                                              "rules": [NEXT_ACTION_RULES, TARGET_RULES]}},
        "click_target": {"type": "choice", "criteria": clicks,
                         "instructions": {"goal": GOAL, "operation": "CLICK",
                                          "rules": [NEXT_ACTION_RULES, TARGET_RULES]}},
    },
    "B_short_string": {
        "operation": {"type": "choice", "criteria": dict(OPERATIONS),
                      "instructions": f"Goal: {GOAL}  Which action should be taken next?"},
        "type_text_target": {"type": "choice", "criteria": fills,
                             "instructions": f"Goal: {GOAL}  Which field should be typed into?"},
        "click_target": {"type": "choice", "criteria": clicks,
                         "instructions": f"Goal: {GOAL}  Which element should be clicked?"},
    },
    "C_empty_form_hint": {
        "operation": {"type": "choice", "criteria": dict(OPERATIONS),
                      "instructions": {"goal": GOAL,
                                       "rules": "The search field is empty. Fill it before submitting. "
                                                "Fill required fields before submitting."}},
        "type_text_target": {"type": "choice", "criteria": fills,
                             "instructions": f"Goal: {GOAL}  Which field should be typed into?"},
        "click_target": {"type": "choice", "criteria": clicks,
                         "instructions": f"Goal: {GOAL}  Which element should be clicked?"},
    },
    "D_no_rules": {
        "operation": {"type": "choice", "criteria": dict(OPERATIONS),
                      "instructions": {"goal": GOAL}},
        "type_text_target": {"type": "choice", "criteria": fills,
                             "instructions": {"goal": GOAL, "operation": "TYPE_TEXT"}},
        "click_target": {"type": "choice", "criteria": clicks,
                         "instructions": {"goal": GOAL, "operation": "CLICK"}},
    },
}

decider = Decider()
print(f"goal: {GOAL}")
print(f"offered fields: {list(fills)}\noffered clicks: {list(clicks)}\n")
print(f"{'variant':<22} {'operation':<11} p(op)   target                      p(target)")
print("-" * 88)
for name, questions in variants.items():
    result = decider.decide(state, questions)
    if not result.ok:
        print(f"{name:<22} FAILED       {result.error}")
        continue
    assert result.answers is not None
    operation = result.answers.choice("operation")
    p_op = result.answers.probabilities("operation")[operation]
    target, p_target, label = "-", 0.0, ""
    key = f"{operation.lower()}_target"
    if key in result.answers.raw:
        target = result.answers.choice(key)
        p_target = result.answers.probabilities(key)[target]
        element = table.by_index().get(target)
        label = element.label[:26] if element else "?"
    print(f"{name:<22} {operation:<11} {p_op:<7.3f} {target:<4} {label:<22} {p_target:.3f}")
