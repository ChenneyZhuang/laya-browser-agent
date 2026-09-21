"""Is the TYPE_TEXT reluctance about the label, or about the model?

The label in the failing case is "Search products" (the field) next to "Search" (the
submit button) - nearly the same string. A decision model reading a short option list may
simply not distinguish them. This tests that hypothesis by changing ONLY the labels, with
the goal and the page role held constant.
"""

from localdecide import Decider, build_element_table

state = {"page": {"url": "https://shop.example/search", "title": "Flow Shop",
                  "text": "Step 1 - Find a product. Cart: 0 items."},
         "recent_actions": []}

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

scenarios = {
    "similar labels (current fixture)": {
        "fill": {"3": "[3] Search products (textbox)"},
        "click": {"1": "[1] Search (button)", "2": "[2] Delete my account (button)"},
    },
    "distinct field label": {
        "fill": {"3": "[3] Product search query (searchbox)"},
        "click": {"1": "[1] Search (button)", "2": "[2] Delete my account (button)"},
    },
    "field listed before button": {
        "fill": {"1": "[1] Search products (textbox)"},
        "click": {"2": "[2] Search (button)", "3": "[3] Delete my account (button)"},
    },
    "no submit button at all": {
        "fill": {"1": "[1] Search products (textbox)"},
        "click": {"2": "[2] Delete my account (button)"},
    },
    "goal names the field explicitly": {
        "fill": {"1": "[1] Search products (textbox)"},
        "click": {"2": "[2] Search (button)", "3": "[3] Delete my account (button)"},
    },
}

decider = Decider()
print(f"{'scenario':<34} {'operation':<11} {'p':<7} target")
print("-" * 78)
for name, parts in scenarios.items():
    goal = ("Type 'kettle' into the 'Search products' field." if name.startswith("goal names")
            else GOAL)
    questions = {
        "operation": {"type": "choice", "criteria": dict(OPERATIONS),
                      "instructions": {"goal": goal}},
        "type_text_target": {"type": "choice", "criteria": parts["fill"],
                             "instructions": {"goal": goal, "operation": "TYPE_TEXT"}},
        "click_target": {"type": "choice", "criteria": parts["click"],
                         "instructions": {"goal": goal, "operation": "CLICK"}},
    }
    result = decider.decide(state, questions)
    if not result.ok:
        print(f"{name:<34} FAILED: {result.error}")
        continue
    assert result.answers is not None
    operation = result.answers.choice("operation")
    probability = result.answers.probabilities("operation")[operation]
    target = "-"
    key = f"{operation.lower()}_target"
    if key in result.answers.raw:
        target = result.answers.choice(key)
    print(f"{name:<34} {operation:<11} {probability:<7.3f} {target}")
