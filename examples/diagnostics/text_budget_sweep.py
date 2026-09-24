"""Page text is a first-class knob. Find the setting that keeps decisions honest.

Finding this measures: with the full fixture text visible, the model answers CLICK on a
submit button while the search field is empty; with no page text it answers TYPE_TEXT. The
text length steers the decision more than any instruction wording does.

Two variables are swept here because they interact:
  * `text_chars`  - how much page prose the model sees
  * "field visible" - whether a `type_text_target` question was offered at all

The second one matters because `table_to_questions` only offers an operation when the page
has a control for it. A state built from a full-page table but questions built from a
scoped table is an inconsistent pair, and the earlier version of this script made exactly
that mistake - which is why its numbers disagreed with the decision-level experiments.
"""

from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

import pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parents[2]
def _fx(name: str) -> str:
    return (_REPO / "tests" / "fixtures" / name).as_uri()

FIXTURE = _fx("flow_shop.html")
GOAL = "Search products for 'kettle' and then show the results."

driver = PlaywrightDriver(headless=True, start_url=FIXTURE)
observation = driver.observe()
driver.close()

for scope_label, scope in (("unscoped", None), ("scoped(25)", Scope(max_elements=25))):
    scoped = scope.apply(observation, goal=GOAL) if scope else observation
    table = build_element_table(scoped)
    fills = list(table.targets_for("TYPE_TEXT"))
    clicks = list(table.targets_for("CLICK"))
    print(f"=== {scope_label}: {len(table.elements)} offered "
          f"({len(clicks)} clickable, {len(fills)} fillable) ===")

    print(f"{'text_chars':<12} {'operation':<11} {'p':<7} target")
    print("-" * 62)
    for text_chars in (0, 80, 300, 1200):
        questions = table_to_questions(table, GOAL)   # SAME table -> consistent pair
        result = Decider().decide(table.state(text_chars=text_chars), questions)
        if not result.ok:
            print(f"{text_chars:<12} FAILED   {result.error}")
            continue
        assert result.answers is not None
        operation = result.answers.choice("operation")
        probability = result.answers.probabilities("operation")[operation]
        target = "-"
        key = f"{operation.lower()}_target"
        if key in result.answers.raw:
            element = table.by_index().get(result.answers.choice(key))
            target = element.label[:24] if element else "?"
        print(f"{text_chars:<12} {operation:<11} {probability:<7.3f} {target}")
    print()
