"""Diagnostic: why does the model CLICK instead of TYPE_TEXT on a search flow?

Finding to explain: with the flow_shop fixture and the goal "Search products for 'kettle'",
the model chose CLICK on the Search *button* four times instead of typing into the field
first. Reproduced here in isolation so the cause is visible: is it the scoping order, the
question wording, or the checkpoint's bias?
"""
import time

from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

URL = "file:///Volumes/SSD/localdecide/tests/fixtures/flow_shop.html"
GOAL = "Search products for 'kettle' and then show the results."

driver = PlaywrightDriver(headless=True, start_url=URL)
observation = driver.observe()
driver.close()

print(f"observed {len(observation['actions'])} elements")
for action in observation["actions"]:
    print(f"  [{action['index']}] {action['kind']:<7} {action['label'][:50]!r}")

for max_elements in (10, 25, 40):
    scoped = Scope(max_elements=max_elements).apply(observation, goal=GOAL)
    table = build_element_table(scoped)
    questions = table_to_questions(table, GOAL)
    print(f"\n--- max_elements={max_elements}: offered {len(table.elements)} "
          f"({len(table.targets_for('CLICK'))} clickable, {len(table.targets_for('TYPE_TEXT'))} fillable)")
    for element in table.elements:
        print(f"    [{element.index}] {','.join(element.operations):<20} {element.label[:45]!r}")
    result = Decider().decide(table.state(text_chars=1200), questions)
    if result.ok:
        assert result.answers is not None
        operation = result.answers.choice("operation")
        probabilities = result.answers.probabilities("operation")
        print(f"    -> {operation}  {dict(sorted(probabilities.items(), key=lambda kv: -kv[1])[:4])}")
        if operation in ("CLICK", "TYPE_TEXT", "SELECT"):
            target = result.answers.choice(f"{operation.lower()}_target")
            print(f"       target {target}: {table.by_index()[target].label[:50]!r}")
    else:
        print(f"    -> FAILED: {result.error}")

print("\n--- same question with the goal's own words removed from scoping (no goal) ---")
scoped = Scope(max_elements=25).apply(observation)
table = build_element_table(scoped)
questions = table_to_questions(table, GOAL)
result = Decider().decide(table.state(text_chars=1200), questions)
if result.ok:
    assert result.answers is not None
    print(f"    -> {result.answers.choice('operation')}")
    print(f"       offered: {[e.label[:35] for e in table.elements]}")
