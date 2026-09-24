
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

import pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parents[2]
def _fx(name: str) -> str:
    return (_REPO / "tests" / "fixtures" / name).as_uri()

drv = PlaywrightDriver(headless=True, start_url=_fx("element_gym.html"))
obs = drv.observe(); drv.close()
GOAL = "Accept the terms by ticking 'Terms accepted'."
table = build_element_table(Scope(max_elements=25).apply(obs, goal=GOAL))
qs = table_to_questions(table, GOAL)

# what does the model see for that checkbox?
for i, e in table.targets_for("CLICK").items():
    if "Terms" in e.label or "Newsletter" in e.label:
        print(f"  [{i}] {e.describe()!r}")

r = Decider().decide(table.state(text_chars=1200), qs)
if r.ok:
    op = r.answers.choice("operation")
    print(f"\noperation: {op} p={r.answers.probabilities('operation')[op]:.3f}")
    print("all op probs:", {k: round(v,3) for k,v in sorted(r.answers.probabilities("operation").items(), key=lambda kv:-kv[1])})
    if "click_target" in r.answers.raw:
        t = r.answers.choice("click_target")
        print(f"target: {t} = {table.by_index()[t].label!r} p={r.answers.probabilities('click_target')[t]:.3f}")
        print("top targets:", {table.by_index()[k].label[:30]: round(v,3) for k,v in sorted(r.answers.probabilities("click_target").items(), key=lambda kv:-kv[1])[:4]})
else:
    print("FAILED", r.error)
