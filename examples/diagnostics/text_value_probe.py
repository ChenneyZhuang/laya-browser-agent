
import sys
sys.path.insert(0, "/Volumes/SSD/localdecide")
from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

drv = PlaywrightDriver(headless=True, start_url="file:///Volumes/SSD/localdecide/tests/fixtures/flow_shop.html")
obs = drv.observe(); drv.close()
GOAL = "Search products for 'kettle' and then show the results."
table = build_element_table(Scope(max_elements=25).apply(obs, goal=GOAL))
qs = table_to_questions(table, GOAL)
print("questions offered:", list(qs))
print("fill criteria:", qs.get("type_text_target", {}).get("criteria"))
print("click criteria:", qs.get("click_target", {}).get("criteria"))
full = table.state(text_chars=1200)
print("\npage text repr:", repr(full["page"]["text"])[:200])
print("url:", full["page"]["url"])
print("title:", full["page"]["title"])

d = Decider()
variants = {
  "as produced by the driver": full,
  "text replaced with empty string": {**full, "page": {**full["page"], "text": ""}},
  "no text key at all": {**full, "page": {k: v for k, v in full["page"].items() if k != "text"}},
}
for name, state in variants.items():
    r = d.decide(state, qs)
    if r.ok:
        op = r.answers.choice("operation")
        print(f"{name:<34} -> {op:<10} p={r.answers.probabilities('operation')[op]:.3f}")
    else:
        print(f"{name:<34} -> FAILED {r.error}")
