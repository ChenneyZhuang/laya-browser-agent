
import sys
sys.path.insert(0, "/Volumes/SSD/localdecide")
from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

drv = PlaywrightDriver(headless=True, start_url="file:///Volumes/SSD/localdecide/tests/fixtures/flow_shop.html")
obs = drv.observe(); drv.close()
table = build_element_table(obs)
qs = table_to_questions(table, "Search products for kettle and then show the results.")
print("operation instructions type:", type(qs["operation"]["instructions"]).__name__)
print("operation instructions keys:", list(qs["operation"]["instructions"].keys()) if isinstance(qs["operation"]["instructions"], dict) else "n/a")
print("rules length:", len(str(qs["operation"]["instructions"].get("rules", ""))))
print()
print("=== what the model actually receives (raw sequence) ===")
d = Decider()
agent = d.backend._load()
state = table.state(text_chars=1200)
items, internal = agent.prepare(state, qs)
q = internal[0]
text = agent.tok if False else None
# decode the token ids back to text so we can SEE the prompt
try:
    ids = items[0]["ids"]
    decoded = agent.tok.decode(ids) if hasattr(agent.tok, "decode") else None
except Exception as e:
    decoded = f"(decode failed: {e})"
if decoded:
    print(decoded[:2500])
