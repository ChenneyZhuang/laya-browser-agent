
import sys, json
sys.path.insert(0, "/Volumes/SSD/localdecide")
from localdecide import build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver
from laya_mlx.common import build_prefix, render_options, QTYPES

drv = PlaywrightDriver(headless=True, start_url="file:///Volumes/SSD/localdecide/tests/fixtures/flow_shop.html")
obs = drv.observe(); drv.close()
table = build_element_table(obs)
qs = table_to_questions(table, "Search products for kettle and then show the results.")

from localdecide import Decider
d = Decider(); agent = d.backend._load()
q = agent._to_internal(qs["operation"])
print("instructions value type:", type(q["ins"]).__name__)
print("instructions total chars:", len(str(q["ins"])))
opts = render_options(q)
print("options:", len(opts), "| head_max_len:", agent.cfg.get("head_max_len"))
ids, markers = build_prefix(agent.tok, q, agent.cfg.get("head_max_len", 192))
# how many tokens did the INSTRUCTIONS actually get?
head_only = agent.tok("%s question: %s" % (q["t"], str(q["ins"]).replace(agent.tok.mask_token," ")), add_special_tokens=False)["input_ids"]
print("raw instruction tokens:", len(head_only))
print("tokens kept in prefix for instructions:", len([i for i in ids[:len(head_only)]]) - 0)
print()
print("=== FIRST 160 TOKENS OF THE ACTUAL PROMPT ===")
print(agent.tok.decode(ids[:160]) if hasattr(agent.tok, "decode") else "(no decode)")
