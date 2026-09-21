
import sys
sys.path.insert(0, "/Volumes/SSD/localdecide")
from localdecide import Decider, build_element_table

GOAL = "Search products for 'kettle' and then show the results."
OPS = {
  "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
  "TYPE_TEXT": "Enter or replace text in an editable field. A text model will supply the value.",
  "SCROLL_DOWN": "Scroll the page down to reveal more content.",
  "SCROLL_UP": "Scroll the page up.",
  "WAIT": "The needed control is absent or disabled, or submitted results are still loading.",
  "DONE": "Every requirement in the goal is visibly satisfied.",
  "BLOCKED": "No supported operation can make progress.",
}
fills = {"1": "[1] Search products (text)"}
clicks = {"2": "[2] Search (button)", "3": "[3] Delete my account (button)"}

def ask(text, label):
    state = {"page": {"url": "https://shop.example/x", "title": "Shop", "text": text}, "recent_actions": []}
    qs = {"operation": {"type":"choice","criteria":dict(OPS),"instructions":{"goal":GOAL}},
          "type_text_target": {"type":"choice","criteria":fills,"instructions":{"goal":GOAL,"operation":"TYPE_TEXT"}},
          "click_target": {"type":"choice","criteria":clicks,"instructions":{"goal":GOAL,"operation":"CLICK"}}}
    r = Decider().decide(state, qs)
    if not r.ok: return print(f"{label:<44} FAILED {r.error}")
    op = r.answers.choice("operation")
    p = r.answers.probabilities("operation")[op]
    extra = dict(sorted(r.answers.probabilities("operation").items(), key=lambda kv:-kv[1])[:3])
    print(f"{label:<44} {op:<10} p={p:.3f}  {extra}")

print("--- identical option list; only the page text changes ---")
ask("", "empty text")
ask("Welcome to our shop.", "neutral text")
ask("Cart: 0 items", "cart status")
ask("Search products  Search", "text echoing the field and button")
ask("Step 1 — Find a product\nSearch products  Search", "the fixture's actual text")
