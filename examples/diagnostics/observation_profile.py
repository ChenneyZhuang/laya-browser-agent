"""Diagnostic: what does the model actually see, and where does latency go?"""
import time

from localdecide import Decider
from localdecide.drivers import PlaywrightDriver
from localdecide.page import build_element_table, table_to_questions

URL = "https://en.wikipedia.org/wiki/Main_Page"

driver = PlaywrightDriver(headless=True, start_url=URL)
observation = driver.observe()
actions = observation["actions"]
print(f"observed elements: {len(actions)}")
for action in actions[:18]:
    print(f"  [{action['index']:>3}] {action['kind']:<7} {action['label'][:66]!r}")

targets = [a for a in actions if "contents" in a["label"].lower() or "random" in a["label"].lower()]
print(f"\nelements matching 'contents'/'random': {len(targets)}")
for action in targets:
    print(f"  [{action['index']}] {action['label'][:66]!r}")

print("\n--- latency vs observation size (goal: open the Contents page) ---")
decider = Decider()
for cap in (12, 20, 30, 60, 120):
    table = build_element_table({**observation, "actions": actions[:cap]})
    questions = table_to_questions(table, "Open the Contents page.")
    started = time.time()
    result = decider.decide(table.state(text_chars=1200), questions)
    elapsed = (time.time() - started) * 1000
    if result.ok:
        chosen = result.answers.choice("click_target") if "click_target" in result.answers.raw else "-"
        element = table.by_index().get(chosen)
        passes = result.answers.raw.get("click_target", {}).get("coarse_to_fine")
        print(f"  cap={cap:<4} {elapsed:6.0f} ms  target={chosen:<4} {element.label[:42] if element else ''!r}"
              + (f"  [{passes['chunks']} chunks, 2 passes]" if passes else "  [1 pass]"))
    else:
        print(f"  cap={cap:<4} failed: {result.error}")

print("\n--- does the state text length matter? ---")
table = build_element_table({**observation, "actions": actions[:20]})
questions = table_to_questions(table, "Open the Contents page.")
for chars in (200, 600, 1200, 3000):
    started = time.time()
    result = decider.decide(table.state(text_chars=chars), questions)
    print(f"  text_chars={chars:<5} {(time.time()-started)*1000:6.0f} ms  ok={result.ok}")

driver.close()
