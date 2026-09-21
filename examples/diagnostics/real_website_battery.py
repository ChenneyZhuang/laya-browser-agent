"""Real-website battery: the model against the live, messy, unscripted web.

Each target is a real production site chosen to exercise a different challenge.
All goals are read-only navigation - no POST, purchase, or deletion paths.

Safety rules:
  * one driver at a time, closed in a finally block
  * scoped observations (max 25 elements) to keep memory bounded
  * any failure is recorded, never raised into a retry loop

Run:  .venv/bin/python examples/diagnostics/real_website_battery.py
"""

from __future__ import annotations

import time

from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

# (url, goal, label-fragment that marks success)
CASES = [
    ("https://news.ycombinator.com",
     "Open the newest submissions page.", "new"),
    ("https://en.wikipedia.org/wiki/Python_(programming_language)",
     "View the edit history of this page.", "history"),
    ("https://www.python.org",
     "Go to the downloads page.", "download"),
    ("https://stackoverflow.com",
     "Browse the questions tagged Python.", "python"),
    ("https://www.bbc.com/news",
     "Open the business news section.", "business"),
    ("https://duckduckgo.com",
     "Type a query into the search box.", "search"),
]

SUCCESS_MARKERS = {
    "newest submissions": "new",
    "edit history": "history",
    "downloads page": "download",
    "tagged python": "python",
    "business news": "business",
    "search box": "search",
}


def is_hit(goal: str, label: str) -> bool:
    for fragment, marker in SUCCESS_MARKERS.items():
        if fragment in goal:
            return marker.lower() in label.lower()
    return False


decider = Decider()
print(f"{'site':<32} {'goal':<42} {'op':<10} {'hit':<5} {'ms':<6} {'p':<6} chose")
print("-" * 120)

correct = 0
total = 0

for url, goal, want in CASES:
    driver = None
    try:
        driver = PlaywrightDriver(headless=True, start_url=url)
        observation = driver.observe()
    except Exception as error:
        print(f"{url[:30]:<32} {goal[:40]:<42} OBSERVE FAILED: {type(error).__name__}: {str(error)[:50]}")
        continue

    try:
        scoped = Scope(max_elements=25).apply(observation, goal=goal)
        table = build_element_table(scoped)
        questions = table_to_questions(table, goal)
        started = time.time()
        result = decider.decide(table.state(text_chars=1200), questions)
        elapsed = (time.time() - started) * 1000
    finally:
        driver.close()

    if not result.ok:
        print(f"{url[:30]:<32} {goal[:40]:<42} DECIDE FAILED  {str(result.error)[:60]}")
        continue

    assert result.answers is not None
    operation = result.answers.choice("operation")
    key = f"{operation.lower()}_target"
    if key not in result.answers.raw:
        print(f"{url[:30]:<32} {goal[:40]:<42} {operation:<10} (no target question)")
        continue

    total += 1
    target = result.answers.choice(key)
    element = table.by_index().get(target)
    label = element.label if element else "?"
    probability = result.answers.probabilities(key).get(target, 0.0)
    hit = is_hit(goal, label)
    correct += 1 if hit else 0
    print(f"{url[:30]:<32} {goal[:40]:<42} {operation:<10} {'HIT ' if hit else 'miss'} {elapsed:<6.0f} {probability:<6.3f} {label[:32]!r}")

print(f"\n=== summary: {correct}/{total} correct target selections on real production sites ===")
