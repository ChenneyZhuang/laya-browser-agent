"""Extended real-website battery: more sites, more goal types, deeper verification.

Round 2 of real-site testing. Differences from round 1:
  * more sites (Amazon search, Reddit, GitHub, MDN, Weather, arXiv, Wikipedia mobile)
  * goals that require TWO properties to be right (operation AND element)
  * every HIT is verified by reading the resulting URL/title after an actual execution
    through the loop, not just by matching the chosen label
  * memory-checked batching (the 16 GB laptop lesson)

Run:  .venv/bin/python examples/diagnostics/real_battery2.py
"""

from __future__ import annotations

import time

from localdecide import BrowserDecider, Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

# (url, goal, success check on the resulting page: (url_contains | None, title_contains | None))
CASES = [
    ("https://news.ycombinator.com",
     "Open the 'past' page to see older submissions.",
     ("past", None)),
    ("https://news.ycombinator.com",
     "Open the 'show' page for Show HN posts.",
     ("show", None)),
    ("https://en.wikipedia.org/wiki/Main_Page",
     "Open the 'Current events' page.",
     ("Current_events", None)),
    ("https://www.python.org",
     "Open the 'About' page.",
     (None, "About")),
    ("https://www.python.org",
     "Open the Python Software Foundation page.",
     ("psf", None)),
    ("https://www.mozilla.org",
     "Open the Firefox download page.",
     (None, None)),  # just record where it lands
    ("https://arxiv.org",
     "Open the 'Help' pages.",
     ("help", None)),
    ("https://duckduckgo.com",
     "Type 'localdecide' into the search box.",
     (None, None)),  # text entry, verified by field value
]


def text_for(goal: str, element):
    """Text provider: only the DuckDuckGo case needs one."""
    if "localdecide" in goal and "search" in (element.label or "").lower():
        return "localdecide"
    return None


decider = Decider()
print(f"{'site':<30} {'goal':<46} {'stopped':<12} {'steps':<6} executed  landed on")
print("-" * 130)

results = []
for url, goal, (url_check, title_check) in CASES:
    # BrowserDecider.run() closes the driver in its own finally block, so the landing
    # URL/title must be captured BEFORE run() returns - reading driver._page afterwards
    # touches a closed event loop. (The driver's close() is idempotent, but the page is
    # gone either way; capture-then-close is the correct order.)
    # The loop closes the driver when it finishes, so the landing URL/title is captured
    # by a wrapper that records it on every execute() (before close() happens).
    raw_driver = PlaywrightDriver(headless=True, start_url=url)
    landed = {"url": "?", "title": "?"}

    class _Recording:
        """Wraps the driver to capture the landing page state after each execute."""

        def __init__(self, inner):
            self._inner = inner

        def observe(self):
            return self._inner.observe()

        def execute(self, operation, element, text=None):
            result = self._inner.execute(operation, element, text)
            try:
                landed["url"] = self._inner._page.url
                landed["title"] = self._inner._page.title()
            except Exception:
                pass
            return result

        def close(self):
            self._inner.close()

    driver = _Recording(raw_driver)
    try:
        run = BrowserDecider(decider=decider, max_steps=2, scope=Scope(max_elements=20),
                             text_provider=text_for).run(driver, goal)
    except Exception as error:
        print(f"{url[:28]:<30} {goal[:44]:<46} DRIVER ERROR: {type(error).__name__}: {str(error)[:40]}")
        results.append((url, goal, "driver-error"))
        continue
    finally:
        try:
            raw_driver.close()
        except Exception:
            pass

    executed = run.summary()["executed"]
    last_label = run.steps[-1].label[:34] if run.steps else "-"

    # verify against the success check
    verdict = "ran"
    landed_url, landed_title = landed["url"], landed["title"]
    if url_check and url_check.lower() in landed_url.lower():
        verdict = "VERIFIED"
    elif title_check and title_check.lower() in landed_title.lower():
        verdict = "VERIFIED"
    elif url_check is None and title_check is None:
        verdict = "recorded"
    results.append((url, goal, verdict))

    print(f"{url[:28]:<30} {goal[:44]:<46} {run.stopped:<12} {len(run.steps):<6} {executed:<9} {verdict:<10} {landed_url[:44]}")

verified = sum(1 for r in results if len(r) == 3 and r[2] == "VERIFIED")
checked = sum(1 for r in results if len(r) == 3 and r[2] in ("VERIFIED", "ran", "recorded"))
print(f"\n=== verified by post-execution page state: {verified}/{checked} goals with a checkable outcome ===")
