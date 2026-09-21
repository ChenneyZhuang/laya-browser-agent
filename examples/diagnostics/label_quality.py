"""Label quality on real pages: what does the observation reader actually produce?

Accuracy starts here. If labels are empty, duplicated, or 200 characters of prose, the
decision has nothing to work with - and that failure looks like a model failure. This
dumps what the reader sees on real sites and counts the problems.
"""

from __future__ import annotations

import collections

from localdecide.drivers import PlaywrightDriver

URLS = [
    "https://news.ycombinator.com",
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://duckduckgo.com",
    "https://www.python.org",
]

for url in URLS:
    try:
        driver = PlaywrightDriver(headless=True, start_url=url)
        observation = driver.observe()
        driver.close()
    except Exception as error:
        print(f"\n=== {url}: FAILED: {type(error).__name__}: {error}")
        continue

    actions = observation["actions"]
    print(f"\n=== {url} : {len(actions)} actions ===")
    for action in actions[:16]:
        value = f" = {action.get('current_value', '')!r}" if action.get("current_value") else ""
        print(f"  [{action['index']:>3}] {action['kind']:<7} {str(action['label'])[:56]!r}{value}")

    empty = [a for a in actions if not str(a.get("label", "")).strip()]
    long_labels = [a for a in actions if len(str(a.get("label", ""))) > 80]
    counts = collections.Counter(str(a.get("label", "")) for a in actions)
    duplicated = {label: count for label, count in counts.items() if count > 1}
    print(f"  -- empty labels: {len(empty)} | over 80 chars: {len(long_labels)} "
          f"| duplicated: {sum(duplicated.values())} across {len(duplicated)} label(s)")
    for label, count in list(duplicated.items())[:4]:
        print(f"       x{count}  {label[:60]!r}")
    for action in long_labels[:3]:
        print(f"       prose: {str(action['label'])[:110]!r}")
