
import time
from localdecide import BrowserDecider, Decider, Scope
from localdecide.drivers import PlaywrightDriver

GOALS = [
    ("https://en.wikipedia.org/wiki/Main_Page", "Open the Contents page.", "Contents"),
    ("https://news.ycombinator.com", "Open the newest submissions page.", "new"),
]

for url, goal, want in GOALS:
    drv = PlaywrightDriver(headless=True, start_url=url)
    obs = drv.observe()
    scoped = Scope(max_elements=20).apply(obs, goal=goal)
    offered = [a["label"] for a in scoped["actions"]]
    visible = any(want.lower() in l.lower() for l in offered)
    print(f"\n{url.split('//')[1][:30]}")
    print(f"  goal: {goal}")
    print(f"  observed {len(obs['actions'])} -> offered {len(offered)} | '{want}' offered: {visible}")
    for l in offered[:6]:
        print(f"     - {l[:55]!r}")
    drv.close()
