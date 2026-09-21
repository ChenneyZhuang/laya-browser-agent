"""End-to-end: a real browser, a real page, decisions from the local model.

Run it with the browser visible to watch the decisions land:

    .venv/bin/python examples/live_test.py
    .venv/bin/python examples/live_test.py --headless
"""

import argparse
import sys

from localdecide import BrowserDecider, Decider
from localdecide.drivers import PlaywrightDriver

GOAL = "Click the 'Random article' link in the navigation."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--goal", default=GOAL)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()

    print("launching chromium...", flush=True)
    driver = PlaywrightDriver(headless=args.headless, start_url="https://en.wikipedia.org/wiki/Main_Page")
    observation = driver.observe()
    print(f"page: {observation['title']}")
    print(f"elements observed: {len(observation['actions'])}")
    for action in observation["actions"][:8]:
        print(f"  [{action['index']}] {action['kind']:<7} {action['label'][:50]!r}")

    print(f"\n--- running the loop for: {args.goal!r} ---", flush=True)
    decider = BrowserDecider(decider=Decider(), max_steps=args.max_steps)
    run = decider.run(driver, args.goal)

    print(f"\nstopped: {run.stopped}" + (f"  ({run.error})" if run.error else ""))
    for step in run.steps:
        print(f"  step {step.n}: {step.operation:<9} -> {step.label[:45]!r} "
              f"conf={step.confidence:.3f} {step.latency_ms}ms ok={step.executed}")
    print("\nsummary:", run.summary())
    return 0 if run.stopped in ("done", "max_steps") else 1


if __name__ == "__main__":
    sys.exit(main())
