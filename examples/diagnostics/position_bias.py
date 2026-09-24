"""Is the answer a decision, or a position bias? Measure it before tuning anything.

Two things to separate:

1. **Order sensitivity** - does the same option set, shuffled, produce different answers?
   If yes, the model reads position as well as content, and multi-permutation voting can
   cancel that noise out.
2. **Whether voting actually helps** - averaging over permutations costs N forward passes,
   so it has to buy accuracy, not just feel safer.

Ground truth here is "does any permutation get the intended element right", and the metric
we care about is top-1 across permutations vs top-1 of a single ordering.
"""

from __future__ import annotations

import json
import random

from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

import pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parents[2]
def _fx(name: str) -> str:
    return (_REPO / "tests" / "fixtures" / name).as_uri()

CASES = [
    (_fx("element_gym.html"),
     "Click the control labelled 'ARIA role=button'.", "ARIA role=button"),
    (_fx("element_gym.html"),
     "Click the 'Reveal more options' button.", "Reveal more options"),
    (_fx("element_gym.html"),
     "Type a name into the 'Full name' field.", "Full name"),
    (_fx("flow_shop.html"),
     "Open the account section.", "Delete my account"),
    (_fx("multilingual.html"),
     "Click the control labelled '登录'.", "登录"),
]

PERMUTATIONS = 6
decider = Decider()

print(f"{'goal':<44} {'first-try':<10} {'voted':<10} {'spread':<8} gold rank (avg)")
print("-" * 92)

for url, goal, want in CASES:
    driver = PlaywrightDriver(headless=True, start_url=url)
    observation = driver.observe()
    driver.close()
    table = build_element_table(Scope(max_elements=25).apply(observation, goal=goal))
    base_questions = table_to_questions(table, goal)

    gold = [index for index, element in table.by_index().items() if want.lower() in element.label.lower()]
    if not gold:
        print(f"{goal[:42]:<44} (gold not observable)")
        continue

    # vote per option index across permutations
    votes: dict[str, float] = {}
    answers_seen = []
    first_answer = None
    for seed in range(PERMUTATIONS):
        questions = json.loads(json.dumps(base_questions))
        rng = random.Random(seed)
        for question in questions.values():
            if question["type"] == "choice" and isinstance(question["criteria"], dict):
                items = list(question["criteria"].items())
                rng.shuffle(items)
                question["criteria"] = dict(items)
        result = decider.decide(table.state(text_chars=1200), questions)
        if not result.ok:
            continue
        assert result.answers is not None
        operation = result.answers.choice("operation")
        key = f"{operation.lower()}_target"
        chosen = result.answers.choice(key) if key in result.answers.raw else None
        answers_seen.append(chosen)
        if first_answer is None:
            first_answer = chosen
        if chosen:
            for option, probability in result.answers.probabilities(key).items():
                votes[option] = votes.get(option, 0.0) + probability / PERMUTATIONS

    voted = max(votes, key=votes.get) if votes else None
    first_hit = first_answer in gold
    voted_hit = voted in gold
    spread = len(set(answers_seen))
    # average rank of the gold option in the single-order probability lists
    rank_note = f"{'HIT' if first_hit else 'miss'} -> {'HIT' if voted_hit else 'miss'}"

    print(f"{goal[:42]:<44} {str(first_hit):<10} {str(voted_hit):<10} {spread}/{PERMUTATIONS:<6} {rank_note}")

print("\nInterpretation:")
print("  'first-try' - did the first ordering pick the intended element")
print("  'voted'     - did averaging over 6 shuffled orderings")
print("  'spread'    - how many DIFFERENT elements were chosen across orderings.")
print("                High spread = the model reads position; voting is worth paying for.")
