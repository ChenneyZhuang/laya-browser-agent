"""localdecide — run a browser-deciding System 1 model entirely on your own machine.

A decision model answers typed questions about a state (`choice` / `score` / `noul`)
in one forward pass and returns calibrated probabilities. It never writes text. That
makes it a good fit for the *deciding* half of a browser agent: given a numbered table
of the controls on a page, pick the next operation and the element to act on.

    from localdecide import Decider, choice
    d = Decider()                          # auto-detects a local runtime
    r = d.decide("Pool company in Sydney", {"is_pool": choice("Is this pool related?",
                                                              {"yes": "pool business", "no": "unrelated"})})
    print(r.answers.choice("is_pool"))     # -> yes

For the full browser loop:

    from localdecide import BrowserDecider
    from localdecide.drivers import PlaywrightDriver
    with PlaywrightDriver(start_url="https://en.wikipedia.org") as driver:
        run = BrowserDecider().run(driver, "Open the Random article link")

Nothing here sends your page content to a cloud service.
"""

from .decider import Answers, Decider, Decision, DecisionError, Question, choice, noul, score
from .loop import BrowserDecider, ElementRef, Run, Step
from .page import (
    OPERATIONS,
    Element,
    ElementTable,
    build_element_table,
    table_to_questions,
)
from .grounding import ground_goal, overlap_score, same_script, script_of
from .scope import Scope, full_page_scope, goal_tokens


def _version() -> str:
    """Single source of truth: the installed package metadata (set from pyproject).

    Falls back to 0.0.0.dev0 for a source checkout that was never installed —
    the value is informational, and a wrong constant here is worse than an
    honest "unknown".
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - stdlib on every supported Python
        return "0.0.0.dev0"
    try:
        return version("laya-browser-agent")
    except PackageNotFoundError:  # pragma: no cover - plain source checkout
        return "0.0.0.dev0"


__version__ = _version()
__all__ = [
    # deciding
    "Decider", "Decision", "DecisionError", "Answers", "Question", "choice", "score", "noul",
    # browser loop
    "BrowserDecider", "Run", "Step", "ElementRef",
    # observations
    "ElementTable", "Element", "build_element_table", "table_to_questions", "OPERATIONS",
    # observation scoping
    "Scope", "full_page_scope",
    # cross-language grounding
    "ground_goal", "overlap_score", "same_script", "script_of",
    # goal tokenization
    "goal_tokens",
    "__version__",
]
