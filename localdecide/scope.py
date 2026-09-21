"""Scoping: keep the observation small enough that decisions stay fast and accurate.

Measured on a 120-element page (Wikipedia main page, M4):

| elements | latency | passes |
|---|---|---|
| 10  | 183 ms | 1 |
| 20  | 333 ms | 1 |
| 30  | 487 ms | 2 (chunked) |
| 60  | 678 ms | 2 (chunked) |
| 120 | 1204 ms | 2 (chunked) |

Two things fall off as the observation grows: latency (more chunks to answer) and
decision quality (more lookalike candidates to confuse). The fix is not a bigger model;
it is a **smaller, better-scoped observation**.

## The trap this module avoids

The obvious implementation is a static list of "chrome to drop" - nav bars, footer
links, keyboard hints. That is wrong, and dangerously so: **"Random article" is a
navigation link, and it is also exactly what a user asks an agent to click.** A filter
that drops it turns a working agent into one that silently cannot do the task, and the
failure is indistinguishable from a model error.

So scoping here is **goal-aware**. An element whose label overlaps the goal is never
dropped, whatever else it looks like. Chrome patterns only remove things the goal shows
no interest in, and goal overlap also promotes an element to the top of the ranking -
because `max_elements` truncates, and whatever ranks last is invisible to the model.

The rule of thumb: **the goal always outranks the heuristic.**
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

# Structural noise: page furniture that is essentially never the subject of a goal.
# Deliberately does NOT include the names of real navigable destinations - those are
# legitimate targets and are only ever dropped when the goal ignores them.
CHROME_PATTERNS: Sequence[str] = (
    r"\bctrl-option-\w+",          # MediaWiki keyboard hints, e.g. "View source [ctrl-option-e]"
    r"\bctrl-\w+",                 # other keyboard hints
    r"^skip to (content|main)",    # skip links
    r"^jump to",
    r"^(privacy policy|disclaimers|code of conduct|mobile view|cookie statement)$",
    r"^(terms of use|creative commons|powered by|hosted by)",
    r"^toggle the (table of )?contents",
    r"^\[?edit\]?$",
    r"^\s*<.*>\s*$",
)

# Signals that an element is an action a goal might plausibly want, so it survives
# even if it also matches a chrome pattern.
KEEP_PATTERNS: Sequence[str] = (
    r"(add to cart|buy now|checkout|purchase|subscribe|sign up)",
    r"(next page|previous page|load more|show more)",
    r"(submit|continue|confirm|save changes|apply)",
    r"(download|install|log ?in|sign ?in|register|log ?out)",
)

# Words that appear in almost every goal and name nothing on a page. Matching on them
# would protect half the observation, defeating the purpose of goal-scoring.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by",
    "from", "into", "onto", "this", "that", "these", "those", "is", "are", "be", "it",
    "its", "then", "than", "please", "click", "open", "go", "navigate", "find", "page",
    "link", "button", "field", "box", "site", "website", "menu", "option", "use",
})


def goal_tokens(goal: str) -> List[str]:
    """The words in a goal that could name something on a page.

    `"Click the 'Random article' link in the navigation."` -> `["random", "article", "navigation"]`.

    Quotes and possessives are stripped rather than kept: a label is written "Random
    article", and a token like `article'` (quote glued on by the regex) would never match
    it - silently disabling goal protection and letting the chrome filter eat the target.
    """
    words = re.findall(r"[a-z0-9']+", (goal or "").lower())
    tokens = []
    for word in words:
        cleaned = word.strip("'")
        if len(cleaned) > 2 and cleaned not in _STOPWORDS:
            tokens.append(cleaned)
    return tokens


@dataclass
class Scope:
    """Filter an observation down to the elements worth deciding among.

    Example:
        scope = Scope(max_elements=20)
        table = build_element_table(scope.apply(driver.observe(), goal=goal))

    Why this is an object rather than an inline list comprehension: the elements you
    *drop* are part of your agent's behaviour, so they should be named, reviewable, and
    testable. And when a drop is wrong it is nearly invisible - the agent simply never
    considers the control the user wanted.
    """

    max_elements: int = 20
    # Drop anything matching these (case-insensitive substrings) before ranking.
    exclude_words: List[str] = field(default_factory=list)
    # Keep only elements whose label contains one of these, if non-empty. This is an
    # explicit caller request, so it is not overridden by goal protection.
    include_words: List[str] = field(default_factory=list)
    # Drop structural page furniture (see CHROME_PATTERNS). Goal-matching elements survive.
    drop_chrome: bool = True
    # Prefer these labels first (case-insensitive substring), keeping their order.
    prefer_words: List[str] = field(default_factory=list)
    # Hard ceiling on label length, to stop prose-heavy elements eating the token budget.
    max_label_chars: int = 60
    # If True (default), an element matching the goal is never removed by drop_chrome.
    protect_goal_matches: bool = True
    # When the goal is written in a non-Latin script, restrict the offered elements to
    # the ones whose labels share that script. Measured: the browser checkpoint cannot
    # bridge scripts (a Chinese goal picked a Hindi button at p=0.06), and reordering is
    # not enough - the distractors have to leave the option list. Latin goals are unaffected.
    ground_non_latin: bool = True

    # -- pattern helpers ------------------------------------------------------

    def _is_chrome(self, label: str) -> bool:
        return any(re.search(pattern, label, re.IGNORECASE) for pattern in CHROME_PATTERNS)

    def _is_action(self, label: str) -> bool:
        return any(re.search(pattern, label, re.IGNORECASE) for pattern in KEEP_PATTERNS)

    def _matches_goal(self, label: str, tokens: Sequence[str]) -> bool:
        lowered = label.lower()
        return any(token in lowered for token in tokens)

    # -- the decision ---------------------------------------------------------

    def keep(self, action: Mapping[str, Any], tokens: Sequence[str] = ()) -> bool:
        """Would this element be worth deciding among? The goal gets the final say it can."""
        label = str(action.get("label", "") or "")
        if not label.strip():
            return False
        lowered = label.lower()
        if self.include_words and not any(word.lower() in lowered for word in self.include_words):
            return False
        if any(word.lower() in lowered for word in self.exclude_words):
            return False
        if self.drop_chrome and self._is_chrome(label):
            goal_hit = self.protect_goal_matches and self._matches_goal(label, tokens)
            if not (self._is_action(label) or goal_hit):
                return False
        return True

    def rank(self, actions: Sequence[Mapping[str, Any]], tokens: Sequence[str] = ()) -> List[Mapping[str, Any]]:
        """Order by usefulness for the goal, not by position on the page.

        Ranking matters because `max_elements` truncates: whatever is at the bottom is
        invisible to the model. Goal overlap outranks everything, then editable fields,
        then general links, then namespaced meta-pages ("Help:", "Special:").
        """
        def key(action: Mapping[str, Any]):
            label = str(action.get("label", "") or "").lower()
            for position, word in enumerate(self.prefer_words):
                if word.lower() in label:
                    return (0, position)
            if tokens and self._matches_goal(label, tokens):
                return (0, len(self.prefer_words))
            if action.get("kind") == "fill":
                return (1, 0)
            if re.search(r"^(help|special|wikipedia|file|template|category|talk):", label):
                return (3, 0)
            return (2, 0)

        return sorted(actions, key=key)

    def apply(self, observation: Mapping[str, Any], goal: str = "") -> Dict[str, Any]:
        """Return a copy of the observation with a scoped action list.

        Pass the goal whenever you have one: it protects a legitimate target from being
        mistaken for chrome, it puts the elements you probably want at the top, and it
        lets cross-language grounding shortlist elements whose labels are written in the
        goal's own script (see `grounding.ground_goal`).
        """
        from .grounding import ground_goal

        actions = list(observation.get("actions", []) or [])
        tokens = goal_tokens(goal) if goal else []
        kept = [action for action in actions if self.keep(action, tokens)]

        # Cross-language grounding. Measured twice on the browser checkpoint:
        #   * reordering same-script candidates first changes nothing (1/9 -> 1/9);
        #   * what works is removing the other-script distractors entirely - the model
        #     cannot bridge scripts, and a Latin or Devanagari label in the option list
        #     actively pulls the answer away from the goal's script.
        # Only applied to non-Latin goals, where the script signal is reliable. Latin is
        # left alone: English goals work fine and the fallback risk is not worth it.
        grounding: Dict[str, Any] = {}
        if goal and self.ground_non_latin:
            grounding = ground_goal(goal, kept)
            script = grounding.get("script", "")
            candidates = grounding.get("candidates", [])
            if script and script != "latin" and candidates:
                # Strict candidate filtering: keep only what the shortlist names, mapping
                # through the scores' own indices (ground_goal assigns positional indices
                # to actions that lack one, so this works whether or not the observation
                # carried explicit indices). An earlier version kept everything of the same
                # script family instead — that let a Japanese label survive a Chinese goal's
                # filter and pull the answer (measured 8/9 -> 5/9 on mixed Han pages), which
                # is exactly the regression this strict form prevents.
                keep_indices = set(candidates)
                filtered = []
                for position, action in enumerate(kept, start=1):
                    index = str(action.get("index") or position)
                    if index in keep_indices:
                        filtered.append({**action, "index": index})
                # Never filter everything out - a wrong-but-usable list beats an empty one.
                if filtered:
                    kept = filtered
                # Keep only the strong candidates in the report so "grounded" counts what
                # actually survived, not what was scored.
                grounding = {**grounding, "candidates": sorted(
                    keep_indices, key=lambda index: candidates.index(index))}
            elif script == "latin":
                # Latin goals work without grounding; drop the report so callers can tell
                # the difference between "no grounding needed" and "grounding found nothing".
                grounding = {}

        ranked = self.rank(kept, tokens)[: self.max_elements]
        # Renumber so the indices the model sees are dense and start at 1.
        renumbered = [{**action, "index": index} for index, action in enumerate(ranked, start=1)]
        scoped = {**observation, "actions": renumbered}
        scoped["scope"] = {
            "observed": len(actions),
            "kept": len(kept),
            "offered": len(renumbered),
            "dropped": len(actions) - len(renumbered),
            "goal_protected": sum(1 for action in ranked
                                  if tokens and self._matches_goal(str(action.get("label", "")), tokens)),
        }
        if grounding:
            scoped["scope"]["script"] = grounding.get("script", "")
            scoped["scope"]["grounded"] = len(grounding.get("candidates", []))
            if grounding.get("diagnosis"):
                scoped["scope"]["diagnosis"] = grounding["diagnosis"]
        return scoped


def full_page_scope(*, max_elements: int = 60) -> Scope:
    """A permissive scope for pages you already trust, or for exploring an unknown site.

    Measured trade-off: ~60 elements costs roughly 680 ms per decision and is noticeably
    more error-prone than 20, because every extra lookalike candidate is a chance to
    pick the wrong one. Use it to see the whole page once, then write a tight scope.
    """
    return Scope(max_elements=max_elements, drop_chrome=False)
