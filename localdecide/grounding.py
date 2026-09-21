"""Cross-language grounding: when the model cannot read the script, help it with code.

Measured: the browser checkpoint picks the right element when the goal's own characters
appear in the label (`点击"登录"` -> the 登录 button, HIT) and fails when they do not
(`「カートに追加」をクリック` -> wrong element; p≈0.06, i.e. no signal at all). The
multilingual checkpoint was no better in this harness.

That is a solvable problem *outside* the model. A goal in one language and a label in
another can be connected by code before the decision is made:

* Unicode-script detection tells you which candidates are in the goal's script.
* When candidates share the goal's script, keyword overlap almost always contains the
  target, so the shortlist can be pre-ranked and the model only has to confirm.
* When they do not overlap at all, the honest answer is "translate this first" - and the
  harness can say so instead of letting the model guess at p=0.06.

This module does the first two. It is deliberately small and readable: a translation
layer belongs in the caller's stack, not buried here.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# Unicode script blocks we can name. Enough to separate the scripts real sites use; not a
# full Unicode database, and deliberately so - an unknown script returns "" and the caller
# falls back to plain textual matching.
_SCRIPT_RANGES: Sequence[Tuple[str, int, int]] = (
    ("latin", 0x0041, 0x024F),
    ("latin", 0x1E00, 0x1EFF),
    ("greek", 0x0370, 0x03FF),
    ("cyrillic", 0x0400, 0x04FF),
    ("hebrew", 0x0590, 0x05FF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("devanagari", 0x0900, 0x097F),
    ("thai", 0x0E00, 0x0E7F),
    ("hangul", 0xAC00, 0xD7AF),
    ("hangul", 0x1100, 0x11FF),
    ("kana", 0x3040, 0x30FF),
    ("han", 0x3400, 0x4DBF),
    ("han", 0x4E00, 0x9FFF),
    ("han", 0xF900, 0xFAFF),
)

# Han and kana co-occur in Japanese text, so they are treated as one family for the
# "same script" test. Hangul stands alone; Chinese characters inside Japanese text are
# still han, which is why the family matters.
_SCRIPT_FAMILIES = {
    "han": {"han", "kana"},
    "kana": {"han", "kana"},
}

# Scripts written without spaces between words. Splitting on whitespace destroys them, so
# they get character-level matching instead.
_UNSPACED_SCRIPTS = {"han", "kana", "hangul", "thai"}


def script_of(text: str) -> str:
    """The dominant script of a string, or "" if it is punctuation/digits only.

    Counting characters rather than testing the first one matters: a label like
    `[3] ログイン (button)` mixes Latin furniture with the real content, and the content is
    what identifies the target.
    """
    counts: Dict[str, int] = {}
    for character in text or "":
        code = ord(character)
        if character.isspace() or unicodedata.category(character).startswith(("P", "N", "S")):
            continue
        for name, start, end in _SCRIPT_RANGES:
            if start <= code <= end:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return ""
    return max(counts, key=lambda name: counts[name])


def same_script(left: str, right: str) -> bool:
    """True when two strings are written in the same script family."""
    left_script, right_script = script_of(left), script_of(right)
    if not left_script or not right_script:
        return False
    family = _SCRIPT_FAMILIES.get(left_script, {left_script})
    return right_script in family


def _code_units(text: str) -> List[str]:
    """Matching units for a string: words for spaced scripts, characters for unspaced ones."""
    script = script_of(text)
    if script in _UNSPACED_SCRIPTS:
        return [character for character in (text or "")
                if not character.isspace() and not unicodedata.category(character).startswith(("P", "N"))]
    return [word for word in re.findall(r"[^\W_]+", (text or "").lower()) if len(word) > 1]


def overlap_score(goal: str, label: str) -> float:
    """Fraction of the goal's content that appears in the label.

    Character-level for unspaced scripts: with no spaces, "加入购物车按钮" shares most of its
    characters with the label "加入购物车", so overlap is a usable signal. Word-level
    elsewhere. Returns 0 when the two are written in scripts that share nothing.
    """
    if not goal or not label:
        return 0.0
    if not same_script(goal, label):
        return 0.0
    goal_units = _code_units(goal)
    if not goal_units:
        return 0.0
    label_text = label if script_of(label) in _UNSPACED_SCRIPTS else label.lower()
    label_units = set(_code_units(label))
    hits = 0
    for unit in goal_units:
        if unit in label_text or unit in label_units or any(unit in existing or existing in unit
                                                           for existing in label_units if len(existing) > 1):
            hits += 1
    return hits / len(goal_units)


def ground_goal(goal: str, actions: Sequence[Mapping[str, Any]], *, floor: float = 0.34) -> Dict[str, Any]:
    """Shortlist the elements whose labels share the goal's language, best first.

    Returns a dict describing what was found, so a caller can decide what to do:
    `candidates` (indices, ranked), `script`, and a `diagnosis` explaining a shortfall.
    Nothing here replaces the model - it feeds it a smaller, cleaner option list.

    Elements without an explicit `index` are assigned one from their position, because
    raw observations frequently omit it and a silent `None` key would make the whole
    shortlist unusable.
    """
    script = script_of(goal)
    scored: List[Tuple[float, str]] = []
    for position, action in enumerate(actions, start=1):
        label = str(action.get("label", "") or "")
        score = overlap_score(goal, label)
        if score > 0:
            scored.append((score, str(action.get("index") or position)))
    scored.sort(key=lambda pair: -pair[0])
    strong = [(score, index) for score, index in scored if score >= floor]

    diagnosis = ""
    if not script:
        diagnosis = "goal has no identifiable script"
    elif not scored:
        diagnosis = (f"no element label is written in the goal's script ({script}); "
                     "the model cannot cross scripts on its own - translate the goal, "
                     "or ask with the label text")
    elif not strong:
        best = scored[0][0] if scored else 0.0
        diagnosis = (f"only weak matches (best overlap {best:.2f}); "
                     "the goal may be phrased very differently from the labels")

    return {
        "script": script,
        "candidates": [index for _, index in (strong or scored)],
        "scores": {index: round(score, 2) for score, index in scored},
        "diagnosis": diagnosis,
    }
