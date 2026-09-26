#!/usr/bin/env python3
"""Export the 95-case wide suite (handoff bench_wide.py) as wire items for the 3080.

The Mac bench feeds each case through the project's own harness:

    table = build_element_table(payload)
    questions = table_to_questions(table, goal)
    answers = backend.answer(table.state(text_chars=1500), questions)

This script reproduces that exact pipeline once, on the machine that owns the
harness, and writes one JSONL line per case so a GPU-side evaluator can feed
`state` + `questions` straight to a checkpoint.

Scoring fields per pick case:
  * `accepted`            — accepted labels (the handoff's correctness rule)
  * `expected_op_semantic`— the operation the accepted element actually supports
                            (CLICK or TYPE_TEXT; "ANY" when ambiguous). The
                            handoff's own scoring required op == "CLICK" for
                            every case, which made 8 fill-target cases (search
                            boxes, compose fields) impossible to score; the
                            semantic rule fixes that. Both are reported.
  * `index_ops`           — operations per element index, for diagnostics.

`scoreable` = reachable under the handoff's original CLICK-only rule.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_wide", HERE / "handoff" / "bench_wide.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_wide"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    sys.path.insert(0, str(REPO))
    from localdecide import build_element_table, table_to_questions

    bw = load_bench()
    out_path = HERE / "wide_items.jsonl"

    states = {}
    for name, fn in bw.STATES.items():
        url, title, text, actions = fn()
        states[name] = {"url": url, "title": title, "text": text, "actions": actions}

    lines = []
    stats = {"pick": 0, "pick_scoreable": 0, "restraint": 0, "semantic_CLICK": 0, "semantic_TYPE_TEXT": 0, "any": 0}

    for state_name, slug, goal, accepted in bw.SCENARIOS:
        info = states[state_name]
        payload = {"actions": info["actions"], "url": info["url"],
                   "title": info["title"], "text": info["text"]}
        table = build_element_table(payload)
        questions = table_to_questions(table, goal)
        state = table.state(text_chars=1500)

        by_idx = table.by_index()
        index_labels = {str(i): e.label for i, e in by_idx.items()}
        index_ops = {str(i): sorted(e.operations) for i, e in by_idx.items()}
        click_idx = set(questions.get("click_target", {}).get("criteria", {}).keys())
        accepted_hit = [i for i, e in by_idx.items() if e.label in accepted]
        scoreable = any(i in click_idx for i in accepted_hit)

        # semantic expected operation: union of ops (CLICK/TYPE_TEXT) on accepted elements
        ops: set[str] = set()
        for i in accepted_hit:
            ops.update(op for op in by_idx[i].operations if op in ("CLICK", "TYPE_TEXT"))
        if ops == {"CLICK"}:
            semantic = "CLICK"
        elif ops == {"TYPE_TEXT"}:
            semantic = "TYPE_TEXT"
        else:
            semantic = "ANY"

        stats["pick"] += 1
        stats["pick_scoreable"] += int(scoreable)
        stats[f"semantic_{semantic}" if semantic != "ANY" else "any"] += 1

        lines.append({
            "id": f"{state_name}/{slug}", "suite": "pick", "state_name": state_name,
            "slug": slug, "goal": goal, "state": state, "questions": questions,
            "accepted": sorted(accepted), "expected_operation": "CLICK",
            "expected_op_semantic": semantic,
            "index_labels": index_labels, "index_ops": index_ops,
            "accepted_indices": sorted(accepted_hit),
            "click_indices": sorted(click_idx),
            "type_indices": sorted(questions.get("type_text_target", {}).get("criteria", {}).keys()),
            "scoreable": scoreable,
        })

    for state_name, slug, goal, expected in bw.RESTRAINT:
        info = states[state_name]
        payload = {"actions": info["actions"], "url": info["url"],
                   "title": info["title"], "text": info["text"]}
        table = build_element_table(payload)
        questions = table_to_questions(table, goal)
        state = table.state(text_chars=1500)
        by_idx = table.by_index()
        index_labels = {str(i): e.label for i, e in by_idx.items()}
        index_ops = {str(i): sorted(e.operations) for i, e in by_idx.items()}
        stats["restraint"] += 1
        lines.append({
            "id": f"{state_name}/{slug}", "suite": "restraint", "state_name": state_name,
            "slug": slug, "goal": goal, "state": state, "questions": questions,
            "expected_operation": expected, "index_labels": index_labels,
            "index_ops": index_ops, "scoreable": True, "accepted": [],
        })

    with open(out_path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    summary = {
        "items": len(lines), "pick": stats["pick"], "restraint": stats["restraint"],
        "pick_scoreable_handoff_rule": stats["pick_scoreable"],
        "semantic_CLICK": stats["semantic_CLICK"], "semantic_TYPE_TEXT": stats["semantic_TYPE_TEXT"],
        "any": stats["any"],
        "out": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
