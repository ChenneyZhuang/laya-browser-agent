#!/usr/bin/env python3
"""Export the 95 wide cases with BOTH state layouts, for a factorized probe.

The wide_items.jsonl was exported with the deploy state layout (`v3`: page +
recent_actions only, ~1.5k chars of text). Training items for our checkpoints
carried the `v1` layout instead (same page + recent_actions, PLUS the element
table as JSON inside the state). This script re-runs the same pipeline and,
for every case, writes the v1-layout state alongside the existing v0 record so
the GPU probe can separate "state layout" from "option rendering" as causes.

Output: wide_items_rich.jsonl  (same records + "state_rich" field)
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
    src = HERE / "wide_items.jsonl"
    items = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]

    states = {}
    for name, fn in bw.STATES.items():
        url, title, text, actions = fn()
        states[name] = {"actions": actions, "url": url, "title": title, "text": text}

    rich_by_state = {}
    for name, info in states.items():
        table = build_element_table(info)
        state = table.state(text_chars=1500, layout="v1")  # v1 = elements inside state
        rich_by_state[name] = state

    n_rich = 0
    with open(HERE / "wide_items_rich.jsonl", "w", encoding="utf-8") as fh:
        for item in items:
            item = dict(item)
            item["state_rich"] = rich_by_state[item["state_name"]]
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_rich += 1

    # sanity: check one
    first = json.loads(open(HERE / "wide_items_rich.jsonl", encoding="utf-8").readline())
    el = first["state_rich"].get("elements", [])
    print(json.dumps({
        "items": n_rich,
        "rich_states": len(rich_by_state),
        "elements_in_first_rich_state": len(el),
        "sample_element": el[0] if el else None,
        "lean_keys": sorted(first["state"].keys()),
        "rich_keys": sorted(first["state_rich"].keys()),
    }, ensure_ascii=False, indent=1)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
