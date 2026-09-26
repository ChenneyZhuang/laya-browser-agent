#!/usr/bin/env python3
"""Score a Laya checkpoint on the 95-case wide suite (wire items from export_wide_items.py).

Runs on the 3080 (CUDA). Reads one JSONL produced by the Mac-side exporter:

    {"id","suite","state_name","slug","goal","state","questions","accepted",
     "expected_operation","expected_op_semantic","index_labels","index_ops", ...}

Scores two rules side by side:
  * handoff rule  — ok = (op == "CLICK") and resolved label in accepted.
                    (The original bench's correctness rule; 8 fill-target
                    cases are unscorable under it.)
  * semantic rule — ok = (op == expected_op_semantic) and resolved label in
                    accepted. This is the honest measure for fill targets.
Restraint cases: ok = op == expected_operation (DONE / BLOCKED).

Usage:
    uv run --offline --no-sync python evaluate_wide.py <model_dir> <items.jsonl> \
        --device cuda --output <report.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch  # noqa: E402

try:
    torch.backends.python_native.disable_operations("bmm")
except Exception:
    pass

import laya  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("items")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.items, encoding="utf-8") if line.strip()]
    agent = laya.load(args.model_dir, device=args.device)
    agent.cfg["max_len"] = 4096
    agent.cfg["head_max_len"] = 1024

    rows = []
    latencies = []
    for item in items:
        questions = {}
        for qid, q in item["questions"].items():
            questions[qid] = {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}
        t0 = time.perf_counter()
        answers = agent.predict(item["state"], questions)["answers"]
        latencies.append((time.perf_counter() - t0) * 1000)

        ans = answers.get("operation") or {}
        op = ans.get("choice") if isinstance(ans, dict) else ans
        target_idx = None
        label = None
        if op in ("CLICK", "TYPE_TEXT", "SELECT"):
            qid = f"{op.lower()}_target"
            if qid in questions:
                tb = answers.get(qid) or {}
                target_idx = tb.get("choice") if isinstance(tb, dict) else tb
                if target_idx is not None:
                    label = item["index_labels"].get(str(target_idx))

        if item["suite"] == "pick":
            accepted = set(item["accepted"])
            handoff_ok = (op == "CLICK") and (label in accepted)
            exp_sem = item.get("expected_op_semantic", "CLICK")
            sem_ok = (exp_sem == "ANY" or op == exp_sem) and (label in accepted)
        else:
            handoff_ok = sem_ok = (op == item["expected_operation"])

        rows.append({
            "id": item["id"], "suite": item["suite"], "state": item["state_name"],
            "slug": item["slug"], "goal": item["goal"][:110],
            "op": op, "target": target_idx, "label": label,
            "accepted_or_expected": item["accepted"] if item["suite"] == "pick" else [item["expected_operation"]],
            "handoff_ok": bool(handoff_ok), "semantic_ok": bool(sem_ok),
        })

    def acc(rows_sub):
        n = len(rows_sub)
        if not n:
            return {}
        return {
            "n": n,
            "handoff": round(sum(r["handoff_ok"] for r in rows_sub) / n, 4),
            "semantic": round(sum(r["semantic_ok"] for r in rows_sub) / n, 4),
        }

    picks = [r for r in rows if r["suite"] == "pick"]
    restraint = [r for r in rows if r["suite"] == "restraint"]
    by_state = {s: acc([r for r in picks if r["state"] == s]) for s in sorted({r["state"] for r in picks})}

    n = len(rows)
    summary = {
        "model_dir": args.model_dir,
        "items": args.items,
        "n": n,
        "overall": acc(rows),
        "pick": acc(picks),
        "restraint": acc(restraint),
        "by_state": by_state,
        "latency_ms": {
            "p50": sorted(latencies)[n // 2] if latencies else None,
            "mean": round(sum(latencies) / n, 1) if latencies else None,
        },
        "restraint_detail": [
            {"slug": r["slug"], "want": r["accepted_or_expected"][0], "got": r["op"], "ok": r["semantic_ok"]}
            for r in restraint
        ],
        "rows": rows,
    }
    out = args.output
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"model   {args.model_dir}")
    print(f"overall {summary['overall'].get('semantic')} semantic | {summary['overall'].get('handoff')} handoff")
    print(f"pick    {summary['pick']['semantic']} semantic | {summary['pick']['handoff']} handoff  (n={summary['pick']['n']})")
    print(f"restraint {summary['restraint']['semantic']}  (n={summary['restraint']['n']})")
    for s, a in by_state.items():
        print(f"  {s:10s} {a['handoff']} handoff / {a['semantic']} semantic")
    print("restraint detail:")
    for r in summary["restraint_detail"]:
        mark = "OK " if r["ok"] else "MISS"
        print(f"  {mark} {r['slug']:34s} want={r['want']:8s} got={r['got']}")
    miss_picks = [r for r in picks if not r["semantic_ok"]]
    print(f"pick misses ({len(miss_picks)}):")
    for r in miss_picks:
        print(f"  MISS {r['slug']:32s} op={str(r['op']):9s} label={str(r['label'])[:40]!r} want={r['accepted_or_expected']}")
    opmix = Counter(r["op"] for r in picks)
    print("op mix (pick):", dict(opmix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
