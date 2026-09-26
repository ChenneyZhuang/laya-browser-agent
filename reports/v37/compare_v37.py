#!/usr/bin/env python3
"""Render the v37 comparison table from the 3080 eval logs.

Usage (on the 3080, from /mnt/d/v37):
    python compare_v37.py

Reads: logs/wide-*.json (wide suite), reports/generated/*-eval/browser-suite-v5.json,
       reports/generated/*-eval/recovery2-holdout.json.
Prints: pick/restraint/overall table for baselines vs every v37 candidate, plus
        the per-case flips vs v10s (wins and losses), so the winner is obvious.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

LOGS = pathlib.Path("/mnt/d/v37/logs")
D = pathlib.Path("/mnt/d/Jev-Training")

BASELINES = {
    "v10s (upstream)": LOGS / "v10s-wide-baseline.json",
    "v32b (published)": LOGS / "wide-v32b.json",
    "v24a-b06 (base)": LOGS / "wide-v24a.json",
}
CANDIDATES = [
    "wide-v37a-solo.json", "wide-v37b-solo.json",
    "wide-v37a-b03.json", "wide-v37a-b08.json", "wide-v37a-b15.json",
    "wide-v37b-b03.json", "wide-v37b-b08.json", "wide-v37b-b15.json",
]
TAG2CKPT = {
    "v37a-b03": "/mnt/d/v37_blends/v37a-b03", "v37a-b08": "/mnt/d/v37_blends/v37a-b08",
    "v37a-b15": "/mnt/d/v37_blends/v37a-b15", "v37b-b03": "/mnt/d/v37_blends/v37b-b03",
    "v37b-b08": "/mnt/d/v37_blends/v37b-b08", "v37b-b15": "/mnt/d/v37_blends/v37b-b15",
}


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def wide_row(tag, rep):
    if not rep:
        return None
    picks = [r for r in rep["rows"] if r["suite"] == "pick"]
    res = [r for r in rep["rows"] if r["suite"] == "restraint"]
    return {
        "pick_handoff": sum(r["handoff_ok"] for r in picks),
        "pick_sem": sum(r["semantic_ok"] for r in picks),
        "n_pick": len(picks),
        "restraint": sum(r["semantic_ok"] for r in res),
        "n_res": len(res),
        "handoff_total": sum(r["handoff_ok"] for r in rep["rows"]),
        "sem_total": sum(r["semantic_ok"] for r in rep["rows"]),
        "n": len(rep["rows"]),
    }


def fmt_w(w) -> str:
    return (f"{w['pick_handoff']:3d}/{w['n_pick']:<3d} {w['pick_sem']:3d}/{w['n_pick']:<3d} "
            f"{w['restraint']:2d}/{w['n_res']:<2d} {w['handoff_total']:3d}/{w['n']:<2d} {w['sem_total']:3d}/{w['n']:<2d}")


def suite5(tag):
    p = D / f"reports/generated/{tag}-eval/browser-suite-v5.json"
    rep = load(p)
    return rep["accuracy"] if rep else None


def holdout(tag):
    p = D / f"reports/generated/{tag}-eval/recovery2-holdout.json"
    rep = load(p)
    return rep.get("accuracy") if rep else None


def main() -> int:
    print(f"{'model':26s} {'pick-h':>7s} {'pick-s':>7s} {'res':>5s} {'95-h':>6s} {'95-s':>6s} {'v5':>7s} {'hold':>7s}")
    base_rows = {}
    for name, p in BASELINES.items():
        rep = load(p)
        if rep is None:
            print(f"{name:26s}  (no report)")
            continue
        w = wide_row(name, rep)
        if w is None:
            continue
        base_rows[name] = rep
        print(f"{name:26s} {fmt_w(w)}")

    print()
    for fname in CANDIDATES:
        rep = load(LOGS / fname)
        if rep is None:
            continue
        tag = fname.replace("wide-", "").replace(".json", "")
        w = wide_row(tag, rep)
        if w is None:
            continue
        v5 = suite5(tag) if tag in TAG2CKPT else None
        ho = holdout(tag) if tag in TAG2CKPT else None
        v5s = f"{v5:.4f}" if v5 is not None else "-"
        hos = f"{ho:.4f}" if ho is not None else "-"
        print(f"{tag:26s} {fmt_w(w)} {v5s:>7s} {hos:>7s}")

    # flips vs v10s for the best candidate (by sem_total)
    cands = []
    for fname in CANDIDATES:
        rep = load(LOGS / fname)
        if rep is None:
            continue
        tag = fname.replace("wide-", "").replace(".json", "")
        w = wide_row(tag, rep)
        if w is not None:
            cands.append((w["sem_total"], tag, rep))
    if cands:
        cands.sort(reverse=True)
        _, tag, rep = cands[0]
        v10 = base_rows.get("v10s (upstream)")
        if v10:
            v10map = {r["id"]: r for r in v10["rows"]}
            print(f"\n=== flips vs v10s for best candidate: {tag} ===")
            wins, losses = [], []
            for r in rep["rows"]:
                b = v10map.get(r["id"])
                if b is None:
                    continue
                if r["semantic_ok"] and not b["semantic_ok"]:
                    wins.append(r)
                elif b["semantic_ok"] and not r["semantic_ok"]:
                    losses.append(r)
            print(f"wins ({len(wins)}):")
            for r in wins:
                print(f"  + {r['id']:44s} now {r['op']}->{str(r['label'])[:40]!r}")
            print(f"losses ({len(losses)}):")
            for r in losses:
                bb = v10map.get(r["id"]) or {}
                print(f"  - {r['id']:44s} v10s {bb.get('op')}->{str(bb.get('label'))[:40]!r} vs ours {r['op']}->{str(r['label'])[:40]!r}")
            # restraint detail
            print("restraint (best):")
            for r in [x for x in rep["rows"] if x["suite"] == "restraint"]:
                mark = "OK " if r["semantic_ok"] else "MISS"
                print(f"  {mark} {r['slug']:34s} got={r['op']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
