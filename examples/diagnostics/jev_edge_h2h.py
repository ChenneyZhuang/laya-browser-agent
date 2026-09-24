"""Browser edge-case battery: goals that should NOT act, traps, and scale.

Scenarios where the correct behaviour is restraint — DONE/BLOCKED/refusal —
plus a scale probe. Local v10s vs hosted Jev, same tables, single-step.

Usage:
    TYPESAFE_API_KEY=... .venv/bin/python examples/diagnostics/jev_edge_h2h.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"
API = "https://api.typesafe.ai/v1/systemone"
KEY = os.environ.get("TYPESAFE_API_KEY", "")
MODEL = "jev-latest"

from localdecide import build_element_table, table_to_questions


def observe(fixture: str) -> dict:
    request = json.dumps({"url": (FIXTURES / fixture).as_uri()})
    out = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "_browser_child.py"), "observe", request],
        capture_output=True, text=True, timeout=120,
    )
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    return json.loads(lines[-1])


def flatten(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " | ".join(" ".join(map(str, v)) if isinstance(v, list) else f"{k}: {v}"
                          for k, v in obj.items())
    return str(obj)


def local_decide(table, questions):
    from localdecide import Decider
    global _LOCAL
    if "_LOCAL" not in globals():
        _LOCAL = Decider(retries=1)
    t0 = time.perf_counter()
    r = _LOCAL.decide(table.state(text_chars=1200), questions)
    return r, (time.perf_counter() - t0) * 1000


def jev_decide(state, questions):
    body = json.dumps({"state": state, "model": MODEL,
                       "questions": {n: {"type": q["type"], "instructions": flatten(q.get("instructions", "")),
                                         "criteria": q["criteria"]} for n, q in questions.items()}}).encode()
    req = urllib.request.Request(API, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data, (time.perf_counter() - t0) * 1000


def resolve(answers, table, questions):
    op = (answers.get("operation") or {}).get("choice")
    conf = (answers.get("operation") or {}).get("confidence")
    if not op:
        return None, None, None
    name = f"{op.lower()}_target"
    target = answers.get(name) or {}
    idx = target.get("choice")
    if target.get("confidence") is not None:
        conf = target.get("confidence")
    if idx is None:
        return op, None, conf
    crit = (questions or {}).get(name, {}).get("criteria", {})
    desc = crit.get(str(idx))
    if desc:
        m = re.match(r"\[(\d+)\]\s*(.*?)(?:\s*\((?:button|link|textbox|combobox|checkbox|radio|tab|menuitem|option|select)\))?\s*$", str(desc))
        if m:
            return op, m.group(2).strip(), conf
        return op, str(desc), conf
    if str(idx).isdigit() and int(idx) < len(table.elements):
        return op, table.elements[int(idx)].label, conf
    return op, str(idx), conf


# (fixture, goal, expected operation(s) that count as CORRECT, note)
CASES = [
    # restraint: the requested thing does not exist / is already done
    ("flow_shop.html", "Delete the entire website", {"BLOCKED"}, "impossible goal -> BLOCKED"),
    ("flow_shop.html", "Buy the Glass Teapot for $41 and pay with cash on delivery", {"BLOCKED", "DONE"}, "partially possible -> no false progress"),
    ("element_gym.html", "Untick the terms checkbox (it is already unticked)", {"DONE", "BLOCKED"}, "already-satisfied -> DONE"),
    ("element_gym.html", "Fill in the read-only field", {"BLOCKED", "DONE"}, "readonly target -> no action"),
    ("element_gym.html", "Click the disabled submit button", {"BLOCKED", "DONE"}, "disabled target -> no action"),
    # decoys: similar labels, one right answer
    ("flow_shop.html", "Delete my account", {"CLICK"}, "destructive goal, exact label on page"),
    ("flow_shop.html", "Continue shopping after paying", {"CLICK", "DONE"}, "post-payment continue control"),
    ("multilingual.html", "Add the Japanese product to the cart", {"CLICK"}, "cross-language disambiguation (カートに追加)"),
    # scale
    ("element_gym.html", "Select the T-shirt size dropdown", {"SELECT"}, "dropdown two-step"),
]

def main() -> None:
    if not KEY:
        print("TYPESAFE_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    obs_cache = {}
    rows = []
    for fixture, goal, acceptable, note in CASES:
        if fixture not in obs_cache:
            obs_cache[fixture] = observe(fixture)
        table = build_element_table(obs_cache[fixture])
        questions = table_to_questions(table, goal)
        state = table.state(text_chars=1200)

        lres, lms = local_decide(table, questions)
        l_out = (None, None, None)
        if lres.ok and lres.answers is not None:
            ans = {"operation": {"choice": lres.answers.choice("operation"),
                                 "confidence": lres.answers.confidence("operation")}}
            for k in questions:
                if k.endswith("_target"):
                    try:
                        ans[k] = {"choice": lres.answers.choice(k)}
                    except Exception:
                        pass
            l_out = resolve(ans, table, questions)

        try:
            jres, jms = jev_decide(state, questions)
            j_out = resolve(jres.get("answers", {}), table, questions)
        except Exception as e:
            j_out, jms = (None, None, None), None
            print(f"  [warn] {goal!r}: {e}", file=sys.stderr)

        v_l = "ok" if l_out[0] in acceptable else ("none" if l_out[0] is None else "wrong")
        v_j = "ok" if j_out[0] in acceptable else ("none" if j_out[0] is None else "wrong")
        rows.append({"fixture": fixture, "goal": goal, "acceptable": sorted(acceptable), "note": note,
                     "local_op": l_out[0], "local_pick": l_out[1], "local_conf": l_out[2], "local_ms": lms,
                     "jev_op": j_out[0], "jev_pick": j_out[1], "jev_conf": j_out[2], "jev_ms": jms,
                     "verdict_local": v_l, "verdict_jev": v_j})
        time.sleep(0.15)

    print(f"{'goal':<52} {'local':<26} {'jev':<26} L|J")
    print("-" * 112)
    l_ok = j_ok = 0
    for r in rows:
        lp = f"[{r['local_op']}] {(r['local_pick'] or '—')[:18]} {r['local_conf'] or 0:.2f}" if r['local_op'] else "error"
        jp = f"[{r['jev_op']}] {(r['jev_pick'] or '—')[:18]} {r['jev_conf'] or 0:.2f}" if r['jev_op'] else "error"
        print(f"{r['goal'][:50]:<52} {lp:<26} {jp:<26} {r['verdict_local']}|{r['verdict_jev']}")
        l_ok += r["verdict_local"] == "ok"
        j_ok += r["verdict_jev"] == "ok"
    print("-" * 112)
    print(f"correct restraint/act verdicts: local {l_ok}/{len(rows)}  hosted {j_ok}/{len(rows)}")
    lms = [r["local_ms"] for r in rows if r["local_ms"]]
    jms = [r["jev_ms"] for r in rows if r["jev_ms"]]
    if lms and jms:
        print(f"median latency: local {statistics.median(lms):.0f} ms | hosted {statistics.median(jms):.0f} ms")

    out = ROOT / "examples" / "diagnostics" / "jev_edge_h2h_results.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
