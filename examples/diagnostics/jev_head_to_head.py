"""Head-to-head: official hosted Jev (jev-1.13.0) vs local Laya v10s.

Same element tables, same goals, same wire contract — two engines.
The question dict from `table_to_questions` is already almost Jev-shaped:
{"name": {"type": "choice", "criteria": {...}, "instructions": ...}} —
the only fix-ups are flattening instructions to a string and scoring
no-target operations (BLOCKED/DONE/WAIT) as valid verdicts.

Usage:
    TYPESAFE_API_KEY=... .venv/bin/python examples/diagnostics/jev_head_to_head.py
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

from localdecide import build_element_table, table_to_questions  # noqa: E402


def observe(fixture: str) -> dict:
    """Observation via the same subprocess browser the live tests use."""
    request = json.dumps({"url": (FIXTURES / fixture).as_uri()})
    out = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "_browser_child.py"), "observe", request],
        capture_output=True, text=True, timeout=120,
    )
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    return json.loads(lines[-1])


def flatten(obj):
    """Instructions arrive as {goal, rules} dicts; Jev takes a string."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            if isinstance(v, list):
                parts.append(" ".join(str(x) for x in v))
            else:
                parts.append(f"{k}: {v}")
        return " | ".join(parts)
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
    body = json.dumps({
        "state": state,
        "model": MODEL,
        "questions": {
            name: {
                "type": q["type"],
                "instructions": flatten(q.get("instructions", "")),
                "criteria": q["criteria"],
            }
            for name, q in questions.items()
        },
    }).encode()
    req = urllib.request.Request(API, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data, (time.perf_counter() - t0) * 1000


def resolve(answers, table, questions=None):
    """(op, label|None, conf) — mapping the raw choice back to a real label.

    The choice is a criteria KEY (the element's table index, which is NOT the
    list position — keys are sparse). The label lives in the criteria VALUE
    ("[26] 🔐 Log in (button)"), so parse it from there, falling back to
    elements[int(key)] when the key is in range.
    """
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
    key = str(idx)
    desc = crit.get(key)
    if desc:                                    # canonical path: parse "[26] label (role)"
        m = re.match(r"\[(\d+)\]\s*(.*?)(?:\s*\((?:button|link|textbox|combobox|checkbox|radio|tab|menuitem|option|select)\))?\s*$",
                     str(desc))
        if m:
            return op, m.group(2).strip(), conf
        return op, str(desc), conf
    if key.isdigit() and int(key) < len(table.elements):
        return op, table.elements[int(key)].label, conf
    # model echoed a value instead of a key: reverse lookup
    for k, d in crit.items():
        if key and (key in str(d) or str(d) in key):
            m = re.match(r"\[(\d+)\]\s*(.*?)\s*$", str(d))
            if m and m.group(1).isdigit() and int(m.group(1)) < len(table.elements):
                return op, table.elements[int(m.group(1))].label, conf
    return op, key, conf


# (fixture, goal, regex over the fixture's REAL labels, required operation or None)
CASES = [
    ("multilingual.html", "Click the login button (zh)", r"^登录$", "CLICK"),
    ("multilingual.html", "Log in (English)", r"Log in$", "CLICK"),
    ("multilingual.html", "Search for products (ja)", r"商品を検索", "CLICK"),
    ("multilingual.html", "Найдите кнопку входа (ru)", r"Войти$", "CLICK"),
    ("multilingual.html", "Iniciar sesión (ar)", r"تسجيل الدخول", "CLICK"),
    ("multilingual.html", "ログインボタンをクリック", r"^ログイン$", "CLICK"),
    ("multilingual.html", "Add to cart", r"加入购物车|カートに追加|장바구니|Add to basket|أضف", "CLICK"),
    ("element_gym.html", "Click the search button", r"Search", "CLICK"),
    ("element_gym.html", "Accept the terms", r"Terms", "CLICK"),
    ("element_gym.html", "Type the username into its field", r"Full name", "TYPE_TEXT"),  # no username field exists; Full name is the nearest text field
    ("element_gym.html", "Select a country from the dropdown", r"Country", "SELECT"),
    ("flow_shop.html", "Proceed to checkout", r"Checkout", "CLICK"),
]


def main() -> None:
    if not KEY:
        print("TYPESAFE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    obs_cache: dict[str, dict] = {}
    rows = []
    for fixture, goal, expect_re, want_op in CASES:
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
            print(f"  [warn] hosted call failed on {goal!r}: {e}", file=sys.stderr)

        def verdict(out):
            op, label, _ = out
            if op is None:
                return "error"
            if want_op and op != want_op:
                return "wrong-op"
            if label is None:
                return "no-target"
            return "HIT" if re.search(expect_re, str(label), re.I) else "miss"

        v_l, v_j = verdict(l_out), verdict(j_out)
        rows.append({
            "fixture": fixture, "goal": goal, "expect_re": expect_re, "want_op": want_op,
            "n_elements": len(table.elements),
            "local_op": l_out[0], "local_pick": l_out[1], "local_conf": l_out[2], "local_ms": lms,
            "jev_op": j_out[0], "jev_pick": j_out[1], "jev_conf": j_out[2], "jev_ms": jms,
            "verdict_local": v_l, "verdict_jev": v_j,
        })
        time.sleep(0.15)

    print(f"{'fixture':<17} {'goal':<40} {'local (v10s)':<34} {'hosted Jev':<34} L|J")
    print("-" * 132)
    for r in rows:
        lp = f"[{r['local_op']}] {(r['local_pick'] or '—')}"[:32]
        jp = f"[{r['jev_op']}] {(r['jev_pick'] or '—')}"[:32]
        print(f"{r['fixture']:<17} {r['goal']:<40} {lp:<34} {jp:<34} {r['verdict_local']}|{r['verdict_jev']}")
    print("-" * 132)
    l_hits = sum(1 for r in rows if r["verdict_local"] == "HIT")
    j_hits = sum(1 for r in rows if r["verdict_jev"] == "HIT")
    print(f"strict element hits: local {l_hits}/{len(rows)}   hosted {j_hits}/{len(rows)}")
    lms = [r["local_ms"] for r in rows if r["local_ms"]]
    jms = [r["jev_ms"] for r in rows if r["jev_ms"]]
    if lms and jms:
        print(f"latency: local median {statistics.median(lms):.0f} ms | hosted median {statistics.median(jms):.0f} ms")
    lcs = [r["local_conf"] for r in rows if isinstance(r["local_conf"], (int, float))]
    jcs = [r["jev_conf"] for r in rows if isinstance(r["jev_conf"], (int, float))]
    if lcs and jcs:
        print(f"mean confidence: local {statistics.mean(lcs):.2f} | hosted {statistics.mean(jcs):.2f}")
    agree = sum(1 for r in rows
                if r["local_pick"] and r["jev_pick"] and str(r["local_pick"]) == str(r["jev_pick"]))
    print(f"same-element agreement: {agree}/{len(rows)}")

    out = ROOT / "examples" / "diagnostics" / "jev_head_to_head_results.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
