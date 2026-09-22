"""Multi-step head-to-head: full task flows, local v10s vs hosted Jev.

The child process drives a real Chromium through BrowserDecider's loop; every
decision round-trips to THIS process, where the engine answers:

  engine=local  -> the local MLX checkpoint (Decider)
  engine=jev    -> TypeSafe production /v1/systemone over HTTP

Same goals, same pages, same loop/guards/scoping for both engines. This is the
regime v10s was fine-tuned for — history, scoping, multi-step state.

Usage:
    TYPESAFE_API_KEY=... .venv/bin/python examples/diagnostics/jev_flow_h2h.py [local|jev|both]
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

FLOWS = [
    {
        "name": "shop: search -> add kettle -> checkout",
        "fixture": "flow_shop.html",
        "goal": "Search for a kettle and add the Steel Kettle to the cart, then proceed to checkout",
        "text_for": {"Search products": "kettle"},
        "max_steps": 8,
        "expect_labels": ["Steel Kettle", "Checkout"],
    },
    {
        "name": "shop: search -> add cheapest (Mug Set) -> checkout",
        "fixture": "flow_shop.html",
        "goal": "Search for products and add the cheapest item to the cart, then checkout",
        "text_for": {"Search products": "tea"},
        "max_steps": 8,
        "expect_labels": ["Mug Set", "Checkout"],
    },
    {
        "name": "shop: checkout -> pay (card flow)",
        "fixture": "flow_shop.html",
        "goal": "Checkout and pay for the order with the card",
        "text_for": {"card": "4242424242424242"},
        "max_steps": 8,
        "expect_labels": ["Pay now"],
    },
    {
        "name": "gym: fill name -> accept terms",
        "fixture": "element_gym.html",
        "goal": "Fill in the full name field and accept the terms checkbox",
        "text_for": {"Type your name": "Chenney Zhuang"},
        "max_steps": 8,
        "expect_labels": ["Full name", "Terms accepted"],
    },
    {
        "name": "gym: search the docs",
        "fixture": "element_gym.html",
        "goal": "Search the documentation for deployment guides",
        "text_for": {"Search the docs": "deployment guides"},
        "max_steps": 6,
        "expect_labels": ["Search the docs"],
    },
    {
        "name": "multi: 中文打开帮助中心",
        "fixture": "multilingual.html",
        "goal": "打开帮助中心",
        "text_for": {},
        "max_steps": 5,
        "expect_labels": ["帮助中心"],
    },
]


def flatten(obj):
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


class LocalEngine:
    name = "local v10s"

    def __init__(self):
        from localdecide import Decider
        self.decider = Decider(retries=1)

    def answer(self, state, questions):
        t0 = time.perf_counter()
        r = self.decider.decide(state, questions)
        if not r.ok or r.answers is None:
            return {"ok": False, "error": r.error or "decision failed"}
        return {"ok": True, "answers": r.answers.raw,
                "latency_ms": (time.perf_counter() - t0) * 1000, "backend": self.name}


class JevEngine:
    name = "hosted jev-1.13.0"

    def answer(self, state, questions):
        body = json.dumps({
            "state": state, "model": MODEL,
            "questions": {n: {"type": q["type"], "instructions": flatten(q.get("instructions", "")),
                              "criteria": q["criteria"]} for n, q in questions.items()},
        }).encode()
        req = urllib.request.Request(API, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {KEY}",
                                              "Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        answers = {}
        for n, a in data.get("answers", {}).items():
            if a.get("type") == "choice":
                entry = {"choice": a.get("choice")}
            else:
                entry = {"noul": a.get("noul")}
            if a.get("confidence") is not None:
                entry["confidence"] = a["confidence"]
            answers[n] = entry
        return {"ok": True, "answers": answers,
                "latency_ms": (time.perf_counter() - t0) * 1000, "backend": self.name}


def run_one(engine: str, flow: dict) -> dict:
    request = json.dumps({
        "url": (FIXTURES / flow["fixture"]).as_uri(),
        "goal": flow["goal"],
        "text_for": flow["text_for"],
        "max_steps": flow["max_steps"],
        "remote_decide": True,
        "max_elements": 25,
        "confirm": True,
    })
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tests" / "_browser_child.py"), "run", request],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None
    engine_impl = LocalEngine() if engine == "local" else JevEngine()
    latencies: list[float] = []
    result = None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("want") != "decide":
                if "solved" in msg:
                    result = msg
                continue
            reply = engine_impl.answer(msg["state"], msg["questions"])
            latencies.append(reply.get("latency_ms", 0))
            proc.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        if result is None and proc.stdout is not None:
            rest = proc.stdout.read() or ""
            for line in rest.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        msg = json.loads(line)
                        if "solved" in msg:
                            result = msg
                    except json.JSONDecodeError:
                        pass
        if proc.poll() is None:
            proc.kill()
    steps = (result or {}).get("steps", [])
    solved = bool((result or {}).get("solved"))
    labels = [s.get("label") or "" for s in steps]
    touched = all(any(re.search(re.escape(exp), ln, re.I) for ln in labels)
                  for exp in flow["expect_labels"])
    return {
        "engine": engine_impl.name, "flow": flow["name"],
        "solved": solved,
        "error": (result or {}).get("error"),
        "steps": [{"n": s.get("n"), "op": s.get("op"), "label": (s.get("label") or "")[:36]} for s in steps],
        "expect": flow["expect_labels"],
        "touched_all": touched,
        "hit": solved or touched,
        "median_ms": round(statistics.median(latencies)) if latencies else None,
        "n_decisions": len(latencies),
    }


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    engines = ["local", "jev"] if which == "both" else [which]
    if "jev" in engines and not KEY:
        print("TYPESAFE_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    out = []
    for flow in FLOWS:
        for engine in engines:
            t0 = time.perf_counter()
            result = run_one(engine, flow)
            result["wall_s"] = round(time.perf_counter() - t0, 1)
            out.append(result)
            steps_str = " > ".join(f"{s['op']}:{s['label'][:18]}" for s in result["steps"][:5])
            print(f"[{result['engine']:<15}] {result['flow'][:44]:<44} "
                  f"hit={result['hit']!s:<5} solved={result['solved']!s:<5} "
                  f"steps={result['n_decisions']} {result['median_ms']}ms  {steps_str}")
            time.sleep(0.2)
    (ROOT / "examples" / "diagnostics" / "jev_flow_h2h_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    # summary
    for engine in engines:
        rows = [r for r in out if r["engine"].startswith("local" if engine == "local" else "hosted")]
        hits = sum(1 for r in rows if r["hit"])
        solved = sum(1 for r in rows if r["solved"])
        ms = [r["median_ms"] for r in rows if r["median_ms"]]
        print(f"\n== {engine}: flows hit {hits}/{len(rows)}, loop-solved {solved}/{len(rows)}, "
              f"median decision {statistics.median(ms) if ms else 0:.0f} ms")


if __name__ == "__main__":
    main()
