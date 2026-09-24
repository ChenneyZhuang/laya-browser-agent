"""Text-classification head-to-head: local v10s vs hosted Jev on real business texts.

Same Decider question contract, two engines, three task families:

1. pool-lead triage (English business enquiries; pool industry vs not,
   lead quality score 0-4) — the user's actual production use case.
2. Chinese SMS triage (transaction vs not, expense/income/transfer, phishing)
   — multilingual + safety-relevant.
3. Robustness probes (empty state, 5000-char state, adversarial wording).

Usage:
    TYPESAFE_API_KEY=... .venv/bin/python examples/diagnostics/jev_text_h2h.py
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

API = "https://api.typesafe.ai/v1/systemone"
KEY = os.environ.get("TYPESAFE_API_KEY", "")
MODEL = "jev-latest"

from localdecide import Decider, choice, noul, score

# ── task 1: pool-lead triage (English) ───────────────────────────────────────
POOL_CASES = [
    # (text, is_pool_expected, quality_floor 0-4 or None for not-pool)
    ("Hi, my pool pump stopped working and there's water everywhere. I need someone today. Do you service Crestwood?", True, 3),
    ("Looking for a quote on a new concrete pool, about 8m, backyard is 12m wide.", True, 3),
    ("Do you sell pool tablets in bulk? We run a motel with two pools.", True, 2),
    ("My neighbour's pool fence is broken, can you repair glass fencing?", True, 2),
    ("Hi, I need my car's coolant flushed. Do you do radiators?", False, None),
    ("We install solar panels. Are you interested in a commercial quote?", False, None),
    ("Looking for a puppy grooming appointment this weekend.", False, None),
    ("URGENT: your domain is about to expire, click here to renew.", False, None),
    ("", False, None),  # empty state robustness
    ("泳池水泵坏了，需要今天上门维修，在Crestwood附近。", True, 2),  # Chinese pool lead
]

POOL_QUESTIONS = {
    "relevant": noul("Is this business enquiry about swimming pools, spa, or pool equipment?"),
    "lead_quality": score("Rate this as a sales lead for a pool service company.",
                          ["not relevant", "weak", "moderate", "strong", "excellent"]),
    "category": choice("Which category fits best?", {
        "service": "Repairs, maintenance, cleaning",
        "construction": "New pools, renovations",
        "supplies": "Chemicals, equipment, parts",
        "not_pool": "Not a pool enquiry at all",
    }),
}

# ── task 2: Chinese SMS triage ───────────────────────────────────────────────
SMS_CASES = [
    # (text, is_txn, txn_type, is_phishing)
    ("【招商银行】您尾号1234的账户于09月22日10:01支出人民币299.00元，余额10,234.56元", True, "expense", False),
    ("【支付宝】转账收款：已收到来自张*的转账 ￥500.00", True, "income", False),
    ("妈妈，我手机坏了，这个是新号码，急用钱，快转5000到这个卡号6222...", True, "expense", True),
    ("【工商银行】尊敬的客户，您已成功办理定期存款业务", True, "other", False),
    ("今天中午一起吃饭吗？老地方。", False, None, False),
    ("【京东】您的订单已发货，预计明天送达", False, None, False),
    ("恭喜您被抽中幸运用户，点击链接领取888元现金红包！", False, None, True),
    ("【建设银行】您尾号8888的信用卡账单已出，最低还款额1,200元", True, "expense", False),
]

SMS_QUESTIONS = {
    "is_transaction": noul("这是一条与资金变动有关的交易短信吗？"),
    "txn_type": choice("资金变动类型是？", {
        "expense": "支出/消费/还款",
        "income": "收入/收款/退款",
        "transfer": "转账（转出或转入）",
        "other": "其他资金相关",
    }),
    "is_phishing": noul("这条短信有钓鱼或诈骗嫌疑吗？"),
}

# ── task 3: robustness probes ────────────────────────────────────────────────
def robustness_cases():
    long_state = ("Quarterly report: " + "revenue grew 3.2%% quarter over quarter. " * 180)  # ~5.4k chars
    return [
        ("empty state", "", POOL_QUESTIONS, {"relevant": False}),
        ("5000+ char state", long_state, POOL_QUESTIONS, {"relevant": False}),
        ("adversarial: pool wording on a non-pool business",
         "Our pool of candidates is deep. We staff accounting roles. Pool table sales also welcome.",
         POOL_QUESTIONS, {"relevant": False}),
    ]


def flatten_local(questions):
    """local Decider expects raw question dicts; nothing to flatten."""
    return questions


def flatten_jev(questions):
    out = {}
    for name, q in questions.items():
        if q["type"] == "choice":
            crit = q["criteria"]
            if isinstance(crit, dict):
                # already a map of option->desc
                out[name] = {"type": "choice", "instructions": _flat(q["instructions"]), "criteria": crit}
            else:
                out[name] = {"type": "choice", "instructions": _flat(q["instructions"]),
                             "criteria": {c: c for c in crit}}
        elif q["type"] == "score":
            out[name] = {"type": "score", "instructions": _flat(q["instructions"]),
                         "criteria": list(q["criteria"])}
        else:
            out[name] = {"type": "noul", "instructions": _flat(q["instructions"])}
    return out


def _flat(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " | ".join(" ".join(map(str, v)) if isinstance(v, list) else f"{k}: {v}"
                          for k, v in obj.items())
    return str(obj)


class LocalEngine:
    name = "local v10s"

    def __init__(self):
        self.decider = Decider(retries=1)

    def decide(self, state, questions):
        t0 = time.perf_counter()
        r = self.decider.decide({"text": state}, questions)
        ms = (time.perf_counter() - t0) * 1000
        if not r.ok or r.answers is None:
            return {"ok": False, "error": r.error, "ms": ms}
        out = {"ok": True, "ms": ms, "answers": {}}
        for name, q in questions.items():
            try:
                if q["type"] == "choice":
                    out["answers"][name] = {"value": r.answers.choice(name),
                                            "conf": r.answers.confidence(name)}
                elif q["type"] == "score":
                    raw = r.answers.raw.get(name, {})
                    out["answers"][name] = {"value": raw.get("score"), "conf": raw.get("confidence")}
                else:
                    out["answers"][name] = {"p": r.answers.noul(name)}
            except Exception as e:
                out["answers"][name] = {"error": str(e)[:80]}
        return out


class JevEngine:
    name = "hosted jev-1.13.0"

    def decide(self, state, questions):
        body = json.dumps({"state": state, "model": MODEL,
                           "questions": flatten_jev(questions)}).encode()
        req = urllib.request.Request(API, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {KEY}",
                                              "Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        ms = (time.perf_counter() - t0) * 1000
        out = {"ok": True, "ms": ms, "answers": {}}
        for name, a in data.get("answers", {}).items():
            if a.get("type") == "choice":
                out["answers"][name] = {"value": a.get("choice"), "conf": a.get("confidence")}
            elif a.get("type") == "score":
                out["answers"][name] = {"value": a.get("score"), "conf": a.get("confidence")}
            else:
                out["answers"][name] = {"p": a.get("noul")}
        return out


def score_family(rows, family):
    """Aggregate correctness per family."""
    correct = total = 0
    notes = []
    for r in rows:
        exp = r.get("expect", {})
        got = {k: v.get("value", v.get("p")) for k, v in r["local"]["answers"].items()} if r["local"]["ok"] else {}
        got_j = {k: v.get("value", v.get("p")) for k, v in r["jev"]["answers"].items()} if r["jev"]["ok"] else {}
        for key, want in exp.items():
            if key not in got and key not in got_j:
                continue
            total += 1
            l_ok = _match(got.get(key), want, key)
            j_ok = _match(got_j.get(key), want, key)
            correct += int(l_ok) + int(j_ok)
            notes.append((r["case"][:28], key, want, got.get(key), got_j.get(key), l_ok, j_ok))
    return correct, total, notes


def _match(got, want, key):
    if got is None:
        return False
    if isinstance(want, bool):
        p = got if isinstance(got, float) else None
        return bool(p is not None and (p >= 0.5) == want)
    if key == "lead_quality" and isinstance(want, int):
        try:
            return int(round(float(got))) >= want
        except (TypeError, ValueError):
            return False
    if isinstance(want, str):
        return str(got).strip().lower() == want.lower()
    return False


def main() -> None:
    if not KEY:
        print("TYPESAFE_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    local, jev = LocalEngine(), JevEngine()
    all_rows = []

    print("== POOL LEADS (English + one Chinese) ==")
    for text, exp_pool, exp_q in POOL_CASES:
        state = text or "(empty message)"
        l = local.decide(state, POOL_QUESTIONS)
        j = jev.decide(state, POOL_QUESTIONS)
        row = {"family": "pool", "case": text[:50] or "(empty)", "expect": {"relevant": exp_pool},
               "local": l, "jev": j}
        if exp_pool and exp_q is not None:
            row["expect"]["lead_quality"] = exp_q
        if not exp_pool:
            row["expect"]["category"] = "not_pool"
        all_rows.append(row)
        lr = l["answers"].get("relevant", {}) if l["ok"] else {}
        jr = j["answers"].get("relevant", {}) if j["ok"] else {}
        lc = l["answers"].get("category", {}).get("value", "-") if l["ok"] else "ERR"
        jc = j["answers"].get("category", {}).get("value", "-") if j["ok"] else "ERR"
        print(f"  {text[:44]!r:<48} local p={lr.get('p','-')} cat={lc} | jev p={jr.get('p','-')} cat={jc}")
        time.sleep(0.12)

    print("== CHINESE SMS ==")
    for text, exp_txn, exp_type, exp_phish in SMS_CASES:
        l = local.decide(text, SMS_QUESTIONS)
        j = jev.decide(text, SMS_QUESTIONS)
        expect = {"is_transaction": exp_txn, "is_phishing": exp_phish}
        if exp_type:
            expect["txn_type"] = exp_type
        all_rows.append({"family": "sms", "case": text[:36], "expect": expect, "local": l, "jev": j})
        lt = l["answers"].get("is_transaction", {}) if l["ok"] else {}
        jt = j["answers"].get("is_transaction", {}) if j["ok"] else {}
        lph = l["answers"].get("is_phishing", {}).get("p", "-") if l["ok"] else "-"
        jph = j["answers"].get("is_phishing", {}).get("p", "-") if j["ok"] else "-"
        print(f"  {text[:30]!r:<34} local txn={lt.get('p','-')} phish={lph} | jev txn={jt.get('p','-')} phish={jph}")
        time.sleep(0.12)

    print("== ROBUSTNESS ==")
    for name, state, questions, expect in robustness_cases():
        l = local.decide(state, questions)
        j = jev.decide(state, questions)
        all_rows.append({"family": "robust", "case": name, "expect": expect, "local": l, "jev": j})
        print(f"  {name:<44} local ok={l['ok']} | jev ok={j['ok']}")

    # aggregate
    for fam in ("pool", "sms", "robust"):
        rows = [r for r in all_rows if r["family"] == fam]
        l_ok = sum(1 for r in rows if r["local"]["ok"])
        j_ok = sum(1 for r in rows if r["jev"]["ok"])
        lms = [r["local"]["ms"] for r in rows if r["local"]["ok"]]
        jms = [r["jev"]["ms"] for r in rows if r["jev"]["ok"]]
        print(f"\n{fam}: engines ok local {l_ok}/{len(rows)}, jev {j_ok}/{len(rows)}; "
              f"median ms local {statistics.median(lms):.0f} vs jev {statistics.median(jms):.0f}"
              if lms and jms else f"{fam}: local {l_ok}/{len(rows)} jev {j_ok}/{len(rows)}")

    c, t, notes = score_family(all_rows, "all")
    print(f"\ncorrectness on labelled fields: local+jev combined {c}/{t*2}")
    print(f"{'case':<30} {'field':<15} {'want':<12} {'local':<22} {'jev':<22} L|J")
    for case, key, want, lg, jg, l_ok, j_ok in notes:
        print(f"{case:<30} {key:<15} {str(want):<12} {str(lg)[:20]:<22} {str(jg)[:20]:<22} {int(l_ok)}|{int(j_ok)}")

    out = ROOT / "examples" / "diagnostics" / "jev_text_h2h_results.json"
    out.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
