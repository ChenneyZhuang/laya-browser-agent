"""Does grounding actually improve the multilingual results, or just look tidy?

Runs the same failing cases as `multilingual_accuracy.py`, with and without the grounding
shortlist, and prints both so the claim can be checked rather than believed.
"""

from __future__ import annotations

from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

import pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parents[2]
def _fx(name: str) -> str:
    return (_REPO / "tests" / "fixtures" / name).as_uri()

URL = _fx("multilingual.html")

CASES = [
    ("点击标着“登录”的按钮。", "登录"),
    ("Click the button labelled '登录'.", "登录"),
    ("点击“加入购物车”按钮。", "加入购物车"),
    ("「カートに追加」をクリック。", "カートに追加"),
    ("장바구니에 담기 버튼을 클릭하세요.", "장바구니에 담기"),
    ("Нажмите кнопку «Войти».", "Войти"),
    ("اضغط على زر تسجيل الدخول.", "تسجيل الدخول"),
    ("Κάντε κλικ στο κουμπί «Σύνδεση».", "Σύνδεση"),
    ("คลิกปุ่มเข้าสู่ระบบ", "เข้าสู่ระบบ"),
]

driver = PlaywrightDriver(headless=True, start_url=URL)
observation = driver.observe()
driver.close()
decider = Decider()

print(f"{'goal':<36} {'grounding':<10} {'offered':<8} {'chose':<22} {'p':<7} result")
print("-" * 96)

wins = {"plain": 0, "grounded": 0}
total = 0

for goal, want in CASES:
    row = {}
    for mode in ("plain", "grounded"):
        scoped = Scope(max_elements=25, ground_non_latin=(mode == "grounded"))
        applied = scoped.apply(observation, goal=goal)
        table = build_element_table(applied)
        questions = table_to_questions(table, goal)
        result = decider.decide(table.state(text_chars=1200), questions)
        if not result.ok:
            row[mode] = (False, f"FAILED: {result.error}", 0.0, "-", len(table.elements))
            continue
        assert result.answers is not None
        operation = result.answers.choice("operation")
        key = f"{operation.lower()}_target"
        chosen = result.answers.choice(key) if key in result.answers.raw else "-"
        element = table.by_index().get(chosen)
        probability = result.answers.probabilities(key).get(chosen, 0.0) if key in result.answers.raw else 0.0
        hit = bool(element and want.lower() in element.label.lower())
        row[mode] = (hit, element.label[:20] if element else "-", probability, chosen, len(table.elements))
        wins[mode] += 1 if hit else 0

    total += 1
    for mode in ("plain", "grounded"):
        hit, shown, probability, chosen, offered = row[mode]
        tag = f"{goal[:34]:<36}" if mode == "plain" else f"{'':<36}"
        print(f"{tag} {mode:<10} {offered:<8} {shown:<22} {probability:<7.3f} {'HIT' if hit else 'miss'}")

print(f"\ntotals: plain {wins['plain']}/{total}   grounded {wins['grounded']}/{total}")
