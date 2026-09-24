"""Why does the Chinese goal fail, and does a different checkpoint fix it?

The browser checkpoint (cklxx/laya-browser v10s) is built on mmBERT-base, which claims
100+ language coverage. The position-bias sweep showed it failing on a Chinese goal while
nailing every English one. Before tuning anything, find out whether this is:

  (a) the checkpoint cannot read Chinese at all -> needs the multilingual checkpoint, or
  (b) the goal phrasing is unusual -> fixable in how we ask, or
  (c) the element labels are genuinely ambiguous -> fixable in how we observe.

The test: same page, same elements, ask in Chinese, English, and a literal label copy, on
both the browser checkpoint and the multilingual one.
"""

from __future__ import annotations

import json

from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

import pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parents[2]
def _fx(name: str) -> str:
    return (_REPO / "tests" / "fixtures" / name).as_uri()

URL = _fx("multilingual.html")

# (goal, expected label substring, language tag)
CASES = [
    ("点击标着“登录”的按钮。", "登录", "zh-goal"),
    ("Click the button labelled '登录'.", "登录", "en-goal-zh-label"),
    ("登录", "登录", "bare-label"),
    ("点击“加入购物车”按钮。", "加入购物车", "zh-cart"),
    ("「カートに追加」をクリック。", "カートに追加", "ja"),
    ("장바구니에 담기 버튼을 클릭하세요.", "장바구니에 담기", "ko"),
    ("Нажмите кнопку «Войти».", "Войти", "ru"),
    ("اضغط على زر تسجيل الدخول.", "تسجيل الدخول", "ar"),
    ("Κάντε κλικ στο κουμπί «Σύνδεση».", "Σύνδεση", "el"),
    ("คลิกปุ่มเข้าสู่ระบบ", "เข้าสู่ระบบ", "th"),
]

driver = PlaywrightDriver(headless=True, start_url=URL)
observation = driver.observe()
driver.close()

CHECKPOINTS = {
    "browser (v10s)": "browser",
    "multilingual": ("convaiinnovations/laya", "multilingual"),
}

print(f"{'language':<20} {'checkpoint':<16} {'chose':<28} {'p':<7} result")
print("-" * 88)

for label, checkpoint in CHECKPOINTS.items():
    try:
        decider = Decider(backend="laya-mlx", model=checkpoint) if isinstance(checkpoint, str) \
            else Decider(backend="laya-mlx", model=checkpoint[0], subfolder=checkpoint[1])
    except Exception as error:
        print(f"{'':<20} {label:<16} could not load: {error}")
        continue

    for goal, want, language in CASES:
        table = build_element_table(Scope(max_elements=25).apply(observation, goal=goal))
        questions = table_to_questions(table, goal)
        result = decider.decide(table.state(text_chars=1200), questions)
        if not result.ok:
            print(f"{language:<20} {label:<16} FAILED: {result.error}")
            continue
        assert result.answers is not None
        operation = result.answers.choice("operation")
        key = f"{operation.lower()}_target"
        chosen = result.answers.choice(key) if key in result.answers.raw else "-"
        element = table.by_index().get(chosen)
        probability = result.answers.probabilities(key).get(chosen, 0.0) if key in result.answers.raw else 0.0
        hit = "HIT " if element and want.lower() in element.label.lower() else "miss"
        shown = (element.label[:26] if element else "-")
        print(f"{language:<20} {label:<16} {shown:<28} {probability:<7.3f} {hit}")
    print()
