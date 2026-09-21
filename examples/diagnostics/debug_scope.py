
import sys
sys.path.insert(0, "/Volumes/SSD/localdecide")
sys.path.insert(0, "/Volumes/SSD/localdecide/tests")
from localdecide import BrowserDecider, Decider, Scope, goal_tokens
from test_contract import _SequenceBackend, FakeDriver, TestScope

t = TestScope()
obs = t.observation()
print("goal tokens:", goal_tokens("Click the 'Random article' link."))
scoped = Scope(max_elements=4).apply(obs, goal="Click the 'Random article' link.")
print("scoped actions:")
for a in scoped["actions"]:
    print("   ", a["index"], a["label"], a["kind"])
print("scope report:", scoped["scope"])

backend = _SequenceBackend([{"operation": "CLICK", "click_target": "1"}], on_exhausted={"operation": "DONE"})
driver = FakeDriver([obs])
run = BrowserDecider(decider=Decider(backend=backend), max_steps=1, scope=Scope(max_elements=4)).run(driver, "Click the 'Random article' link.")
print("\nstopped:", run.stopped, "| error:", run.error)
print("executed:", driver.executed)
for s in run.steps:
    print(f"  step {s.n}: {s.operation} target={s.target} label={s.label!r} conf={s.confidence} ok={s.executed} detail={s.detail[:80]}")
