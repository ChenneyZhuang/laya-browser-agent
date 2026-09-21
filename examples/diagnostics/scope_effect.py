
import time
from localdecide import Decider, Scope, build_element_table, table_to_questions
from localdecide.drivers import PlaywrightDriver

d = Decider()
drv = PlaywrightDriver(headless=True, start_url="https://en.wikipedia.org/wiki/Main_Page")
obs = drv.observe()
print(f"observed: {len(obs['actions'])} elements")

for name, scope in [("default Scope()", Scope()),
                    ("tight (10)", Scope(max_elements=10)),
                    ("no scoping", None)]:
    o = scope.apply(obs) if scope else obs
    t = build_element_table(o)
    q = table_to_questions(t, "Open the Contents page.")
    t0 = time.time(); r = d.decide(t.state(text_chars=1200), q); dt = (time.time()-t0)*1000
    info = o.get("scope", {})
    print(f"{name:<16} offered={len(t.elements):<4} {dt:6.0f} ms  kept={info.get('kept','-')}/{info.get('observed','-')}")
drv.close()
