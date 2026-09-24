
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from localdecide.drivers import PlaywrightDriver
import pathlib
FIX = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
d = PlaywrightDriver(headless=True, start_url=f"file://{FIX/'element_gym.html'}")
obs = d.observe()
labels = [a["label"] for a in obs["actions"]]
print("total observed:", len(labels))
for bad in ["Hidden button", "display:none", "visibility:hidden", "Zero-size"]:
    hits = [l for l in labels if bad in l]
    print(f"  {bad:<18} leaked: {bool(hits)} {hits if hits else ''}")
print("\nvisible controls found:")
for l in labels: print("   ", l[:60])
d.close()
