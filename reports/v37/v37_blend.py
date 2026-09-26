# v37 blends: replicate the v32 recipe — low-alpha head blend of the patch
# checkpoint into the v24a-b06 base (the step that produced all champions).
#   v37a-blends: v24a-b06 x v37a (lr 1e-4, 2ep) at alpha 0.03 / 0.08 / 0.15
#   v37b-blends: v24a-b06 x v37b (lr 5e-5, 1ep) at alpha 0.03 / 0.08 / 0.15
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

V24 = Path("/mnt/d/v24_blends/v24a-b06")
V37A = Path("/mnt/d/v37_ckpts/v37a-lr1e-4")
V37B = Path("/mnt/d/v37_ckpts/v37b-lr5e-5")
OUT = Path("/mnt/d/v37_blends")
OUT.mkdir(parents=True, exist_ok=True)


def blend_heads(ckpt_x: Path, ckpt_y: Path, alpha: float, out_dir: Path):
    sx = load_file(str(ckpt_x / "model.safetensors"))
    sy = load_file(str(ckpt_y / "model.safetensors"))
    assert set(sx.keys()) == set(sy.keys()), "key mismatch"
    mixed = {}
    for k in sx:
        if "encoder." in k:
            mixed[k] = sx[k].clone()
        else:
            mixed[k] = ((1.0 - alpha) * sx[k].float() + alpha * sy[k].float()).to(sx[k].dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(mixed, str(out_dir / "model.safetensors"))
    for sub in ("encoder", "tokenizer"):
        dst = out_dir / sub
        if not dst.exists():
            shutil.copytree(ckpt_x / sub, dst)
    shutil.copy(ckpt_x / "rl_agent_config.json", out_dir / "rl_agent_config.json")
    print(f"wrote {out_dir.name}: alpha={alpha}", flush=True)


for alpha in (0.03, 0.08, 0.15):
    blend_heads(V24, V37A, alpha, OUT / f"v37a-b{int(alpha*100):02d}")
for alpha in (0.03, 0.08, 0.15):
    blend_heads(V24, V37B, alpha, OUT / f"v37b-b{int(alpha*100):02d}")
print("V37_BLENDS_DONE", flush=True)
