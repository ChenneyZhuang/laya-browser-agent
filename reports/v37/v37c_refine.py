# v37c: continue v37a training — the restraint class is optimization-limited.
#
# Evidence: in-sample v37 probe after v37a's 2 epochs: BLOCKED 0->16/40,
# DONE 0->3/40 (moving), but the wide suite still shows 0/16 with gold of
# DONE/BLOCKED ranked 6-7/7 at p_gold=0.0000 — the class is losing to the CLICK
# prior, not missing data. Per the diagnosis playbook: low head LR over many
# epochs on the natural mix, NOT a higher LR.
#
# This script continues from v37a-lr1e-4 for +6 epochs at the same LR,
# checkpointing after EVERY epoch (v37c-e1 .. v37c-e6) so the wide suite can be
# evaluated as the class climbs.
import os, sys, json, time, random, shutil, torch
sys.path.insert(0, "/mnt/d/Jev-Training/vendor/laya-browser/code")
os.environ.setdefault("DISABLE_TORCH_NATIVE_BMM", "1")
_d = getattr(getattr(torch.backends, "python_native", None), "disable_operations", None)
if _d: _d("bmm")
from safetensors.torch import load_file, save_file
from laya.common import build_model, proper_reward
from transformers import AutoTokenizer

START = "/mnt/d/v37_ckpts/v37a-lr1e-4"   # continue from here
CKPT = "/mnt/d/v37_ckpts"
MIX = "/mnt/d/Jev-Training/data/generated/v37-mix-train.pt"
LR = 1e-4
EPOCHS = 8

def log(*a):
    print(*a, flush=True)
    with open("/mnt/d/v37c_out.log", "a") as f:
        print(*a, file=f, flush=True)

def collate(batch, pad_id):
    n, L = len(batch), max(len(it["ids"]) for it in batch)
    kmax = max(len(it["markers"]) for it in batch)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax))
    for i, it in enumerate(batch):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"]); att[i, :len(it["ids"])] = 1
        mpos[i, :len(it["markers"])] = torch.tensor(it["markers"]); mmask[i, :len(it["markers"])] = True
        target[i, :len(it["target"])] = torch.tensor(it["target"])
    return dict(input_ids=ids, attention_mask=att, marker_pos=mpos, marker_mask=mmask,
                target=target, qtype=torch.tensor([it["qtype"] for it in batch]))

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_ops(model, subset, tag, tok):
    model.eval(); correct = {}
    for b in range(0, len(subset), 16):
        batch = collate(subset[b:b + 16], tok.pad_token_id)
        batch = {k: v.to(DEV) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
            logits, act = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
        pi = logits.float().argmax(-1).tolist(); gi = batch["target"].argmax(-1).tolist()
        for i, it in enumerate(subset[b:b + 16]):
            g = it["gold_op"]
            correct.setdefault(g, [0, 0]); correct[g][0] += int(pi[i] == gi[i]); correct[g][1] += 1
    model.train()
    log(f"[{tag}] " + str({g: f"{correct[g][0]}/{correct[g][1]}" for g in correct}))
    return correct

def save_ckpt(model, name, start_dir):
    sd = {k: v.detach().to("cpu").clone().contiguous() for k, v in model.state_dict().items()}
    d = os.path.join(CKPT, name)
    os.makedirs(d, exist_ok=True)
    save_file(sd, os.path.join(d, "model.safetensors"))
    for sub_ in ("encoder", "tokenizer"):
        src = os.path.join(start_dir, sub_); dst = os.path.join(d, sub_)
        if not os.path.exists(dst): shutil.copytree(src, dst)
    shutil.copy(os.path.join(start_dir, "rl_agent_config.json"), os.path.join(d, "rl_agent_config.json"))
    log(f"[ckpt] saved {name}")

log(f"\n########## v37c: continue from {START}, lr={LR}, +{EPOCHS} epochs ##########")
cfg = json.load(open(START + "/rl_agent_config.json"))
cfg.update(max_len=1024, head_max_len=768)
tok = AutoTokenizer.from_pretrained(START + "/tokenizer")
base_sd = load_file(START + "/model.safetensors")
items = torch.load(MIX, weights_only=False)
log(f"[v37c] mix items: {len(items)}")
model = build_model(cfg, encoder_dir=START + "/encoder")
model.load_state_dict(base_sd, strict=True); model.to(DEV)
for p in [p for n, p in model.named_parameters() if "encoder." in n]: p.requires_grad = False
head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
for p in head_params: p.requires_grad = True
opt = torch.optim.AdamW([{"params": head_params, "lr": LR}], weight_decay=0.0)

# probes: v37 class probe + fixture guard
items37 = torch.load("/mnt/d/v37/v37-items.pt", weights_only=False)
op37 = [it for it in items37 if it.get("qid") == "operation"]
def pick37(gold, n, seed=11):
    xs = [it for it in op37 if it["gold_op"] == gold]
    random.Random(seed).shuffle(xs)
    return xs[:n]
v37_probe = (pick37("CLICK", 60) + pick37("TYPE_TEXT", 40) +
             pick37("DONE", 40) + pick37("BLOCKED", 40))
random.Random(43).shuffle(v37_probe)
items17 = torch.load("/mnt/d/Jev-Training/data/generated/v17-mix-train.pt", weights_only=False)
op17 = [it for it in items17 if it.get("qid") == "operation"]
def pick17(gold, n, seed=7):
    xs = [it for it in op17 if it["gold_op"] == gold]
    random.Random(seed).shuffle(xs)
    return xs[:n]
fixture_probe = pick17("SCROLL_DOWN", 48) + pick17("CLICK", 32) + pick17("WAIT", 8) + pick17("DONE", 8)
random.Random(42).shuffle(fixture_probe)
probes = {"fixture": fixture_probe, "v37": v37_probe}

t0 = time.time(); BS, G, sigma = 8, 4, 0.2
for ptag, psub in probes.items():
    evaluate_ops(model, psub, f"v37c BEFORE {ptag}", tok)
for epoch in range(EPOCHS):
    order = list(range(len(items))); random.Random(700 + epoch).shuffle(order)
    tot = 0.0; nb = 0
    for b in range(0, len(order), BS):
        batch = collate([items[j] for j in order[b:b + BS]], tok.pad_token_id)
        batch = {k: v.to(DEV) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV.type == "cuda")):
            logits, act = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
        logits = logits.float(); mask = batch["marker_mask"]; target = batch["target"]
        loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
        k = mask.sum(-1, keepdim=True).float()
        eps = torch.randn((G,) + logits.shape, device=logits.device) * sigma * mask
        eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
        z = logits.detach().unsqueeze(0) + eps
        q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
        with torch.no_grad():
            r = proper_reward(q, target.unsqueeze(0), batch["qtype"], mask, w_sph=0.75, w_rps=1.0)
            adv = r - r.mean(0, keepdim=True); adv = adv / (adv.std() + 1e-6)
        logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
        loss = -(adv * logp).mean() + loss_ce
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += float(loss.detach()); nb += 1
        if nb % 1000 == 0: log(f"[v37c] ep{epoch+1} step {nb} loss {float(loss.detach()):.3f} {time.time()-t0:.0f}s")
    log(f"[v37c] epoch {epoch+1} avg loss {tot/max(1,nb):.4f} {time.time()-t0:.0f}s")
    for ptag, psub in probes.items():
        evaluate_ops(model, psub, f"v37c ep{epoch+1} {ptag}", tok)
    save_ckpt(model, f"v37c-e{epoch+1}", START)
    # evaluate this epoch's checkpoint on the wide suite (the acceptance gate)
    ckdir = os.path.join(CKPT, f"v37c-e{epoch+1}")
    wide_out = f"/mnt/d/v37/logs/wide-v37c-e{epoch+1}.json"
    try:
        import subprocess
        cmd = ("cd /mnt/d/Jev-Training/vendor/laya-browser/code && "
               "export DISABLE_TORCH_NATIVE_BMM=1 PATH=\"$HOME/.local/bin:$PATH\" TOKENIZERS_PARALLELISM=false && "
               f"timeout 900 uv run --offline --no-sync python -u /mnt/d/v37/evaluate_wide.py {ckdir} "
               f"/mnt/d/v37/wide_items.jsonl --device cuda --output {wide_out}")
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=950)
        tail = "\n".join(r.stdout.strip().splitlines()[:6])
        log(f"[v37c] wide ep{epoch+1}:\n{tail}")
        if r.returncode != 0:
            log(f"[v37c] wide eval rc={r.returncode} stderr={r.stderr[-500:]}")
    except Exception as e:
        log(f"[v37c] wide eval EXC {type(e).__name__}: {e}")
log(f"[v37c] FINAL ({time.time()-t0:.0f}s)")
log("V37C_ALL_DONE")
