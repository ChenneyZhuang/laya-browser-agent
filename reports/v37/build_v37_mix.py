# Build the v37 mix: v18c + v23 + v31-noul + v37-items.
# Full-retention composition mirroring the v31-mix recipe that produced the
# v32 champions, with the new v37 deployment patch added.
import torch, random
from collections import Counter

G = "/mnt/d/Jev-Training/data/generated"
i18 = torch.load(G + "/v18c-mix-train.pt", weights_only=False)
i23 = torch.load("/mnt/d/Jev-Training/data/v23/v23-items.pt", weights_only=False)
i31n = torch.load("/mnt/d/Jev-Training/data/v31/v31-noul-items.pt", weights_only=False)
i37 = torch.load("/mnt/d/v37/v37-items.pt", weights_only=False)
print("sizes:", len(i18), len(i23), len(i31n), len(i37), flush=True)

mix = i18 + i23 + i31n + i37
random.Random(53).shuffle(mix)
torch.save(mix, G + "/v37-mix-train.pt")

ops = Counter(it.get("gold_op") for it in mix if it.get("qid") == "operation")
qids = Counter(it.get("qid") for it in mix)
print("v37-mix:", len(mix), "qids:", dict(qids), flush=True)
print("operation golds:", {k: v for k, v in sorted(ops.items())}, flush=True)
print("BUILD_V37_MIX_DONE", flush=True)
