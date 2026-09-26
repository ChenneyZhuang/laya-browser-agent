#!/bin/bash
# v37 chain: build mix -> train v37a/v37b -> blend -> evaluate battery.
# Idempotent: every step skips when its output exists. One log, markers per step.
exec > /mnt/d/v37/logs/v37_chain.log 2>&1
echo "=== V37 CHAIN start $(date) ==="
cd /mnt/d/Jev-Training/vendor/laya-browser/code
export DISABLE_TORCH_NATIVE_BMM=1
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/mnt/d/Jev-Training
PY=$D/vendor/laya-browser/code/.venv/bin/python
LOG=/mnt/d/v37/logs
mkdir -p $LOG

# ── step 1: build mix ─────────────────────────────────────────────────────────
if [ ! -f $D/data/generated/v37-mix-train.pt ]; then
  echo "--- build v37 mix ---"
  $PY /mnt/d/v37/build_v37_mix.py
  echo "mix rc=$?"
else
  echo "--- mix exists, skip ---"
fi

# ── step 2: train ─────────────────────────────────────────────────────────────
if [ ! -d /mnt/d/v37_ckpts/v37b-lr5e-5 ]; then
  echo "--- train v37a/v37b ---"
  $PY /mnt/d/v37/v37_refine.py
  echo "train rc=$?"
else
  echo "--- ckpts exist, skip ---"
fi

# ── step 3: blends ────────────────────────────────────────────────────────────
if [ ! -d /mnt/d/v37_blends/v37b-b15 ]; then
  echo "--- blends ---"
  $PY /mnt/d/v37/v37_blend.py
  echo "blend rc=$?"
else
  echo "--- blends exist, skip ---"
fi

# ── step 4: eval battery (wide suite first: the acceptance gate) ──────────────
EVAL=$D/vendor/laya-browser/code/.venv/bin/python
for pair in \
  "v37a-solo:/mnt/d/v37_ckpts/v37a-lr1e-4" \
  "v37b-solo:/mnt/d/v37_ckpts/v37b-lr5e-5" \
  "v37a-b03:/mnt/d/v37_blends/v37a-b03" "v37a-b08:/mnt/d/v37_blends/v37a-b08" "v37a-b15:/mnt/d/v37_blends/v37a-b15" \
  "v37b-b03:/mnt/d/v37_blends/v37b-b03" "v37b-b08:/mnt/d/v37_blends/v37b-b08" "v37b-b15:/mnt/d/v37_blends/v37b-b15" \
; do
  TAG=${pair%%:*}; M=${pair#*:}
  [ -d "$M" ] || { echo "MISSING $TAG $M"; continue; }
  [ -f $LOG/wide-$TAG.json ] && { echo "wide-$TAG exists, skip"; continue; }
  echo "##### wide $TAG #####"
  timeout 600 uv run --offline --no-sync python -u /mnt/d/v37/evaluate_wide.py "$M" /mnt/d/v37/wide_items.jsonl \
    --device cuda --output $LOG/wide-$TAG.json >> $LOG/wide-$TAG.stdout 2>&1
  echo "wide $TAG rc=$?"
done
echo "--- wide battery done $(date) ---"
touch /mnt/d/v37/logs/wide_done.marker

# ── step 5: regression battery (v5 suite + holdout) ───────────────────────────
for pair in \
  "v37a-b03:/mnt/d/v37_blends/v37a-b03" "v37a-b08:/mnt/d/v37_blends/v37a-b08" "v37a-b15:/mnt/d/v37_blends/v37a-b15" \
  "v37b-b03:/mnt/d/v37_blends/v37b-b03" "v37b-b08:/mnt/d/v37_blends/v37b-b08" "v37b-b15:/mnt/d/v37_blends/v37b-b15" \
; do
  TAG=${pair%%:*}; M=${pair#*:}
  [ -d "$M" ] || continue
  OUTD=$D/reports/generated/$TAG-eval
  mkdir -p $OUTD
  [ -f "$OUTD/browser-suite-v5.json" ] || {
    echo "##### suite5 $TAG #####"
    timeout 400 $EVAL $D/scripts/evaluate_browser_suite.py $M $D/reports/generated/browser-suite-v5.jsonl \
      --device cuda --output "$OUTD/browser-suite-v5.json" > $OUTD/suite5.stdout 2>&1
    echo "suite5 $TAG rc=$?"
  }
  [ -f "$OUTD/recovery2-holdout.json" ] || {
    echo "##### holdout $TAG #####"
    timeout 420 $EVAL $D/scripts/evaluate_laya_records.py $D/data/generated/recovery2-holdout.jsonl $M \
      --device cuda --max-len 1024 --head-max-len 768 \
      --output "$OUTD/recovery2-holdout.json" > $OUTD/holdout.stdout 2>&1
    echo "holdout $TAG rc=$?"
  }
done
echo "--- regression battery done $(date) ---"
touch /mnt/d/v37/logs/reg_done.marker

echo "=== V37 CHAIN DONE $(date) ==="
touch /mnt/d/v37/logs/V37_CHAIN_DONE.marker
