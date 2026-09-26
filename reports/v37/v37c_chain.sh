#!/bin/bash
# v37c chain: continue training from v37a for +6 epochs (low LR, many epochs —
# the diagnosis-playbook fix for a class stuck at 0 due to optimization distance).
# Each epoch saves a checkpoint AND evaluates the wide suite so the class climb
# is observable. Then a final regression battery on the best epochs.
exec > /mnt/d/v37/logs/v37c_chain.log 2>&1
echo "=== V37C CHAIN start $(date) ==="
cd /mnt/d/Jev-Training/vendor/laya-browser/code
export DISABLE_TORCH_NATIVE_BMM=1
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/mnt/d/Jev-Training
PY=$D/vendor/laya-browser/code/.venv/bin/python
LOG=/mnt/d/v37/logs

if [ ! -f $LOG/V37C_ALL_DONE.marker ]; then
  echo "--- train v37c (+6 epochs from v37a) ---"
  $PY /mnt/d/v37/v37c_refine.py
  echo "v37c train rc=$?"
else
  echo "--- v37c already done, skip ---"
fi

# regression battery for the epochs that look best on wide (by hand pick after
# inspecting wide-v37c-e*.json; here: run all six so nothing is missed)
EVAL=$D/vendor/laya-browser/code/.venv/bin/python
for e in 1 2 3 4 5 6; do
  M=/mnt/d/v37_ckpts/v37c-e$e
  [ -d "$M" ] || continue
  OUTD=$D/reports/generated/v37c-e$e-eval
  mkdir -p $OUTD
  [ -f "$OUTD/browser-suite-v5.json" ] || {
    echo "##### suite5 v37c-e$e #####"
    timeout 400 $EVAL $D/scripts/evaluate_browser_suite.py $M $D/reports/generated/browser-suite-v5.jsonl \
      --device cuda --output "$OUTD/browser-suite-v5.json" > $OUTD/suite5.stdout 2>&1
    echo "suite5 v37c-e$e rc=$?"
  }
  [ -f "$OUTD/recovery2-holdout.json" ] || {
    echo "##### holdout v37c-e$e #####"
    timeout 420 $EVAL $D/scripts/evaluate_laya_records.py $D/data/generated/recovery2-holdout.jsonl $M \
      --device cuda --max-len 1024 --head-max-len 768 \
      --output "$OUTD/recovery2-holdout.json" > $OUTD/holdout.stdout 2>&1
    echo "holdout v37c-e$e rc=$?"
  }
done
echo "--- regression battery done $(date) ---"
touch $LOG/V37C_ALL_DONE.marker
echo "=== V37C CHAIN DONE $(date) ==="
