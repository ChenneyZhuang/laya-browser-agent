#!/bin/bash
# Run the live suite in small batches, checking memory between each one.
#
# Why: a full live run puts a model checkpoint and a Chromium on the machine at once. On
# a 16 GB box that is enough to trigger a kernel watchdog panic and reboot. Batching keeps
# the peak low, and the memory check aborts before the machine is under real pressure.

set -u
cd /Volumes/SSD/localdecide || exit 1

BATCHES=(
  "TestObservationReader"
  "TestDecisionsOnRealElements"
  "TestMultilingual"
  "TestShadowDOM"
  "TestHiddenMenus"
  "TestRTLAndExtendedScripts"
  "TestGroundingMultiLanguage"
  "TestFlows"
)

mem_free_percent() {
  # "System-wide memory free percentage" from memory_pressure is the cheapest honest signal
  memory_pressure 2>/dev/null | awk -F': ' '/System-wide memory free percentage/{gsub(/%/,"",$2); print $2}' | head -1
}

swap_used_mb() {
  sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}' | tr -d 'M'
}

for batch in "${BATCHES[@]}"; do
  free=$(mem_free_percent)
  free=${free:-100}
  if [ "$free" -lt 25 ]; then
    echo "ABORT before $batch: only ${free}% memory free (refusing to risk a reboot)"
    exit 2
  fi
  echo "=== $batch (memory free: ${free}%, swap: $(swap_used_mb)MB) ==="
  .venv/bin/python -m pytest "tests/test_live.py::$batch" -q --tb=line -p no:cacheprovider 2>&1 | tail -18
  # Let the batch's processes and memory settle before the next one.
  sleep 4
done

echo "=== memory after the run ==="
memory_pressure 2>/dev/null | tail -3
