#!/bin/bash
# Poll the 3080 for the v37 chain result. Exit when:
#   - V37_CHAIN_DONE.marker appears (success), or
#   - the chain process died without the marker (incident), or
#   - max wall time (~4h) reached.
# Echoes a compact progress digest every poll so the tail is readable.
SSH="ssh -i $HOME/.ssh/<key> -o BatchMode=yes -o ConnectTimeout=15 <user>@<host>"
MAX_POLLS=120
for i in $(seq 1 $MAX_POLLS); do
  STATE=$($SSH "wsl -e bash -c 'if [ -f /mnt/d/v37/logs/V37_CHAIN_DONE.marker ]; then echo DONE; else if pgrep -f v37_chain >/dev/null; then echo RUNNING; else echo DEAD; fi; fi'" 2>/dev/null)
  echo "[poll $i $(date +%H:%M:%S)] $STATE"
  case "$STATE" in
    DONE)
      echo "===== CHAIN COMPLETE ====="
      $SSH "wsl -e bash -c 'tail -60 /mnt/d/v37/logs/v37_chain.log'" 2>/dev/null
      exit 0
      ;;
    DEAD)
      echo "===== CHAIN DIED WITHOUT MARKER ====="
      $SSH "wsl -e bash -c 'tail -80 /mnt/d/v37/logs/v37_chain.log'" 2>/dev/null
      exit 2
      ;;
    *)
      # progress digest: last meaningful lines of the chain log + refine log
      $SSH "wsl -e bash -c 'tail -3 /mnt/d/v37/logs/v37_chain.log; echo ---; tail -4 /mnt/d/v37r_out.log 2>/dev/null'" 2>/dev/null | sed 's/^/    /'
      ;;
  esac
  sleep 120
done
echo "===== WATCHER TIMEOUT ====="
exit 3
