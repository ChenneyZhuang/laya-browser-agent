#!/bin/bash
# Watch v37c: poll every 5 min. Report per-epoch wide results as they land.
# Exit on V37C_ALL_DONE.marker, process death, or ~3h timeout.
SSH="ssh -i $HOME/.ssh/<key> -o BatchMode=yes -o ConnectTimeout=15 <user>@<host>"
MAX=$((36))
for i in $(seq 1 $MAX); do
  STATE=$($SSH "wsl -e bash -c 'ls /mnt/d/v37/logs/V37C_ALL_DONE.marker 2>/dev/null && echo DONE || (pgrep -f v37c_ >/dev/null && echo RUNNING || echo DEAD)'" 2>/dev/null)
  echo "[poll $i $(date +%H:%M:%S)] $STATE"
  case "$STATE" in
    DONE)
      echo "===== V37C COMPLETE ====="
      $SSH "wsl -e bash -c 'grep -a \"wide ep\\|v37c ep\\|EPOCHS\\|rc=\" /mnt/d/v37c_out.log | tail -40'" 2>/dev/null
      exit 0 ;;
    DEAD)
      echo "===== V37C DIED ====="
      $SSH "wsl -e bash -c 'tail -30 /mnt/d/v37/logs/v37c_chain.log; echo ---; tail -20 /mnt/d/v37c_out.log'" 2>/dev/null
      exit 2 ;;
    *)
      $SSH "wsl -e bash -c 'echo -n \"ckpts: \"; ls /mnt/d/v37_ckpts/ | grep v37c | tr \"\\n\" \" \"; echo; echo -n \"wide files: \"; ls /mnt/d/v37/logs/ | grep \"wide-v37c\" | tr \"\\n\" \" \"; echo; tail -4 /mnt/d/v37c_out.log 2>/dev/null | head -4'" 2>/dev/null | sed 's/^/    /' ;;
  esac
  sleep 300
done
echo "===== WATCHER TIMEOUT ====="
exit 3
