#!/usr/bin/env bash
# One-off: hand over from the old (non-checkpointing) gtb-sweep unit to the new pipeline.
# Waits for the running PI-MBRL seed 1 to finish (or a pause request), then stops gtb-sweep and starts the pipeline.
cd "$(dirname "$0")/.."
say() { echo "[$(date '+%F %T')] switch: $*"; }
while systemctl --user is-active --quiet gtb-sweep; do
  if [ -e PAUSE ]; then say "pause requested"; break; fi
  if ls results/pimbrl/*/seed1/eval.json >/dev/null 2>&1; then say "pimbrl seed1 finished"; break; fi
  sleep 15
done
systemctl --user stop gtb-sweep
say "old sweep stopped"
if [ -e PAUSE ]; then say "paused; run scripts/gtb resume to continue"; exit 0; fi
systemctl --user reset-failed gtb-pipeline 2>/dev/null
systemd-run --user --unit=gtb-pipeline --collect -p MemoryMax=19G -p MemorySwapMax=2G \
  -p WorkingDirectory="$PWD" -E PATH="$HOME/.local/bin:/usr/bin:/bin" -E HOME="$HOME" \
  bash -c 'scripts/pipeline.sh >> logs/pipeline.out 2>&1'
say "pipeline started"
