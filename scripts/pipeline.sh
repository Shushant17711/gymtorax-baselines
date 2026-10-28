#!/usr/bin/env bash
# The whole study, in order. Every stage skips work that is already finished, and every training job
# resumes from its checkpoint, so this script can be stopped (scripts/gtb pause) and rerun at any time.
#   1. original reward:  tune classical, reference controllers, main sweep           -> results/
#   2. capped reward (PREREGISTRATION amendment 1): the same                         -> results/capped/
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PAUSED=75
say() { echo "[$(date '+%F %T')] $*"; }
check_pause() { if [ -e PAUSE ]; then say "paused"; exit $PAUSED; fi; }
run() {  # run a stage; stop the pipeline if it paused or failed
  "$@"; local rc=$?
  if [ $rc -eq $PAUSED ]; then say "paused"; exit $PAUSED; fi
  if [ $rc -ne 0 ]; then say "stage failed (exit $rc): $*"; exit $rc; fi
}

stage() {  # stage <tag> <q-cap or ""> <classical yaml> <results root> <sweep yaml>
  local tag=$1 cap=$2 cls=$3 root=$4 sweep=$5 capargs=()
  [ -n "$cap" ] && capargs=(--q-cap "$cap")
  check_pause
  if [ ! -e "$cls" ]; then
    say "[$tag] tuning classical controller"
    run uv run python scripts/tune_classical.py "${capargs[@]}" --workers 14 --out "$cls" >> "logs/tune_$tag.log" 2>&1
  fi
  for r in random scripted classical; do
    check_pause
    if [ ! -e "$root/reference/$r/eval.json" ]; then
      say "[$tag] reference: $r"
      extra=(); [ $r = classical ] && extra=(--params "$cls")
      run uv run python scripts/evaluate.py --reference $r "${extra[@]}" "${capargs[@]}" --root "$root" \
        --workers 14 >> "logs/ref_${r}_$tag.log" 2>&1
    fi
  done
  check_pause
  say "[$tag] sweep $sweep"
  run uv run python scripts/sweep.py "$sweep" --root "$root"
}

say "pipeline start"
stage original "" configs/agent/classical.yaml results configs/sweep/main.yaml
stage capped 10 configs/agent/classical_capped.yaml results/capped configs/sweep/capped.yaml
say "ALL DONE"
