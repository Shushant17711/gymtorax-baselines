#!/usr/bin/env bash
# Amendment 1 pipeline (capped-Q reward). Waits for the main sweep unit, then:
# tune classical -> evaluate references -> capped sweep. Everything under results/capped/. Logs in logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
while systemctl --user is-active --quiet gtb-sweep; do sleep 60; done
echo "[$(date +%T)] tuning classical (capped)"
uv run python scripts/tune_classical.py --q-cap 10 --workers 14 --out configs/agent/classical_capped.yaml > logs/tune_classical_capped.log 2>&1
echo "[$(date +%T)] references (capped)"
for r in random scripted; do
  uv run python scripts/evaluate.py --reference $r --q-cap 10 --root results/capped --workers 14 > logs/ref_${r}_capped.log 2>&1
done
uv run python scripts/evaluate.py --reference classical --params configs/agent/classical_capped.yaml --q-cap 10 \
  --root results/capped --workers 14 > logs/ref_classical_capped.log 2>&1
echo "[$(date +%T)] sweep (capped)"
uv run python scripts/sweep.py configs/sweep/capped.yaml --root results/capped
echo "[$(date +%T)] all done"
