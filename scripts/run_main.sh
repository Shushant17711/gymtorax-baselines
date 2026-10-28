#!/usr/bin/env bash
# Full preregistered pipeline: tune classical -> evaluate references -> main sweep. Logs in logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
echo "[$(date +%T)] tuning classical"; uv run python scripts/tune_classical.py > logs/tune_classical.log 2>&1
echo "[$(date +%T)] references"
uv run python scripts/evaluate.py --reference random   > logs/ref_random.log 2>&1
uv run python scripts/evaluate.py --reference scripted > logs/ref_scripted.log 2>&1
uv run python scripts/evaluate.py --reference classical --params configs/agent/classical.yaml > logs/ref_classical.log 2>&1
echo "[$(date +%T)] sweep"; uv run python scripts/sweep.py configs/sweep/main.yaml
echo "[$(date +%T)] all done"
