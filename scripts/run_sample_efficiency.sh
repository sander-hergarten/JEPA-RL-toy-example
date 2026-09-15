#!/usr/bin/env bash
# Sample-efficiency matrix: score at a fixed 500k-decision budget for five recipes, 3 seeds each.
# Every arm trains from scratch with its own collection policy; no data is shared between arms.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
for s in ${SEEDS:-0 1 2}; do
  for c in se_base se_nstep se_rr2 se_plan se_all; do
    scripts/run_pipeline.sh configs/sample_eff/breakout_${c}_500k.yaml "$s" "${THREADS:-1}" &
  done
done
wait
echo "${MARKER:-SAMPLE_EFF_COMPLETE}"
