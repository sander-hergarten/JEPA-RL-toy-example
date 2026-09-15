#!/usr/bin/env bash
# n-step sweep (n=3,10), the late-planner / replay-ratio follow-up matrix, and the 2.5M n-step run.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
for s in ${SEEDS:-0 1 2}; do
  for c in se_nstep3 se_nstep10 se_n5_lateplan se_n5_rr2 se_n5_rr2_reset se_n5_rr2_bigbuf; do
    scripts/run_pipeline.sh configs/sample_eff/breakout_${c}_500k.yaml "$s" "${THREADS:-1}" &
  done
  scripts/run_pipeline.sh configs/long/breakout_long_n5.yaml "$s" "${THREADS:-1}" &
done
wait
echo "${MARKER:-WAVE9_COMPLETE}"
