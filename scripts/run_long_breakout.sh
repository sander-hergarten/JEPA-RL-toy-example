#!/usr/bin/env bash
# The 1.5M-decision Breakout comparison: model-free, live delta+motion, and offline delta+motion
# (fine-tuned and frozen), 3 seeds each. Idempotent: finished runs are skipped and unfinished ones
# resume from their last checkpoint, so this can be re-run after a crash or reboot.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
for s in ${SEEDS:-0 1 2}; do
  for c in breakout_long_q breakout_long_live_delta_motion breakout_long_offline_dm_finetune breakout_long_offline_dm_frozen; do
    scripts/run_pipeline.sh configs/long/$c.yaml "$s" "${THREADS:-1}" &
  done
done
wait
echo WAVE6_COMPLETE
