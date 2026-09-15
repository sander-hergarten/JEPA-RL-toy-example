#!/usr/bin/env bash
# H-JEPA arm (level-2 jumpy model) vs multi-step planning on the flat model, at a matched 500k budget.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
for s in 0 1 2; do
  scripts/run_pipeline.sh configs/sample_eff/breakout_se_hjepa_500k.yaml "$s" "${THREADS:-2}" &
done
# meanwhile: beam H=3/H=5 on the already-trained flat n-step 500k checkpoints (evaluation only)
for s in 0 1 2; do
  for h in 3 5; do
    d=runs/breakout_se_nstep_500k/seed$s
    [ -f $d/eval_lookahead_h${h}_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt \
      --controller lookahead --horizon $h --epsilon 0.01 --max-episode-decisions 10000 --threads 1 \
      --out $d/eval_lookahead_h${h}_eps001.json >> $d/eval_h.log 2>&1 &
  done
done
wait
echo "${MARKER:-WAVE10_COMPLETE}"
