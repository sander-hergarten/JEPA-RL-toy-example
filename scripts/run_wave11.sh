#!/usr/bin/env bash
# H-JEPA with level 2 trained on detached latents, so it cannot corrupt level 1.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
for s in 0 1 2; do
  scripts/run_pipeline.sh configs/sample_eff/breakout_se_hjepa_detached_500k.yaml "$s" "${THREADS:-2}" &
done
wait
for s in 0 1 2; do
  d=runs/breakout_se_hjepa_detached_500k/seed$s
  for c in q hierarchical; do
    [ -f $d/eval_${c}_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt --controller $c \
      --epsilon 0.01 --max-episode-decisions 10000 --threads 1 --out $d/eval_${c}_eps001.json >> $d/eval_eps001.log 2>&1 &
  done
  for h in 1 5; do
    [ -f $d/eval_lookahead_h${h}_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt \
      --controller lookahead --horizon $h --epsilon 0.01 --max-episode-decisions 10000 --threads 1 \
      --out $d/eval_lookahead_h${h}_eps001.json >> $d/eval_h.log 2>&1 &
  done
done
wait
echo "${MARKER:-WAVE11_COMPLETE}"
