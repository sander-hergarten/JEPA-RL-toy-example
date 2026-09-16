#!/usr/bin/env bash
# Successor features: does a bootstrapped (unbounded) value horizon help where unrolling did not?
#
# The delta probe showed the long-range wall is stochasticity, not training coverage: a head trained
# directly at delta=100 cannot beat a constant. What survives at that range is a discounted average
# over futures -- which is what psi(z,a) estimates. And unlike every long-horizon objective tried here
# (H-JEPA level 2, longer K), psi gets its horizon from bootstrapping, so its gradient signature is the
# one-step TD that has never made the representation slower.
#
# Arms are breakout_se_sf_500k (identical to breakout_se_nstep_500k plus the successor head, trained on
# detached latents). Each checkpoint is scored with:
#   q                    the existing Q head, unchanged           -- must match breakout_se_nstep_500k
#   sf                   greedy on w . psi, no rollout at all
#   lookahead h1/h5      beam search bootstrapping on Q           -- the +40.4 / +65.1 comparison
#   lookahead h1/h5 sf   the same search bootstrapping on w . psi -- the actual question
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}

for s in $SEEDS; do
  scripts/run_pipeline.sh configs/sample_eff/breakout_se_sf_500k.yaml "$s" "${THREADS:-2}" &
done
wait

for s in $SEEDS; do
  d=runs/breakout_se_sf_500k/seed$s
  [ -d "$d" ] || continue
  for c in q sf; do
    [ -f $d/eval_${c}_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt \
      --controller $c --epsilon 0.01 --max-episode-decisions 10000 --threads 1 \
      --out $d/eval_${c}_eps001.json >> $d/eval_eps001.log 2>&1 &
  done
  for h in 1 5; do
    [ -f $d/eval_lookahead_h${h}_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt \
      --controller lookahead --horizon $h --bootstrap q --epsilon 0.01 --max-episode-decisions 10000 \
      --threads 1 --out $d/eval_lookahead_h${h}_eps001.json >> $d/eval_h.log 2>&1 &
    [ -f $d/eval_lookahead_h${h}_sfboot_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt \
      --controller lookahead --horizon $h --bootstrap sf --epsilon 0.01 --max-episode-decisions 10000 \
      --threads 1 --out $d/eval_lookahead_h${h}_sfboot_eps001.json >> $d/eval_h.log 2>&1 &
  done
  wait
done
wait
echo "${MARKER:-WAVE15_COMPLETE}"
