#!/usr/bin/env bash
# Depth-weighted latent loss at K=30: the intervention the gradient anatomy implies.
#
# Measured on the K sweep (gradient_anatomy, 3 seeds each): the fraction of depth-term pairs whose root
# gradients *oppose* each other is 0% at K=5, 0% at K=10, 3.3% at K=20 and 8.4% at K=30; mean pairwise
# alignment falls +0.425 -> +0.278; and the action's share of credit from the deepest term falls
# 0.915 -> 0.415. Uniform weighting treats a K-step rollout as K equally important tasks, and past
# K ~= 20 they start fighting. The feature they all agree on is one that barely changes -- exactly the
# smoothing measured by the delta probe, and exactly what control cannot use.
#
# These arms keep the K=30 rollout and change only the weighting of its depth terms. Predictions, so
# the result can falsify the account rather than absorb it:
#   * control recovers toward K=5 (+65.1) from K=30's +36.9
#   * conflicting-pair fraction drops back toward 0
#   * delta-probe persistence at delta=1 falls back toward K=5's -0.067 (latents move again)
#   * long-horizon prediction stays better than K=5's, since the rollout is still 30 steps
# If control recovers but the conflict fraction does not, the weighting helped for another reason.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
HORIZONS=${HORIZONS:-"1 5 10 30"}
ARMS=${ARMS:-"breakout_se_k30_depth_inverse_500k breakout_se_k30_depth_discount_500k"}

for cfg in $ARMS; do
  for s in $SEEDS; do
    scripts/run_pipeline.sh configs/sample_eff/$cfg.yaml "$s" "${THREADS:-2}" &
  done
  wait
done

for cfg in $ARMS; do
  for s in $SEEDS; do
    d=runs/$cfg/seed$s
    [ -d "$d" ] || continue
    [ -f $d/eval_q_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt \
      --controller q --epsilon 0.01 --max-episode-decisions 10000 --threads 1 \
      --out $d/eval_q_eps001.json >> $d/eval_eps001.log 2>&1 &
    for h in $HORIZONS; do
      [ -f $d/eval_lookahead_h${h}_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt \
        --controller lookahead --horizon $h --epsilon 0.01 --max-episode-decisions 10000 --threads 1 \
        --out $d/eval_lookahead_h${h}_eps001.json >> $d/eval_h.log 2>&1 &
    done
    wait
  done
  # the mechanism checks: did the conflict actually go away, and did the latents start moving again?
  for s in $SEEDS; do
    d=runs/$cfg/seed$s
    [ -f $d/gradient_anatomy.json ] || $PYTHON -m atari_jepa.gradient_anatomy --checkpoint $d/checkpoint.pt \
      --threads 2 >> $d/gradient_anatomy.log 2>&1
    [ -f $d/delta_probe.json ] || [ ! -f $d/replay.npz ] || $PYTHON -m atari_jepa.delta_probe \
      --checkpoint $d/checkpoint.pt --deltas 1,5,10,30,100 --train-pairs 40000 --test-pairs 4000 \
      --updates 20000 --lr 3e-4 --threads 4 >> $d/delta_probe.log 2>&1
  done
done
echo "${MARKER:-WAVE19_COMPLETE}"
