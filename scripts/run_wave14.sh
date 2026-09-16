#!/usr/bin/env bash
# Separate the rollout length from the value horizon.
#
# The K sweep (wave 13) found a dissociation: longer training rollouts give a strictly better world
# model -- lower latent-prediction error at every depth, ~2x the root action sensitivity, a better
# movement probe -- and strictly worse control, with the loss concentrated in the shallow controllers
# (H=1: +40.4 at K=5, +30.7 at K=10, +21.6 at K=20). q_imagined supervises Q on z_hat[0..K-1], so a
# long-K arm trains its value head mostly on deep imagined latents; loss.q_imagined_depth caps that.
#
# If capping restores control while keeping the better model, the confound is confirmed and long
# rollouts become usable. If control stays low, the rollout length itself is what hurts.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
HORIZONS=${HORIZONS:-"1 3 5 10 20 30"}

for cfg in breakout_se_k10_qd5_500k breakout_se_k20_qd5_500k; do
  for s in $SEEDS; do
    scripts/run_pipeline.sh configs/sample_eff/$cfg.yaml "$s" "${THREADS:-2}" &
  done
done
wait

for cfg in breakout_se_k10_qd5_500k breakout_se_k20_qd5_500k; do
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
done
wait
echo "${MARKER:-WAVE14_COMPLETE}"
