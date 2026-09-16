#!/usr/bin/env bash
# Rollout-length sweep on the LMU dynamics, against the conv prior.
#
# The conv sweep (wave 13) found a strictly better model that played strictly worse as K grew: latent
# error fell at every depth while the best controller went +65.1 (K=5) -> +36.9 (K=30), and the
# persistence column showed why -- long-rollout pressure makes the latents temporally smooth. A
# memoryless core has to encode "how fast is the ball moving" into the latent itself, which is exactly
# the pressure that smooths it. An LMU keeps that history in a separate state, so the question is
# whether the dissociation weakens.
#
# theta tracks K, so the memory window spans exactly the rollout it is supervised on. That does mean
# two things move together; a fixed theta would instead confound "the LMU does not help at K=30" with
# "its window covered 5 of the 30 steps".
#
# K=5 comes from wave 16 (breakout_se_lmu_500k). Compare against, at the same budget and seeds:
#   conv K=5  Q +19.10  H=1 +40.43  H=5 +65.07
#   conv K=10 Q +19.90  H=1 +30.73  H=5 +61.83
#   conv K=20 Q +17.63  H=1 +21.63  H=5 +49.13
#   conv K=30 Q +16.20  H=1 +18.30  H=5 +34.03
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
HORIZONS=${HORIZONS:-"1 5 10 30"}
ARMS=${ARMS:-"breakout_se_lmu_k10_500k breakout_se_lmu_k20_500k breakout_se_lmu_k30_500k"}

for cfg in $ARMS; do
  for s in $SEEDS; do
    scripts/run_pipeline.sh configs/sample_eff/$cfg.yaml "$s" "${THREADS:-2}" &
  done
done
wait

for cfg in $ARMS breakout_se_lmu_500k; do
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
echo "${MARKER:-WAVE17_COMPLETE}"
