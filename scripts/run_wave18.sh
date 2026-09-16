#!/usr/bin/env bash
# Offline "learn by observing" x rollout length x LMU, frozen and fine-tuned encoder.
#
# The conv K sweep found a strictly better model that played strictly worse as K grew, and the delta
# probe's persistence column suggested why: long-rollout pressure makes the latents temporally smooth.
# That is an *encoder* story, so freezing the encoder is the direct test -- a frozen encoder cannot be
# reshaped by K at all. If control still degrades with K under a frozen encoder, the explanation is
# wrong and the damage lives somewhere else (the dynamics, or the heads the planner queries).
#
# Three conditions per K, all with LMU dynamics and all at 500k decisions:
#   live      (wave 16/17)  encoder trained from scratch by this arm's own RL
#   frozen                  offline-pretrained encoder, held fixed
#   finetune                offline-pretrained encoder, further trained by RL
#
# The offline encoder is shared by every K, so K stays the only thing varying across the sweep. It was
# pretrained with a conv dynamics, so only the encoder transfers and the LMU dynamics starts fresh --
# the run log says so, and it is an asymmetry against the conv offline prior, which inherited both.
#
# Prior to compare against (conv, 500k, 3 seeds, eps=0.01):
#   live conv     K=5 Q +19.10 / H5 +65.07    K=10 +19.90 / +61.83    K=30 +16.20 / +34.03
#   offline conv  frozen and fine-tuned arms exist at K=5 only (breakout_offline_dm_*_500k)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
HORIZONS=${HORIZONS:-"1 5 10"}
ARMS=${ARMS:-""}
if [ -z "$ARMS" ]; then
  for K in 5 10 30; do
    for mode in frozen finetune; do
      ARMS="$ARMS breakout_se_lmu_k${K}_offline_${mode}_500k"
    done
  done
fi

for cfg in $ARMS; do
  for s in $SEEDS; do
    scripts/run_pipeline.sh configs/sample_eff/$cfg.yaml "$s" "${THREADS:-2}" &
  done
  wait   # one arm at a time: 3 concurrent runs, not 18
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
done
echo "${MARKER:-WAVE18_COMPLETE}"
