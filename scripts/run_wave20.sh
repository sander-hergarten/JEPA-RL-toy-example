#!/usr/bin/env bash
# Partition the rollout-length effect at K=30 between the latent loss and the heads.
#
# Wave 19's discount arm falsified the smoothing premise (see README): re-weighting the latent loss
# returned the latent to K=5's temporal statistics with a healthy Q and the highest action sensitivity
# of any arm, and the planning gain did not move. What a 30-step window does that that arm did not undo
# is supervise the reward/continuation/Q heads on 30 imagined depths and train the dynamics through 30
# steps. Three arms, each capping the depth-summed losses differently (see the configs):
#   k30_short       every loss capped at 5  -> must reproduce K=5 (+40 at H=1); if not, the pipeline is at fault
#   k30_jepa_deep   latent loss deep, heads shallow  -> isolates the latent loss through dynamics + encoder
#   k30_heads_deep  latent loss shallow, heads deep  -> isolates the heads on deep imagined latents
# Predictions, written before running: exactly one of jepa_deep / heads_deep lands at the K=30 level
# (H=1 ~ +18-22) and the other near K=5 (~ +35-40); k30_short near K=5. If both land low, the two
# effects are each sufficient; if both land high, the damage needs both, or lives in the pipeline.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
HORIZONS=${HORIZONS:-"1 5 10 30"}
ARMS=${ARMS:-"breakout_se_k30_short_500k breakout_se_k30_jepa_deep_500k breakout_se_k30_heads_deep_500k"}

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
  for s in $SEEDS; do
    d=runs/$cfg/seed$s
    [ -f $d/embeddings.json ] || $PYTHON -m atari_jepa.embeddings --checkpoint $d/checkpoint.pt --threads 2 \
      --out $d/embeddings.json >> $d/embeddings.log 2>&1
    [ -f $d/action_effect.json ] || $PYTHON -m atari_jepa.action_effect --checkpoint $d/checkpoint.pt \
      --episodes 3 --max-probes 400 --threads 2 >> $d/action_effect.log 2>&1
    [ -f $d/delta_probe.json ] || [ ! -f $d/replay.npz ] || $PYTHON -m atari_jepa.delta_probe \
      --checkpoint $d/checkpoint.pt --deltas 1,5,10,30,100 --train-pairs 40000 --test-pairs 4000 \
      --updates 20000 --lr 3e-4 --threads 4 >> $d/delta_probe.log 2>&1
  done
done
echo "${MARKER:-WAVE20_COMPLETE}"
