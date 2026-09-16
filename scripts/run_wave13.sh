#!/usr/bin/env bash
# Rollout-horizon (K) sweep at a fixed 500k-decision budget, Breakout.
#
# On the 2.5M K=5 checkpoints, beam-search return rose with depth to H=5 and fell at H=10 -- exactly
# where the search leaves the horizon the dynamics were trained on. This sweep asks whether that
# ceiling moves with K: train identical recipes at K=5 (control, already run), 10, 20 and 30, then
# score every checkpoint at H=1,3,5,10,20,30. If the ceiling tracks K, longer rollouts buy usable
# planning depth; if it does not, the depth limit comes from something else.
#
# Each arm trains from scratch and collects its own data: no arm sees another's samples.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
HORIZONS=${HORIZONS:-"1 3 5 10 20 30"}

for K in 10 20 30; do
  for s in $SEEDS; do
    scripts/run_pipeline.sh configs/sample_eff/breakout_se_k${K}_500k.yaml "$s" "${THREADS:-2}" &
  done
done
wait

# Evaluation at the shared protocol (eps=0.01, 10 episodes, seed_base 10000). The K=5 control is
# included so every cell of the K x H table comes from the same evaluation path.
for cfg in breakout_se_nstep_500k breakout_se_k10_500k breakout_se_k20_500k breakout_se_k30_500k; do
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
    wait   # one seed's horizons at a time: 7 concurrent evals, not 28
  done
done
wait
echo "${MARKER:-WAVE13_COMPLETE}"
