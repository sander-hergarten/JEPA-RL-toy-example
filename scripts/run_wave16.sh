#!/usr/bin/env bash
# The old rollout strategy with a Legendre Memory Unit in place of the memoryless dynamics core.
#
# g(z, a) currently sees the latent and nothing about how the rollout reached it, so a K-step unroll is
# a chain of amnesiac steps. An LMU threads a state whose *fixed* Legendre Delay Network matrices hold
# an orthogonal representation of the rollout's own history over a window of lmu_theta steps. The basis
# is derived rather than learned, which is what separates this from "add a recurrent layer".
#
# Everything else matches breakout_se_nstep_500k (K=5, n_step 5, delta target, motion channels, 500k
# decisions), so the comparison is against +22.1 / +39.2 / +60.6 (Q / H=1 / H=5) at 3 seeds.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2"}
HORIZONS=${HORIZONS:-"1 5 10"}

for s in $SEEDS; do
  scripts/run_pipeline.sh configs/sample_eff/breakout_se_lmu_500k.yaml "$s" "${THREADS:-2}" &
done
wait

for s in $SEEDS; do
  d=runs/breakout_se_lmu_500k/seed$s
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

# Does the memory actually extend the model's usable horizon? Same probe as the conv core, so the
# iterated column is directly comparable (it threads the LMU state through the whole rollout).
for s in $SEEDS; do
  d=runs/breakout_se_lmu_500k/seed$s
  [ -f $d/delta_probe.json ] || [ ! -f $d/replay.npz ] || $PYTHON -m atari_jepa.delta_probe \
    --checkpoint $d/checkpoint.pt --deltas 1,5,10,30,100 --train-pairs 40000 --test-pairs 4000 \
    --updates 20000 --lr 3e-4 --threads 4 >> $d/delta_probe.log 2>&1
done
echo "${MARKER:-WAVE16_COMPLETE}"
