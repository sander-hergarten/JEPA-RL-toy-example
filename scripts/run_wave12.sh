#!/usr/bin/env bash
# Does a gradient-isolated level 2 cost anything at all?
#
# With loss.macro_detach the level-2 loss reaches only macro_dynamics/macro_return/macro_continuation,
# so level 1 should train exactly as it does without a hierarchy -- and the diagnostics agree (action
# distance 0.0686 vs 0.0661, shuffled-action penalty 51% vs 48%). But at 3 seeds the detached arm's
# 5-step beam search came out 15 points below the flat n-step model (+50.1 vs +65.1), which the
# diagnostics do not explain. Seeds 3/4/5 of both arms decide whether that gap is real or seed noise.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-$PWD/.venv/bin/python}
for s in 3 4 5; do
  for cfg in breakout_se_nstep_500k breakout_se_hjepa_detached_500k; do
    scripts/run_pipeline.sh configs/sample_eff/$cfg.yaml "$s" "${THREADS:-1}" &
  done
done
wait
for s in 3 4 5; do
  for cfg in breakout_se_nstep_500k breakout_se_hjepa_detached_500k; do
    d=runs/$cfg/seed$s
    [ -f $d/eval_q_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt --controller q \
      --epsilon 0.01 --max-episode-decisions 10000 --threads 1 --out $d/eval_q_eps001.json >> $d/eval_eps001.log 2>&1 &
    for h in 1 5; do
      [ -f $d/eval_lookahead_h${h}_eps001.json ] || $PYTHON -m atari_jepa.evaluate --checkpoint $d/checkpoint.pt \
        --controller lookahead --horizon $h --epsilon 0.01 --max-episode-decisions 10000 --threads 1 \
        --out $d/eval_lookahead_h${h}_eps001.json >> $d/eval_h.log 2>&1 &
    done
  done
done
wait
echo "${MARKER:-WAVE12_COMPLETE}"
