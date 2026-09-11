#!/usr/bin/env bash
# Train (or resume) one config/seed, then evaluate and run diagnostics. Idempotent: finished steps are
# skipped, so the script can simply be re-run after an interruption.
#   scripts/run_pipeline.sh configs/pong_world_model.yaml 0 [threads]
set -euo pipefail
CONFIG=$1
SEED=$2
THREADS=${3:-4}
PY=${PYTHON:-python}
NAME=$(grep -m1 '^name:' "$CONFIG" | awk '{print $2}')
RUN="runs/$NAME/seed$SEED"
CKPT="$RUN/checkpoint.pt"
mkdir -p "$RUN"

$PY -m atari_jepa.train --config "$CONFIG" --seed "$SEED" --auto-resume --set torch_threads="$THREADS" \
  2>&1 | grep --line-buffered -v -e '^A.L.E' -e '^\[Powered by Stella\]' >> "$RUN/train.log"

[ -f "$RUN/eval_q.json" ] || $PY -m atari_jepa.evaluate --checkpoint "$CKPT" --controller q --threads "$THREADS" \
  2>&1 | grep --line-buffered -v -e '^A.L.E' -e '^\[Powered by Stella\]' >> "$RUN/eval.log"
if grep -q 'reward: true' "$CONFIG" && grep -q 'continuation: true' "$CONFIG"; then
  [ -f "$RUN/eval_lookahead.json" ] || $PY -m atari_jepa.evaluate --checkpoint "$CKPT" --controller lookahead \
    --threads "$THREADS" 2>&1 | grep --line-buffered -v -e '^A.L.E' -e '^\[Powered by Stella\]' >> "$RUN/eval.log"
fi
[ -f "$RUN/diagnostics.json" ] || $PY -m atari_jepa.diagnostics --checkpoint "$CKPT" --threads "$THREADS" \
  2>&1 | grep --line-buffered -v -e '^A.L.E' -e '^\[Powered by Stella\]' >> "$RUN/diagnostics.log"
echo "$RUN done"
