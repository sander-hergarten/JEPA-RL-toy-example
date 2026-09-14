#!/usr/bin/env bash
# One ablation arm: train, evaluate, diagnostics and the embedding probe for a config + --set overrides.
#   scripts/run_ablation.sh configs/ball/pong_wm_100k.yaml pong_delta 0 2 loss.jepa_target=delta
# Idempotent: finished steps are skipped, so re-running resumes.
set -uo pipefail
CONFIG=$1; NAME=$2; SEED=$3; THREADS=${4:-2}; shift 4 || shift 3
PY=${PYTHON:-python}
RUN="runs/$NAME/seed$SEED"
SETS=(--set "name=$NAME")
for kv in "$@"; do SETS+=(--set "$kv"); done
mkdir -p "$RUN"
$PY -m atari_jepa.train --config "$CONFIG" --seed "$SEED" --run-dir "$RUN" --auto-resume \
  --set torch_threads="$THREADS" "${SETS[@]}" 2>&1 | grep --line-buffered -v -e '^A.L.E' -e '^\[Powered by Stella\]' >> "$RUN/train.log"
[ -f "$RUN/eval_q.json" ] || $PY -m atari_jepa.evaluate --checkpoint "$RUN/checkpoint.pt" --controller both \
  --threads "$THREADS" 2>&1 | grep --line-buffered -v -e '^A.L.E' -e '^\[Powered by Stella\]' >> "$RUN/eval.log"
[ -f "$RUN/diagnostics.json" ] || $PY -m atari_jepa.diagnostics --checkpoint "$RUN/checkpoint.pt" --threads "$THREADS" \
  2>&1 | grep --line-buffered -v -e '^A.L.E' -e '^\[Powered by Stella\]' >> "$RUN/diagnostics.log"
[ -f "$RUN/embeddings.json" ] || $PY -m atari_jepa.embeddings --checkpoint "$RUN/checkpoint.pt" --threads "$THREADS" \
  2>&1 | grep --line-buffered -v -e '^A.L.E' -e '^\[Powered by Stella\]' >> "$RUN/embeddings.log"
echo "$RUN done"
