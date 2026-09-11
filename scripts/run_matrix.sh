#!/usr/bin/env bash
# The A/B/C comparison over seeds 0,1,2, then the aggregate report. Re-running continues where it stopped.
#   MODE=lanes    (default) two sequential lanes, for an 8-core CPU-only machine
#   MODE=parallel all nine runs at once, for a machine with a GPU and many cores
set -uo pipefail
THREADS=${THREADS:-4}
MODE=${MODE:-lanes}
SEEDS=${SEEDS:-"0 1 2"}
# CONFIGS: space-separated, world model first (it is the slowest); e.g. the 500k follow-up:
#   CONFIGS="configs/extended/pong_world_model_500k.yaml configs/extended/pong_temporal_jepa_500k.yaml configs/extended/pong_q_500k.yaml"
read -r -a CONFIGS <<< "${CONFIGS:-configs/pong_world_model.yaml configs/pong_temporal_jepa.yaml configs/pong_q.yaml}"

if [ "$MODE" = parallel ]; then
  for c in "${CONFIGS[@]}"; do
    for s in $SEEDS; do scripts/run_pipeline.sh "$c" "$s" "$THREADS" & done
  done
  wait
else
  lane1() { for s in $SEEDS; do scripts/run_pipeline.sh "${CONFIGS[0]}" "$s" "$THREADS"; done; }
  lane2() {
    for c in "${CONFIGS[@]:1}"; do
      for s in $SEEDS; do scripts/run_pipeline.sh "$c" "$s" "$THREADS"; done
    done
  }
  lane1 & lane2 & wait
fi
${PYTHON:-python} -m atari_jepa.report runs
