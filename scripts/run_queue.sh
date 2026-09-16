#!/usr/bin/env bash
# Run the pending waves in order under one supervised process. Each wave is idempotent (finished
# training resumes from its checkpoint, finished evaluations are skipped), so re-running the queue
# after a crash or a reboot picks up where it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."
scripts/run_wave13.sh      # K sweep: rollout_steps 10/20/30, scored at H=1..30
scripts/run_wave14.sh      # q_imagined_depth cap: long rollout, short value horizon
echo "QUEUE_COMPLETE"
