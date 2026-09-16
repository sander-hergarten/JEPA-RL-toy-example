#!/usr/bin/env bash
# Run the pending waves in order under one supervised process. Each wave is idempotent (finished
# training resumes from its checkpoint, finished evaluations are skipped), so re-running the queue
# after a crash or a reboot picks up where it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."
scripts/run_wave15.sh      # successor features: a bootstrapped, unbounded value horizon
scripts/run_wave16.sh      # Legendre Memory Unit in place of the memoryless dynamics core
echo "QUEUE_COMPLETE"
