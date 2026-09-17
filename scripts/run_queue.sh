#!/usr/bin/env bash
# Run the pending waves in order under one supervised process. Each wave is idempotent (finished
# training resumes from its checkpoint, finished evaluations are skipped), so re-running the queue
# after a crash or a reboot picks up where it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."
scripts/run_wave15.sh      # successor features: a bootstrapped, unbounded value horizon
scripts/run_wave16.sh      # Legendre Memory Unit in place of the memoryless dynamics core
scripts/run_wave17.sh      # rollout-length sweep on the LMU, against the conv K sweep
scripts/run_wave18.sh      # the same sweep with an offline encoder, frozen and fine-tuned
scripts/run_wave19.sh      # depth-weighted latent loss at K=30: the gradient-conflict intervention
echo "QUEUE_COMPLETE"
