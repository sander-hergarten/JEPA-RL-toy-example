#!/usr/bin/env bash
# Restart a long experiment script if it is not running and not yet finished.
# Meant for a systemd timer (or cron @reboot plus a periodic check), so a reboot, a crash or a dropped
# connection does not end the experiment:
#   */10 * * * * flock -n /tmp/atari_jepa_supervise.lock SCRIPTS/supervise.sh SCRIPT runs/waveN.log MARKER
# The work itself is idempotent (training resumes from the last checkpoint, finished steps are skipped).
#
# The run holds RUN_LOCK for its whole lifetime, and we refuse to start when that lock is taken. A
# pgrep for training alone is not enough: between the training and evaluation phases of a wave there
# is no atari_jepa.train process, and a timer firing in that window would launch a second copy of the
# same script. Note that a systemd unit running this must set KillMode=process, or the launched run is
# killed along with the (oneshot) unit that started it.
set -uo pipefail
SCRIPT=${1:?usage: supervise.sh SCRIPT LOG MARKER}
LOG=${2:?}
MARKER=${3:?}
RUN_LOCK=${RUN_LOCK:-/tmp/atari_jepa_run.lock}
cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$LOG")"
if grep -q "$MARKER" "$LOG" 2>/dev/null; then
  exit 0                                   # experiment already finished
fi
if ! flock -n "$RUN_LOCK" true 2>/dev/null; then
  exit 0                                   # a run holds the lock: training, evaluating or diagnosing
fi
if pgrep -f "atari_jepa[.](train|evaluate|diagnostics)" >/dev/null; then
  exit 0                                   # belt and braces, e.g. a run started outside this script
fi
echo "[$(date -Is)] supervise: (re)starting $SCRIPT" >> "$LOG"
setsid nohup flock -n "$RUN_LOCK" "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
