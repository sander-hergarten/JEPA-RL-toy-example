#!/usr/bin/env bash
# Restart a long experiment script if it is not running and not yet finished.
# Meant for cron (@reboot and a periodic check), so a reboot or a crash does not end the experiment:
#   */10 * * * * flock -n /tmp/atari_jepa_supervise.lock SCRIPTS/supervise.sh SCRIPTS/run_long_breakout.sh runs/wave6.log WAVE6_COMPLETE
# The work itself is idempotent (training resumes from the last checkpoint, finished steps are skipped).
set -uo pipefail
SCRIPT=${1:?usage: supervise.sh SCRIPT LOG MARKER}
LOG=${2:?}
MARKER=${3:?}
cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$LOG")"
if grep -q "$MARKER" "$LOG" 2>/dev/null; then
  exit 0                                   # experiment already finished
fi
if pgrep -f "atari_jepa[.]train" >/dev/null; then
  exit 0                                   # still running, leave it alone
fi
echo "[$(date -Is)] supervise: (re)starting $SCRIPT" >> "$LOG"
setsid nohup "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
