#!/bin/zsh
set -euo pipefail

# Absolute project root
ROOT="/Users/youngchuljoo/Desktop/power_pred"
VENV="$ROOT/.venv"
LOGDIR="$ROOT/runs"
mkdir -p "$LOGDIR"

# Activate venv
if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

# Run preliminary checks (writes timestamped MD/JSON under runs/)
python3 "$ROOT/tests/run_prelim_checks.py"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] daily prelim run completed" >> "$LOGDIR/cron.log"


