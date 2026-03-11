#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — background pipeline launcher
#
# Activates the project virtual environment, then runs pipeline.py in the
# background under nohup, streaming output to a timestamped log file.
#
# Usage:
#   ./run.sh                             # run all stages
#   ./run.sh --stages feature train      # only feature + train
#   ./run.sh --stages train --force      # force re-train
#   ./run.sh --stages all --data-dir /mnt/data
#
# All arguments are forwarded to pipeline.py (see --help for options).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Project root (directory of this script) ────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Virtual environment ────────────────────────────────────────────────────
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_ACTIVATE"
    echo "       Create it with: python -m venv .venv && pip install -r requirements.txt"
    exit 1
fi
source "$VENV_ACTIVATE"

# ── Log directory ──────────────────────────────────────────────────────────
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/run_${TIMESTAMP}.log"

# ── Launch pipeline in background ──────────────────────────────────────────
echo "Starting pipeline at $(date)"
echo "Log file: $LOG_FILE"
echo "Args: $*"
echo ""

nohup python -u "$SCRIPT_DIR/crop_mapping_pipeline/pipeline.py" "$@" \
    > >(tee "$LOG_FILE") \
    2>&1 &

PID=$!
echo "Pipeline PID: $PID"
echo "$PID" > "$LOGS_DIR/pipeline_${TIMESTAMP}.pid"

echo ""
echo "Pipeline running in background."
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  kill $PID   # to stop"
