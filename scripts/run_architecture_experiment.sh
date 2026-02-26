#!/usr/bin/env bash
# run_architecture_experiment.sh -- One-command architecture experiment runner.
#
# Usage:
#   bash scripts/run_architecture_experiment.sh [--probes <p1,p2,...>] [--output <path>]
#
# Defaults:
#   --probes  all
#   --output  results/outputs.json

set -euo pipefail

# --- Default values ---
PROBES="all"
OUTPUT="results/outputs.json"

# --- Parse optional arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --probes) PROBES="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Install dependencies ---
echo "[run_architecture_experiment.sh] Installing dependencies..."
if command -v uv &>/dev/null; then
    uv sync
else
    pip install -e ".[dev]" --quiet
fi

# --- Create results directories ---
mkdir -p results/figures

# --- Run experiments ---
echo "[run_architecture_experiment.sh] Starting experiments (device=auto, probes=$PROBES)"
python -m src.architecture_deepdive.experiment_runner \
    --device auto \
    --probes "$PROBES" \
    --output "$OUTPUT"

echo "[run_architecture_experiment.sh] Done! Results in results/"
