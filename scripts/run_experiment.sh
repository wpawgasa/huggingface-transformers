#!/usr/bin/env bash
# run_experiment.sh — One-command experiment runner.
#
# Usage:
#   bash scripts/run_experiment.sh [--tasks <task1,task2,...>] [--output <path>]
#
# Defaults:
#   --tasks  all
#   --output results/outputs.json

set -euo pipefail

# --- Default values ---
TASKS="all"
OUTPUT="results/outputs.json"
BENCHMARK_OUTPUT="results/benchmarks.csv"

# --- Parse optional arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)            TASKS="$2";            shift 2 ;;
        --output)           OUTPUT="$2";           shift 2 ;;
        --benchmark-output) BENCHMARK_OUTPUT="$2"; shift 2 ;;
        *)                  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Install dependencies ---
echo "[run_experiment.sh] Installing dependencies..."
if command -v uv &>/dev/null; then
    uv sync
else
    pip install -e ".[dev]" --quiet
fi

# --- Create results directory ---
mkdir -p results

# --- Run experiments ---
echo "[run_experiment.sh] Starting experiments (device=auto, tasks=$TASKS)"
python -m src.pipeline_exploration.experiment_runner \
    --device auto \
    --tasks "$TASKS" \
    --output "$OUTPUT" \
    --benchmark-output "$BENCHMARK_OUTPUT"

echo "[run_experiment.sh] Done! Results in results/"
