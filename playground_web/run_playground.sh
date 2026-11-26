#!/usr/bin/env bash
# Start the Multi-Method Nimble & Kev SOTA Playground on an idle GPU (default GPU 0, port 23834).
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export JEV_DATA_DIR="${JEV_DATA_DIR:-/root/projects/jev-real-ad-1000/kev_data}"
export LEGACY_RUNS_DIR="${LEGACY_RUNS_DIR:-/root/projects/jev-real-ad-1000/runs}"
export SOTA_RUNS_DIR="${SOTA_RUNS_DIR:-/data/nvme0/nimble_mm_exp/runs}"
export KEV_ROOT="${KEV_ROOT:-/root/projects/kev}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="${PYTHON_BIN:-/root/projects/kev/.venv/bin/python}"
GPU_ID="${GPU:-0}"
PORT="${PORT:-23834}"

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup "$PY" -u playground_web/server.py --host 0.0.0.0 --port "$PORT" > playground_web/server.log 2>&1 < /dev/null &
echo "Started Playground PID=$! on http://100.80.18.39:${PORT} (GPU ${GPU_ID})"
