#!/usr/bin/env bash
# Start the Kev multimodal web UI on all interfaces, pinned to one idle GPU (default 7). Logs to web/server.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES="${GPU:-7}" nohup /root/projects/kev/.venv/bin/python web/server.py --host 0.0.0.0 --port "${PORT:-8080}" > web/server.log 2>&1 &
echo "started pid $! on port ${PORT:-8080}, gpu ${GPU:-7}"
