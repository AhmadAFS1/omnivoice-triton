#!/bin/bash
set -euo pipefail
TASK_ROOT=/workspace/omnivoice-4070-rps
exec > >(tee -a "$TASK_ROOT/bootstrap.log") 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip curl ffmpeg libsndfile1 git rsync libgomp1 > "$TASK_ROOT/apt.log" 2>&1
if [ ! -x /root/.local/bin/uv ]; then
 curl --retry 4 --connect-timeout 15 --max-time 300 -LsSf https://astral.sh/uv/install.sh -o "$TASK_ROOT/uv-install.sh"
 sh "$TASK_ROOT/uv-install.sh"
fi
export PATH="/root/.local/bin:$PATH"
export HF_HOME="$TASK_ROOT/hf-cache"
export UV_CACHE_DIR="$TASK_ROOT/uv-cache"
export UV_LINK_MODE=hardlink
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=3
mkdir -p "$TASK_ROOT/source"
tar xzf "$TASK_ROOT/source.tar.gz" -C "$TASK_ROOT/source"
cd "$TASK_ROOT/source"
uv sync --frozen --no-dev --python 3.12
uv pip install --python .venv/bin/python httpx==0.28.1 psutil==7.0.0 pynvml==12.0.0 pytest==8.4.2 jiwer==4.0.0
uv pip freeze --python .venv/bin/python > "$TASK_ROOT/pip-freeze.txt"
.venv/bin/python - <<'PY'
import json,pathlib
from huggingface_hub import snapshot_download
r=pathlib.Path('/workspace/omnivoice-4070-rps');p=json.loads((r/'model-pin.json').read_text())
snapshot_download(p['repo_id'],revision=p['revision'],local_dir=str(r/'model'))
PY
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,power.limit --format=csv > "$TASK_ROOT/nvidia-smi.csv"
touch "$TASK_ROOT/READY"
