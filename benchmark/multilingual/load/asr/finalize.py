#!/usr/bin/env python3
"""Wait for this task's CTC prefetch, hash cached assets and verify packages.

CPU/network preparation only: this never constructs a model or starts inference.
"""

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ctc_log = root / "asr-prefetch-ctc.log"
deadline = time.monotonic() + 1800
while True:
    if ctc_log.exists() and '"sha256"' in ctc_log.read_text():
        break
    if time.monotonic() > deadline:
        raise TimeoutError("CTC asset preparation exceeded30-minute finite wait")
    time.sleep(2)
with (root / "asr-prefetch-final.log").open("w") as log:
    subprocess.run(
        [
            sys.executable,
            str(root / "asr/prefetch.py"),
            "--output",
            str(root / "asr-assets.json"),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
subprocess.run([sys.executable, str(root / "asr/verify_assets.py")], check=True)
marker = {
    "exit_code": 0,
    "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "baseline_package_freeze_unchanged": True,
    "method": "Separate ASR virtualenv with UV cache reuse and online index metadata",
    "gpu_model_load_or_inference_performed": False,
    "verified_marker": str(root / "ASR_ASSETS_VERIFIED.json"),
}
(root / "ASR_SETUP_DONE.json").write_text(json.dumps(marker, indent=2) + "\n")
print(json.dumps(marker), flush=True)
