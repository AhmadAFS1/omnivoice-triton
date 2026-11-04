#!/usr/bin/env python3
"""Fetch one official CTC asset concurrently, with no model load/inference."""

import hashlib
import json
import time
from pathlib import Path

from fairseq2.assets.download_manager import StandardAssetDownloadManager
from fairseq2.utils.uri import Uri

started = time.perf_counter()
name = "omniASR_CTC_1B_v2"
url = "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B-v2.pt"
manager = StandardAssetDownloadManager(Path.home() / ".cache/fairseq2/assets")
uri = Uri.maybe_parse(url)
assert uri is not None
path = manager.download_model(uri, name, progress=False)
digest = hashlib.sha256()
with path.open("rb") as asset:
    for chunk in iter(lambda: asset.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
assert path.stat().st_size == 3902956068
assert actual == "354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c"
print(
    json.dumps(
        {
            "measurement_kind": "CPU/network download, no GPU inference",
            "name": name,
            "url": url,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "wall_s": time.perf_counter() - started,
        }
    ),
    flush=True,
)
