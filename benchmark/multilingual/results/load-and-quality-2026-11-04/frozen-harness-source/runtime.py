"""Provenance/runtime utilities reused from the corrected prior benchmark."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def append_json(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation, matching NumPy's default percentile definition."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def collect_hardware() -> dict[str, Any]:
    import importlib.metadata
    import platform

    import torch

    def command(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, text=True, stderr=subprocess.STDOUT, timeout=15
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unverified: {exc}"

    packages = {}
    for package in [
        "torch",
        "torchaudio",
        "triton",
        "transformers",
        "omnivoice",
        "fastapi",
        "starlette",
        "anyio",
        "uvicorn",
        "httpx",
        "numpy",
        "soundfile",
    ]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"
    devices = []
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": prop.name,
                "visible_vram_bytes": prop.total_memory,
                "compute_capability": [prop.major, prop.minor],
            }
        )
    return {
        "collected_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_logical": os.cpu_count(),
        "cpu": command(["lscpu"]),
        "packages": packages,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_devices": devices,
        "nvidia_smi_gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,memory.used,driver_version,power.limit",
                "--format=csv",
            ]
        ),
        "nvidia_smi_processes": command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv",
            ]
        ),
        "physical_vram_evidence": (
            "nvidia-smi memory.total; visible memory independently from torch device "
            "properties"
        ),
    }


def source_identity(source: Path) -> dict[str, Any]:
    source_hash = hashlib.sha256()
    files = []
    for path in sorted((source / "src").rglob("*.py")):
        relative = str(path.relative_to(source))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_hash.update(relative.encode() + b"\0" + digest.encode() + b"\n")
        files.append({"path": relative, "sha256": digest})
    try:
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.SubprocessError:
        head = "archive; see --source-ref and source digest"
    return {
        "source_ref": head,
        "source_tree_sha256": source_hash.hexdigest(),
        "files": files,
    }


async def wait_ready(
    url: str, process: subprocess.Popen, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server exited with status {process.returncode}; inspect "
                    f"server.log"
                )
            try:
                ready = await client.get(url + "/healthz")
                if ready.status_code == 200:
                    return (await client.get(url + "/health")).json()
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(1)
    raise TimeoutError("Server readiness deadline exceeded; inspect server.log")
