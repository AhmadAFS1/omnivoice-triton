"""3-Tier verification runner for OmniVoice Triton.

Tier 1: Kernel unit tests via pytest tests/kernels/
Tier 2: Model parity tests via pytest tests/test_model_parity.py
Tier 3: Quality evaluation via eval_quality.py (UTMOS, CER, Speaker Sim).

Saves structured JSON to benchmark/results/verification_report.json.

Usage:
    uv run python benchmark/run_verification.py
    uv run python benchmark/run_verification.py --tier 1
    uv run python benchmark/run_verification.py --tier 1,2
    uv run python benchmark/run_verification.py --output /tmp/report.json
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _get_hardware_info() -> dict[str, Any]:
    """Collect GPU and platform info for the report header."""
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "gpu": None,
        "cuda_version": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except Exception as exc:
        logger.debug("Could not collect GPU info: %s", exc)
    return info


def _parse_pytest_line(line: str) -> dict[str, Any] | None:
    """Parse one line of pytest -v output into a test result dict.

    Args:
        line: A single line from pytest stdout.

    Returns:
        Dict with name/fullname/status keys, or None if not a test line.
    """
    match = re.match(
        r"^(tests/\S+::(\S+))\s+(PASSED|FAILED|SKIPPED|ERROR)",
        line.strip(),
    )
    if not match:
        return None
    return {
        "name": match.group(2),
        "fullname": match.group(1),
        "status": match.group(3),
    }


def _run_pytest(
    args: list[str],
    timeout: int = 300,
) -> tuple[list[dict[str, Any]], int, str]:
    """Run pytest with given args and parse test results.

    Args:
        args: Additional pytest arguments (e.g. test paths, flags).
        timeout: Max seconds to wait for pytest to complete.

    Returns:
        Tuple of (test result list, return code, combined stdout+stderr).
    """
    cmd = (
        [sys.executable, "-m", "pytest"] + args + ["-v", "--tb=short", "-q", "--no-cov"]
    )
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tests: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parsed = _parse_pytest_line(line)
        if parsed:
            tests.append(parsed)

    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    return tests, result.returncode, output


# ─────────────────────────────────────────────
# Tier 1: Kernel unit tests
# ─────────────────────────────────────────────


def run_tier1() -> dict[str, Any]:
    """Tier 1: Run pytest kernel tests.

    Returns:
        Dict with status, passed/failed/total counts, duration, and test list.
    """
    t_start = time.perf_counter()
    logger.info("Tier 1: Running kernel unit tests (tests/kernels/)...")

    try:
        tests, returncode, output = _run_pytest(
            ["tests/kernels/", "--ignore=tests/test_model_parity.py"],
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        logger.error("Tier 1 timed out after 300s")
        return {
            "status": "TIMEOUT",
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
            "duration_s": 300,
            "tests": [],
        }

    duration = round(time.perf_counter() - t_start, 2)
    passed = sum(1 for t in tests if t["status"] == "PASSED")
    failed = sum(1 for t in tests if t["status"] == "FAILED")
    skipped = sum(1 for t in tests if t["status"] == "SKIPPED")
    status = "PASS" if failed == 0 and returncode == 0 else "FAIL"

    logger.info(
        "Tier 1: %s  passed=%d  failed=%d  skipped=%d  (%.1fs)",
        status,
        passed,
        failed,
        skipped,
        duration,
    )
    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": len(tests),
        "duration_s": duration,
        "returncode": returncode,
        "tests": tests,
    }


# ─────────────────────────────────────────────
# Tier 2: Model parity tests
# ─────────────────────────────────────────────


def run_tier2() -> dict[str, Any]:
    """Tier 2: Run model parity tests.

    Returns:
        Dict with status, passed/failed counts, duration, and test list.
    """
    t_start = time.perf_counter()
    logger.info("Tier 2: Running model parity tests (tests/test_model_parity.py)...")

    try:
        tests, returncode, output = _run_pytest(
            ["tests/test_model_parity.py"],
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        logger.error("Tier 2 timed out after 600s")
        return {
            "status": "TIMEOUT",
            "passed": 0,
            "failed": 0,
            "total": 0,
            "duration_s": 600,
            "tests": [],
        }

    duration = round(time.perf_counter() - t_start, 2)
    passed = sum(1 for t in tests if t["status"] == "PASSED")
    failed = sum(1 for t in tests if t["status"] == "FAILED")
    status = "PASS" if failed == 0 and returncode == 0 else "FAIL"

    logger.info(
        "Tier 2: %s  passed=%d  failed=%d  (%.1fs)",
        status,
        passed,
        failed,
        duration,
    )
    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "total": len(tests),
        "duration_s": duration,
        "returncode": returncode,
        "tests": tests,
    }


# ─────────────────────────────────────────────
# Tier 3: Quality evaluation
# ─────────────────────────────────────────────


def run_tier3(mode: str = "fast") -> dict[str, Any]:
    """Tier 3: Run quality evaluation (UTMOS, CER, Speaker Sim).

    Args:
        mode: 'fast' (~5min, whisper-small) or 'full' (~30min, whisper-large-v3).

    Returns:
        Dict with status, duration, and output path.
    """
    t_start = time.perf_counter()
    logger.info("Tier 3: Running quality evaluation (mode=%s)...", mode)

    output_path = RESULTS_DIR / "tier3_fast_multi.json"
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "eval_quality.py"),
        "--mode",
        mode,
        "--output",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        duration = round(time.perf_counter() - t_start, 2)

        if result.returncode == 0:
            logger.info("Tier 3: PASS (%.1fs)", duration)
            status = "PASS"
        else:
            logger.error("Tier 3: FAIL (%.1fs)\n%s", duration, result.stderr[-500:])
            status = "FAIL"

        return {
            "status": status,
            "mode": mode,
            "duration_s": duration,
            "returncode": result.returncode,
            "output": str(output_path),
        }
    except subprocess.TimeoutExpired:
        logger.error("Tier 3 timed out after 3600s")
        return {
            "status": "TIMEOUT",
            "mode": mode,
            "duration_s": 3600,
        }


# ─────────────────────────────────────────────
# Summary + report
# ─────────────────────────────────────────────


def _print_summary(report: dict[str, Any]) -> None:
    """Print a concise pass/fail summary to the logger.

    Args:
        report: Verification report dict with optional tier1/tier2/tier3 keys.
    """
    logger.info("=" * 50)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 50)
    for tier_key, label in [
        ("tier1", "TIER1 (Kernel Tests)"),
        ("tier2", "TIER2 (Model Parity)"),
        ("tier3", "TIER3 (Quality Eval)"),
    ]:
        t = report.get(tier_key)
        if t is None:
            logger.info("  %s: SKIPPED", label)
        else:
            status = t.get("status", "UNKNOWN")
            details = ""
            if "passed" in t:
                details = f"  passed={t['passed']} failed={t.get('failed', 0)}"
            elif "message" in t:
                details = f"  → {t['message'][:80]}"
            logger.info("  %s: %s%s", label, status, details)
    logger.info("=" * 50)


def main() -> None:
    """CLI entry point for the 3-Tier verification runner."""
    parser = argparse.ArgumentParser(
        description="3-Tier verification runner for OmniVoice Triton.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1,2,3",
        help="Comma-separated tier numbers to run (e.g. '1,2' or '1').",
    )
    parser.add_argument(
        "--skip-tier3",
        action="store_true",
        help="Skip Tier 3 (equivalent to --tier 1,2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: benchmark/results/verification_report.json).",
    )
    args = parser.parse_args()

    tiers = {int(t.strip()) for t in args.tier.split(",")}
    if args.skip_tier3:
        tiers.discard(3)

    report: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "hardware": _get_hardware_info(),
    }

    if 1 in tiers:
        logger.info("=" * 50)
        logger.info("Running Tier 1...")
        logger.info("=" * 50)
        report["tier1"] = run_tier1()

    if 2 in tiers:
        logger.info("=" * 50)
        logger.info("Running Tier 2...")
        logger.info("=" * 50)
        try:
            report["tier2"] = run_tier2()
        except Exception as exc:
            logger.error("Tier 2 error: %s", exc)
            report["tier2"] = {"status": "ERROR", "error": str(exc), "duration_s": 0}

    if 3 in tiers:
        logger.info("=" * 50)
        logger.info("Tier 3...")
        logger.info("=" * 50)
        report["tier3"] = run_tier3()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.output) if args.output else RESULTS_DIR / "verification_report.json"
    )
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report saved to %s", out_path)

    _print_summary(report)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    main()
