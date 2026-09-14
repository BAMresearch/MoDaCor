from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MODACOR_RUN_CHUNK_MEMORY_TESTS") != "1",
    reason="Set MODACOR_RUN_CHUNK_MEMORY_TESTS=1 for subprocess RSS scaling checks.",
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_chunked_hdf.py"


def _run_benchmark(output: Path, frames: int) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--shape",
            f"{frames},256,256",
            "--output",
            str(output),
            "--chunk-size",
            "1",
            "--storage-chunks",
            "1,256,256",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_peak_rss_scales_with_chunk_not_total_frame_count(tmp_path: Path) -> None:
    small = _run_benchmark(tmp_path / "small.h5", frames=4)
    large = _run_benchmark(tmp_path / "large.h5", frames=32)

    assert large["logical_bytes"] == 8 * small["logical_bytes"]
    assert large["memory"]["peak_bytes"] is not None
    assert small["memory"]["peak_bytes"] is not None
    allowance = max(64 * 1024**2, 2 * 256 * 256 * 4)
    assert large["memory"]["peak_bytes"] <= small["memory"]["peak_bytes"] + allowance
