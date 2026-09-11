from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_chunked_hdf.py"
SPEC = importlib.util.spec_from_file_location("benchmark_chunked_hdf", SCRIPT_PATH)
benchmark_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark_mod
SPEC.loader.exec_module(benchmark_mod)


def test_synthetic_benchmark_cli_writes_validated_json_report(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.h5"
    report_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--shape",
            "5,3,2",
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--chunk-size",
            "2",
            "--compression",
            "gzip",
            "--compression-level",
            "1",
            "--storage-chunks",
            "2,3,2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_report = json.loads(completed.stdout)
    file_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert file_report["status"] == "passed"
    assert file_report["chunk_count"] == 3
    assert file_report["storage_layout"] == {
        "chunks": [2, 3, 2],
        "compression": "gzip",
        "compression_opts": 1,
    }
    assert file_report["memory"]["measurement"] == "process_peak_rss"


def test_external_hdf_benchmark_reads_and_validates_by_slice(tmp_path: Path) -> None:
    source = tmp_path / "source.nxs"
    output = tmp_path / "external.h5"
    expected = np.arange(30, dtype=np.float32).reshape(5, 3, 2)
    with h5py.File(source, "w") as h5:
        h5.create_dataset("entry/data", data=expected)

    report = benchmark_mod.run_benchmark(
        benchmark_mod.BenchmarkConfig(
            output=output,
            source_hdf=source,
            source_dataset="entry/data",
            chunk_size=2,
            write_order="reverse",
        )
    )

    assert report["status"] == "passed"
    assert report["source"]["kind"] == "hdf"
    assert report["write_order"] == "reverse"
    with h5py.File(output, "r") as h5:
        np.testing.assert_array_equal(h5["processing/result/benchmark/sample/signal/signal"], expected)
