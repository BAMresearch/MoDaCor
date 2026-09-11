# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark and validate HDF chunk assembly without repository test data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import h5py
import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide resource
    resource = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))

from modacor import __version__, ureg  # noqa: E402
from modacor.dataclasses.basedata import BaseData  # noqa: E402
from modacor.dataclasses.databundle import DataBundle  # noqa: E402
from modacor.dataclasses.processing_data import ProcessingData  # noqa: E402
from modacor.io.chunking import (  # noqa: E402
    AxisSelector,
    ChunkArrayLayout,
    ChunkOutputLayout,
    ChunkPlacement,
    ChunkPlan,
    ChunkSpec,
)
from modacor.io.hdf import HDFChunkedProcessingSink  # noqa: E402


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    output: Path
    shape: tuple[int, ...] | None = None
    source_hdf: Path | None = None
    source_dataset: str = "entry/data"
    chunk_axis: int = 0
    chunk_size: int = 1
    rank_of_data: int = 2
    units: str = "dimensionless"
    dtype: str = "float32"
    compression: str | None = None
    compression_level: int | None = None
    storage_chunks: tuple[int, ...] | bool = True
    write_order: str = "forward"
    plan_id: str = "benchmark"
    run_name: str = "benchmark"
    overwrite: bool = False


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("dimensions must be positive integers")
    return values


def _resolve_source(config: BenchmarkConfig) -> tuple[tuple[int, ...], np.dtype[Any]]:
    if config.source_hdf is not None:
        with h5py.File(config.source_hdf, "r") as h5:
            if config.source_dataset not in h5 or not isinstance(h5[config.source_dataset], h5py.Dataset):
                raise KeyError(f"Source dataset {config.source_dataset!r} was not found.")
            dataset = h5[config.source_dataset]
            shape = tuple(int(size) for size in dataset.shape)
            dtype = np.dtype(dataset.dtype)
    else:
        if config.shape is None:
            raise ValueError("Synthetic benchmarks require shape.")
        shape = tuple(config.shape)
        dtype = np.dtype(config.dtype)

    if not shape or any(size < 1 for size in shape):
        raise ValueError("The source must be a non-scalar array with positive dimensions.")
    if config.chunk_axis < 0 or config.chunk_axis >= len(shape):
        raise ValueError("chunk_axis is outside the source rank.")
    if config.chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if config.rank_of_data < 0 or config.rank_of_data > len(shape):
        raise ValueError("rank_of_data must be between zero and the source rank.")
    data_axes = set(range(max(0, len(shape) - config.rank_of_data), len(shape)))
    if config.chunk_axis in data_axes:
        raise ValueError("chunk_axis must identify a non-data dimension before the trailing image/data dimensions.")
    if isinstance(config.storage_chunks, tuple) and len(config.storage_chunks) != len(shape):
        raise ValueError("storage_chunks must have one extent per source dimension.")
    if config.compression_level is not None and config.compression != "gzip":
        raise ValueError("compression_level is only valid with gzip compression.")
    return shape, dtype


def _selectors(rank: int, axis: int, start: int, stop: int) -> tuple[AxisSelector, ...]:
    return tuple(AxisSelector.sliced(start, stop) if index == axis else AxisSelector.all() for index in range(rank))


def _build_plan(config: BenchmarkConfig, shape: tuple[int, ...], dtype: np.dtype[Any]) -> ChunkPlan:
    total_chunks = math.ceil(shape[config.chunk_axis] / config.chunk_size)
    width = max(6, len(str(total_chunks - 1)))
    source_name = f"{config.source_hdf}::{config.source_dataset}" if config.source_hdf is not None else "synthetic"
    data_axes = tuple(range(max(0, len(shape) - config.rank_of_data), len(shape)))
    normalized_units = str(ureg.Unit(config.units))
    return ChunkPlan(
        schema_version="1.0",
        plan_id=config.plan_id,
        total_chunks=total_chunks,
        expected_chunk_ids=tuple(f"c{ordinal:0{width}d}" for ordinal in range(total_chunks)),
        outputs=(
            ChunkOutputLayout(
                output_id="signal",
                processing_path="/sample/signal",
                destination_path="sample/signal",
                units=normalized_units,
                rank_of_data=config.rank_of_data,
                arrays=(ChunkArrayLayout(component="signal", final_shape=shape, dtype=dtype.str),),
            ),
        ),
        driver={"source": source_name, "full_shape": list(shape), "dtype": dtype.str},
        batch_axes=(config.chunk_axis,),
        data_axes=data_axes,
        axis_rules=({"axis": config.chunk_axis, "chunk_size": config.chunk_size},),
    )


def _chunk_specs(config: BenchmarkConfig, plan: ChunkPlan, shape: tuple[int, ...]) -> list[ChunkSpec]:
    specs = []
    for ordinal, chunk_id in enumerate(plan.expected_chunk_ids):
        start = ordinal * config.chunk_size
        stop = min(start + config.chunk_size, shape[config.chunk_axis])
        selection = _selectors(len(shape), config.chunk_axis, start, stop)
        selected_shape = list(shape)
        selected_shape[config.chunk_axis] = stop - start
        specs.append(
            ChunkSpec(
                schema_version=plan.schema_version,
                plan_id=plan.plan_id,
                plan_hash=plan.plan_hash,
                chunk_id=chunk_id,
                ordinal=ordinal,
                grid_index=(ordinal,),
                source_selection=selection,
                expected_input_shape=tuple(selected_shape),
                placements=(
                    ChunkPlacement(
                        output_id="signal",
                        destination_selection=selection,
                        expected_shape=tuple(selected_shape),
                    ),
                ),
            )
        )
    if config.write_order == "reverse":
        specs.reverse()
    return specs


def _synthetic_values(spec: ChunkSpec, dtype: np.dtype[Any]) -> np.ndarray:
    return np.full(spec.expected_input_shape, spec.ordinal, dtype=dtype)


def _source_values(
    config: BenchmarkConfig,
    specs: list[ChunkSpec],
    dtype: np.dtype[Any],
) -> Iterator[tuple[ChunkSpec, np.ndarray]]:
    if config.source_hdf is None:
        for spec in specs:
            yield spec, _synthetic_values(spec, dtype)
        return
    with h5py.File(config.source_hdf, "r") as h5:
        dataset = h5[config.source_dataset]
        for spec in specs:
            yield spec, np.asarray(dataset[tuple(selector.to_index() for selector in spec.source_selection)])


def _processing_data(values: np.ndarray, config: BenchmarkConfig) -> ProcessingData:
    processing_data = ProcessingData()
    bundle = DataBundle()
    bundle["signal"] = BaseData(
        signal=values,
        units=ureg.Unit(config.units),
        rank_of_data=config.rank_of_data,
    )
    processing_data["sample"] = bundle
    return processing_data


def _validate_output(
    config: BenchmarkConfig,
    specs: list[ChunkSpec],
    dtype: np.dtype[Any],
) -> None:
    source = h5py.File(config.source_hdf, "r") if config.source_hdf is not None else None
    try:
        with h5py.File(config.output, "r") as assembled:
            output = assembled[f"processing/result/{config.run_name}/sample/signal/signal"]
            for spec in specs:
                selection = tuple(selector.to_index() for selector in spec.source_selection)
                expected = (
                    _synthetic_values(spec, dtype)
                    if source is None
                    else np.asarray(source[config.source_dataset][selection])
                )
                np.testing.assert_array_equal(output[selection], expected)
    finally:
        if source is not None:
            source.close()


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run one identity assembly benchmark and return a JSON-ready report."""

    shape, dtype = _resolve_source(config)
    if config.source_hdf is not None and config.source_hdf.resolve() == config.output.resolve():
        raise ValueError("output must not overwrite the source HDF5 file.")
    if config.output.exists():
        if not config.overwrite:
            raise FileExistsError(f"Output already exists: {config.output}")
        config.output.unlink()
    config.output.parent.mkdir(parents=True, exist_ok=True)

    plan = _build_plan(config, shape, dtype)
    specs = _chunk_specs(config, plan, shape)
    sink = HDFChunkedProcessingSink(
        resource_location=config.output,
        iosink_method_kwargs={
            "compression": config.compression,
            "compression_opts": config.compression_level,
            "chunks": config.storage_chunks,
        },
    )
    rss_before = _peak_rss_bytes()
    started = _utc_now()
    start = perf_counter()
    sink.initialize_chunked(config.run_name, plan)
    initialized = perf_counter()
    for spec, values in _source_values(config, specs, dtype):
        sink.write_chunk(config.run_name, _processing_data(values, config), plan=plan, chunk=spec)
    written = perf_counter()
    sink.finalize_chunked(config.run_name, plan=plan)
    finalized = perf_counter()
    _validate_output(config, specs, dtype)
    validated = perf_counter()
    rss_peak = _peak_rss_bytes()
    logical_bytes = math.prod(shape) * dtype.itemsize

    with h5py.File(config.output, "r") as h5:
        dataset = h5[f"processing/result/{config.run_name}/sample/signal/signal"]
        storage_layout = {
            "chunks": None if dataset.chunks is None else list(dataset.chunks),
            "compression": dataset.compression,
            "compression_opts": dataset.compression_opts,
        }

    write_seconds = written - initialized
    return {
        "schema_version": "1.0",
        "status": "passed",
        "started_utc": started,
        "finished_utc": _utc_now(),
        "modacor_version": __version__,
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "source": {
            "kind": "hdf" if config.source_hdf is not None else "synthetic",
            "path": None if config.source_hdf is None else str(config.source_hdf),
            "dataset": None if config.source_hdf is None else config.source_dataset,
        },
        "output": str(config.output),
        "shape": list(shape),
        "dtype": dtype.str,
        "rank_of_data": config.rank_of_data,
        "chunk_axis": config.chunk_axis,
        "chunk_size": config.chunk_size,
        "chunk_count": plan.total_chunks,
        "write_order": config.write_order,
        "logical_bytes": logical_bytes,
        "output_file_bytes": config.output.stat().st_size,
        "storage_layout": storage_layout,
        "timing_s": {
            "initialize": initialized - start,
            "write": write_seconds,
            "finalize": finalized - written,
            "validate": validated - finalized,
            "total": validated - start,
        },
        "write_throughput_mib_s": logical_bytes / (1024**2) / write_seconds if write_seconds else None,
        "memory": {
            "measurement": "process_peak_rss",
            "baseline_bytes": rss_before,
            "peak_bytes": rss_peak,
            "peak_growth_bytes": None if rss_peak is None or rss_before is None else max(0, rss_peak - rss_before),
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-hdf", type=Path, help="External HDF5/NeXus input file.")
    source.add_argument("--shape", type=_parse_int_tuple, help="Synthetic array shape, e.g. 24,1679,1475.")
    parser.add_argument("--source-dataset", default="entry/data", help="Dataset within --source-hdf.")
    parser.add_argument("--output", type=Path, required=True, help="Assembled benchmark HDF5 file.")
    parser.add_argument("--report", type=Path, help="Optional JSON report path; JSON is always printed.")
    parser.add_argument("--chunk-axis", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--rank-of-data", type=int, default=2)
    parser.add_argument("--units", default="dimensionless")
    parser.add_argument("--dtype", default="float32", help="Synthetic dtype; ignored for HDF input.")
    parser.add_argument("--compression", choices=("none", "gzip", "lzf"), default="none")
    parser.add_argument("--compression-level", type=int)
    parser.add_argument("--storage-chunks", type=_parse_int_tuple, help="Explicit HDF chunk shape; default is auto.")
    parser.add_argument("--write-order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--plan-id", default="benchmark")
    parser.add_argument("--run-name", default="benchmark")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        output=args.output,
        shape=args.shape,
        source_hdf=args.source_hdf,
        source_dataset=args.source_dataset.strip("/"),
        chunk_axis=args.chunk_axis,
        chunk_size=args.chunk_size,
        rank_of_data=args.rank_of_data,
        units=args.units,
        dtype=args.dtype,
        compression=None if args.compression == "none" else args.compression,
        compression_level=args.compression_level,
        storage_chunks=True if args.storage_chunks is None else args.storage_chunks,
        write_order=args.write_order,
        plan_id=args.plan_id,
        run_name=args.run_name,
        overwrite=args.overwrite,
    )
    try:
        report = run_benchmark(config)
    except Exception as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        if args.report.resolve() == args.output.resolve():
            parser.error("report must not overwrite the HDF5 output file")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
