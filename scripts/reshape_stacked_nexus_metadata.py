# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""Make stacked NeXus metadata arrays broadcast cleanly with MoDaCor data arrays.

Standard stacked NeXus files often store per-stack metadata as ``(N,)`` or ``(N, 1)``
while the main detector data is shaped like ``(N, 1, Y, X)``. NumPy aligns
broadcasting from the right, so those metadata arrays do not broadcast to the data
shape. This script appends trailing singleton dimensions, e.g. ``(N, 1)`` becomes
``(N, 1, 1, 1)``.
"""

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_with_local_venv() -> None:
    """Retry with a project-local interpreter if current Python misses deps."""
    if os.environ.get("MODACOR_NEXUS_RESHAPE_REEXEC") == "1":
        return

    for candidate in (PROJECT_ROOT / ".venv" / "bin" / "python", PROJECT_ROOT / ".venv-dev" / "bin" / "python"):
        if not candidate.exists():
            continue
        if Path(sys.executable).resolve() == candidate.resolve():
            continue

        env = dict(os.environ)
        env["MODACOR_NEXUS_RESHAPE_REEXEC"] = "1"
        completed = subprocess.run([str(candidate), *sys.argv], env=env, check=False)
        raise SystemExit(completed.returncode)


try:
    import h5py
except ModuleNotFoundError:
    _maybe_reexec_with_local_venv()
    raise


DEFAULT_DATA_PATH = "entry1/instrument/detector00/data"


@dataclass(frozen=True)
class ReshapePlan:
    path: str
    old_shape: tuple[int, ...]
    new_shape: tuple[int, ...]


def _clean_hdf_path(path: str) -> str:
    return path.strip("/")


def _is_numeric_dataset(dataset: h5py.Dataset) -> bool:
    return np.issubdtype(dataset.dtype, np.number) or np.issubdtype(dataset.dtype, np.bool_)


def _planned_shape(
    dataset_shape: tuple[int, ...],
    data_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    if not dataset_shape or len(dataset_shape) >= len(data_shape):
        return None
    if data_shape[: len(dataset_shape)] != dataset_shape:
        return None

    trailing_dims = len(data_shape) - len(dataset_shape)
    return dataset_shape + (1,) * trailing_dims


def discover_reshape_plan(
    h5: h5py.File,
    *,
    data_path: str = DEFAULT_DATA_PATH,
    metadata_paths: Sequence[str] | None = None,
    exclude_metadata_paths: Sequence[str] | None = None,
    include_non_numeric: bool = False,
) -> list[ReshapePlan]:
    """Find metadata datasets that should gain trailing singleton dimensions."""
    data_path = _clean_hdf_path(data_path)
    if data_path not in h5:
        raise KeyError(f"Main data path not found: {data_path}")
    if not isinstance(h5[data_path], h5py.Dataset):
        raise TypeError(f"Main data path is not a dataset: {data_path}")

    data_shape = tuple(h5[data_path].shape)
    plans: list[ReshapePlan] = []

    explicit_paths = bool(metadata_paths)
    excluded_paths = {_clean_hdf_path(path) for path in exclude_metadata_paths or ()}

    def maybe_add(path: str, dataset: h5py.Dataset) -> None:
        if path == data_path:
            return
        if not explicit_paths and not include_non_numeric and not _is_numeric_dataset(dataset):
            return
        new_shape = _planned_shape(tuple(dataset.shape), data_shape)
        if new_shape is None or new_shape == tuple(dataset.shape):
            return
        plans.append(ReshapePlan(path=path, old_shape=tuple(dataset.shape), new_shape=new_shape))

    if metadata_paths:
        for raw_path in metadata_paths:
            path = _clean_hdf_path(raw_path)
            if path not in h5:
                raise KeyError(f"Metadata path not found: {path}")
            dataset = h5[path]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"Metadata path is not a dataset: {path}")
            maybe_add(path, dataset)
    else:

        def visitor(path: str, obj: object) -> None:
            if path in excluded_paths:
                return
            if isinstance(obj, h5py.Dataset):
                maybe_add(path, obj)

        h5.visititems(visitor)

    return plans


def _creation_kwargs(dataset: h5py.Dataset, new_shape: tuple[int, ...]) -> dict:
    kwargs = {
        "dtype": dataset.dtype,
        "compression": dataset.compression,
        "compression_opts": dataset.compression_opts,
        "shuffle": dataset.shuffle,
        "fletcher32": dataset.fletcher32,
        "scaleoffset": dataset.scaleoffset,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    if dataset.chunks is not None:
        trailing_dims = len(new_shape) - len(dataset.shape)
        kwargs["chunks"] = tuple(dataset.chunks) + (1,) * trailing_dims
        kwargs["maxshape"] = tuple(dataset.maxshape) + (1,) * trailing_dims

    return kwargs


def apply_reshape_plan(h5: h5py.File, plans: Iterable[ReshapePlan]) -> list[ReshapePlan]:
    """Rewrite each planned dataset with the requested shape, preserving values and attrs."""
    applied: list[ReshapePlan] = []

    for plan in plans:
        dataset = h5[plan.path]
        data = dataset[()].reshape(plan.new_shape)
        attrs = {key: dataset.attrs[key] for key in dataset.attrs}
        kwargs = _creation_kwargs(dataset, plan.new_shape)

        parent_path, dataset_name = plan.path.rsplit("/", 1) if "/" in plan.path else ("", plan.path)
        parent = h5[parent_path] if parent_path else h5
        del parent[dataset_name]
        new_dataset = parent.create_dataset(dataset_name, data=data, **kwargs)
        for key, value in attrs.items():
            new_dataset.attrs[key] = value
        applied.append(plan)

    return applied


def convert_file(
    input_path: Path,
    output_path: Path | None,
    *,
    data_path: str = DEFAULT_DATA_PATH,
    metadata_paths: Sequence[str] | None = None,
    exclude_metadata_paths: Sequence[str] | None = None,
    include_non_numeric: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[ReshapePlan]:
    """Copy *input_path* to *output_path* and reshape compatible metadata arrays."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    target_path = Path(output_path) if output_path is not None else input_path
    if target_path != input_path and not dry_run:
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {target_path}")
        shutil.copy2(input_path, target_path)

    mode = "r" if dry_run else "r+"
    with h5py.File(target_path if not dry_run else input_path, mode) as h5:
        plans = discover_reshape_plan(
            h5,
            data_path=data_path,
            metadata_paths=metadata_paths,
            exclude_metadata_paths=exclude_metadata_paths,
            include_non_numeric=include_non_numeric,
        )
        if dry_run:
            return plans
        return apply_reshape_plan(h5, plans)


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_modacor{input_path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append trailing singleton dimensions to stacked NeXus metadata arrays so they "
            "broadcast against MoDaCor's main detector data."
        )
    )
    parser.add_argument("input", type=Path, help="Input HDF5/NeXus file.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output file. Defaults to '<input stem>_modacor<input suffix>'.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the input file directly instead of writing a converted copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report datasets that would be reshaped without writing changes.",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"Main detector data dataset path. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--metadata-path",
        action="append",
        dest="metadata_paths",
        help=(
            "Metadata dataset to reshape. Repeat to list several paths. "
            "When omitted, numeric prefix-shaped datasets are discovered automatically."
        ),
    )
    parser.add_argument(
        "--exclude-metadata-path",
        action="append",
        dest="exclude_metadata_paths",
        help=("Metadata dataset to leave untouched during automatic discovery. " "Repeat to exclude several paths."),
    )
    parser.add_argument(
        "--include-non-numeric",
        action="store_true",
        help="Also reshape non-numeric datasets during automatic discovery.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.in_place and args.output is not None:
        raise SystemExit("Use either --in-place or an output path, not both.")

    output_path = None if args.in_place else (args.output or _default_output_path(args.input))
    plans = convert_file(
        args.input,
        output_path,
        data_path=args.data_path,
        metadata_paths=args.metadata_paths,
        exclude_metadata_paths=args.exclude_metadata_paths,
        include_non_numeric=args.include_non_numeric,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    target = args.input if args.in_place or args.dry_run else output_path
    action = "Would reshape" if args.dry_run else "Reshaped"
    print(f"{action} {len(plans)} dataset(s) in {target}:")
    for plan in plans:
        print(f"  {plan.path}: {plan.old_shape} -> {plan.new_shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
