# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pytest

from modacor.cli import main


def _write_empty_pipeline(path: Path) -> None:
    path.write_text("name: empty\nsteps: {}\n", encoding="utf-8")


def test_cli_run_smoke(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yml"
    _write_empty_pipeline(pipeline_path)

    rc = main(["run", "--pipeline", str(pipeline_path)])

    assert rc == 0


def test_cli_write_hdf_requires_write_path(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yml"
    output_path = tmp_path / "out.h5"
    _write_empty_pipeline(pipeline_path)

    with pytest.raises(ValueError, match="--write-hdf requires at least one --write-path"):
        main(
            [
                "run",
                "--pipeline",
                str(pipeline_path),
                "--write-hdf",
                str(output_path),
            ]
        )
