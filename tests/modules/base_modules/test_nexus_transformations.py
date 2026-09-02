# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "02/09/2026"
__status__ = "Development"

import numpy as np
import pytest

from modacor.io.hdf.hdf_source import HDFSource
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.nexus_transformations import (
    load_nexus_detector_frame_inputs,
    resolve_nexus_transform_chain,
)


def _write_transform(dataset, *, transformation_type, units, vector, depends_on=".", offset=None, offset_units="mm"):
    dataset.attrs["transformation_type"] = transformation_type
    dataset.attrs["units"] = units
    dataset.attrs["vector"] = np.asarray(vector, dtype=float)
    dataset.attrs["depends_on"] = depends_on
    if offset is not None:
        dataset.attrs["offset"] = np.asarray(offset, dtype=float)
        dataset.attrs["offset_units"] = offset_units


@pytest.fixture
def nexus_detector_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "detector.nxs"
    with h5py.File(path, "w") as h5:
        module = h5.require_group("/entry1/instrument/detector/detector_module")
        transformations = h5.require_group("/entry1/instrument/detector/transformations")

        origin = transformations.create_dataset("origin_offset", data=100.0)
        _write_transform(
            origin,
            transformation_type="translation",
            units="mm",
            vector=[1.0, 0.0, 0.0],
        )

        rotation = transformations.create_dataset("euler_c", data=90.0)
        _write_transform(
            rotation,
            transformation_type="rotation",
            units="degree",
            vector=[0.0, 0.0, 1.0],
            depends_on="./origin_offset",
        )

        offset = module.create_dataset("module_offset", data=0.0)
        _write_transform(
            offset,
            transformation_type="translation",
            units="mm",
            vector=[0.0, 0.0, 1.0],
            depends_on="../transformations/euler_c",
        )

        fast = module.create_dataset("fast_pixel_direction", data=1.0)
        _write_transform(
            fast,
            transformation_type="translation",
            units="mm",
            vector=[1.0, 0.0, 0.0],
            depends_on="./module_offset",
        )

        slow = module.create_dataset("slow_pixel_direction", data=2.0)
        _write_transform(
            slow,
            transformation_type="translation",
            units="mm",
            vector=[0.0, 1.0, 0.0],
            depends_on="./module_offset",
        )
    return path


@pytest.fixture
def io_sources(nexus_detector_file):
    sources = IoSources()
    sources.register_source(HDFSource(source_reference="calibration", resource_location=nexus_detector_file))
    return sources


def test_resolve_nexus_transform_chain_translation_and_rotation(io_sources):
    result = resolve_nexus_transform_chain(
        io_sources,
        "calibration",
        "/entry1/instrument/detector/detector_module/module_offset",
    )

    np.testing.assert_allclose(result.translation, [0.1, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(result.rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)
    assert result.paths == (
        "/entry1/instrument/detector/transformations/origin_offset",
        "/entry1/instrument/detector/transformations/euler_c",
        "/entry1/instrument/detector/detector_module/module_offset",
    )


def test_load_nexus_detector_frame_inputs_first_pixel_center_origin(io_sources):
    frame = load_nexus_detector_frame_inputs(
        io_sources,
        source_reference="calibration",
        detector_path="/entry1/instrument/detector",
        module_origin="first_pixel_center",
    )

    np.testing.assert_allclose(frame.basis_fast, [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(frame.basis_slow, [-1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(frame.basis_normal, [0.0, -0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(frame.pixel_pitch_fast.signal, 0.001, atol=1e-15)
    np.testing.assert_allclose(frame.pixel_pitch_slow.signal, 0.002, atol=1e-15)

    # PixelCoordinates3D adds +0.5 fast and +0.5 slow internally, so a
    # first-pixel-centre module origin is converted back to a corner here.
    np.testing.assert_allclose(
        [frame.det_coord_x.signal, frame.det_coord_y.signal, frame.det_coord_z.signal],
        [0.101, -0.0005, 0.0],
        atol=1e-15,
    )
