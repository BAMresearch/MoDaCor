# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw", "Anja F. Hörmann"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "04/08/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports
__version__ = "20260804.1"

import numpy as np
import tempfile
import unittest
from pathlib import Path
from os import unlink
import h5py

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.io.hdf.hdf_source import HDFSource
from modacor.modules.technique_modules.grazing_incidence.pixel_coordinates_with_roll import PixelCoordinatesWithRoll
from modacor.modules.technique_modules.scattering.pixel_coordinates_3d import PixelCoordinates3D
# ----------------------------
# tests: pixel coordinate math
# ----------------------------


class TestPixelCoordinatesWithRoll(unittest.TestCase):
    """Testing class for modacor/modules/technique_modules/grazing_incidence/pixel_coordinates_with_roll.py"""

    def setUp(self):
        self.temp_file_handle = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False)
        self.temp_file_path = self.temp_file_handle.name
        self.temp_file_handle.close()
        self.temp_dataset_name = "dataset"
        self.temp_dataset_shape = (10, 1)
        datasets = {"det_coord_z": {"data": np.ones(self.temp_dataset_shape) * 0.3, "units": "m"},
                    "det_coord_x": {"data": np.ones(self.temp_dataset_shape) * 0.03, "units": "m"},
                    "det_coord_y": {"data": np.ones(self.temp_dataset_shape) * 0.07, "units": "m"},
                    "pixel_pitch_fast": {"data": np.ones(self.temp_dataset_shape) * 0.075, "units": "mm"},
                    "pixel_pitch_slow": {"data": np.ones(self.temp_dataset_shape) * 0.075, "units": "mm"},
                    }
        with h5py.File(self.temp_file_path, "w") as hdf_file:
            for key, properties in datasets.items():
                hdf_file.create_dataset(
                    key, data=properties["data"], dtype="float64", compression="gzip"
                )
                hdf_file[key].attrs.update({"units": properties["units"]})

        self.test_hdf_source = HDFSource(source_reference="testdata", resource_location=self.temp_file_path)
        self.test_processing_data = ProcessingData()
        signal = np.ones((2, 3), dtype=np.uint32)
        db = DataBundle(
            signal=BaseData(signal=signal, units=ureg.dimensionless, uncertainties={}),
        )
        self.test_processing_data["sample"] = db
        self.sources = IoSources()
        self.sources.register_source(self.test_hdf_source)

    def tearDown(self):
        self.test_hdf_source = None
        self.test_file_path = None
        self.test_dataset_name = None
        self.test_dataset_shape = None
        self.sources = None
        unlink(self.temp_file_path)

    def _make_step(self) -> PixelCoordinatesWithRoll:
        step = PixelCoordinatesWithRoll(io_sources=self.sources)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "det_coord_z_source": "testdata::det_coord_z",
            "det_coord_z_units_source": "testdata::det_coord_z@units",
            "det_coord_x_source": "testdata::det_coord_x",
            "det_coord_x_units_source": "testdata::det_coord_x@units",
            "det_coord_y_source": "testdata::det_coord_y",
            "det_coord_y_units_source": "testdata::det_coord_y@units",
            "pixel_pitch_fast_source": "testdata::pixel_pitch_fast",
            "pixel_pitch_fast_units_source": "testdata::pixel_pitch_fast@units",
            "pixel_pitch_slow_source": "testdata::pixel_pitch_slow",
            "pixel_pitch_slow_units_source": "testdata::pixel_pitch_slow@units",
            "basis_fast": [-1.0, 0.0, 0.0],
            "basis_slow": [0.0, -1.0, 0.0],
            "basis_normal": [0.0, 0.0, 1.0],
            "sample_roll": 0.0,
        }
        step.processing_data = self.test_processing_data
        return step

    def _make_pixel_coordinate_3d_step(self) -> PixelCoordinates3D:
        step = PixelCoordinates3D(io_sources=self.sources)
        step.configuration = {
            "with_processing_keys": ["sample"],
            "det_coord_z_source": "testdata::det_coord_z",
            "det_coord_z_units_source": "testdata::det_coord_z@units",
            "det_coord_x_source": "testdata::det_coord_x",
            "det_coord_x_units_source": "testdata::det_coord_x@units",
            "det_coord_y_source": "testdata::det_coord_y",
            "det_coord_y_units_source": "testdata::det_coord_y@units",
            "pixel_pitch_fast_source": "testdata::pixel_pitch_fast",
            "pixel_pitch_fast_units_source": "testdata::pixel_pitch_fast@units",
            "pixel_pitch_slow_source": "testdata::pixel_pitch_slow",
            "pixel_pitch_slow_units_source": "testdata::pixel_pitch_slow@units",
            "basis_fast": [-1.0, 0.0, 0.0],
            "basis_slow": [0.0, -1.0, 0.0],
            "basis_normal": [0.0, 0.0, 1.0],
        }
        step.processing_data = self.test_processing_data
        return step

    def test_pixel_coordinates_no_roll(self):
        """
        Ensure the module returns the same result as PixelCoordinates3D
        if the roll angle is zero.
        """

        step = self._make_step()
        step.execute(self.test_processing_data)
        out = step.processing_data["sample"]
        step_processing_data_3d = self._make_pixel_coordinate_3d_step()
        step_processing_data_3d.execute(self.test_processing_data)
        out_processing_data_3d = step_processing_data_3d.processing_data["sample"]

        np.testing.assert_array_equal(out["coord_x"].signal, out_processing_data_3d["coord_x"].signal)
        np.testing.assert_array_equal(out["coord_y"].signal, out_processing_data_3d["coord_y"].signal)
        np.testing.assert_array_equal(out["coord_z"].signal, out_processing_data_3d["coord_z"].signal)

    def test_pixel_coordinates_90deg_roll(self):
        """
        Compare outcome of 90 degree rotation to expectations
        """

        step = self._make_step()
        step.configuration["sample_roll"] = 90.0
        frame = step._load_canonical_frame(RoD=2, detector_shape=self.test_processing_data["sample"]["signal"].signal.shape,
                                           reference_signal=self.test_processing_data["sample"]["signal"])

        np.testing.assert_array_almost_equal(frame.e_fast, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_almost_equal(frame.e_slow, np.array([-1.0, 0.0, 0.0]))
