# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "06/06/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import tempfile
import unittest
from os import unlink
from pathlib import Path

import h5py
import numpy as np

from modacor.io.hdf.hdf_source import HDFSource


class TestHDFSource(unittest.TestCase):
    """Testing class for modacor/io/hdf/hdf_source.py"""

    def setUp(self):
        self.temp_file_handle = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False)
        self.temp_file_path = self.temp_file_handle.name
        self.temp_file_handle.close()
        self.temp_dataset_name = "dataset"
        self.temp_dataset_shape = (10, 2)
        with h5py.File(self.temp_file_path, "w") as hdf_file:
            hdf_file.create_dataset(
                self.temp_dataset_name, data=np.zeros(self.temp_dataset_shape), dtype="float64", compression="gzip"
            )

        self.test_hdf_source = HDFSource(source_reference="Test Data", resource_location=self.temp_file_path)

    def tearDown(self):
        self.test_hdf_source = None
        self.test_file_path = None
        self.test_dataset_name = None
        self.test_dataset_shape = None
        unlink(self.temp_file_path)

    def test_open_file(self):
        self.test_hdf_source._file_path = Path(self.temp_file_path)
        self.test_hdf_source._preload()
        self.assertEqual(Path(self.temp_file_path), self.test_hdf_source._file_path)
        # self.assertEqual(self.temp_dataset_name, list(self.test_hdf_source._data_cache.keys())[0])
        self.assertEqual(self.temp_dataset_shape, self.test_hdf_source._file_datasets_shapes[self.temp_dataset_name])

    def test_get_data(self):
        self.test_hdf_source._file_path = Path(self.temp_file_path)
        self.test_hdf_source._preload()
        data_array = self.test_hdf_source.get_data(self.temp_dataset_name)
        self.assertTrue(isinstance(data_array, np.ndarray))
        self.assertEqual(self.temp_dataset_shape, data_array.shape)

    def test_get_data_returns_copy_of_cached_array(self):
        data_array = self.test_hdf_source.get_data(self.temp_dataset_name)
        data_array[:] = 1.0

        second_data_array = self.test_hdf_source.get_data(self.temp_dataset_name)

        self.assertFalse(np.any(second_data_array))

    def test_get_data_with_slice(self):
        self.test_hdf_source._file_path = Path(self.temp_file_path)
        self.test_hdf_source._preload()
        data_array = self.test_hdf_source.get_data(self.temp_dataset_name, load_slice=(slice(0, 5), slice(None)))
        self.assertTrue(isinstance(data_array, np.ndarray))
        self.assertEqual((5, 2), data_array.shape)

    def test_get_data_cache_keeps_sliced_and_full_reads_separate(self):
        sliced = self.test_hdf_source.get_data(self.temp_dataset_name, load_slice=(slice(0, 5), slice(None)))
        full = self.test_hdf_source.get_data(self.temp_dataset_name)

        self.assertEqual((5, 2), sliced.shape)
        self.assertEqual(self.temp_dataset_shape, full.shape)

    def test_get_data_shape(self):
        self.test_hdf_source._file_path = Path(self.temp_file_path)
        self.test_hdf_source._preload()
        data_shape = self.test_hdf_source.get_data_shape(self.temp_dataset_name)
        self.assertEqual(self.temp_dataset_shape, data_shape)

    def test_get_data_dtype(self):
        self.test_hdf_source._file_path = Path(self.temp_file_path)
        self.test_hdf_source._preload()
        data_dtype = self.test_hdf_source.get_data_dtype(self.temp_dataset_name)
        self.assertEqual(np.dtype("float64"), data_dtype)

    def test_get_static_metadata(self):
        self.test_hdf_source._file_path = Path(self.temp_file_path)
        self.test_hdf_source._preload()
        static_metadata = self.test_hdf_source.get_static_metadata(self.temp_dataset_name)
        self.assertTrue(isinstance(static_metadata, np.ndarray))
        self.assertEqual(self.temp_dataset_shape, static_metadata.shape)

    def test_get_data_attributes(self):
        self.test_hdf_source._file_path = Path(self.temp_file_path)
        self.test_hdf_source._preload()
        data_attributes = self.test_hdf_source.get_data_attributes(self.temp_dataset_name)
        self.assertEqual({}, data_attributes)  # No attributes set, should return empty dict

    def test_get_static_metadata_decodes_scalar_bytes_dataset(self):
        with h5py.File(self.temp_file_path, "a") as hdf_file:
            hdf_file.create_dataset("unit_name", data=np.bytes_("mm"))

        source = HDFSource(source_reference="Test Data", resource_location=self.temp_file_path)

        self.assertEqual("mm", source.get_static_metadata("unit_name"))

    def test_get_data_attributes_decodes_numpy_bytes_values(self):
        with h5py.File(self.temp_file_path, "a") as hdf_file:
            hdf_file[self.temp_dataset_name].attrs["units"] = np.bytes_("mm")
            hdf_file[self.temp_dataset_name].attrs["labels"] = np.asarray([np.bytes_("x"), np.bytes_("y")])

        source = HDFSource(source_reference="Test Data", resource_location=self.temp_file_path)
        data_attributes = source.get_data_attributes(self.temp_dataset_name)

        self.assertEqual("mm", data_attributes["units"])
        np.testing.assert_array_equal(data_attributes["labels"], np.asarray(["x", "y"]))
