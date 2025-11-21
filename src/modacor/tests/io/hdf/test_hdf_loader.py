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

from ....io.hdf.hdf_loader import HDFLoader


class TestHDFLoader(unittest.TestCase):
    """Testing class for modacor/io/hdf/hdf_loader.py"""

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

        self.test_hdf_loader = HDFLoader(source_reference="Test Data", resource_location=self.temp_file_path)

    def tearDown(self):
        self.test_hdf_loader = None
        self.test_file_path = None
        self.test_dataset_name = None
        self.test_dataset_shape = None
        unlink(self.temp_file_path)

    def test_open_file(self):
        self.test_hdf_loader._file_path = Path(self.temp_file_path)
        self.test_hdf_loader._preload()
        self.assertEqual(Path(self.temp_file_path), self.test_hdf_loader._file_path)
        self.assertEqual(self.temp_dataset_name, self.test_hdf_loader._file_datasets[0])
        self.assertEqual(self.temp_dataset_shape, self.test_hdf_loader._file_datasets_shapes[self.temp_dataset_name])

    def test_get_data(self):
        self.test_hdf_loader._file_path = Path(self.temp_file_path)
        self.test_hdf_loader._preload()
        data_array = self.test_hdf_loader.get_data(self.temp_dataset_name)
        self.assertTrue(isinstance(data_array, np.ndarray))
        self.assertEqual(self.temp_dataset_shape, data_array.shape)

    def test_get_data_with_slice(self):
        self.test_hdf_loader._file_path = Path(self.temp_file_path)
        self.test_hdf_loader._preload()
        data_array = self.test_hdf_loader.get_data(self.temp_dataset_name, load_slice=(slice(0, 5), slice(None)))
        self.assertTrue(isinstance(data_array, np.ndarray))
        self.assertEqual((5, 2), data_array.shape)

    def test_get_data_shape(self):
        self.test_hdf_loader._file_path = Path(self.temp_file_path)
        self.test_hdf_loader._preload()
        data_shape = self.test_hdf_loader.get_data_shape(self.temp_dataset_name)
        self.assertEqual(self.temp_dataset_shape, data_shape)

    def test_get_data_dtype(self):
        self.test_hdf_loader._file_path = Path(self.temp_file_path)
        self.test_hdf_loader._preload()
        data_dtype = self.test_hdf_loader.get_data_dtype(self.temp_dataset_name)
        self.assertEqual(np.dtype("float64"), data_dtype)

    def test_get_static_metadata(self):
        self.test_hdf_loader._file_path = Path(self.temp_file_path)
        self.test_hdf_loader._preload()
        static_metadata = self.test_hdf_loader.get_static_metadata(self.temp_dataset_name)
        self.assertTrue(isinstance(static_metadata, np.ndarray))
        self.assertEqual(self.temp_dataset_shape, static_metadata.shape)

    def test_get_data_attributes(self):
        self.test_hdf_loader._file_path = Path(self.temp_file_path)
        self.test_hdf_loader._preload()
        data_attributes = self.test_hdf_loader.get_data_attributes(self.temp_dataset_name)
        self.assertEqual({}, data_attributes)  # No attributes set, should return empty dict
