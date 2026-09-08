# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "15/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import unittest

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.poisson_uncertainties import PoissonUncertainties

# import h5py

TEST_IO_SOURCES = IoSources()


class TestPoissonUncertainties(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/poisson_uncertainties.py"""

    def setUp(self):
        self.test_processing_data = ProcessingData()
        self.test_data = BaseData(signal=np.arange(0, 100).reshape((10, 10)), units=ureg.Unit("count"))
        self.test_data_bundle = DataBundle(signal=self.test_data)
        self.test_processing_data["bundle"] = self.test_data_bundle

    def tearDown(self):
        pass

    def test_poisson_calculation(self):
        poisson_uncertainties = PoissonUncertainties(io_sources=TEST_IO_SOURCES)
        poisson_uncertainties.modify_config_by_kwargs(with_processing_keys=["bundle"])
        poisson_uncertainties.processing_data = self.test_processing_data
        poisson_uncertainties.calculate()
        assert "Poisson" in self.test_processing_data["bundle"]["signal"].variances.keys()

    def test_poisson_execution(self):
        poisson_uncertainties = PoissonUncertainties(io_sources=TEST_IO_SOURCES)
        poisson_uncertainties.modify_config_by_kwargs(with_processing_keys=["bundle"])
        poisson_uncertainties(self.test_processing_data)
        assert "Poisson" in self.test_processing_data["bundle"]["signal"].variances.keys()

    def test_poisson_result_values(self):
        poisson_uncertainties = PoissonUncertainties(io_sources=TEST_IO_SOURCES)
        poisson_uncertainties.modify_config_by_kwargs(with_processing_keys=["bundle"])
        poisson_uncertainties(self.test_processing_data)
        expected_variances = np.arange(0, 100).reshape((10, 10)).astype(float).clip(min=1)
        actual_variances = self.test_processing_data["bundle"]["signal"].variances["Poisson"]
        np.testing.assert_allclose(expected_variances, actual_variances)
