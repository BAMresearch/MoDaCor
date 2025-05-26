# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2025 MoDaCor Authors
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__license__ = "BSD-3-Clause"
__copyright__ = "Copyright 2025 MoDaCor Authors"
__status__ = "Alpha"


from modacor.modules.base_modules.poisson_uncertainties import *
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources

from os.path import abspath
from logging import WARNING
from os.path import abspath
from os import unlink
import numpy as np
import tempfile
import unittest
import h5py

TEST_IO_SOURCES = IoSources()


class TestPoissonUncertainties(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/poisson_uncertainties.py"""

    def setUp(self):
        self.test_processing_data = ProcessingData()
        self.test_data = BaseData(signal=np.arange(0, 100).reshape((10, 10)))
        self.test_data_bundle = DataBundle(signal=self.test_data)
        self.test_processing_data["bundle"] = self.test_data_bundle

    def tearDown(self):
        pass

    def test_poisson_calculation(self):
        poisson_uncertainties = PoissonUncertainties(io_sources=TEST_IO_SOURCES)
        poisson_uncertainties.modify_config("with_processing_keys", ["bundle"])
        poisson_uncertainties.processing_data = self.test_processing_data
        poisson_uncertainties.calculate()
