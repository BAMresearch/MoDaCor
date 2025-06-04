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


# import tempfile
import unittest

# from os.path import abspath
# from logging import WARNING
# from os.path import abspath
# from os import unlink
import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.poisson_uncertainties import PoissonUncertainties


class TestPoissonUncertainties(unittest.TestCase):
    """Testing class for modacor/modules/base_modules/poisson_uncertainties.py"""

    def setUp(self):
        self.test_data = BaseData(
            signal=np.arange(100, dtype=float).reshape((10, 10)),
            uncertainties={
                "SEM": 0.2,  # scalar uncertainty, not used in this test
            },
            units=ureg.counts,
        )
        self.test_data_bundle = DataBundle(signal=self.test_data)

    def tearDown(self):
        pass

    def test_poisson_calculation(self):
        poisson_uncertainties = PoissonUncertainties(IoSources())
        poisson_uncertainties.calculate(self.test_data_bundle)  # adds to variance
        assert "Poisson" in self.test_data_bundle["signal"].uncertainties
        assert np.isclose(self.test_data_bundle["signal"].uncertainties["Poisson"].mean(), 6.625, atol=0.01)
