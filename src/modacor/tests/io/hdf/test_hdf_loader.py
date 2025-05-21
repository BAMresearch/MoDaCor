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

from modacor.io.hdf.hdf_loader import *

from os.path import abspath
from logging import WARNING
import numpy as np
import h5py


class TestHDFLoader(unittest.TestCase):
    """Testing class for modacor/io/hdf/hdf_loader.py"""

    def setUp(self):
        self.test_hdf_loader = HDFLoader()
        self.test_file_path = "tbd - some form of temp file"
        self.test_dataset_name = "dataset"
        self.test_dataset_shape = (10, 2)


    def tearDown(self):
        self.test_h5_loader = None
        self.test_file_path = None
        self.test_dataset_name = None
        self.test_dataset_shape = None


    def test_open_file(self):
        absolute_test_file_path = abspath(self.test_file_path)
        self.test_h5_loader._open_file(self.test_file_path)

        self.assertEqual(absolute_test_file_path, self.test_h5_loader._file_path)
        self.assertEqual(self.test_dataset_name, self.test_h5_loader._file_datasets[0])
        self.assertEqual(self.test_dataset_shape, self.test_h5_loader._file_datasets_shapes[self.test_dataset_name])


    def test_close_file(self):
        self.test_open_file()
        self.test_h5_loader._close_file()

        self.assertEqual(None, self.test_h5_loader._file_path)
        self.assertEqual(None, self.test_h5_loader._file_reference)
        self.assertEqual([], self.test_h5_loader._file_datasets)
        self.assertEqual({}, self.test_h5_loader._file_datasets_shapes)
