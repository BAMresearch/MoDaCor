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


from logging import WARNING
from os.path import abspath

import h5py

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.messagehandler import MessageHandler

from ..io_source import IoSource


class HDFLoader(IoSource):
    def __init__(self, source_reference: str, logging_level=WARNING):
        super().__init__(source_reference)
        self.hdf_logger = MessageHandler(level=logging_level, name="hdf5logger")
        self._file_path = None
        self._file_reference = None
        self._file_datasets = []
        self._file_datasets_shapes = {}

    def _open_file(self, file_path=None):
        if file_path is None:
            error = "No filepath given"
            self.hdf_logger.log.error(error)
            raise OSError(error)

        try:
            self._file_reference = h5py.File(file_path, "r")
            self._file_path = abspath(file_path)
            self._file_reference.visititems(self._find_datasets)
        except OSError as error:
            self.hdf_logger.logger.error(error)
            raise OSError(error)

    def _close_file(self):
        try:
            self._file_reference.close()
            self._file_path = None
            self._file_reference = None
            self._file_datasets.clear()
            self._file_datasets_shapes.clear()
        except OSError as error:
            self.hdf_logger.log.error(error)
            raise OSError(error)

    def _find_datasets(self, path_name, path_object):
        """
        An internal function to be used to walk the tree of an HDF5 file and return a list of
        the datasets within
        """
        if isinstance(self._file_reference[path_name], h5py._hl.dataset.Dataset):
            self._file_datasets.append(path_name)
            self._file_datasets_shapes[path_name] = self._file_reference[path_name].shape

    def get_data(self, data_key: str) -> BaseData:
        raise (NotImplementedError("get_data method not yet implemented in HDFLoader class."))

    def get_static_metadata(self, data_key):
        raise (
            NotImplementedError(
                "get_static_metadata method not yet implemented in HDFLoader class."
            )
        )
