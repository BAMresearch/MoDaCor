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
__all__ = []

import numpy as np
import pytest

from modacor.io import IoSources
from modacor.io.io_source import IoSource

data_shape = (20, 20)


class OnesSource(IoSource):
    type_reference = "ones_source"

    def __attrs_post_init__(self):
        self.configuration["shape"] = data_shape

    def get_data(self, index, key):
        return np.ones(self.configuration["shape"])

    def get_metadata(self, key):
        return key


class IndexSource(IoSource):
    type_reference = "index_source"

    def __attrs_post_init__(self):
        self.configuration["shape"] = data_shape

    def get_data(self, index, key):
        return index * np.ones(self.configuration["shape"])

    def get_metadata(self, key):
        return key


@pytest.fixture
def io_sources():
    return IoSources()


def configure_test_sources(io_sources):
    _source_30x10 = OnesSource()
    _source_30x10.configuration["shape"] = (30, 10)
    io_sources.register_source("ones_30x10", _source_30x10)

    _source_20x20 = OnesSource()
    _source_20x20.configuration["shape"] = (20, 20)
    io_sources.register_source("ones_20x20", _source_20x20)

    _index_source_25x25 = IndexSource()
    _index_source_25x25.configuration["shape"] = (25, 25)
    io_sources.register_source("index_25x25", _index_source_25x25)
    return io_sources


@pytest.mark.parametrize("ref", [42, None, ["a", "ref2"]])
def test_register_source__wrong_ref_type(io_sources, ref):
    with pytest.raises(TypeError):
        io_sources.register_source(ref, OnesSource())


@pytest.mark.parametrize("source", [IoSource, object()])
def test_register_source__wrong_source_type(io_sources, source):
    with pytest.raises(TypeError):
        io_sources.register_source("source", source)


def test_register_source__duplicate_ref(io_sources):
    io_sources.register_source("source", OnesSource())
    with pytest.raises(ValueError):
        io_sources.register_source("source", OnesSource())


def test_register_source__valid(io_sources):
    _source = OnesSource()
    io_sources.register_source("ones", _source)
    assert "ones" in io_sources.defined_sources
    assert io_sources.defined_sources["ones"] == _source


def test_get_source__invalid(io_sources):
    with pytest.raises(KeyError):
        io_sources.get_source("invalid_source")


def test_get_source__valid(io_sources):
    io_sources = configure_test_sources(io_sources)
    for _key in io_sources.defined_sources:
        source = io_sources.get_source(_key)
        assert isinstance(source, IoSource)


def test_split_data_reference__invalid(io_sources):
    with pytest.raises(ValueError):
        io_sources.split_data_reference("invalid_reference")


@pytest.mark.parametrize("ref", ["test::ref", "test::ref::extra", "test::/entry/data", "test::/entry/data::extra"])
def test_split_data_reference__valid(io_sources, ref):
    source_ref, data_key = io_sources.split_data_reference(ref)
    assert source_ref == "test"
    assert data_key == ref.lstrip("test::")


if __name__ == "__main__":
    pytest.main()
