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

import pytest

from modacor.io import IoSource
from modacor.io.io_registry import IoRegistry, register_as_io_source


@pytest.fixture
def io_registry():
    """
    Fixture to create a mock IoRegistry for testing.
    """
    _initial_keys = tuple(IoRegistry.keys())
    yield IoRegistry
    for _key in [_key for _key in IoRegistry.keys() if _key not in _initial_keys]:
        del IoRegistry[_key]


def test_register_as_io_source__any_class(io_registry):
    """
    Test the register_as_io_source decorator.
    """
    with pytest.raises(TypeError):

        @register_as_io_source
        class MockIoSource:
            type_reference = "mock_source"


@pytest.mark.parametrize("reference", [42, None, ["spam", "ham"]])
def test_register_as_io_source__wrong_type_reference(io_registry, reference):
    with pytest.raises(AttributeError):

        @register_as_io_source
        class MockIoSource(IoSource):
            type_reference = reference


def test_register_as_io_source__valid(io_registry):
    """
    Test the register_as_io_source decorator with a valid class.
    """

    @register_as_io_source
    class MockIoSource(IoSource):
        type_reference = "mock_source"

    assert "mock_source" in io_registry
    assert io_registry["mock_source"] == MockIoSource


def test_register_as_io_source__multiple_classes(io_registry):
    _test_classes = {}
    for _n in range(5):
        _Class = type(f"MockIoSource{_n}", (IoSource,), {"type_reference": f"mock_source{_n}"})
        _Class = register_as_io_source(_Class)
        _test_classes[_n] = _Class

    for _n in range(5):
        assert f"mock_source{_n}" in io_registry
        assert io_registry[f"mock_source{_n}"] == _test_classes[_n]


def test_register_as_io_source__duplicate(io_registry):
    """
    Test the register_as_io_source decorator with a duplicate class.
    """

    @register_as_io_source
    class MockIoSource(IoSource):
        type_reference = "mock_source"

    with pytest.raises(ValueError):

        @register_as_io_source
        class DuplicateMockIoSource(IoSource):
            type_reference = "mock_source"


if __name__ == "__main__":
    pytest.main()
