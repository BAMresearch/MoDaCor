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

from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_registry import ProcessStepRegistry, register_as_process_step


@pytest.fixture
def ps_registry():
    """
    Fixture to create a mock IoRegistry for testing.
    """
    _initial_keys = tuple(ProcessStepRegistry.keys())
    yield ProcessStepRegistry
    for _key in [_key for _key in ProcessStepRegistry.keys() if _key not in _initial_keys]:
        del ProcessStepRegistry[_key]


def test_register_as_process_step__any_class(ps_registry):
    """
    Test the register_as_process_step decorator.
    """
    with pytest.raises(TypeError):

        @register_as_process_step
        class MockProcessStep:
            type_reference = "mock_step"


@pytest.mark.parametrize("reference", [42, None, ["spam", "ham"]])
def test_register_as_process_step__wrong_type_reference(ps_registry, reference):
    with pytest.raises(AttributeError):

        @register_as_process_step
        class MockProcessStep(ProcessStep):
            type_reference = reference


def test_register_as_process_step__valid(ps_registry):
    """
    Test the register_as_process_step decorator with a valid class.
    """

    @register_as_process_step
    class MockProcessStep(ProcessStep):
        type_reference = "mock_step"

    assert "mock_step" in ps_registry
    assert ps_registry["mock_step"] == MockProcessStep


def test_register_as_process_step__multiple_classes(ps_registry):
    _test_classes = {}
    for _n in range(5):
        _Class = type(f"MockProcessStep{_n}", (ProcessStep,), {"type_reference": f"mock_step{_n}"})
        _Class = register_as_process_step(_Class)
        _test_classes[_n] = _Class

    for _n in range(5):
        assert f"mock_step{_n}" in ps_registry
        assert ps_registry[f"mock_step{_n}"] == _test_classes[_n]


def test_register_as_process_step__duplicate(ps_registry):
    """
    Test the register_as_process_step decorator with a duplicate class.
    """

    @register_as_process_step
    class MockProcessStep(ProcessStep):
        type_reference = "mock_step"

    with pytest.raises(ValueError):

        @register_as_process_step
        class DuplicateMockProcessStep(ProcessStep):
            type_reference = "mock_step"


if __name__ == "__main__":
    pytest.main()
