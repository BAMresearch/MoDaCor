# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import pytest

from modacor.dataclasses.process_step import ProcessStep
from modacor.runner.process_step_registry import ProcessStepRegistry, _pascal_to_snake


def test_register_and_get_process_step():
    registry = ProcessStepRegistry()

    class DummyStep(ProcessStep):
        def execute(self, **kwargs):
            pass

    registry.register(DummyStep)
    cls = registry.get("DummyStep")

    assert cls is DummyStep
    assert "DummyStep" in registry


def test_register_with_custom_name():
    registry = ProcessStepRegistry()

    class DummyStep(ProcessStep):
        def execute(self, **kwargs):
            pass

    registry.register(DummyStep, name="custom_name")
    cls = registry.get("custom_name")

    assert cls is DummyStep
    assert "custom_name" in registry


def test_register_non_process_step_raises():
    registry = ProcessStepRegistry()

    class NotAStep:
        pass

    with pytest.raises(TypeError):
        registry.register(NotAStep)


def test_get_unknown_without_base_package_raises():
    registry = ProcessStepRegistry()

    with pytest.raises(KeyError):
        registry.get("DoesNotExistStep")


def test_get_unknown_with_filesystem_discovery_disabled_raises_policy_message():
    registry = ProcessStepRegistry(curated_module=None, allow_filesystem_discovery=False)

    with pytest.raises(KeyError, match="filesystem discovery is disabled"):
        registry.get("DoesNotExistStep")


def test_pascal_to_snake_handles_digit_named_process_steps():
    assert _pascal_to_snake("Plot1DVisualization") == "plot_1d_visualization"
    assert _pascal_to_snake("Plot2DVisualization") == "plot_2d_visualization"
    assert _pascal_to_snake("Q2Mapper") == "q2_mapper"


def test_get_digit_named_plot_steps_with_filesystem_discovery():
    registry = ProcessStepRegistry(curated_module=None)

    assert registry.get("Plot1DVisualization").__name__ == "Plot1DVisualization"
    assert registry.get("Plot2DVisualization").__name__ == "Plot2DVisualization"
