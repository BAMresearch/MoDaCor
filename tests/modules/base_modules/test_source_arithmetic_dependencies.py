from __future__ import annotations

import pytest

from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.divide import Divide
from modacor.modules.base_modules.multiply import Multiply
from modacor.modules.base_modules.subtract import Subtract


@pytest.mark.parametrize(
    ("step_class", "source_prefix"),
    [
        (Divide, "divisor"),
        (Multiply, "multiplier"),
        (Subtract, "subtrahend"),
    ],
)
def test_source_arithmetic_dependency_contract_is_exact(step_class, source_prefix: str) -> None:
    step = step_class(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample", "background"],
            f"{source_prefix}_source": "measurement::/value",
            f"{source_prefix}_units_source": "calibration::/units",
            f"{source_prefix}_uncertainties_sources": {"SEM": "statistics::/sem"},
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        source_refs={"measurement", "calibration", "statistics"},
        processing_reads={"sample.signal", "background.signal"},
        processing_writes={"sample.signal", "background.signal"},
    )
