from __future__ import annotations

from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.divide_databundles import DivideDatabundles
from modacor.modules.base_modules.multiply_databundles import MultiplyDatabundles
from modacor.modules.base_modules.subtract_databundles import SubtractDatabundles


def test_divide_databundles_dependency_contract_is_asymmetric() -> None:
    step = DivideDatabundles(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["sample", "normalization"],
        dividend_data_key="intensity",
        divisor_data_key="transmission",
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"sample.intensity", "normalization.transmission"},
        processing_writes={"sample.intensity"},
    )


def test_multiply_databundles_dependency_contract_is_asymmetric() -> None:
    step = MultiplyDatabundles(io_sources=IoSources())
    step.modify_config_by_kwargs(
        with_processing_keys=["sample", "calibration"],
        multiplicand_data_key="intensity",
        multiplier_data_key="scale",
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"sample.intensity", "calibration.scale"},
        processing_writes={"sample.intensity"},
    )


def test_subtract_databundles_dependency_contract_is_asymmetric() -> None:
    step = SubtractDatabundles(io_sources=IoSources())
    step.modify_config_by_kwargs(with_processing_keys=["sample", "background"])

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"sample.signal", "background.signal"},
        processing_writes={"sample.signal"},
    )
