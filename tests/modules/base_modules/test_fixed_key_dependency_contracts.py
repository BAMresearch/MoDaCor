from __future__ import annotations

from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.io.io_sources import IoSources
from modacor.modules.base_modules.poisson_uncertainties import PoissonUncertainties
from modacor.modules.technique_modules.scattering.solid_angle_correction import SolidAngleCorrection


def test_poisson_uncertainties_dependency_contract_is_exact() -> None:
    step = PoissonUncertainties(io_sources=IoSources())
    step.modify_config_by_kwargs(with_processing_keys=["sample", "background"])

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"sample.signal", "background.signal"},
        processing_writes={"sample.signal", "background.signal"},
    )


def test_solid_angle_dependency_contract_is_exact() -> None:
    step = SolidAngleCorrection(io_sources=IoSources())
    step.modify_config_by_kwargs(with_processing_keys=["sample", "background"])

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={
            "sample.signal",
            "sample.Omega",
            "background.signal",
            "background.Omega",
        },
        processing_writes={"sample.signal", "background.signal"},
    )
