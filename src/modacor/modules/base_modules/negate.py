# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["Negate"]
__version__ = "20260929.1"

from pathlib import Path

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class Negate(ProcessStep):
    """Negate a selected BaseData entry while preserving uncertainty magnitudes."""

    documentation = ProcessStepDescriber(
        calling_name="Negate BaseData values",
        calling_id="Negate",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={"configured data key": ["signal"]},
        arguments={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": True,
                "default": None,
                "doc": "DataBundle key or keys to update.",
            },
            "data_key": {
                "type": str,
                "default": "signal",
                "doc": "BaseData key whose nominal values are negated.",
            },
        },
        step_keywords=["negate", "sign", "BaseData"],
        step_doc="Negate a BaseData entry using its uncertainty-preserving unary operation.",
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        processing_keys = normalize_processing_key_values(self.configuration.get("with_processing_keys"))
        if not processing_keys:
            return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})
        data_key = str(self.configuration.get("data_key", "signal"))
        paths = {f"{processing_key}.{data_key}" for processing_key in processing_keys}
        return ProcessStepDependencies(processing_reads=paths, processing_writes=paths)

    def calculate(self) -> dict[str, DataBundle]:
        data_key = str(self.configuration.get("data_key", "signal"))
        output: dict[str, DataBundle] = {}
        for processing_key in self._normalised_processing_keys():
            bundle = self.processing_data[processing_key]
            bundle[data_key] = -bundle[data_key]
            output[processing_key] = bundle
        return output
