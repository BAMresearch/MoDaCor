# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "12/12/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["MultiplyDatabundles"]
__version__ = "20260927.2"

from pathlib import Path

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class MultiplyDatabundles(ProcessStep):
    """
    Multiply a DataBundle with another DataBundle, useful for scaling or combining data
    """

    documentation = ProcessStepDescriber(
        calling_name="Multiply another DataBundle",
        calling_id="MultiplyDatabundles",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={"signal": ["signal", "uncertainties", "units"]},
        arguments={
            "multiplicand_data_key": {
                "type": str,
                "default": "signal",
                "doc": "BaseData key to modify in the multiplicand DataBundle.",
            },
            "multiplier_data_key": {
                "type": str,
                "default": "signal",
                "doc": "BaseData key to read from the multiplier DataBundle.",
            },
        },
        step_keywords=["multiply", "scaling", "databundle"],
        step_doc="Multiply a DataBundle element using another DataBundle",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="""
            This multiplies one DataBundle's signal with another, useful for scaling or combining data.
            'with_processing_keys' in the configuration should contain two keys, the operation
            will multiply the first key's DataBundle by the second key's DataBundle.
        """,
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        processing_keys = normalize_processing_key_values(self.configuration.get("with_processing_keys"))
        if len(processing_keys) != 2:
            return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})

        multiplicand_key, multiplier_key = processing_keys
        multiplicand_path = f"{multiplicand_key}.{self.configuration['multiplicand_data_key']}"
        multiplier_path = f"{multiplier_key}.{self.configuration['multiplier_data_key']}"
        return ProcessStepDependencies(
            processing_reads={multiplicand_path, multiplier_path},
            processing_writes={multiplicand_path},
        )

    def calculate(self) -> dict[str, DataBundle]:
        # actual work happens here:
        keys = self._normalised_processing_keys()
        if len(keys) != 2:
            raise ValueError(
                "MultiplyDatabundles requires exactly two processing keys in 'with_processing_keys': "
                "the first is the multiplicand, the second is the multiplier."
            )
        multiplicand_key = keys[0]
        multiplicand = self.processing_data.get(multiplicand_key)
        multiplier = self.processing_data.get(keys[1])
        # multiply the data
        multiplicand[self.configuration["multiplicand_data_key"]] *= multiplier[
            self.configuration["multiplier_data_key"]
        ]
        output: dict[str, DataBundle] = {multiplicand_key: multiplicand}
        return output
