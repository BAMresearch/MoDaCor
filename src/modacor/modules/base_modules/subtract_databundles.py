# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw", "Armin Moser"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "29/10/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["SubtractDatabundles"]
__version__ = "20251029.1"

from pathlib import Path

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class SubtractDatabundles(ProcessStep):
    """
    Subtract a DataBundle from a DataBundle, useful for background subtraction
    """

    documentation = ProcessStepDescriber(
        calling_name="Subtract another DataBundle",
        calling_id="SubtractDatabundles",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={"signal": ["signal", "uncertainties", "units"]},
        default_configuration={},  # no arguments needed
        argument_specs={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "doc": "Two processing keys: minuend then subtrahend.",
            },
        },
        step_keywords=["subtract", "background", "databundle"],
        step_doc="Subtract a DataBundle element using another DataBundle",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="""
            This subtracts one DataBundle's signal from another, useful for background subtraction.
            'with_processing_keys' in the configuration should contain two keys, the operation
            will subtract the second key's DataBundle from the first key's DataBundle.
        """,
    )

    def calculate(self) -> dict[str, DataBundle]:
        # actual work happens here:
        assert len(self.configuration["with_processing_keys"]) == 2, (
            "SubtractDatabundles requires exactly two processing keys in 'with_processing_keys': "
            "the first is the minuend, the second is the subtrahend."
        )
        minuend_key = self.configuration["with_processing_keys"][0]
        minuend = self.processing_data.get(minuend_key)
        subtrahend = self.processing_data.get(self.configuration["with_processing_keys"][1])
        # subtract the data
        minuend["signal"] -= subtrahend["signal"]
        output: dict[str, DataBundle] = {minuend_key: minuend}
        return output
