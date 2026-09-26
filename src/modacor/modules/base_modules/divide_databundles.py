# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "25/09/2026"
__status__ = "Development"

__all__ = ["DivideDatabundles"]
__version__ = "20260925.1"

from pathlib import Path

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class DivideDatabundles(ProcessStep):
    """Divide a BaseData entry in one DataBundle by an entry in another."""

    documentation = ProcessStepDescriber(
        calling_name="Divide by another DataBundle",
        calling_id="DivideDatabundles",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={"signal": ["signal", "uncertainties", "units"]},
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": None,
                "doc": "Two processing keys: dividend then divisor.",
            },
            "dividend_data_key": {
                "type": str,
                "default": "signal",
                "doc": "BaseData key to modify in the dividend DataBundle.",
            },
            "divisor_data_key": {
                "type": str,
                "default": "signal",
                "doc": "BaseData key to read from the divisor DataBundle.",
            },
        },
        step_keywords=["divide", "normalize", "databundle"],
        step_doc="Divide a DataBundle entry using another DataBundle.",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note=(
            "with_processing_keys contains the dividend first and divisor second. "
            "BaseData supplies broadcasting, unit handling, and uncertainty propagation."
        ),
    )

    def calculate(self) -> dict[str, DataBundle]:
        keys = self._normalised_processing_keys()
        assert len(keys) == 2, (
            "DivideDatabundles requires exactly two processing keys in "
            "'with_processing_keys': the first is the dividend, the second is the divisor."
        )

        dividend_key, divisor_key = keys
        dividend = self.processing_data.get(dividend_key)
        divisor = self.processing_data.get(divisor_key)
        dividend_data_key = self.configuration["dividend_data_key"]
        divisor_data_key = self.configuration["divisor_data_key"]
        dividend[dividend_data_key] /= divisor[divisor_data_key]
        return {dividend_key: dividend}
