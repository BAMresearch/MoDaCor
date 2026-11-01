# SPDX-License-Identifier: BSD-3-Clause
"""Convert selected data in place, retaining references to coordinate axes."""

from __future__ import annotations

__all__ = ["ConvertUnits"]
__version__ = "20261101.1"

from pathlib import Path

import pint

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class ConvertUnits(ProcessStep):
    """Convert signal and absolute uncertainties while retaining relative weights."""

    documentation = ProcessStepDescriber(
        calling_name="Convert units",
        calling_id="ConvertUnits",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={},
        arguments={
            "units": {
                "dependency_role": "processing_read_write_basedata_key",
                "type": dict,
                "default": {},
                "required": True,
                "doc": "Mapping of BaseData key to target units, for example {Q: '1/nm'}.",
            },
            "with_processing_keys": {
                "type": (list, str),
                "default": "*",
                "doc": "List of DataBundle keys, or '*' for all bundles. Missing data keys are skipped only with '*'.",
            },
        },
        step_keywords=["units", "conversion", "output"],
        step_doc="Convert selected BaseData signals and all their named uncertainties to compatible units.",
        step_note=(
            "Conversion mutates each BaseData in place, preserving axis references and dimensionless relative weights. "
            "With '*' missing data keys are skipped and logged at INFO; with explicit bundles they raise KeyError."
        ),
    )

    def calculate(self) -> dict[str, DataBundle]:
        targets = self.configuration["units"]
        if any(not isinstance(key, str) or not key.strip() for key in targets):
            raise ValueError("ConvertUnits: units keys must be non-empty BaseData keys.")
        selection = self.configuration["with_processing_keys"]
        wildcard = selection == "*"
        if not wildcard and not isinstance(selection, list):
            raise TypeError("ConvertUnits: with_processing_keys must be a list or '*'.")
        bundle_keys = list(self.processing_data) if wildcard else self._normalised_processing_keys()
        output = {}
        for bundle_key in bundle_keys:
            bundle = self.processing_data[bundle_key]
            for key, target in targets.items():
                if key not in bundle:
                    if wildcard:
                        self.logger.info(f"ConvertUnits: skipping missing key {key!r} in bundle {bundle_key!r}.")
                        continue
                    raise KeyError(f"ConvertUnits: bundle {bundle_key!r} is missing key {key!r}.")
                data = bundle[key]
                try:
                    data.to_units(target)
                except (ValueError, TypeError, pint.PintError) as exc:
                    raise ValueError(
                        f"ConvertUnits: bundle {bundle_key!r}, key {key!r}: cannot convert {data.units} to {target!r}."
                    ) from exc
            output[bundle_key] = bundle
        return output
