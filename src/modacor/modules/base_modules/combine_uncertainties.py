# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "20/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["CombineUncertainties"]
__version__ = "20260120.1"

from pathlib import Path

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.dataclasses.uncertainty_tools import (
    combine_uncertainty_keys,
    normalize_uncertainty_combinations,
    quadrature_aggregator,
)


class CombineUncertainties(ProcessStep):
    """Combine multiple uncertainty entries on a :class:`~modacor.dataclasses.basedata.BaseData` element.

    The configured combinations are evaluated as root-sum-of-squares of the listed one-sigma
    uncertainties. Each combination writes (or overwrites) the target uncertainty key.

    Example configuration::

        combinations:
          stat_total: ["poisson", "readout"]
          geometry: ["pixel_index_slow", "pixel_index_fast"]

    The example above will create/update the keys ``"stat_total"`` and ``"geometry"`` on the
    target BaseData, combining the referenced uncertainties in quadrature.
    """

    documentation = ProcessStepDescriber(
        calling_name="Combine uncertainties in quadrature",
        calling_id="CombineUncertainties",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={"signal": ["uncertainties"]},
        default_configuration={
            "target_basedata_key": "signal",
            "combinations": {},
            "drop_source_keys": False,
            "ignore_missing": False,
        },
        argument_specs={
            "target_basedata_key": {
                "type": str,
                "required": False,
                "doc": "Name of the BaseData entry within each DataBundle to modify (default: 'signal').",
            },
            "combinations": {
                "type": dict,
                "required": True,
                "doc": "Mapping of output uncertainty key to an iterable of source keys to combine.",
            },
            "drop_source_keys": {
                "type": bool,
                "required": False,
                "doc": "Remove source uncertainty keys after combination (default: False).",
            },
            "ignore_missing": {
                "type": bool,
                "required": False,
                "doc": (
                    "If True, missing source keys are ignored (combinations use the available ones). "
                    "If all listed keys are missing, the combination is skipped."
                ),
            },
        },
        step_keywords=["uncertainties", "combine", "quadrature", "propagation"],
        step_doc="Combine selected uncertainties in quadrature and expose the result under new keys.",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note=(
            "Designed for SAXS/SANS pipelines where uncertainties such as Poisson, readout noise, "
            "and flat-field corrections are stored separately (see the MOUSE notebook examples)."
        ),
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def calculate(self) -> dict[str, DataBundle]:
        combinations_raw = self.configuration.get("combinations", {})
        combinations = normalize_uncertainty_combinations(combinations_raw)
        if not combinations:
            raise ValueError("CombineUncertainties requires a non-empty 'combinations' mapping in its configuration.")

        target_basedata_key = str(self.configuration.get("target_basedata_key", "signal"))
        drop_sources = bool(self.configuration.get("drop_source_keys", False))
        ignore_missing = bool(self.configuration.get("ignore_missing", False))

        output: dict[str, DataBundle] = {}

        for processing_key in self._normalised_processing_keys():
            databundle: DataBundle = self.processing_data.get(processing_key)
            if target_basedata_key not in databundle:
                raise KeyError(f"DataBundle '{processing_key}' does not contain BaseData '{target_basedata_key}'.")

            combine_uncertainty_keys(
                basedata=databundle[target_basedata_key],
                combinations=combinations,
                aggregator=quadrature_aggregator,
                drop_sources=drop_sources,
                ignore_missing=ignore_missing,
                logger=self.logger,
                target_name=f"BaseData '{target_basedata_key}' in DataBundle '{processing_key}'",
            )

            output[processing_key] = databundle

        return output
