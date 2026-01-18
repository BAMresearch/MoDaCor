# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Malte Storm", "Tim Snow", "Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__version__ = "20250522.1"
__all__ = ["PoissonUncertainties"]

from pathlib import Path

import numpy as np

# from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

# from typing import Any


class PoissonUncertainties(ProcessStep):
    """
    Adding Poisson uncertainties to the data
    """

    documentation = ProcessStepDescriber(
        calling_name="Add Poisson Uncertainties",
        calling_id="PoissonUncertainties",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={"variances": ["Poisson"]},
        argument_specs={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "doc": "ProcessingData keys to update with Poisson variances.",
            },
        },
        step_keywords=["uncertainties", "Poisson"],
        step_doc="Add Poisson uncertainties to the data",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="This is a simple Poisson uncertainty calculation based on the signal intensity",
    )

    def calculate(self):
        """
        Calculate the Poisson uncertainties for the data
        """

        # Get the data
        data = self.processing_data
        output = {}
        for key in self.configuration["with_processing_keys"]:
            databundle = data.get(key)
            signal = databundle["signal"].signal

            # Add the variance to the data
            databundle["signal"].variances["Poisson"] = np.clip(signal, 1, None)
            output[key] = databundle
        return output
