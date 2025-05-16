# src/modacor/modules/base_modules/poisson_uncertainty.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path

import pint
from modacor.dataclasses import ProcessStepExecutor, ProcessStepDescriber
import dask
import dask.array as da
from modacor.dataclasses.validators import check_data_element_and_units


class PoissonUncertainty(ProcessStepExecutor):
    """
    A class to add Poisson uncertainty to a signal.
    This class is designed to be used as a processing step in the Modacor framework.
    """
    documentation = ProcessStepDescriber(
            calling_name="Poisson Uncertainty estimator",
            calling_id="PoissonUncertainty",
            calling_module_path=Path("src/modacor/modules/base_modules/poisson_uncertainty.py"),
            calling_version="0.1",
            required_data_keys=["Signal"],
            works_on={'Signal': ['internal_data', 'uncertainties']},
            step_keywords=['uncertainty', 'poisson', 'error', 'estimation', 'counting statistics'],
            step_doc="Adds the Poisson uncertainty on data if the internal_data is in units of counts",
            step_reference="DOI 10.1088/0953-8984/25/38/383201",
            step_note="This is a simple Poisson uncertainty calculator, the uncertainty of a measurement cannot be lower than this",
        )
    
    def __attrs_post_init__(self):
        super().__attrs_post_init__(self)
        self.documentation.calling_arguments = self.kwargs

    def can_apply(self) -> bool:
        """
        Check if the process can be applied to the given data.
        """
        return check_data_element_and_units(self.data, "Signal", pint.Unit("counts"), self.message_handler)

    def apply(self, **kwargs):
        # intensity_object: BaseData = self.kwargs["Signal"]
        # self.start() # this timing doesn't make a lot of sense with dask delayed
        self.data.data["Signal"].uncertainties += [
            dask.delayed(
                da.clip(self.data.data["Signal"].internal_data, 1, da.inf)**0.5
            )
        ]
        self.data.data["Signal"].uncertainties_origins += ["PoissonUncertainty"]
        self.data.provenance += [self.documentation] # should be enough to recreate?
        # self.stop()