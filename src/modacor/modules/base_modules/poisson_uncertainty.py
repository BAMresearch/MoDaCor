# src/modacor/modules/base_modules/poisson_uncertainty.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from modacor.dataclasses import ProcessStepExecutor, BaseData, ProcessStepDescriber
import dask
from modacor import ureg
import dask.array as da


class PoissonUncertainty(ProcessStepExecutor):
    """
    A class to add Poisson uncertainty to a signal.
    This class is designed to be used as a processing step in the Modacor framework.
    """
    documentation = None

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.documentation = ProcessStepDescriber(
            calling_name = "Poisson Uncertainty estimator",
            calling_id   = "PoissonUncertainty",
            calling_module_path = Path("src/modacor/modules/base_modules/poisson_uncertainty.py"),
            calling_version = "0.1",
            required_data_keys=["Signal"],
            calling_arguments=['Signal'],
            step_keywords=['uncertainty', 'poisson', 'error', 'estimation', 'counting statistics'],
            step_doc="Adds the Poisson uncertainty on data if the internal_data is in units of counts",
            step_reference="DOI 10.1088/0953-8984/25/38/383201",
            step_note="This is a simple Poisson uncertainty calculator, the uncertainty of a measurement cannot be lower than this",
        )

    def can_apply(self) -> bool:
        """
        Check if the process can be applied to the given data.
        """
        # Check if the required data is available.. these checks should probably be abstracted and made generally available.
        if (intensity_object := self.kwargs.get("Signal", None)) is None:
            self.message_handler.error(
                "Signal data is required for PoissonUncertainty."
            )
            return False
        if not (intensity_object.internal_units == ureg.counts):
            self.message_handler.error(
                "Signal data should have units of counts."
            )
            return False

        return True

    def apply(self):
        # intensity_object: BaseData = self.kwargs["Signal"]
        self.kwargs["Signal"].uncertainties += [
            dask.delayed(
                da.clip(self.kwargs["Signal"].internal_data, 1, da.inf)**0.5
            )
        ]
        self.kwargs["Signal"].uncertainties_origins += ["PoissonUncertainty"] 