# src/modacor/modules/base_modules/poisson_uncertainty.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from modacor.dataclasses import ProcessStepExecutor, BaseData
import dask
from modacor import ureg
import dask.array as da


class PoissonUncertainty(ProcessStepExecutor):
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