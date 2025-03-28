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
        if (intensity_object := self.kwargs.get("I", None)) is None:
            self.message_handler.error(
                "Signal ('I') data is required for PoissonUncertainty."
            )
            return False
        if not (intensity_object.internal_units == ureg.counts):
            self.message_handler.error(
                "Signal data ('I') should have units of counts."
            )
            return False

        return True

    @dask.delayed
    def apply(self):
        intensity_object: BaseData = self.kwargs["I"]
        intensity_object.uncertainties.append(
            da.clip(intensity_object.internal_data, 1, da.inf)**0.5
            )
        # self.message_handler.info("PoissonUncertainty applied successfully.")