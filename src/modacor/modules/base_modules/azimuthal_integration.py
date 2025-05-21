from __future__ import annotations

__author__ = "Jerome Kieffer"
__copyright__ = "MoDaCor team"
__license__ = "BSD3"
__date__ = "21/05/2025"

import warnings
from pathlib import Path

import numpy as np

from ...dataclasses.integrated_data import IntegratedData
from ...dataclasses.process_step import ProcessStep
from ...dataclasses.process_step_describer import ProcessStepDescriber
from ...dataclasses.validators import check_data

# import pint
# from scipy.sparse import csc_matrix


class AzimuthalIntegration(ProcessStep):
    """
    A class that performes azimuthal integreation with variance propagation
    """

    documentation = ProcessStepDescriber(
        calling_name="Azimuthal Integration",
        calling_id="AzimuthalIntegration",
        calling_module_path=Path(__file__),
        calling_version="0.0.1",
        required_data_keys=["signal"],
        works_on={"signal": ["raw_data", "variances", "normalization", "normalization_factor"]},
        step_keywords=["average"],
        step_doc="Add azimuthal integration date with variance propagated",
        step_reference="DOI 10.1107/S1600576724011038",
        step_note=(
            "This is a simple Azimuthal integration step based on sparse matrix multiplication"
        ),
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__(self)
        self.documentation.calling_arguments = self.kwargs

    def can_apply(self) -> bool:
        """
        Check if the process can be applied to the given data.
        """
        return check_data(self.bundle, "Signal", None, self.message_handler)

    def apply(self, apply_scalers, **kwargs):
        source = self.bundle["signal"]
        signal = source.signal
        normalization = source.normalization
        integrated = self.bundle["integrated"] = IntegratedData(
            sum_signal=self.sparse.dot(signal),
            sum_normalization=self.sparse.dot(normalization),
            sum_normalization_squared=self.sparse_squared.dot(normalization * normalization),
            normalization_factor=source.normalization_factor,
            normalization_factor_variance=source.normalization_factor_variance,
            sem={},
            std={},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integrated.mean = integrated.sum_signal / integrated.sum_normalization
            for key, var in source.variances.items():
                integrated.sum_variance[key] = self.sparse_squared.dot(var)
                integrated.std = np.sqrt(integrated.sum_signal) / integrated.sum_normalization
                integrated.sem = np.sqrt(
                    integrated.sum_signal / integrated.sum_normalization_squared
                )
