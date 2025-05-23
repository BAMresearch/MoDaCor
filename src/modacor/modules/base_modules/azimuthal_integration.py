from __future__ import annotations

__author__ = "Jérôme Kieffer"
__copyright__ = "MoDaCor team"
__license__ = "BSD3"
__date__ = "23/05/2025"

import warnings
from pathlib import Path

import numpy as np

# import pint
from scipy.sparse import csc_matrix

from ...dataclasses.integrated_data import IntegratedData
from ...dataclasses.process_step import ProcessStep
from ...dataclasses.process_step_describer import ProcessStepDescriber
from ...dataclasses.validators import check_data


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
        works_on={"data": ["signal", "variances", "normalization"]},
        step_keywords=["azimuthal integration"],
        step_doc="Add azimuthal integration data with variance propagated",
        step_reference="DOI 10.1107/S1600576724011038",
        step_note="This is a simple Azimuthal integration step based on sparse matrix multiplication",
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__(self)
        self.documentation.calling_arguments = self.kwargs

    def can_apply(self) -> bool:
        """
        Check if the process can be applied to the given data.
        """
        return check_data(self.bundle, "signal", None, self.message_handler)

    def _build_sparse(self, name, npt, range_=None):
        """Method which build the two sparse arrays from the name
        of the array in the databundle

        :param name: name of the "Q" dataset in the databundle
        :param npt: number of points expected in the histogram
        :param range_: 2-list of the lower and upper bound in the Q-range
        :return: the sparse matrix
        """
        positions = self.data[name].ravel()
        if range_ is None:
            range_ = [positions.min(), positions.max()]
        # increase slightly the range to include the upper bound pixel
        range_ = [range_[0], range_[1] * (1.0 + np.finfo("float32").eps)]
        bin_boundaries = np.histogram(positions, npt, range=range_)[1]
        row = np.digitize(positions, bin_boundaries) - 1
        size = row.size
        col = np.arange(size)
        dat = np.ones(size)
        self.sparse = csc_matrix(dat, (row, col), shape=(npt, positions.size))
        self.sparse_squared = self.sparse * self.sparse  # actually 1*1 == 1
        self.bin_centers = 0.5 * (bin_boundaries[1:] + bin_boundaries[:-1])
        return self.sparse

    def prepare(self):
        self._build_sparse(**self.configuration)

    def calculate(self, data: DataBundle, dataset="image", **kwargs: Any):
        # work around for `prepare` no being called:
        if "sparse" not in dir(self):
            self.prepare()

        source = data[dataset]
        signal_img = source.signal.ravel()
        normalization_img = source.normalization.ravel()

        integrated = IntegratedData(
            sum_signal=self.sparse.dot(signal_img),
            sum_normalization=self.sparse.dot(normalization_img),
            sum_normalization_squared=self.sparse_squared.dot(normalization_img * normalization_img),
            normalization_factor=source.normalization_factor,
            normalization_factor_variance=source.normalization_factor_variance,
            sem={},
            std={},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integrated.signal = integrated.sum_signal / integrated.sum_normalization
            integrated.normalization = np.ones_like(integrated.sum_signal)

            for key, var in source.variances.items():
                integrated.sum_variance[key] = self.sparse_squared.dot(var)
                integrated.sem[key] = np.sqrt(integrated.sum_signal) / integrated.sum_normalization
                integrated.std[key] = np.sqrt(integrated.sum_signal / integrated.sum_normalization_squared)
                integrated.variance[key] = integrated.sum_variance[key] / integrated.sum_normalization**2

        # now create the variance along an azimuthal ring
        avg_img = self._sparse.T.dot(integrated.signal)  # backproject the average value to the image
        delta = np.divide(signal_img, normalization_img, where=normalization_img != 0) - avg_img
        sum_var = self.sparse_squared.dot((delta * normalization_img) ** 2)
        integrated.sum_variance["azim"] = sum_var
        integrated.sem["azim"] = np.sqrt(sum_var) / integrated.sum_normalization
        integrated.std["azim"] = np.sqrt(sum_var / integrated.sum_normalization_squared)
        integrated.variance["azim"] = sum_var / (integrated.sum_normalization**2)
        return integrated
