__coding__ = "utf-8"
__author__ = "Brian R. Pauw"
__copyright__ = "MoDaCor team"
__license__ = "BSD3"
__date__ = "23/05/2025"
__version__ = "20250523.1"
__status__ = "Development"  # "Development", "Production"


from pathlib import Path

from modacor import ureg
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.dataclasses.processingdata import ProcessingData
from modacor.math.variance_calculations import divide, multiply


class Multiply(ProcessStep):
    """
    Multiply BaseData with another BaseData
    """

    documentation = ProcessStepDescriber(
        calling_name="Multiply by data source variable",
        calling_id="MultiplyByVariable",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        works_on={"signal": ["scalar", "scalar_uncertainty", "signal", "uncertainties", "units"]},
        calling_arguments={
            "multiplier_source": None,
            "multiplier_units_source": None,
            "multiplier_uncertainties_source": None,
        },
        step_keywords=["multiply", "variable", "scalar", "array"],
        step_doc="Multiply by a variable or array loaded from a data source",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="""This loads a scalar (value, units and uncertainty)
            from an IOSource and applies it to the data""",
    )

    def calculate(self) -> dict[str, DataBundle]:
        """ """
        self.processing_data: ProcessingData
        keys = self.configuration.get("with_processing_keys")
        data: DataBundle = self.processing_data[keys[0]]  # type: ignore[assignment]

        # not sure how to get the multiplier BaseData objecct.. like a flatfield w/o uncertainties or units...

        # * * * Multiplication rules with multiplier m: * * *
        # 1. if m.signal has ndim=0, it is (skipped as) assumed to be 1.0 with uncertainty 0.0. IoSource should put 0-sized signals into scalar with scalar_uncertainty
        # 2. otherwise check broadcastability (using method in BaseData), and apply to data.signal.signal
        #   and m.uncertainties to data.signal.uncertainties. Note uncertainties rules
        # 3. apply scalar to data.signal.scalar, and scalar_uncertainty to data.signal.scalar_uncertainty
        # 4. apply units to data.signal.units

        # Uncertainties rules (also applies to variances):
        # - if any item in m.uncertainties is 0.0, we skip that item
        # - if any item in m.uncertainites is named 'propagate_to_all', it is propagated to all uncertainties in data.signal.uncertainties
        # - if m.signal has ndim=0, it is skipped (assumed to be 1.0 with uncertainty 0.0)
        # - remaining keys in m.uncertainties will be propagated to matching keys in data.signal.uncertainties, otherwise skipped

        # apply factor to the data
        f, v = multiply(
            data["signal"].scalar,
            data["signal"].normalization_factor_variances,
            self.configuration.get("scalar"),
            self.configuration.get("scalar_uncertainty", 0),
        )
        data["signal"].scalar = f
        # propagate uncertainties into normalization_factor_
        data["signal"].normalization_factor_variances = v
        # propagate units into normalization_factor_units
        data["signal"].normalization_factor_units /= ureg.Unit(self.configuration.get("units", "dimensionless"))

        return {key: data}
