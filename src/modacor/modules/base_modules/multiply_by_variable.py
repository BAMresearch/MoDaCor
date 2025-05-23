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
from modacor.math.variance_calculations import divide


class MultiplyByVariable(ProcessStep):
    """
    Adding Poisson uncertainties to the data
    """

    documentation = ProcessStepDescriber(
        calling_name="Multiply by data source variable",
        calling_id="MultiplyByVariable",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        works_on={"signal": ["normalization_factor", "variances"]},
        calling_arguments={"scalar": None, "scalar_units": None, "scalar_uncertainty": None},
        step_keywords=["multiply", "variable", "scalar"],
        step_doc="Multiply by a variable loaded from a data source",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="""This loads a scalar (value, units and uncertainty)
            from an IOSource and applies it to the data""",
    )

    def calculate(self) -> dict[str, DataBundle]:
        """ """
        self.processing_data: ProcessingData
        key = self.configuration.get("with_processing_keys")
        data: DataBundle = self.processing_data[key]

        # apply factor to the data
        f, v = divide(
            data["signal"].normalization_factor,
            data["signal"].normalization_factor_variances,
            self.configuration.get("scalar"),
            self.configuration.get("scalar_uncertainty", 0),
        )
        data["signal"].normalization_factor = f
        # propagate uncertainties into normalization_factor_
        data["signal"].normalization_factor_variances = v
        # propagate units into normalization_factor_units
        data["signal"].normalization_factor_units /= ureg.Unit(
            self.configuration.get("units", "dimensionless")
        )

        return {key: data}
