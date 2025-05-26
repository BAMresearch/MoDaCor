__coding__ = "utf-8"
__author__ = "Brian R. Pauw"
__copyright__ = "MoDaCor team"
__license__ = "BSD3"
__date__ = "22/05/2025"
__version__ = "20250522.1"
__status__ = "Development"  # "Development", "Production"


from pathlib import Path
from typing import Any
from pathlib import Path
import numpy as np


from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


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
        works_on={"variances": ["Poisson"]},
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
