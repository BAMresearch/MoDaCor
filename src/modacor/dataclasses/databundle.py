from typing import List

from attrs import define, field
from attrs import validators as v

from .basedata import BaseData
# from .process_step_describer import ProcessStepDescriber

# add a validator that checks if the keys in the axes list are in the data dictionary
def validate_axes(instance, attribute, value):
    """
    Custom validator to check if the keys in the axes list are in the data dictionary.
    """
    if not all(key in instance.data for key in value):
        raise ValueError(f"All keys in axes must be present in data dictionary. Missing: {set(value) - set(instance.data.keys())}")
    return True


class DataBundle(dict):
    """
    DataBundle is a specialized data class for storing related data.
    It contains a dictionary of BaseData data elements, for example Signal, 
    a wavelength and flux spectrum, Qx, Qy, Qz, Psi, etc. Process steps can 
    add further BaseData objects to this bundle. 

    """

    description: str|None = None
    # as per NXcanSAS, matches the data dimensions to axes
    default_plot: str|None = None
