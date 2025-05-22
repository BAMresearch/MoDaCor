# src/modacor/dataclasses/databundle.py
# -*- coding: utf-8 -*-
__author__ = "Jerome Kieffer"
__copyright__ = "MoDaCor team"
__license__ = "BSD3"
__date__ = "21/05/2025"
__version__ = "20250521.1"
__status__ = "Production"  # "Development", "Production"


# add a validator that checks if the keys in the axes list are in the data dictionary
def validate_axes(instance, attribute, value):
    """
    Custom validator to check if the keys in the axes list are in the data dictionary.
    """
    if not all(key in instance.data for key in value):
        raise ValueError(
            f"""Missing axes must be present in data dictionary
                         : {set(value) - set(instance.data.keys())}"""
        )
    return True


class DataBundle(dict):
    """
    DataBundle is a specialized data class for storing related data.
    It contains a dictionary of BaseData data elements, for example Signal,
    a wavelength and flux spectrum, Qx, Qy, Qz, Psi, etc. Process steps can
    add further BaseData objects to this bundle.

    """

    description: str | None = None
    # as per NXcanSAS, matches the data dimensions to axes
    default_plot: str | None = None
