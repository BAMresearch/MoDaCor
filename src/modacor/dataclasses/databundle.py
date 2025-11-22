# src/modacor/dataclasses/databundle.py
# -*- coding: utf-8 -*-
__author__ = "Jerome Kieffer"
__copyright__ = "MoDaCor team"
__license__ = "BSD3"
__date__ = "21/05/2025"
__version__ = "20250521.1"
__status__ = "Production"  # "Development", "Production"
# end of header and standard imports


class DataBundle(dict):
    """
    DataBundle is a specialized data class for storing related data.
    It contains a dictionary of BaseData data elements, for example Signal,
    a wavelength and flux spectrum, Qx, Qy, Qz, Psi, etc. Process steps can
    add further BaseData objects to this bundle.

    """

    description: str | None = None
    # as per NXcanSAS, tells which basedata to plot
    default_plot: str | None = None
