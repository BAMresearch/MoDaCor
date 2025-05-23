# src/modacor/dataclasses/processingdata.py
# -*- coding: utf-8 -*-
__author__ = "Brian R. Pauw"
__copyright__ = "MoDaCor team"
__license__ = "BSD3"
__date__ = "22/05/2025"
__version__ = "20250522.1"
__status__ = "Production"  # "Development", "Production"


class ProcessingData(dict):
    """
    Contains a collection of DataBundles used in a given pipeline
    """

    description: str | None = None
    # as per NXcanSAS, tells which basedata to plot
    default_plot: str | None = None
