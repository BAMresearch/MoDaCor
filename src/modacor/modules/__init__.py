# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "25/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

# official steps are imported here for ease
from modacor.modules.base_modules.divide import Divide
from modacor.modules.base_modules.find_scale_factor1d import FindScaleFactor1D
from modacor.modules.base_modules.multiply import Multiply
from modacor.modules.base_modules.poisson_uncertainties import PoissonUncertainties
from modacor.modules.base_modules.reduce_dimensionality import ReduceDimensionality
from modacor.modules.base_modules.subtract import Subtract
from modacor.modules.base_modules.subtract_databundles import SubtractDatabundles
from modacor.modules.technique_modules.index_pixels import IndexPixels
from modacor.modules.technique_modules.indexed_averager import IndexedAverager
from modacor.modules.technique_modules.solid_angle_correction import SolidAngleCorrection
from modacor.modules.technique_modules.xs_geometry import XSGeometry

__all__ = [
    "Divide",
    "IndexPixels",
    "IndexedAverager",
    "FindScaleFactor1D",
    "Multiply",
    "PoissonUncertainties",
    "ReduceDimensionality",
    "SolidAngleCorrection",
    "SubtractDatabundles",
    "Subtract",
    "XSGeometry",
]
