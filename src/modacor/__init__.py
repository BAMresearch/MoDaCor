#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/11/2025"
__status__ = "Development"  # "Development", "Production"

__version__ = "1.0.0"

from pint import UnitRegistry, set_application_registry

from .units import configure_detector_pixel_units

ureg = UnitRegistry(system="SI")

# Make pixel/px mean "detector element" everywhere in MoDaCor
configure_detector_pixel_units(ureg)

# we need to define an arbitrary intensity unit for scaling of intensity data:
ureg.define("AFU = [flux] = arbitrary_flux_unit")
Q_ = ureg.Quantity
# recommended for pickling and unpickling:
set_application_registry(ureg)
ureg.formatter.default_format = "~P"
ureg.setup_matplotlib(True)
