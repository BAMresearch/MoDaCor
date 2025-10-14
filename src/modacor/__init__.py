# -*- coding: utf-8 -*-
# __init__.py

__version__ = "1.0.0"

from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry(system="SI")
# we need to define an arbitrary intensity unit for scaling of intensity data:
ureg.define("AIU = [intensity] = arbitrary_intensity_unit")
Q_ = ureg.Quantity
# recommended for pickling and unpickling:
set_application_registry(ureg)
ureg.formatter.default_format = "~P"
ureg.setup_matplotlib(True)
