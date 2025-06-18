# -*- coding: utf-8 -*-
# __init__.py

__version__ = "1.0.0"

from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry(system="SI")
Q_ = ureg.Quantity
# recommended for pickling and unpickling:
set_application_registry(ureg)
ureg.formatter.default_format = "~P"
ureg.setup_matplotlib(True)
