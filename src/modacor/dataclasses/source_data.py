# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__author__ = "Brian R. Pauw"
__license__ = "BSD3"
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "24/05/2025"
__version__ = "20250524.1"
__status__ = "Development"  # "Development", "Production"
from typing import Any, Dict

import numpy as np
import pint
from attrs import define, field
from attrs import validators as v

from modacor import ureg
from modacor.administration.licenses import BSD3Clause as __license__  # noqa: F401

from .validators import arrays_are_equal_shape

# end of header and standard imports


@define
class SourceData:
    """
    SourceData is used for a data value or array loaded from an IoSource.
    Punitive defaults have been set to encourage compliance.
    """

    # data, units and variance are required
    data: np.ndarray = field(validator=v.instance_of(np.ndarray))
    units: pint.Unit = field(validator=v.instance_of(ureg.Unit))
    variance: pint.Unit = field(validator=v.instance_of(ureg.Unit))
    attributes: Dict[str, Any] = field(factory=dict)

    def __attrs_post_init__(self):
        if not arrays_are_equal_shape(self.data, self.variance):
            raise ValueError("Data and variance arrays must have the same shape.")
