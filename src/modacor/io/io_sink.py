# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "09/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from logging import WARNING
from typing import Any

import attrs
from attrs import define, field


def default_config() -> dict[str, Any]:
    return {}


@define
class IoSink:
    """
    Base class for IO sinks. Mirrors IoSource.

    Sinks are registered with a resource_location (file/socket/etc.).
    The routed write call passes an optional 'subpath' after '::', which may be empty.
    """

    configuration: dict[str, Any] = field(factory=default_config)
    sink_reference: str = field(default="", converter=str, validator=attrs.validators.instance_of(str))
    type_reference: str = "IoSink"
    iosink_method_kwargs: dict[str, Any] = field(factory=dict, validator=attrs.validators.instance_of(dict))
    logging_level: int = field(default=WARNING, validator=attrs.validators.instance_of(int))

    def write(self, subpath: str, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")
