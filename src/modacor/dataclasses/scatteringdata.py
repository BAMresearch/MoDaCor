from attrs import define, field, validators as v
from typing import List, Union
from .basedata import BaseData
from .process_step import ProcessStepDescriber


@define
class DataBundle:
    """
    DataBundle is a specialized data class for storing and processing scattering data.
    It inherits from the BaseData class and adds additional attributes specific to scattering
    experiments.
    """
    # wavelength + flux can make a spectrum if they are both arrays. Otherwise just float.
    wavelength: BaseData = field(validator=v.instance_of(BaseData))
    flux: BaseData = field(validator=v.instance_of(BaseData))
    description: str = field(default="")
    provenance: List[Union[ProcessStepDescriber, List[ProcessStepDescriber]]] = field(factory=list)
    data: dict[str: BaseData] = field(factory=dict, validator=v.instance_of(dict))
    # as per NXcanSAS, matches the data dimensions to axes
    axes: list[str] = field(factory=list, validator=v.instance_of(list))
