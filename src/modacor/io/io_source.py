# SPDX-License-Identifier: BSD-3-Clause


__license__ = "BSD-3-Clause"
__all__ = ["IoSource"]


from typing import Type, Any

from attrs import field, define


@define
class IoSource:
    """
    IoSource is a base class for all IO sources in the Modacor framework.
    It provides access to a specific IO source and its associated methods.
    """

    type_reference = "IoSource"

    configuration: dict[str, Any] = field(factory=dict)

    def get_data(self, index: int | tuple[int], data_key: str) -> np.ndarray:
        """
        Get data from the IO source using the provided data key.

        Parameters
        ----------
        index : int | tuple[int]
            The index or indices to access the data.
        data_key : str
            The key to access the data.

        Returns
        -------
        np.ndarray :
            The data associated with the provided key.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_static_metadata(self, data_key: str) -> Any:
        """
        Get static metadata from the IO source using the provided data key.

        Parameters
        ----------
        data_key : str
            The key to access the metadata.

        Returns
        -------
        Any :
            The static metadata associated with the provided key.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
