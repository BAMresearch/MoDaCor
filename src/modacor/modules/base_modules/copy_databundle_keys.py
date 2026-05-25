# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "25/05/2026"
__status__ = "Development"

__all__ = ["CopyDataBundleKeys"]
__version__ = "20260525.1"

from copy import deepcopy
from pathlib import Path
from typing import Any

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class CopyDataBundleKeys(ProcessStep):
    """
    Copy selected entries from one DataBundle to another.

    ``with_processing_keys`` contains two keys: target first, source second.
    This keeps static data in a separate branch while allowing a later step to
    attach selected static entries to a dynamic sample bundle.
    """

    documentation = ProcessStepDescriber(
        calling_name="Copy DataBundle keys",
        calling_id="CopyDataBundleKeys",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={},
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": None,
                "doc": "Two processing keys: target then source.",
            },
            "data_keys": {
                "type": (list, str, type(None)),
                "default": None,
                "doc": "Source keys to copy to the target DataBundle under the same names.",
            },
            "key_map": {
                "type": (dict, type(None)),
                "default": None,
                "doc": "Mapping of source key to target key. Mutually exclusive with data_keys.",
            },
            "copy": {
                "type": bool,
                "default": True,
                "doc": "Deep-copy copied values. If False, attach the same objects by reference.",
            },
            "copy_axes": {
                "type": bool,
                "default": True,
                "doc": "Pass with_axes to BaseData.copy when copy is True.",
            },
        },
        step_keywords=["copy", "databundle", "static"],
        step_doc="Copy selected BaseData entries between DataBundles.",
        step_reference="",
        step_note=(
            "Use this to attach static maps such as Q, Psi, Omega, pixel_index, or masks "
            "to each sample without recomputing the static branch."
        ),
    )

    def _key_pairs(self) -> list[tuple[str, str]]:
        data_keys = self.configuration.get("data_keys", None)
        key_map = self.configuration.get("key_map", None)

        if data_keys is not None and key_map is not None:
            raise ValueError("CopyDataBundleKeys accepts either data_keys or key_map, not both.")

        if key_map is not None:
            if not isinstance(key_map, dict) or not key_map:
                raise ValueError("CopyDataBundleKeys key_map must be a non-empty mapping.")
            return [(str(source_key), str(target_key)) for source_key, target_key in key_map.items()]

        if isinstance(data_keys, str):
            keys = [data_keys]
        elif data_keys is None:
            raise ValueError("CopyDataBundleKeys requires data_keys or key_map.")
        else:
            try:
                keys = list(data_keys)
            except TypeError as exc:
                raise TypeError("CopyDataBundleKeys data_keys must be a string or iterable of strings.") from exc

        if not keys:
            raise ValueError("CopyDataBundleKeys data_keys must not be empty.")
        return [(str(key), str(key)) for key in keys]

    def _copy_value(self, value: Any) -> Any:
        if not bool(self.configuration.get("copy", True)):
            return value

        if hasattr(value, "copy"):
            try:
                return value.copy(with_axes=bool(self.configuration.get("copy_axes", True)))
            except TypeError:
                return value.copy()

        return deepcopy(value)

    def calculate(self) -> dict[str, DataBundle]:
        keys = self._normalised_processing_keys()
        assert len(keys) == 2, (
            "CopyDataBundleKeys requires exactly two processing keys in 'with_processing_keys': "
            "the first is the target, the second is the source."
        )

        target_processing_key, source_processing_key = keys
        source = self.processing_data.get(source_processing_key)
        if source is None:
            raise KeyError(f"CopyDataBundleKeys source DataBundle not found: {source_processing_key!r}")

        target = self.processing_data.get(target_processing_key)
        if target is None:
            target = DataBundle()

        for source_key, target_key in self._key_pairs():
            if source_key not in source:
                raise KeyError(f"CopyDataBundleKeys source key not found: {source_processing_key}.{source_key}")
            target[target_key] = self._copy_value(source[source_key])

        return {target_processing_key: target}
