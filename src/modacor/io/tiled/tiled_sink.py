# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from attrs import define, field

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sink import IoSink
from modacor.io.processing_path import infer_units_for_path, parse_processing_path, resolve_processing_path

from .connection import connect_tiled
from .tiled_source import _normalise_path_tokens

__all__ = ["TiledSink"]


@define(kw_only=True)
class TiledSink(IoSink):
    """Write ProcessingData arrays to a writable Tiled catalog.

    Connections accept the same URL, profile, or client mapping as TiledSource.
    ``iosink_method_kwargs`` accepts ``base_item_path`` (or ``base_path``),
    ``overwrite`` (default False), and connection kwargs. Existing arrays can
    be overwritten only with the same shape and dtype; no nodes are deleted.
    BaseData roots export signal, weights, and named uncertainties. Numeric
    leaves export arrays; other leaves export metadata under ``@value``.
    """

    resource_location: str | dict[str, Any] | None = field(default=None)
    root_node: Any | None = field(default=None, repr=False)
    _root_node: Any = field(init=False, default=None, repr=False)
    _base_path: tuple[str, ...] = field(init=False, factory=tuple)
    _overwrite: bool = field(init=False, default=False)

    def __attrs_post_init__(self) -> None:
        options = dict(self.iosink_method_kwargs)
        location_options = self.resource_location if isinstance(self.resource_location, dict) else {}
        self._base_path = _normalise_path_tokens(
            options.pop("base_item_path", options.pop("base_path", None))
            or location_options.get("base_item_path", location_options.get("base_path"))
        )
        self._overwrite = options.pop("overwrite", False)
        if not isinstance(self._overwrite, bool):
            raise TypeError("overwrite must be a boolean.")
        connection_kwargs = options.pop("connection_kwargs", {})
        if not isinstance(connection_kwargs, dict):
            raise TypeError("connection_kwargs must be a dictionary.")
        options.update(connection_kwargs)
        self._root_node = (
            self.root_node if self.root_node is not None else connect_tiled(self.resource_location, options)
        )
        if self._root_node is None:
            raise ValueError("TiledSink requires a root_node or resource_location.")

    def _container(self, tokens: tuple[str, ...]) -> Any:
        node = self._root_node
        for token in tokens:
            try:
                child = node[token]
            except KeyError:
                child = node.create_container(key=token)
            if not callable(getattr(child, "create_container", None)):
                raise ValueError(f"Tiled path component '{token}' is not a writable container.")
            node = child
        return node

    def _put(self, path: str, value: Any, attrs: dict[str, Any]) -> None:
        tokens = self._base_path + _normalise_path_tokens(path)
        parent = self._container(tokens[:-1])
        key = tokens[-1]
        try:
            existing = parent[key]
        except KeyError:
            existing = None
        if existing is not None and not self._overwrite:
            raise FileExistsError(f"Tiled target '{path}' already exists; set overwrite=True to update it.")

        if isinstance(value, (np.ndarray, int, float, complex, bool, np.number)):
            array = np.asarray(value)
            metadata = {"attrs": attrs}
            if existing is None:
                parent.write_array(array, key=key, metadata=metadata)
            else:
                if (
                    not callable(getattr(existing, "write", None))
                    or tuple(existing.shape) != array.shape
                    or np.dtype(existing.dtype) != array.dtype
                ):
                    raise ValueError(f"Cannot overwrite '{path}' with a different node type, shape, or dtype.")
                existing.write(array)
                existing.update_metadata({**dict(existing.metadata), **metadata})
        else:
            metadata = {"attrs": {"value": str(value)}}
            if existing is None:
                parent.create_container(key=key, metadata=metadata)
            elif callable(getattr(existing, "create_container", None)):
                existing.update_metadata({**dict(existing.metadata), **metadata})
            else:
                raise ValueError(f"Cannot overwrite array '{path}' with metadata.")

    def write(
        self,
        subpath: str,
        processing_data: ProcessingData,
        data_paths: Sequence[str] | str | None,
    ) -> dict[str, list[str]]:
        paths = [data_paths] if isinstance(data_paths, str) else list(data_paths or [])
        if not paths:
            raise ValueError("TiledSink.write requires one or more data_paths.")

        # Resolve all requested values before starting remote writes.
        values: dict[str, tuple[Any, dict[str, Any]]] = {}
        for path in paths:
            parsed = parse_processing_path(path)
            value = resolve_processing_path(processing_data, path)
            prefix = "/".join(_normalise_path_tokens(subpath) + _normalise_path_tokens(path))
            if isinstance(value, BaseData):
                for name in ("signal", "weights"):
                    units = str(value.units) if name == "signal" else "dimensionless"
                    values[f"{prefix}/{name}"] = (
                        getattr(value, name),
                        {"units": units, "rank_of_data": value.rank_of_data},
                    )
                for name, uncertainty in value.uncertainties.items():
                    values[f"{prefix}/uncertainties/{name}"] = (uncertainty, {"units": str(value.units)})
            else:
                basedata = processing_data[parsed.databundle_key][parsed.basedata_name]
                values[prefix] = (
                    value,
                    {
                        "units": infer_units_for_path(processing_data, path),
                        "rank_of_data": basedata.rank_of_data,
                    },
                )

        result: dict[str, list[str]] = {"arrays": [], "metadata": []}
        for path, (value, attrs) in values.items():
            self._put(path, value, attrs)
            kind = "arrays" if isinstance(value, (np.ndarray, int, float, complex, bool, np.number)) else "metadata"
            result[kind].append(path)
        return result
