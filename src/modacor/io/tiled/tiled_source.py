# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "20/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["TiledSource"]

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
from attrs import define, field

from modacor.dataclasses.messagehandler import MessageHandler
from modacor.io.io_source import ArraySlice

from ..io_source import IoSource
from .connection import connect_tiled


def _normalise_path_tokens(path: str | Sequence[str] | None) -> tuple[str, ...]:
    if path is None:
        return ()
    if isinstance(path, str):
        tokens = path.split("/")
    else:
        tokens = []
        for element in path:
            tokens.extend(str(element).split("/"))
    return tuple(part for part in (token.strip() for token in tokens) if part)


@define(kw_only=True)
class TiledSource(IoSource):
    """
    IoSource implementation backed by a Tiled data service.

    Parameters
    ----------
    resource_location:
        Connection descriptor for the Tiled service. Supported formats:

        - ``"profile:my-profile"`` or ``"profile://my-profile"``
          → connects via :func:`tiled.client.from_profile`.
        - Any other string (including http/https URLs)
          → connects via :func:`tiled.client.from_uri`.
        - A mapping with one of the keys ``{"profile", "uri", "from_profile", "from_uri"}``.
          Additional keys ``"kwargs"`` or ``"connection_kwargs"`` may provide dicts merged into the
          call.
        - A mapping containing ``"client"`` or ``"node"`` with a pre-constructed Tiled node; in that
          case no connection attempt is made.

        When ``root_node`` is supplied, ``resource_location`` is optional and ignored.

    root_node:
        Existing Tiled node to use as the root. Supplying this allows using in-memory catalogues or
        stubbed objects without importing ``tiled`` at module import time.

    iosource_method_kwargs:
        Optional keywords to control the connection. Recognised entries:

        - ``base_item_path`` or ``base_path``: prefix applied to every ``data_key`` before resolving in
          the Tiled tree.
        - ``connection_kwargs``: nested dict merged into the connection call.
        - Any remaining items are forwarded directly to ``from_uri``/``from_profile``.

    Notes
    -----
    Data and metadata are cached per resolved path when retrieved without an explicit slice to reduce
    repeated network round-trips. Sliced reads bypass the cache. Call ``clear_cache()``
    before reading data that has changed on the server. Empty paths refer to the
    configured base path, including attribute keys such as ``@title``.
    """

    resource_location: str | dict[str, Any] | None = field(default=None)
    root_node: Any | None = field(default=None, repr=False)

    logger: MessageHandler = field(init=False)
    _root_node: Any = field(init=False, default=None, repr=False)
    _base_path: tuple[str, ...] = field(init=False, factory=tuple, repr=False)
    _node_cache: dict[str, Any] = field(init=False, factory=dict, repr=False)
    _data_cache: dict[str, np.ndarray] = field(init=False, factory=dict, repr=False)
    _attribute_cache: dict[str, dict[str, Any]] = field(init=False, factory=dict, repr=False)
    _structure_cache: dict[str, dict[str, Any]] = field(init=False, factory=dict, repr=False)

    def __attrs_post_init__(self) -> None:
        self.logger = MessageHandler(level=self.logging_level, name="TiledSource")

        method_kwargs = dict(self.iosource_method_kwargs or {})
        base_path_setting = method_kwargs.pop(
            "base_item_path", method_kwargs.pop("base_path", None)
        ) or _extract_from_mapping(self.resource_location, ("base_item_path", "base_path"))
        self._base_path = _normalise_path_tokens(base_path_setting)

        connection_kwargs = method_kwargs.pop("connection_kwargs", {})
        if not isinstance(connection_kwargs, dict):
            raise TypeError("connection_kwargs must be a dictionary if provided.")
        method_kwargs.update(connection_kwargs)

        if self.root_node is not None:
            self._root_node = self.root_node
        else:
            self._root_node = self._connect(resource_location=self.resource_location, connection_kwargs=method_kwargs)

        if self._root_node is None:
            raise ValueError("TiledSource requires either a root_node or a valid resource_location to connect to.")

    # ------------------------------------------------------------------
    # IoSource API
    # ------------------------------------------------------------------

    def get_data(self, data_key: str, load_slice: ArraySlice = ...) -> np.ndarray:
        key_path, _ = self._split_key(data_key)

        if load_slice is Ellipsis or load_slice is None:
            if key_path in self._data_cache:
                return np.array(self._data_cache[key_path], copy=True)
            slice_arg: Optional[ArraySlice] = None
        else:
            slice_arg = self._prepare_slice(load_slice)

        node = self._resolve_node(key_path)
        read_kwargs = {}
        if slice_arg is not None:
            read_kwargs["slice"] = slice_arg

        try:
            data_obj = node.read(**read_kwargs)
        except TypeError as exc:
            if slice_arg is not None:
                self.logger.warning(f"Slice {slice_arg} not supported for '{key_path}' ({exc}); slicing locally.")
                return np.array(self._to_numpy(node.read())[slice_arg], copy=True)
            else:
                raise
        except AttributeError as exc:
            raise KeyError(f"Path '{key_path}' does not resolve to a readable Tiled node.") from exc

        array = self._to_numpy(data_obj)

        if slice_arg is None:
            self._data_cache[key_path] = np.array(array, copy=True)
            self._update_structure_cache(key_path, self._data_cache[key_path])
            return np.array(self._data_cache[key_path], copy=True)

        return np.array(array, copy=True)

    def get_data_shape(self, data_key: str) -> tuple[int, ...]:
        key_path, _ = self._split_key(data_key)
        cached = self._structure_cache.get(key_path)
        if cached and "shape" in cached:
            return cached["shape"]

        node = self._resolve_node(key_path)
        shape = self._extract_shape(node)
        if shape:
            self._structure_cache.setdefault(key_path, {})["shape"] = shape
            return shape

        if key_path in self._data_cache:
            shape = tuple(self._data_cache[key_path].shape)
            self._structure_cache.setdefault(key_path, {})["shape"] = shape
            return shape
        return ()

    def get_data_dtype(self, data_key: str) -> np.dtype | None:
        key_path, _ = self._split_key(data_key)
        cached = self._structure_cache.get(key_path)
        if cached and cached.get("dtype") is not None:
            return cached["dtype"]

        node = self._resolve_node(key_path)
        dtype = self._extract_dtype(node)
        if dtype is not None:
            self._structure_cache.setdefault(key_path, {})["dtype"] = dtype
            return dtype

        if key_path in self._data_cache:
            dtype = self._data_cache[key_path].dtype
            self._structure_cache.setdefault(key_path, {})["dtype"] = dtype
            return dtype
        return None

    def get_data_attributes(self, data_key: str) -> dict[str, Any]:
        key_path, _ = self._split_key(data_key)
        if key_path in self._attribute_cache:
            return dict(self._attribute_cache[key_path])

        node = self._resolve_node(key_path)
        attributes = self._extract_attributes(node)
        self._attribute_cache[key_path] = dict(attributes)
        return dict(attributes)

    def get_static_metadata(self, data_key: str) -> Any:
        key_path, attribute = self._split_key(data_key)
        if attribute is None:
            node = self._resolve_node(key_path)
            metadata = getattr(node, "metadata", None)
            return metadata

        attributes = self.get_data_attributes(key_path)
        return attributes.get(attribute)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self, resource_location: str | dict[str, Any] | None, connection_kwargs: dict[str, Any]) -> Any:
        return connect_tiled(resource_location, connection_kwargs)

    def _split_key(self, data_key: str) -> tuple[str, Optional[str]]:
        if "@" in data_key:
            base, attribute = data_key.rsplit("@", 1)
            return base.strip(), attribute.strip()
        return data_key.strip(), None

    def _resolve_node(self, data_key: str) -> Any:
        tokens = self._base_path + _normalise_path_tokens(data_key)
        cache_key = "/".join(tokens)
        if cache_key in self._node_cache:
            return self._node_cache[cache_key]

        node = self._root_node
        try:
            for token in tokens:
                node = node[token]
        except (KeyError, TypeError, AttributeError) as exc:
            raise KeyError(f"Path '{data_key}' could not be resolved in the Tiled tree.") from exc

        self._node_cache[cache_key] = node
        return node

    def clear_cache(self) -> None:
        """Discard locally cached reads before reading data changed on the server."""
        self._node_cache.clear()
        self._data_cache.clear()
        self._attribute_cache.clear()
        self._structure_cache.clear()

    def _prepare_slice(self, load_slice: ArraySlice) -> ArraySlice:
        if isinstance(load_slice, tuple):
            return tuple(load_slice)
        return load_slice

    def _to_numpy(self, data_obj: Any) -> np.ndarray:
        if isinstance(data_obj, np.ndarray):
            return data_obj

        to_records = getattr(data_obj, "to_records", None)
        if callable(to_records):
            try:
                records = to_records(index=False)
                return np.asarray(records)
            except Exception:  # noqa: BLE001 - best-effort conversion
                pass

        to_numpy = getattr(data_obj, "to_numpy", None)
        if callable(to_numpy):
            try:
                array = to_numpy()
                if isinstance(array, np.ndarray):
                    return array
                return np.asarray(array)
            except TypeError:
                pass

        values = getattr(data_obj, "values", None)
        if values is not None:
            try:
                return np.asarray(values)
            except Exception:  # noqa: BLE001
                pass

        return np.asarray(data_obj)

    def _extract_shape(self, node: Any) -> tuple[int, ...]:
        shape = getattr(node, "shape", None)
        if shape is not None:
            return tuple(shape)

        structure = self._call_structure(node)
        if structure is not None:
            shape_attr = getattr(structure, "shape", None)
            if shape_attr is not None:
                return tuple(shape_attr)
        return ()

    def _extract_dtype(self, node: Any) -> np.dtype | None:
        dtype = getattr(node, "dtype", None)
        if dtype is not None:
            try:
                return np.dtype(dtype)
            except TypeError:
                return None

        structure = self._call_structure(node)
        if structure is not None:
            dtype_attr = getattr(structure, "data_type", None)
            if dtype_attr is not None and callable(getattr(dtype_attr, "to_numpy_dtype", None)):
                return dtype_attr.to_numpy_dtype()
            dtype_attr = getattr(structure, "dtype", None)
            if dtype_attr is not None:
                try:
                    return np.dtype(dtype_attr)
                except TypeError:
                    return None
        return None

    def _extract_attributes(self, node: Any) -> dict[str, Any]:
        attributes: dict[str, Any] = {}
        metadata = getattr(node, "metadata", None)
        if isinstance(metadata, Mapping):
            if isinstance(metadata.get("attrs"), Mapping):
                attributes.update(metadata["attrs"])
            else:
                attributes.update(metadata)

        attrs_obj = getattr(node, "attrs", None)
        if isinstance(attrs_obj, Mapping):
            attributes.update(attrs_obj)

        return attributes

    def _update_structure_cache(self, key_path: str, array: np.ndarray) -> None:
        entry = self._structure_cache.setdefault(key_path, {})
        entry["shape"] = tuple(array.shape)
        entry["dtype"] = array.dtype

    def _call_structure(self, node: Any) -> Any:
        structure = getattr(node, "structure", None)
        if callable(structure):
            try:
                return structure()
            except Exception as exc:  # noqa: BLE001
                self.logger.debug(f"Failed to obtain structure: {exc}")
                return None
        return None


def _extract_from_mapping(mapping: str | dict[str, Any] | None, keys: Sequence[str]) -> str | Sequence[str] | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None
