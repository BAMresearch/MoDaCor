# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence, TypeVar

__all__ = ["attach_prepared_data", "get_first_present", "normalize_str_list"]

T = TypeVar("T")


def normalize_str_list(value: Sequence[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def get_first_present(mapping: Mapping[str, T], *keys: str) -> T | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def attach_prepared_data(
    processing_data: MutableMapping[str, Any],
    keys: Sequence[str],
    prepared_data: Mapping[str, Any],
    *,
    logger: Any | None = None,
    module_name: str = "Module",
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in keys:
        databundle = processing_data.get(key)
        if databundle is None:
            if logger is not None:
                logger.warning(f"{module_name}: processing_data has no entry for key={key!r}; skipping.")  # noqa: E702
            continue
        for out_key, bd in prepared_data.items():
            databundle[out_key] = bd
        output[key] = databundle
    return output
