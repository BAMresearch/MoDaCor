# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence, TypeVar

from modacor.modules.helpers.statistics import WeightedScatterEstimates, finalize_weighted_scatter

__all__ = [
    "WeightedScatterEstimates",
    "attach_prepared_data",
    "finalize_weighted_scatter",
    "get_first_present",
    "leading_non_data_axes",
    "normalize_str_list",
]

T = TypeVar("T")


def leading_non_data_axes(ndim: int, rank_of_data: int) -> tuple[int, ...]:
    """Return leading axes that precede the trailing data dimensions."""

    ndim = int(ndim)
    rank_of_data = int(rank_of_data)
    if ndim < 0:
        raise ValueError(f"ndim must be non-negative, got {ndim}.")
    if not 0 <= rank_of_data <= ndim:
        raise ValueError(f"rank_of_data must be between 0 and ndim ({ndim}), got {rank_of_data}.")
    return tuple(range(ndim - rank_of_data))


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
        databundle.update(prepared_data)
        output[key] = databundle
    return output
