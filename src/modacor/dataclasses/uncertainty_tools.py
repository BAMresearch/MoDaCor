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

__all__ = [
    "normalize_uncertainty_combinations",
    "combine_uncertainty_keys",
    "quadrature_aggregator",
    "maximum_aggregator",
]

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from .basedata import BaseData
from .messagehandler import MessageHandler

Aggregator = Callable[[list[np.ndarray], tuple[int, ...]], np.ndarray]


def normalize_uncertainty_combinations(raw: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    """Normalise combination configuration into deterministic tuples."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("'combinations' must be a mapping of output key -> iterable of source keys.")

    normalised: dict[str, tuple[str, ...]] = {}
    for dest_key, sources in raw.items():
        if isinstance(sources, str):
            source_tuple = (sources,)
        elif isinstance(sources, Iterable):
            source_tuple = tuple(str(s).strip() for s in sources if str(s).strip())
        else:
            raise TypeError("Each combinations entry must be a string or iterable of strings.")

        dest_key_str = str(dest_key).strip()
        if not dest_key_str:
            raise ValueError("Combination keys must be non-empty strings.")
        if not source_tuple:
            raise ValueError(f"Combination '{dest_key_str}' must list at least one source uncertainty key.")
        normalised[dest_key_str] = source_tuple
    return normalised


def quadrature_aggregator(uncertainties: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """Combine absolute uncertainties via root-sum-of-squares."""
    total_var: np.ndarray | None = None
    for sigma in uncertainties:
        arr = np.asarray(sigma, dtype=float)
        broadcast = np.broadcast_to(arr, shape).astype(float, copy=False)
        squared = np.square(broadcast)
        total_var = squared if total_var is None else total_var + squared
    if total_var is None:
        raise RuntimeError("Cannot compute quadrature of an empty sequence.")
    return np.sqrt(total_var)


def maximum_aggregator(uncertainties: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """Combine absolute uncertainties by taking the element-wise maximum."""
    if not uncertainties:
        raise RuntimeError("Cannot compute maximum of an empty sequence.")
    broadcasted = [np.broadcast_to(np.asarray(sigma, dtype=float), shape) for sigma in uncertainties]
    return np.maximum.reduce(broadcasted)


def combine_uncertainty_keys(
    *,
    basedata: BaseData,
    combinations: Mapping[str, tuple[str, ...]],
    aggregator: Aggregator,
    drop_sources: bool,
    ignore_missing: bool,
    logger: MessageHandler | None = None,
    target_name: str = "",
) -> None:
    """Apply configured combinations to ``basedata.uncertainties`` in-place."""
    signal_shape = basedata.signal.shape
    new_keys: set[str] = set()
    sources_to_remove: set[str] = set()

    for dest_key, source_keys in combinations.items():
        available: list[np.ndarray] = []
        present_sources: list[str] = []
        missing_sources: list[str] = []

        for src_key in source_keys:
            if src_key in basedata.uncertainties:
                present_sources.append(src_key)
                available.append(basedata.uncertainties[src_key])
            else:
                missing_sources.append(src_key)

        if missing_sources and not ignore_missing:
            missing_formatted = ", ".join(sorted(missing_sources))
            target_descr = target_name or "BaseData"
            raise KeyError(
                f"Missing uncertainties {{{missing_formatted}}} required for '{dest_key}' on {target_descr}."
            )

        if not available:
            if ignore_missing:
                if logger is not None:
                    logger.debug(
                        "Skipping destination '%s' – none of the source keys were present.",
                        dest_key,
                    )
                continue
            target_descr = target_name or "BaseData"
            raise RuntimeError(f"No uncertainties available to combine for destination '{dest_key}' on {target_descr}.")

        combined = aggregator(available, signal_shape)
        basedata.uncertainties[dest_key] = combined
        new_keys.add(dest_key)

        if drop_sources:
            sources_to_remove.update(present_sources)

    if drop_sources:
        for source_key in sources_to_remove - new_keys:
            basedata.uncertainties.pop(source_key, None)
