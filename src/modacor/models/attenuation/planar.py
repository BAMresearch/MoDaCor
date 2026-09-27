# SPDX-License-Identifier: BSD-3-Clause
"""Attenuation kernels for homogeneous planar layers."""

from __future__ import annotations

__all__ = ["planar_absorption_efficiency", "planar_transmission"]

import numpy as np


def planar_transmission(mu_m_inv: float, thickness_m: float, cos_alpha) -> np.ndarray:
    """Return transmission through a planar layer at the supplied incidence cosine."""

    return np.exp((-float(mu_m_inv) * float(thickness_m)) / np.asarray(cos_alpha, dtype=float))


def planar_absorption_efficiency(mu_m_inv: float, thickness_m: float, cos_alpha) -> np.ndarray:
    """Return the absorbed fraction in a planar layer at the supplied incidence cosine."""

    return 1.0 - planar_transmission(mu_m_inv, thickness_m, cos_alpha)
