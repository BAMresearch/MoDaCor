# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = [
    "BeamProfileQuadrature",
    "gaussian_beam_profile",
    "image_beam_profile",
    "trapezoid_beam_profile",
]
__version__ = "20260906.1"

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from modacor.geometry import unit_vector3


@dataclass(frozen=True, slots=True)
class BeamProfileQuadrature:
    """Weighted parallel beam rays in a plane normal to beam propagation."""

    points: np.ndarray
    weights: np.ndarray
    retained_weight_fraction: float = 1.0

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        retained = float(self.retained_weight_fraction)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError(f"points must have shape (ray_count, 3), got {points.shape}.")
        if weights.shape != (points.shape[0],):
            raise ValueError(f"weights must have shape {(points.shape[0],)}, got {weights.shape}.")
        if points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise ValueError("points must be non-empty and finite.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("weights must be finite, non-negative, and contain a positive value.")
        if not np.isfinite(retained) or retained <= 0.0 or retained > 1.0:
            raise ValueError("retained_weight_fraction must be finite and in (0, 1].")

        normalized_weights = weights / np.sum(weights)
        object.__setattr__(self, "points", points.copy())
        object.__setattr__(self, "weights", normalized_weights)
        object.__setattr__(self, "retained_weight_fraction", retained)


def _positive_pair(value, *, name: str, allow_zero: bool = False) -> np.ndarray:
    pair = np.asarray(value, dtype=float)
    if pair.shape != (2,) or not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must be a finite (slow, fast) pair.")
    invalid = pair < 0.0 if allow_zero else pair <= 0.0
    if np.any(invalid):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} entries must be {qualifier}.")
    return pair


def _integer_pair(value, *, name: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        value = (value, value)
    try:
        pair = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer or a pair of positive integers.") from error
    if len(pair) != 2 or any(not isinstance(item, (int, np.integer)) or item <= 0 for item in pair):
        raise ValueError(f"{name} must be a positive integer or a pair of positive integers.")
    return int(pair[0]), int(pair[1])


def _beam_plane_axes(incident_direction, rotation: float) -> tuple[np.ndarray, np.ndarray]:
    incident = unit_vector3(incident_direction, name="incident_direction")
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ incident)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    fast = unit_vector3(reference - (reference @ incident) * incident, name="beam fast axis")
    slow = np.cross(incident, fast)

    rotation = float(rotation)
    if not np.isfinite(rotation):
        raise ValueError("rotation must be finite and expressed in radians.")
    cosine = np.cos(rotation)
    sine = np.sin(rotation)
    return cosine * slow - sine * fast, sine * slow + cosine * fast


def _profile_from_planar_points(
    slow_coordinates,
    fast_coordinates,
    weights,
    *,
    centre,
    incident_direction,
    rotation: float,
    retained_weight_fraction: float = 1.0,
) -> BeamProfileQuadrature:
    slow_coordinates = np.asarray(slow_coordinates, dtype=float).reshape(-1)
    fast_coordinates = np.asarray(fast_coordinates, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if slow_coordinates.shape != fast_coordinates.shape or weights.shape != slow_coordinates.shape:
        raise ValueError("Planar coordinates and weights must have identical flattened shapes.")
    centre = np.asarray(centre, dtype=float)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError("centre must be a finite vector with shape (3,).")
    slow_axis, fast_axis = _beam_plane_axes(incident_direction, rotation)
    points = (
        centre[None, :]
        + slow_coordinates[:, None] * slow_axis[None, :]
        + fast_coordinates[:, None] * fast_axis[None, :]
    )
    return BeamProfileQuadrature(points, weights, retained_weight_fraction)


def image_beam_profile(
    image,
    *,
    pixel_pitch,
    image_centre=None,
    centre=(0.0, 0.0, 0.0),
    incident_direction=(0.0, 0.0, 1.0),
    rotation: float = 0.0,
    downsample=1,
    relative_weight_cutoff: float = 0.0,
) -> BeamProfileQuadrature:
    """Convert a calibrated 2D beam image into weight-preserving beam rays.

    ``pixel_pitch`` and array axes are ordered ``(slow, fast)``. ``image_centre``
    gives the fractional pixel index aligned with the lab-frame ``centre`` and
    defaults to the geometric array centre. Pixel values are integrated
    non-negative intensities. Downsampled blocks are represented at their
    intensity-weighted centroid, preserving total weight and first spatial
    moments. ``rotation`` is in radians in this unit-free model.
    """
    intensities = np.asarray(image, dtype=float)
    if intensities.ndim != 2 or 0 in intensities.shape:
        raise ValueError(f"image must be a non-empty two-dimensional array, got {intensities.shape}.")
    if not np.all(np.isfinite(intensities)) or np.any(intensities < 0.0) or not np.any(intensities > 0.0):
        raise ValueError("image must be finite, non-negative, and contain a positive value.")
    pitch = _positive_pair(pixel_pitch, name="pixel_pitch")
    if image_centre is None:
        image_centre = (np.asarray(intensities.shape, dtype=float) - 1.0) / 2.0
    else:
        image_centre = np.asarray(image_centre, dtype=float)
        if image_centre.shape != (2,) or not np.all(np.isfinite(image_centre)):
            raise ValueError("image_centre must be a finite (slow, fast) pixel-index pair.")
    slow_block, fast_block = _integer_pair(downsample, name="downsample")
    relative_weight_cutoff = float(relative_weight_cutoff)
    if not np.isfinite(relative_weight_cutoff) or relative_weight_cutoff < 0.0 or relative_weight_cutoff >= 1.0:
        raise ValueError("relative_weight_cutoff must be finite and in [0, 1).")

    original_weight = float(np.sum(intensities))
    retained_image = intensities.copy()
    retained_image[retained_image < relative_weight_cutoff * np.max(retained_image)] = 0.0
    retained_weight = float(np.sum(retained_image))
    if retained_weight == 0.0:
        raise ValueError("relative_weight_cutoff removed every positive image pixel.")

    slow_coordinates = (np.arange(intensities.shape[0]) - image_centre[0]) * pitch[0]
    fast_coordinates = (np.arange(intensities.shape[1]) - image_centre[1]) * pitch[1]
    fast_grid, slow_grid = np.meshgrid(fast_coordinates, slow_coordinates)

    output_slow = []
    output_fast = []
    output_weights = []
    for slow_start in range(0, intensities.shape[0], slow_block):
        slow_slice = slice(slow_start, min(slow_start + slow_block, intensities.shape[0]))
        for fast_start in range(0, intensities.shape[1], fast_block):
            fast_slice = slice(fast_start, min(fast_start + fast_block, intensities.shape[1]))
            block_weights = retained_image[slow_slice, fast_slice]
            block_weight = float(np.sum(block_weights))
            if block_weight == 0.0:
                continue
            output_slow.append(float(np.sum(block_weights * slow_grid[slow_slice, fast_slice]) / block_weight))
            output_fast.append(float(np.sum(block_weights * fast_grid[slow_slice, fast_slice]) / block_weight))
            output_weights.append(block_weight)

    return _profile_from_planar_points(
        output_slow,
        output_fast,
        output_weights,
        centre=centre,
        incident_direction=incident_direction,
        rotation=rotation,
        retained_weight_fraction=retained_weight / original_weight,
    )


def gaussian_beam_profile(
    *,
    standard_deviations,
    quadrature_order=12,
    truncation_sigma=(4.0, 4.0),
    centre=(0.0, 0.0, 0.0),
    incident_direction=(0.0, 0.0, 1.0),
    rotation: float = 0.0,
) -> BeamProfileQuadrature:
    """Generate product-Gaussian beam quadrature in its rotated principal axes."""
    sigma = _positive_pair(standard_deviations, name="standard_deviations")
    order = _integer_pair(quadrature_order, name="quadrature_order")
    truncation = _positive_pair(truncation_sigma, name="truncation_sigma")

    coordinates = []
    weighted_nodes = []
    for dimension in range(2):
        nodes, weights = np.polynomial.legendre.leggauss(order[dimension])
        limit = truncation[dimension] * sigma[dimension]
        dimensional_coordinates = limit * nodes
        coordinates.append(dimensional_coordinates)
        weighted_nodes.append(limit * weights * np.exp(-0.5 * (dimensional_coordinates / sigma[dimension]) ** 2))
    fast_grid, slow_grid = np.meshgrid(coordinates[1], coordinates[0])
    fast_weights, slow_weights = np.meshgrid(weighted_nodes[1], weighted_nodes[0])
    retained = erf(truncation[0] / sqrt(2.0)) * erf(truncation[1] / sqrt(2.0))
    return _profile_from_planar_points(
        slow_grid,
        fast_grid,
        slow_weights * fast_weights,
        centre=centre,
        incident_direction=incident_direction,
        rotation=rotation,
        retained_weight_fraction=retained,
    )


def _trapezoid_axis_quadrature(plateau: float, negative_ramp: float, positive_ramp: float, order: int):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half_plateau = plateau / 2.0
    intervals = []
    if negative_ramp > 0.0:
        intervals.append((-half_plateau - negative_ramp, -half_plateau, "rising"))
    if plateau > 0.0:
        intervals.append((-half_plateau, half_plateau, "constant"))
    if positive_ramp > 0.0:
        intervals.append((half_plateau, half_plateau + positive_ramp, "falling"))
    if not intervals:
        raise ValueError("Each trapezoid axis needs a positive plateau or ramp width.")

    axis_coordinates = []
    axis_weights = []
    for lower, upper, shape in intervals:
        midpoint = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        segment_coordinates = midpoint + half_width * nodes
        if shape == "rising":
            intensity = (segment_coordinates - lower) / (upper - lower)
        elif shape == "falling":
            intensity = (upper - segment_coordinates) / (upper - lower)
        else:
            intensity = np.ones_like(segment_coordinates)
        axis_coordinates.append(segment_coordinates)
        axis_weights.append(half_width * weights * intensity)
    return np.concatenate(axis_coordinates), np.concatenate(axis_weights)


def trapezoid_beam_profile(
    *,
    plateau_width,
    ramp_widths,
    quadrature_order_per_region=6,
    centre=(0.0, 0.0, 0.0),
    incident_direction=(0.0, 0.0, 1.0),
    rotation: float = 0.0,
) -> BeamProfileQuadrature:
    """Generate a separable 2D trapezoid with four independent edge ramps.

    Values use ``(slow, fast)`` ordering. ``ramp_widths`` has shape ``(2, 2)``
    and gives ``((slow_negative, slow_positive), (fast_negative,
    fast_positive))``.
    """
    plateau = _positive_pair(plateau_width, name="plateau_width", allow_zero=True)
    ramps = np.asarray(ramp_widths, dtype=float)
    if ramps.shape != (2, 2) or not np.all(np.isfinite(ramps)) or np.any(ramps < 0.0):
        raise ValueError("ramp_widths must be a finite, non-negative array with shape (2, 2).")
    order = _integer_pair(quadrature_order_per_region, name="quadrature_order_per_region")

    slow_coordinates, slow_weights = _trapezoid_axis_quadrature(plateau[0], ramps[0, 0], ramps[0, 1], order[0])
    fast_coordinates, fast_weights = _trapezoid_axis_quadrature(plateau[1], ramps[1, 0], ramps[1, 1], order[1])
    fast_grid, slow_grid = np.meshgrid(fast_coordinates, slow_coordinates)
    fast_weight_grid, slow_weight_grid = np.meshgrid(fast_weights, slow_weights)
    return _profile_from_planar_points(
        slow_grid,
        fast_grid,
        slow_weight_grid * fast_weight_grid,
        centre=centre,
        incident_direction=incident_direction,
        rotation=rotation,
    )
