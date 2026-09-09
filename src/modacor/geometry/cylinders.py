# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = ["ConcentricCylinderGeometry"]
__version__ = "20260906.1"

from dataclasses import dataclass, field

import numpy as np


def _vector3(value, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


@dataclass(frozen=True, slots=True)
class ConcentricCylinderGeometry:
    """Ray intersections with an effectively infinite concentric cylinder.

    ``radii`` contains the ordered outer radius of every radial phase. For
    example, ``(sample_radius, outer_wall_radius)`` describes a central sample
    and one surrounding wall. All positions and radii must use the same length
    unit; this numerical kernel deliberately has no unit policy.

    The cylinder centreline passes through ``centre`` along ``axis``. Ray
    directions are normalized internally, so returned path lengths use the
    same length scale as the positions and radii.
    """

    radii: np.ndarray
    axis: np.ndarray = field(default_factory=lambda: np.array((1.0, 0.0, 0.0)))
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    parallel_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        radii = np.asarray(self.radii, dtype=float)
        if radii.ndim != 1 or radii.size == 0:
            raise ValueError("radii must be a non-empty one-dimensional sequence.")
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("radii must contain only finite, positive values.")
        if np.any(np.diff(radii) <= 0.0):
            raise ValueError("radii must be strictly increasing.")

        axis = _vector3(self.axis, name="axis")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm == 0.0:
            raise ValueError("axis must be non-zero.")

        centre = _vector3(self.centre, name="centre")
        tolerance = float(self.parallel_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= 1.0:
            raise ValueError("parallel_tolerance must be finite and in the interval (0, 1).")

        object.__setattr__(self, "radii", radii.copy())
        object.__setattr__(self, "axis", axis / axis_norm)
        object.__setattr__(self, "centre", centre.copy())
        object.__setattr__(self, "parallel_tolerance", tolerance)

    @property
    def phase_count(self) -> int:
        """Number of concentric radial phases."""
        return int(self.radii.size)

    def _normalized_rays(self, origins, directions) -> tuple[np.ndarray, np.ndarray]:
        origins_array = np.asarray(origins, dtype=float)
        directions_array = np.asarray(directions, dtype=float)
        if origins_array.ndim == 0 or origins_array.shape[-1] != 3:
            raise ValueError(f"origins must have shape (..., 3), got {origins_array.shape}.")
        if directions_array.ndim == 0 or directions_array.shape[-1] != 3:
            raise ValueError(f"directions must have shape (..., 3), got {directions_array.shape}.")
        if not np.all(np.isfinite(origins_array)):
            raise ValueError("origins must contain only finite values.")
        if not np.all(np.isfinite(directions_array)):
            raise ValueError("directions must contain only finite values.")

        origins_array, directions_array = np.broadcast_arrays(origins_array, directions_array)
        direction_norms = np.linalg.norm(directions_array, axis=-1, keepdims=True)
        if np.any(direction_norms == 0.0):
            raise ValueError("directions must be non-zero.")
        return origins_array, directions_array / direction_norms

    def line_boundary_intervals(self, origins, directions) -> np.ndarray:
        """Return full-line parameter intervals inside every radial boundary.

        The rays use normalized directions, so interval parameters have the
        same length scale as ``origins`` and ``radii``. The result shape is
        ``broadcast_shape + (phase_count, 2)`` with lower and upper parameters
        in the last dimension. A boundary not intersected by a line has a pair
        of NaNs.
        """
        origins_array, unit_directions = self._normalized_rays(origins, directions)

        relative = origins_array - self.centre
        relative_perpendicular = relative - np.sum(relative * self.axis, axis=-1, keepdims=True) * self.axis
        direction_perpendicular = unit_directions - (
            np.sum(unit_directions * self.axis, axis=-1, keepdims=True) * self.axis
        )

        quadratic = np.sum(direction_perpendicular**2, axis=-1)
        if np.any(quadratic <= self.parallel_tolerance**2):
            raise ValueError(
                "A ray is parallel or too close to parallel with the infinite cylinder axis; "
                "finite cylinder length and end geometry are required."
            )

        linear_half = np.sum(relative_perpendicular * direction_perpendicular, axis=-1)
        radial_squared = np.sum(relative_perpendicular**2, axis=-1)

        intervals = []
        for radius in self.radii:
            discriminant = linear_half**2 - quadratic * (radial_squared - radius**2)
            intersects = discriminant >= 0.0
            root = np.sqrt(np.maximum(discriminant, 0.0))
            lower = (-linear_half - root) / quadratic
            upper = (-linear_half + root) / quadratic
            intervals.append(
                np.stack((np.where(intersects, lower, np.nan), np.where(intersects, upper, np.nan)), axis=-1)
            )

        return np.stack(intervals, axis=-2)

    def line_phase_path_lengths(self, origins, directions) -> np.ndarray:
        """Return total full-line path length within each radial phase."""
        intervals = self.line_boundary_intervals(origins, directions)
        cumulative = np.nan_to_num(intervals[..., 1] - intervals[..., 0], nan=0.0)
        phase_lengths = cumulative.copy()
        if self.phase_count > 1:
            phase_lengths[..., 1:] -= cumulative[..., :-1]
        return np.maximum(phase_lengths, 0.0)

    def forward_path_lengths(self, origins, directions) -> np.ndarray:
        """Return the path length in each phase along forward semi-infinite rays.

        ``origins`` and ``directions`` accept shape ``(..., 3)`` and follow
        NumPy broadcasting rules. The result has shape ``broadcast_shape +
        (phase_count,)``. A ray may start inside or outside the cylinder; every
        intersection at parameter ``t >= 0`` is included.

        Rays parallel or nearly parallel to an infinite cylinder are rejected:
        their path is either zero or unbounded and a finite-cylinder model is
        required to resolve that geometry.
        """
        intervals = self.line_boundary_intervals(origins, directions)
        lower = intervals[..., 0]
        upper = intervals[..., 1]
        intersects = np.isfinite(lower)
        forward_lower = np.maximum(lower, 0.0)
        cumulative = np.where(intersects, np.maximum(upper - forward_lower, 0.0), 0.0)
        phase_lengths = cumulative.copy()
        if self.phase_count > 1:
            phase_lengths[..., 1:] -= cumulative[..., :-1]

        # Round-off near tangent boundaries can otherwise create tiny negative
        # shell lengths after subtracting nested cumulative intersections.
        return np.maximum(phase_lengths, 0.0)
