# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

__all__ = [
    "AdaptiveDetectorAttenuation",
    "adaptive_attenuation_factors_on_grid",
    "attenuation_factors_at_detectors",
    "attenuation_factors_for_phase_at_detectors",
    "beam_chord_quadrature",
    "direct_beam_transmission",
    "uniform_cross_section_quadrature",
]
__version__ = "20260906.1"

from dataclasses import dataclass, field

import numpy as np

from modacor.geometry import ConcentricCylinderGeometry, unit_vector3


@dataclass(frozen=True, slots=True)
class AdaptiveDetectorAttenuation:
    """Result and diagnostics from adaptive detector-grid evaluation."""

    factors: np.ndarray
    evaluated_mask: np.ndarray
    accepted_cell_count: int
    exact_fallback_pixel_count: int
    max_accepted_validation_relative_error: float


def _weighted_beam_inputs(beam_points, beam_weights) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(beam_points, dtype=float)
    weights = np.asarray(beam_weights, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"beam_points must have shape (beam_count, 3), got {points.shape}.")
    if weights.shape != (points.shape[0],):
        raise ValueError(f"beam_weights must have shape {(points.shape[0],)}, got {weights.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("beam_points must contain only finite values.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("beam_weights must be finite, non-negative, and contain a positive value.")
    return points, weights


def _attenuation_coefficient_sets(
    geometry: ConcentricCylinderGeometry, attenuation_coefficients
) -> tuple[np.ndarray, bool]:
    coefficients = np.asarray(attenuation_coefficients, dtype=float)
    scalar_coefficients = coefficients.ndim == 1
    if scalar_coefficients:
        coefficients = coefficients[None, :]
    if coefficients.ndim != 2 or coefficients.shape[1] != geometry.phase_count:
        raise ValueError(
            "attenuation_coefficients must have shape (phase_count,) or "
            "(coefficient_set_count, phase_count); "
            f"the phase count is {geometry.phase_count}."
        )
    if coefficients.shape[0] == 0:
        raise ValueError("attenuation_coefficients must contain at least one coefficient set.")
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.0):
        raise ValueError("attenuation_coefficients must be finite and non-negative.")
    return coefficients, scalar_coefficients


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic right-handed orthonormal basis normal to axis."""
    reference = np.zeros(3)
    reference[int(np.argmin(np.abs(axis)))] = 1.0
    first = unit_vector3(np.cross(axis, reference), name="cross-section basis")
    second = np.cross(axis, first)
    return first, second


def uniform_cross_section_quadrature(
    *,
    radius: float,
    axis=(1.0, 0.0, 0.0),
    centre=(0.0, 0.0, 0.0),
    radial_order: int = 16,
    azimuthal_order: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return area-quadrature points and weights for a circular cross-section.

    Gauss--Legendre nodes integrate squared normalized radius on ``[0, 1]``;
    uniformly spaced midpoint nodes integrate azimuth. The weights sum to the
    disk area. This is the uniform full-cross-section reference used during
    Stage 2, not the eventual finite-beam profile implementation.
    """
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive.")
    if not isinstance(radial_order, int) or radial_order <= 0:
        raise ValueError("radial_order must be a positive integer.")
    if not isinstance(azimuthal_order, int) or azimuthal_order <= 0:
        raise ValueError("azimuthal_order must be a positive integer.")

    axis_vector = unit_vector3(axis, name="axis")
    centre_vector = np.asarray(centre, dtype=float)
    if centre_vector.shape != (3,) or not np.all(np.isfinite(centre_vector)):
        raise ValueError("centre must be a finite vector with shape (3,).")
    basis_1, basis_2 = _perpendicular_basis(axis_vector)

    legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(radial_order)
    normalized_radius_squared = (legendre_nodes + 1.0) / 2.0
    normalized_radius = np.sqrt(normalized_radius_squared)
    radial_weights = legendre_weights / 2.0
    azimuth = (np.arange(azimuthal_order, dtype=float) + 0.5) * (2.0 * np.pi / azimuthal_order)

    radial_vectors = np.cos(azimuth)[:, None] * basis_1[None, :] + np.sin(azimuth)[:, None] * basis_2[None, :]
    points = centre_vector + radius * normalized_radius[:, None, None] * radial_vectors[None, :, :]
    weights = np.broadcast_to(radial_weights[:, None], (radial_order, azimuthal_order)) * (
        np.pi * radius**2 / azimuthal_order
    )
    return points.reshape(-1, 3), weights.reshape(-1)


def beam_chord_quadrature(
    *,
    geometry: ConcentricCylinderGeometry,
    beam_points,
    beam_weights,
    phase_index: int,
    incident_direction=(0.0, 0.0, 1.0),
    chord_order: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand weighted incident beam rays into volume quadrature for one phase.

    Each beam point identifies a parallel incident line and its beam-plane
    integrated intensity. Analytic line/cylinder intervals exclude empty
    space. Gauss--Legendre nodes are placed along the occupied interval for a
    central phase or along both disjoint intervals of an annular phase.
    Returned weights include the chord-length Jacobian and therefore represent
    illuminated-volume weights up to the beam image's common area scale.
    """
    points, weights = _weighted_beam_inputs(beam_points, beam_weights)
    if not isinstance(phase_index, int) or not 0 <= phase_index < geometry.phase_count:
        raise ValueError(f"phase_index must be in [0, {geometry.phase_count}), got {phase_index!r}.")
    if not isinstance(chord_order, int) or chord_order <= 0:
        raise ValueError("chord_order must be a positive integer.")

    beam_direction = unit_vector3(incident_direction, name="incident_direction")
    intervals = geometry.line_boundary_intervals(points, beam_direction)
    outer = intervals[:, phase_index, :]
    outer_hit = np.all(np.isfinite(outer), axis=1) & (outer[:, 1] > outer[:, 0])

    segment_starts = []
    segment_stops = []
    segment_beam_indices = []
    outer_indices = np.flatnonzero(outer_hit)
    if phase_index == 0:
        segment_starts.append(outer[outer_indices, 0])
        segment_stops.append(outer[outer_indices, 1])
        segment_beam_indices.append(outer_indices)
    else:
        inner = intervals[:, phase_index - 1, :]
        inner_hit = np.all(np.isfinite(inner), axis=1) & (inner[:, 1] > inner[:, 0]) & outer_hit
        annulus_only = outer_hit & ~inner_hit
        annulus_indices = np.flatnonzero(annulus_only)
        if annulus_indices.size:
            segment_starts.append(outer[annulus_indices, 0])
            segment_stops.append(outer[annulus_indices, 1])
            segment_beam_indices.append(annulus_indices)

        crossing_indices = np.flatnonzero(inner_hit)
        if crossing_indices.size:
            segment_starts.extend((outer[crossing_indices, 0], inner[crossing_indices, 1]))
            segment_stops.extend((inner[crossing_indices, 0], outer[crossing_indices, 1]))
            segment_beam_indices.extend((crossing_indices, crossing_indices))

    if not segment_starts:
        raise ValueError(f"No positive-weight beam ray intersects cylindrical phase {phase_index}.")

    starts = np.concatenate(segment_starts)
    stops = np.concatenate(segment_stops)
    beam_indices = np.concatenate(segment_beam_indices)
    positive_segments = (stops > starts) & (weights[beam_indices] > 0.0)
    starts = starts[positive_segments]
    stops = stops[positive_segments]
    beam_indices = beam_indices[positive_segments]
    if starts.size == 0:
        raise ValueError(f"No positive-weight beam ray intersects cylindrical phase {phase_index}.")

    nodes, node_weights = np.polynomial.legendre.leggauss(chord_order)
    half_lengths = (stops - starts) / 2.0
    midpoints = (stops + starts) / 2.0
    parameters = midpoints[:, None] + half_lengths[:, None] * nodes[None, :]
    scattering_points = points[beam_indices, None, :] + parameters[..., None] * beam_direction
    volume_weights = weights[beam_indices, None] * half_lengths[:, None] * node_weights[None, :]
    return scattering_points.reshape(-1, 3), volume_weights.reshape(-1)


def direct_beam_transmission(
    *,
    geometry: ConcentricCylinderGeometry,
    attenuation_coefficients,
    beam_points,
    beam_weights,
    incident_direction=(0.0, 0.0, 1.0),
) -> float:
    """Return beam-area-weighted transmission through all cylindrical phases."""
    coefficients = np.asarray(attenuation_coefficients, dtype=float)
    if coefficients.shape != (geometry.phase_count,):
        raise ValueError(
            "attenuation_coefficients must have one value per cylindrical phase; "
            f"expected {(geometry.phase_count,)}, got {coefficients.shape}."
        )
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.0):
        raise ValueError("attenuation_coefficients must be finite and non-negative.")
    points, weights = _weighted_beam_inputs(beam_points, beam_weights)
    beam_direction = unit_vector3(incident_direction, name="incident_direction")
    through_lengths = geometry.line_phase_path_lengths(points, beam_direction)
    return float(np.average(np.exp(-(through_lengths @ coefficients)), weights=weights))


def attenuation_factors_at_detectors(
    *,
    geometry: ConcentricCylinderGeometry,
    attenuation_coefficients,
    scattering_points,
    volume_weights,
    detector_positions,
    incident_direction=(0.0, 0.0, 1.0),
    detector_chunk_size: int = 256,
) -> np.ndarray:
    """Integrate attenuation for scattering reaching actual detector positions.

    The incoming beam is parallel. Exit directions run from every scattering
    quadrature point to each detector-element centre (point-ray geometry).
    Inputs use one consistent length unit, with attenuation coefficients in
    its reciprocal. ``attenuation_coefficients`` may contain either one vector
    with shape ``(phase_count,)`` or several vectors with shape
    ``(coefficient_set_count, phase_count)``. The latter reuses each geometric
    path calculation and returns shape ``(coefficient_set_count,
    detector_count)``. Detector positions should already be filtered to the
    active elements required by the caller.
    """
    coefficients, scalar_coefficients = _attenuation_coefficient_sets(geometry, attenuation_coefficients)

    points = np.asarray(scattering_points, dtype=float)
    weights = np.asarray(volume_weights, dtype=float)
    detectors = np.asarray(detector_positions, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"scattering_points must have shape (point_count, 3), got {points.shape}.")
    if detectors.ndim != 2 or detectors.shape[1:] != (3,):
        raise ValueError(f"detector_positions must have shape (detector_count, 3), got {detectors.shape}.")
    if weights.shape != (points.shape[0],):
        raise ValueError(f"volume_weights must have shape {(points.shape[0],)}, got {weights.shape}.")
    if points.shape[0] == 0:
        raise ValueError("At least one scattering point is required.")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(detectors)):
        raise ValueError("Scattering points and detector positions must be finite.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("volume_weights must be finite, non-negative, and contain a positive value.")
    if not isinstance(detector_chunk_size, int) or detector_chunk_size <= 0:
        raise ValueError("detector_chunk_size must be a positive integer.")

    beam_direction = unit_vector3(incident_direction, name="incident_direction")
    incoming_lengths = geometry.forward_path_lengths(points, -beam_direction)
    incoming_exponent = incoming_lengths @ coefficients.T
    weight_sum = float(np.sum(weights))
    factors = np.empty((coefficients.shape[0], detectors.shape[0]), dtype=float)

    for start in range(0, detectors.shape[0], detector_chunk_size):
        stop = min(start + detector_chunk_size, detectors.shape[0])
        exit_directions = detectors[start:stop, None, :] - points[None, :, :]
        outgoing_lengths = geometry.forward_path_lengths(points[None, :, :], exit_directions)
        for coefficient_index, coefficient_set in enumerate(coefficients):
            exponent = incoming_exponent[:, coefficient_index][None, :] + outgoing_lengths @ coefficient_set
            factors[coefficient_index, start:stop] = np.sum(np.exp(-exponent) * weights[None, :], axis=1) / weight_sum

    return factors[0] if scalar_coefficients else factors


def attenuation_factors_for_phase_at_detectors(
    *,
    geometry: ConcentricCylinderGeometry,
    attenuation_coefficients,
    beam_points,
    beam_weights,
    origin_phase_index: int,
    detector_positions,
    incident_direction=(0.0, 0.0, 1.0),
    chord_order: int = 16,
    detector_chunk_size: int = 256,
) -> np.ndarray:
    """Calculate detector factors for scattering originating in one phase.

    The beam-plane samples are expanded only along illuminated chords of
    ``origin_phase_index``. Attenuation along the incoming and outgoing paths
    still includes every phase in ``geometry``. Thus, for a sample and wall,
    the same function calculates sample-origin scattering, wall-origin
    scattering in the filled capillary, and wall-origin scattering in an
    empty capillary by changing the origin index and central coefficient.
    """
    scattering_points, volume_weights = beam_chord_quadrature(
        geometry=geometry,
        beam_points=beam_points,
        beam_weights=beam_weights,
        phase_index=origin_phase_index,
        incident_direction=incident_direction,
        chord_order=chord_order,
    )
    return attenuation_factors_at_detectors(
        geometry=geometry,
        attenuation_coefficients=attenuation_coefficients,
        scattering_points=scattering_points,
        volume_weights=volume_weights,
        detector_positions=detector_positions,
        incident_direction=incident_direction,
        detector_chunk_size=detector_chunk_size,
    )


_GridIndex = tuple[int, int]
_AdaptiveCell = tuple[int, int, int, int, int]
_AcceptedCell = tuple[int, int, int, int]


def _adaptive_grid_inputs(
    detector_position_grid,
    active_mask,
    relative_tolerance,
    absolute_tolerance,
    max_depth,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    detector_grid = np.asarray(detector_position_grid, dtype=float)
    if detector_grid.ndim != 3 or detector_grid.shape[-1] != 3:
        raise ValueError(
            "detector_position_grid must have shape (slow_pixels, fast_pixels, 3), " f"got {detector_grid.shape}."
        )
    if detector_grid.shape[0] == 0 or detector_grid.shape[1] == 0:
        raise ValueError("detector_position_grid axes must be non-empty.")
    relative_tolerance = float(relative_tolerance)
    absolute_tolerance = float(absolute_tolerance)
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be finite and non-negative.")
    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer.")

    detector_shape = detector_grid.shape[:2]
    active = np.ones(detector_shape, dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active.shape != detector_shape:
        raise ValueError(
            "active_mask must match the detector grid's first two axes; "
            f"expected {detector_shape}, got {active.shape}."
        )
    if not np.all(np.isfinite(detector_grid[active])):
        raise ValueError("Active detector positions must contain only finite values.")
    return detector_grid, active, relative_tolerance, absolute_tolerance


def _child_cells(cell: _AdaptiveCell) -> list[_AdaptiveCell]:
    slow_0, slow_1, fast_0, fast_1, depth = cell
    slow_mid = (slow_0 + slow_1) // 2
    fast_mid = (fast_0 + fast_1) // 2
    slow_ranges = [(slow_0, slow_1)] if slow_1 - slow_0 <= 1 else [(slow_0, slow_mid), (slow_mid, slow_1)]
    fast_ranges = [(fast_0, fast_1)] if fast_1 - fast_0 <= 1 else [(fast_0, fast_mid), (fast_mid, fast_1)]
    return [
        (child_slow_0, child_slow_1, child_fast_0, child_fast_1, depth + 1)
        for child_slow_0, child_slow_1 in slow_ranges
        for child_fast_0, child_fast_1 in fast_ranges
    ]


def _cell_samples(cell: _AdaptiveCell) -> tuple[set[_GridIndex], set[_GridIndex]]:
    slow_0, slow_1, fast_0, fast_1, _depth = cell
    corners = {
        (slow_0, fast_0),
        (slow_0, fast_1),
        (slow_1, fast_0),
        (slow_1, fast_1),
    }
    slow_mid = (slow_0 + slow_1) // 2
    fast_mid = (fast_0 + fast_1) // 2
    slow_quarters = {(3 * slow_0 + slow_1) // 4, (slow_0 + 3 * slow_1) // 4}
    fast_quarters = {(3 * fast_0 + fast_1) // 4, (fast_0 + 3 * fast_1) // 4}
    checkpoints = {
        (slow_mid, fast_mid),
        (slow_0, fast_mid),
        (slow_1, fast_mid),
        (slow_mid, fast_0),
        (slow_mid, fast_1),
        *((slow, fast) for slow in slow_quarters for fast in fast_quarters),
    }
    return corners | checkpoints, checkpoints - corners


def _bilinear_value(
    corner_values: np.ndarray,
    cell: _AdaptiveCell | _AcceptedCell,
    index: _GridIndex,
) -> np.ndarray:
    slow_0, slow_1, fast_0, fast_1 = cell[:4]
    slow, fast = index
    slow_fraction = 0.0 if slow_1 == slow_0 else (slow - slow_0) / (slow_1 - slow_0)
    fast_fraction = 0.0 if fast_1 == fast_0 else (fast - fast_0) / (fast_1 - fast_0)
    return (
        (1.0 - slow_fraction) * (1.0 - fast_fraction) * corner_values[0, 0]
        + (1.0 - slow_fraction) * fast_fraction * corner_values[0, 1]
        + slow_fraction * (1.0 - fast_fraction) * corner_values[1, 0]
        + slow_fraction * fast_fraction * corner_values[1, 1]
    )


@dataclass(slots=True)
class _AdaptiveGridEvaluator:
    geometry: ConcentricCylinderGeometry
    coefficients: np.ndarray
    scattering_points: object
    volume_weights: object
    detector_grid: np.ndarray
    active: np.ndarray
    incident_direction: object
    relative_tolerance: float
    absolute_tolerance: float
    max_depth: int
    detector_chunk_size: int
    exact_values: dict[_GridIndex, np.ndarray] = field(default_factory=dict, init=False)
    accepted_cells: list[_AcceptedCell] = field(default_factory=list, init=False)
    fallback_indices: set[_GridIndex] = field(default_factory=set, init=False)
    maximum_accepted_relative_error: float = field(default=0.0, init=False)

    @staticmethod
    def _is_indivisible(cell: _AdaptiveCell) -> bool:
        slow_0, slow_1, fast_0, fast_1, _depth = cell
        return slow_1 - slow_0 <= 1 and fast_1 - fast_0 <= 1

    def _evaluate_missing(self, indices: set[_GridIndex]) -> None:
        missing = sorted(index for index in indices if index not in self.exact_values)
        if not missing:
            return
        positions = np.asarray([self.detector_grid[slow, fast] for slow, fast in missing])
        values = attenuation_factors_at_detectors(
            geometry=self.geometry,
            attenuation_coefficients=self.coefficients,
            scattering_points=self.scattering_points,
            volume_weights=self.volume_weights,
            detector_positions=positions,
            incident_direction=self.incident_direction,
            detector_chunk_size=self.detector_chunk_size,
        )
        self.exact_values.update((index, values[:, position]) for position, index in enumerate(missing))

    def _fallback_active_pixels(self, cell: _AdaptiveCell, cell_active: np.ndarray) -> None:
        slow_0, _slow_1, fast_0, _fast_1, _depth = cell
        local_slow, local_fast = np.nonzero(cell_active)
        self.fallback_indices.update(
            (slow_0 + int(slow), fast_0 + int(fast)) for slow, fast in zip(local_slow, local_fast, strict=True)
        )

    def _prepare_cells(
        self, pending_cells: list[_AdaptiveCell]
    ) -> tuple[set[_GridIndex], list[tuple[_AdaptiveCell, set[_GridIndex]]], list[_AdaptiveCell]]:
        sampling_indices: set[_GridIndex] = set()
        cell_samples = []
        next_cells = []
        for cell in pending_cells:
            slow_0, slow_1, fast_0, fast_1, depth = cell
            cell_active = self.active[slow_0 : slow_1 + 1, fast_0 : fast_1 + 1]
            if not np.any(cell_active):
                continue
            if not np.all(cell_active):
                if depth >= self.max_depth or self._is_indivisible(cell):
                    self._fallback_active_pixels(cell, cell_active)
                else:
                    next_cells.extend(_child_cells(cell))
                continue
            samples, checkpoints = _cell_samples(cell)
            sampling_indices.update(samples)
            cell_samples.append((cell, checkpoints))
        return sampling_indices, cell_samples, next_cells

    def _corner_values(self, cell: _AdaptiveCell | _AcceptedCell) -> np.ndarray:
        slow_0, slow_1, fast_0, fast_1 = cell[:4]
        return np.asarray(
            [
                [self.exact_values[(slow_0, fast_0)], self.exact_values[(slow_0, fast_1)]],
                [self.exact_values[(slow_1, fast_0)], self.exact_values[(slow_1, fast_1)]],
            ]
        )

    def _cell_acceptance(self, cell: _AdaptiveCell, checkpoints: set[_GridIndex]) -> tuple[bool, float]:
        corner_values = self._corner_values(cell)
        accepted = True
        cell_relative_error = 0.0
        for index in checkpoints:
            actual = self.exact_values[index]
            absolute_error = np.abs(actual - _bilinear_value(corner_values, cell, index))
            accepted &= bool(
                np.all(absolute_error <= self.absolute_tolerance + self.relative_tolerance * np.abs(actual))
            )
            nonzero = actual != 0.0
            if np.any(nonzero):
                cell_relative_error = max(
                    cell_relative_error,
                    float(np.max(absolute_error[nonzero] / np.abs(actual[nonzero]))),
                )
        return accepted, cell_relative_error

    def _assess_cells(
        self,
        cell_samples: list[tuple[_AdaptiveCell, set[_GridIndex]]],
        next_cells: list[_AdaptiveCell],
    ) -> list[_AdaptiveCell]:
        for cell, checkpoints in cell_samples:
            accepted, relative_error = self._cell_acceptance(cell, checkpoints)
            slow_0, slow_1, fast_0, fast_1, depth = cell
            if accepted or self._is_indivisible(cell):
                self.accepted_cells.append((slow_0, slow_1, fast_0, fast_1))
                self.maximum_accepted_relative_error = max(self.maximum_accepted_relative_error, relative_error)
            elif depth >= self.max_depth:
                self.fallback_indices.update(
                    (slow, fast) for slow in range(slow_0, slow_1 + 1) for fast in range(fast_0, fast_1 + 1)
                )
            else:
                next_cells.extend(_child_cells(cell))
        return next_cells

    def _interpolate_cell(self, factors: np.ndarray, cell: _AcceptedCell) -> None:
        slow_0, slow_1, fast_0, fast_1 = cell
        slow_fraction = (np.zeros(1) if slow_1 == slow_0 else np.linspace(0.0, 1.0, slow_1 - slow_0 + 1))[:, None, None]
        fast_fraction = (np.zeros(1) if fast_1 == fast_0 else np.linspace(0.0, 1.0, fast_1 - fast_0 + 1))[None, :, None]
        corner_values = self._corner_values(cell)
        interpolated = (
            (1.0 - slow_fraction) * (1.0 - fast_fraction) * corner_values[0, 0]
            + (1.0 - slow_fraction) * fast_fraction * corner_values[0, 1]
            + slow_fraction * (1.0 - fast_fraction) * corner_values[1, 0]
            + slow_fraction * fast_fraction * corner_values[1, 1]
        )
        factors[:, slow_0 : slow_1 + 1, fast_0 : fast_1 + 1] = np.moveaxis(interpolated, -1, 0)

    def _render(self) -> tuple[np.ndarray, np.ndarray]:
        factors = np.ones((self.coefficients.shape[0], *self.detector_grid.shape[:2]), dtype=float)
        for cell in self.accepted_cells:
            self._interpolate_cell(factors, cell)
        for slow, fast in self.fallback_indices:
            factors[:, slow, fast] = self.exact_values[(slow, fast)]

        evaluated_mask = np.zeros(self.detector_grid.shape[:2], dtype=bool)
        if self.exact_values:
            evaluated_slow, evaluated_fast = zip(*self.exact_values, strict=True)
            evaluated_mask[np.asarray(evaluated_slow), np.asarray(evaluated_fast)] = True
            for slow, fast in self.exact_values:
                factors[:, slow, fast] = self.exact_values[(slow, fast)]
        return factors, evaluated_mask

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        slow_size, fast_size = self.detector_grid.shape[:2]
        pending_cells = [(0, slow_size - 1, 0, fast_size - 1, 0)]
        while pending_cells:
            sampling_indices, cell_samples, next_cells = self._prepare_cells(pending_cells)
            self._evaluate_missing(sampling_indices)
            pending_cells = self._assess_cells(cell_samples, next_cells)
        self._evaluate_missing(self.fallback_indices)
        return self._render()


def adaptive_attenuation_factors_on_grid(
    *,
    geometry: ConcentricCylinderGeometry,
    attenuation_coefficients,
    scattering_points,
    volume_weights,
    detector_position_grid,
    active_mask=None,
    incident_direction=(0.0, 0.0, 1.0),
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-12,
    max_depth: int = 10,
    detector_chunk_size: int = 256,
) -> AdaptiveDetectorAttenuation:
    """Evaluate smooth attenuation maps on an adaptively refined pixel grid.

    One attenuation-coefficient vector returns one 2D map. Multiple vectors
    return a stack whose leading dimension enumerates coefficient sets, reusing
    geometric path calculations and one refinement mesh. Every map must satisfy
    the checkpoint test before a cell is accepted.

    Each fully active rectangular detector-index cell is sampled at its
    corners, edge midpoints, centre, and four interior quarter points. Bilinear
    interpolation is accepted only when every non-corner checkpoint meets the
    requested mixed absolute/relative tolerance. Failed
    cells subdivide; a cell that reaches ``max_depth`` falls back to exact
    evaluation of all its active pixels. Cells crossing an active-mask
    boundary subdivide until they become uniformly active/inactive or require
    exact active-pixel fallback. Masked pixels retain an identity factor and
    are never passed to the point-ray integrator.

    The tolerance is an a posteriori checkpoint criterion, not a mathematical
    bound between checkpoints. Published correction modules should retain an
    exact point-ray option for validation and difficult geometries.
    """
    detector_grid, active, relative_tolerance, absolute_tolerance = _adaptive_grid_inputs(
        detector_position_grid,
        active_mask,
        relative_tolerance,
        absolute_tolerance,
        max_depth,
    )
    coefficients, scalar_coefficients = _attenuation_coefficient_sets(geometry, attenuation_coefficients)
    evaluator = _AdaptiveGridEvaluator(
        geometry=geometry,
        coefficients=coefficients,
        scattering_points=scattering_points,
        volume_weights=volume_weights,
        detector_grid=detector_grid,
        active=active,
        incident_direction=incident_direction,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        max_depth=max_depth,
        detector_chunk_size=detector_chunk_size,
    )
    factors, evaluated_mask = evaluator.run()

    return AdaptiveDetectorAttenuation(
        factors=factors[0] if scalar_coefficients else factors,
        evaluated_mask=evaluated_mask,
        accepted_cell_count=len(evaluator.accepted_cells),
        exact_fallback_pixel_count=len(evaluator.fallback_indices),
        max_accepted_validation_relative_error=evaluator.maximum_accepted_relative_error,
    )
