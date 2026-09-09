"""Numerical attenuation models."""

from modacor.models.attenuation.beam_profiles import (
    BeamProfileQuadrature,
    gaussian_beam_profile,
    image_beam_profile,
    trapezoid_beam_profile,
)
from modacor.models.attenuation.concentric_cylinder import (
    AdaptiveDetectorAttenuation,
    adaptive_attenuation_factors_on_grid,
    attenuation_factors_at_detectors,
    attenuation_factors_for_phase_at_detectors,
    beam_chord_quadrature,
    direct_beam_transmission,
    uniform_cross_section_quadrature,
)
from modacor.models.attenuation.flat_plate import (
    flat_plate_relative_attenuation,
    flat_plate_relative_attenuation_derivative,
)

__all__ = [
    "AdaptiveDetectorAttenuation",
    "BeamProfileQuadrature",
    "adaptive_attenuation_factors_on_grid",
    "attenuation_factors_at_detectors",
    "attenuation_factors_for_phase_at_detectors",
    "beam_chord_quadrature",
    "direct_beam_transmission",
    "flat_plate_relative_attenuation",
    "flat_plate_relative_attenuation_derivative",
    "gaussian_beam_profile",
    "image_beam_profile",
    "trapezoid_beam_profile",
    "uniform_cross_section_quadrature",
]
