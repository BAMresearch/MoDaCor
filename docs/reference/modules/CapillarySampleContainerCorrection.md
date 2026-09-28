# Capillary sample and container attenuation correction

## Summary
Subtract empty-capillary wall scattering on its filled-capillary attenuation scale, then correct sample-origin attenuation.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.capillary_sample_container_correction.CapillarySampleContainerCorrection`
- **Source:** [`src/modacor/modules/technique_modules/scattering/capillary_sample_container_correction.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/capillary_sample_container_correction.py)
- **Module ID:** CapillarySampleContainerCorrection
- **Module version:** 20260906.3
- **Keywords:** sample, container, capillary, absorption, subtraction

## Required data keys
- signal
- coord_x
- coord_y
- coord_z

## Modifies
- **signal**: signal, uncertainties
- **mask**: signal
- **capillary_sample_attenuation**: signal, uncertainties
- **capillary_wall_attenuation_filled**: signal, uncertainties
- **capillary_wall_attenuation_empty**: signal
- **capillary_wall_subtraction_scale**: signal, uncertainties
- **capillary_filled_calculated_transmission**: signal, uncertainties
- **capillary_empty_calculated_transmission**: signal
- **capillary_effective_sample_mu**: signal, uncertainties
- **capillary_beam_profile_retained_fraction**: signal
- **capillary_sample_attenuation_evaluated**: signal
- **capillary_wall_filled_attenuation_evaluated**: signal
- **capillary_wall_empty_attenuation_evaluated**: signal

## Required arguments
- beam_profile
- filled_processing_key
- empty_processing_key

## Default configuration
```json
{
  "absolute_tolerance": 1e-12,
  "beam_profile": null,
  "capillary_axis": [
    1.0,
    0.0,
    0.0
  ],
  "capillary_centre": [
    0.0,
    0.0,
    0.0
  ],
  "capillary_centre_units": "m",
  "chord_order": 12,
  "coord_x_key": "coord_x",
  "coord_y_key": "coord_y",
  "coord_z_key": "coord_z",
  "detector_chunk_size": 256,
  "effective_sample_mu_key": "capillary_effective_sample_mu",
  "empty_calculated_transmission_key": "capillary_empty_calculated_transmission",
  "empty_centre_mu": 0.0,
  "empty_centre_mu_source": null,
  "empty_centre_mu_units": "1/m",
  "empty_centre_mu_units_source": null,
  "empty_mask_key": null,
  "empty_processing_key": "background",
  "evaluation_mode": "adaptive",
  "filled_calculated_transmission_key": "capillary_filled_calculated_transmission",
  "filled_processing_key": "sample",
  "incident_direction": [
    0.0,
    0.0,
    1.0
  ],
  "mask_key": "mask",
  "max_depth": 10,
  "minimum_attenuation_factor": 1e-12,
  "profile_retained_fraction_key": "capillary_beam_profile_retained_fraction",
  "relative_tolerance": 0.001,
  "sample_attenuation_key": "capillary_sample_attenuation",
  "sample_evaluated_mask_key": "capillary_sample_attenuation_evaluated",
  "sample_mu": null,
  "sample_mu_sensitivity_relative_step": 0.01,
  "sample_mu_source": null,
  "sample_mu_units": "1/m",
  "sample_mu_units_source": null,
  "sample_phase_absorption": null,
  "sample_phase_absorption_source": null,
  "sample_phase_absorption_uncertainties_sources": {},
  "sample_phase_thickness": null,
  "sample_phase_thickness_source": null,
  "sample_phase_thickness_uncertainties_sources": {},
  "sample_phase_thickness_units": "m",
  "sample_phase_thickness_units_source": null,
  "sample_phase_transmission": null,
  "sample_phase_transmission_source": null,
  "sample_phase_transmission_uncertainties_sources": {},
  "sample_radius": null,
  "sample_radius_source": null,
  "sample_radius_units": "m",
  "sample_radius_units_source": null,
  "wall_chord_order": null,
  "wall_empty_attenuation_key": "capillary_wall_attenuation_empty",
  "wall_empty_evaluated_mask_key": "capillary_wall_empty_attenuation_evaluated",
  "wall_filled_attenuation_key": "capillary_wall_attenuation_filled",
  "wall_filled_evaluated_mask_key": "capillary_wall_filled_attenuation_evaluated",
  "wall_mu": 0.0,
  "wall_mu_source": null,
  "wall_mu_units": "1/m",
  "wall_mu_units_source": null,
  "wall_subtraction_scale_key": "capillary_wall_subtraction_scale",
  "wall_thickness": 0.0,
  "wall_thickness_source": null,
  "wall_thickness_units": "m",
  "wall_thickness_units_source": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `absolute_tolerance` | int or float | No | 1e-12 | - | Adaptive absolute tolerance. |
| `beam_profile` | dict | Yes | - | - | Measured-image, gaussian_2d, or trapezoid_2d beam-profile configuration. |
| `capillary_axis` | tuple | No | [1.0, 0.0, 0.0] | - | Capillary centreline direction in the lab frame; horizontal by default. |
| `capillary_centre` | tuple | No | [0.0, 0.0, 0.0] | - | Point on the capillary centreline in capillary_centre_units. |
| `capillary_centre_units` | str | No | m | - | Capillary-centre length units. |
| `chord_order` | int | No | 12 | - | Gauss--Legendre nodes per occupied chord. |
| `coord_x_key` | str | No | coord_x | - | Detector lab x-coordinate key. |
| `coord_y_key` | str | No | coord_y | - | Detector lab y-coordinate key. |
| `coord_z_key` | str | No | coord_z | - | Detector lab z-coordinate key. |
| `detector_chunk_size` | int | No | 256 | - | Expert detector chunk-size override. |
| `effective_sample_mu_key` | str | No | capillary_effective_sample_mu | - | Resolved or derived effective sample attenuation-coefficient key. |
| `empty_calculated_transmission_key` | str | No | capillary_empty_calculated_transmission | - | Output key for calculated empty-capillary transmission. |
| `empty_centre_mu` | int or float or NoneType | No | 0.0 | - | Linear attenuation coefficient inside the nominally empty capillary. |
| `empty_centre_mu_source` | str or NoneType | No | - | - | Optional IoSources reference for empty-centre attenuation. |
| `empty_centre_mu_units` | str | No | 1/m | - | Empty-centre-mu reciprocal-length units. |
| `empty_centre_mu_units_source` | str or NoneType | No | - | - | Optional IoSources reference for empty-centre-mu units. |
| `empty_mask_key` | str or NoneType | No | - | - | Empty-capillary mask key; defaults to mask_key. |
| `empty_processing_key` | str | Yes | background | - | DataBundle containing the matching empty-capillary measurement. |
| `evaluation_mode` | str | No | adaptive | - | Adaptive detector-grid interpolation or exact active-pixel point rays. |
| `filled_calculated_transmission_key` | str | No | capillary_filled_calculated_transmission | - | Output key for calculated filled-capillary transmission. |
| `filled_processing_key` | str | Yes | sample | - | DataBundle containing the filled-capillary measurement and output signal. |
| `incident_direction` | tuple | No | [0.0, 0.0, 1.0] | - | Incident beam propagation direction in the lab frame. |
| `mask_key` | str or NoneType | No | mask | - | Optional boolean mask key; True pixels are inactive and retain identity factors. |
| `max_depth` | int | No | 10 | - | Maximum adaptive subdivision depth. |
| `minimum_attenuation_factor` | int or float | No | 1e-12 | - | Minimum allowed active-pixel attenuation divisor. |
| `profile_retained_fraction_key` | str | No | capillary_beam_profile_retained_fraction | - | Output key for retained beam-profile incident weight. |
| `relative_tolerance` | int or float | No | 0.001 | - | Adaptive relative tolerance. |
| `sample_attenuation_key` | str | No | capillary_sample_attenuation | - | Output key for A_s,sc. |
| `sample_evaluated_mask_key` | str | No | capillary_sample_attenuation_evaluated | - | Exact-evaluation mask for A_s,sc. |
| `sample_mu` | int or float or NoneType | No | - | - | Sample linear attenuation coefficient; alternatively use its source or phase-factor inputs. |
| `sample_mu_sensitivity_relative_step` | int or float | No | 0.01 | - | Relative finite-difference step for derived-sample-mu uncertainty propagation. |
| `sample_mu_source` | str or NoneType | No | - | - | Optional IoSources reference for sample linear attenuation coefficient. |
| `sample_mu_units` | str | No | 1/m | - | Sample-mu reciprocal-length units. |
| `sample_mu_units_source` | str or NoneType | No | - | - | Optional IoSources reference for sample-mu units. |
| `sample_phase_absorption` | int or float or NoneType | No | - | - | Sample-only absorbed fraction A=1-T; use with sample_phase_thickness instead of sample_mu. |
| `sample_phase_absorption_source` | str or NoneType | No | - | - | Optional IoSources reference for the dimensionless sample-only absorbed fraction. |
| `sample_phase_absorption_uncertainties_sources` | dict | No | {} | - | Uncertainty-name to IoSources-reference mapping for sample-only absorption. |
| `sample_phase_thickness` | int or float or NoneType | No | - | - | Transmission path thickness used to derive effective sample mu. |
| `sample_phase_thickness_source` | str or NoneType | No | - | - | Optional IoSources reference for the sample-phase transmission path thickness. |
| `sample_phase_thickness_uncertainties_sources` | dict | No | {} | - | Uncertainty-name to IoSources-reference mapping for sample-phase thickness. |
| `sample_phase_thickness_units` | str | No | m | - | Sample-phase transmission path-thickness units. |
| `sample_phase_thickness_units_source` | str or NoneType | No | - | - | Optional IoSources reference for sample-phase thickness units. |
| `sample_phase_transmission` | int or float or NoneType | No | - | - | Sample-only transmission T; use with sample_phase_thickness instead of sample_mu. |
| `sample_phase_transmission_source` | str or NoneType | No | - | - | Optional IoSources reference for the dimensionless sample-only transmission. |
| `sample_phase_transmission_uncertainties_sources` | dict | No | {} | - | Uncertainty-name to IoSources-reference mapping for sample-only transmission. |
| `sample_radius` | int or float or NoneType | No | - | - | Inner/sample radius; alternatively use sample_radius_source. |
| `sample_radius_source` | str or NoneType | No | - | - | Optional IoSources reference for sample radius. |
| `sample_radius_units` | str | No | m | - | Sample-radius units. |
| `sample_radius_units_source` | str or NoneType | No | - | - | Optional IoSources reference for sample-radius units. |
| `wall_chord_order` | int or NoneType | No | - | - | Wall-origin nodes per occupied chord; defaults to chord_order. |
| `wall_empty_attenuation_key` | str | No | capillary_wall_attenuation_empty | - | Output key for A_c,c. |
| `wall_empty_evaluated_mask_key` | str | No | capillary_wall_empty_attenuation_evaluated | - | Exact-evaluation mask for A_c,c. |
| `wall_filled_attenuation_key` | str | No | capillary_wall_attenuation_filled | - | Output key for A_c,sc. |
| `wall_filled_evaluated_mask_key` | str | No | capillary_wall_filled_attenuation_evaluated | - | Exact-evaluation mask for A_c,sc. |
| `wall_mu` | int or float or NoneType | No | 0.0 | - | Wall linear attenuation coefficient; zero is allowed for limit checks. |
| `wall_mu_source` | str or NoneType | No | - | - | Optional IoSources reference for wall linear attenuation coefficient. |
| `wall_mu_units` | str | No | 1/m | - | Wall-mu reciprocal-length units. |
| `wall_mu_units_source` | str or NoneType | No | - | - | Optional IoSources reference for wall-mu units. |
| `wall_subtraction_scale_key` | str | No | capillary_wall_subtraction_scale | - | Output key for A_c,sc/A_c,c. |
| `wall_thickness` | int or float or NoneType | No | 0.0 | - | Capillary wall thickness, separate from sample radius. |
| `wall_thickness_source` | str or NoneType | No | - | - | Optional IoSources reference for wall thickness. |
| `wall_thickness_units` | str | No | m | - | Wall-thickness units. |
| `wall_thickness_units_source` | str or NoneType | No | - | - | Optional IoSources reference for wall-thickness units. |

## References
https://doi.org/10.1107/S0021889810021114

## Notes
The filled and empty signals must already have comparable exposure and incident-flux normalization. Do not include a separate normalization by measured sample or empty-capillary transmission in the same correction pipeline. The operation is [F - (A_c,sc/A_c,c) E] / A_s,sc.
