# Capillary sample self-absorption correction

## Summary
Correct detector-resolved sample self-absorption in a centred cylindrical capillary.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.capillary_self_absorption_correction.CapillarySelfAbsorptionCorrection`
- **Source:** [`src/modacor/modules/technique_modules/scattering/capillary_self_absorption_correction.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/capillary_self_absorption_correction.py)
- **Module ID:** CapillarySelfAbsorptionCorrection
- **Module version:** 20260906.3
- **Keywords:** sample, self absorption, capillary, cylinder, transmission

## Required data keys
- signal
- coord_x
- coord_y
- coord_z

## Modifies
- **signal**: signal, uncertainties
- **capillary_sample_attenuation**: signal, uncertainties
- **capillary_self_absorption**: signal, uncertainties
- **capillary_calculated_transmission**: signal, uncertainties
- **capillary_effective_sample_mu**: signal, uncertainties
- **capillary_beam_profile_retained_fraction**: signal
- **capillary_attenuation_evaluated**: signal

## Required arguments
- with_processing_keys
- beam_profile

## Default configuration
```json
{
  "absolute_tolerance": 1e-12,
  "attenuation_key": "capillary_sample_attenuation",
  "beam_profile": null,
  "calculated_transmission_key": "capillary_calculated_transmission",
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
  "correction_key": "capillary_self_absorption",
  "detector_chunk_size": 256,
  "effective_sample_mu_key": "capillary_effective_sample_mu",
  "evaluated_mask_key": "capillary_attenuation_evaluated",
  "evaluation_mode": "adaptive",
  "incident_direction": [
    0.0,
    0.0,
    1.0
  ],
  "input_state": "transmission_normalized",
  "mask_key": "mask",
  "max_depth": 10,
  "minimum_attenuation_factor": 1e-12,
  "profile_retained_fraction_key": "capillary_beam_profile_retained_fraction",
  "relative_tolerance": 0.001,
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
  "transmission_source": null,
  "transmission_uncertainties_sources": {},
  "transmission_units_source": null,
  "wall_mu": 0.0,
  "wall_mu_source": null,
  "wall_mu_units": "1/m",
  "wall_mu_units_source": null,
  "wall_thickness": 0.0,
  "wall_thickness_source": null,
  "wall_thickness_units": "m",
  "wall_thickness_units_source": null,
  "with_processing_keys": [
    "sample"
  ]
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `absolute_tolerance` | int or float | No | 1e-12 | - | Adaptive absolute tolerance. |
| `attenuation_key` | str | No | capillary_sample_attenuation | - | Absolute A_s,sc map key. |
| `beam_profile` | dict | Yes | - | - | Measured-image, gaussian_2d, or trapezoid_2d beam-profile configuration. |
| `calculated_transmission_key` | str | No | capillary_calculated_transmission | - | Calculated whole-beam transmission diagnostic key. |
| `capillary_axis` | tuple | No | [1.0, 0.0, 0.0] | - | Capillary centreline direction in the lab frame; horizontal by default. |
| `capillary_centre` | tuple | No | [0.0, 0.0, 0.0] | - | Point on the capillary centreline in capillary_centre_units. |
| `capillary_centre_units` | str | No | m | - | Capillary-centre length units. |
| `chord_order` | int | No | 12 | - | Gauss--Legendre nodes per occupied chord. |
| `coord_x_key` | str | No | coord_x | - | Detector lab x-coordinate key. |
| `coord_y_key` | str | No | coord_y | - | Detector lab y-coordinate key. |
| `coord_z_key` | str | No | coord_z | - | Detector lab z-coordinate key. |
| `correction_key` | str | No | capillary_self_absorption | - | Applied residual-divisor map key. |
| `detector_chunk_size` | int | No | 256 | - | Expert detector chunk-size override. |
| `effective_sample_mu_key` | str | No | capillary_effective_sample_mu | - | Resolved or derived effective sample attenuation-coefficient key. |
| `evaluated_mask_key` | str | No | capillary_attenuation_evaluated | - | Boolean map identifying exactly evaluated detector pixels. |
| `evaluation_mode` | str | No | adaptive | - | Adaptive detector-grid interpolation or exact active-pixel point rays. |
| `incident_direction` | tuple | No | [0.0, 0.0, 1.0] | - | Incident beam propagation direction in the lab frame. |
| `input_state` | str | No | transmission_normalized | - | One of raw, flux_normalized, or transmission_normalized. |
| `mask_key` | str or NoneType | No | mask | - | Optional boolean mask key; True pixels are inactive and retain identity factors. |
| `max_depth` | int | No | 10 | - | Maximum adaptive subdivision depth. |
| `minimum_attenuation_factor` | int or float | No | 1e-12 | - | Minimum allowed active-pixel attenuation divisor. |
| `profile_retained_fraction_key` | str | No | capillary_beam_profile_retained_fraction | - | Retained incident-weight fraction after beam-profile truncation or thresholding. |
| `relative_tolerance` | int or float | No | 0.001 | - | Adaptive relative tolerance. |
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
| `transmission_source` | str or NoneType | No | - | - | Measured effective whole-beam transmission for transmission_normalized input. |
| `transmission_uncertainties_sources` | dict | No | {} | - | Uncertainty-name to IoSources-reference mapping for measured transmission. |
| `transmission_units_source` | str or NoneType | No | - | - | Optional IoSources reference for measured-transmission units. |
| `wall_mu` | int or float or NoneType | No | 0.0 | - | Wall linear attenuation coefficient; zero is allowed for limit checks. |
| `wall_mu_source` | str or NoneType | No | - | - | Optional IoSources reference for wall linear attenuation coefficient. |
| `wall_mu_units` | str | No | 1/m | - | Wall-mu reciprocal-length units. |
| `wall_mu_units_source` | str or NoneType | No | - | - | Optional IoSources reference for wall-mu units. |
| `wall_thickness` | int or float or NoneType | No | 0.0 | - | Capillary wall thickness, separate from sample radius. |
| `wall_thickness_source` | str or NoneType | No | - | - | Optional IoSources reference for wall thickness. |
| `wall_thickness_units` | str | No | m | - | Wall-thickness units. |
| `wall_thickness_units_source` | str or NoneType | No | - | - | Optional IoSources reference for wall-thickness units. |
| `with_processing_keys` | list | Yes | ["sample"] | - | ProcessingData keys whose detector signal should be corrected. |

## References
https://doi.org/10.1021/acs.cgd.5c00551

## Notes
The default capillary axis is horizontal; tilted straight capillaries are supported. Measured transmission is an effective whole-beam value, not a centreline estimate of mu. This step is only for sample-origin scattering after consistent container removal or when wall scattering is negligible; filled/empty attenuation-aware container subtraction requires the separate composite correction.
