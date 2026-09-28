# Detector efficiency correction

## Summary
Divide signal by angle-dependent detector absorption efficiency.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.detector_efficiency_correction.DetectorEfficiencyCorrection`
- **Source:** [`src/modacor/modules/technique_modules/scattering/detector_efficiency_correction.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/detector_efficiency_correction.py)
- **Module ID:** DetectorEfficiencyCorrection
- **Module version:** 20260927.2
- **Keywords:** detector, efficiency, attenuation, absorption

## Required data keys
- signal
- CosAlpha

## Modifies
- **signal**: signal, uncertainties
- **detector_efficiency**: signal, units

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "beam_energy": null,
  "beam_energy_source": null,
  "beam_energy_units": "keV",
  "beam_energy_units_source": null,
  "chemical_composition": null,
  "chemical_composition_source": null,
  "correction_key": "detector_efficiency",
  "cos_alpha_key": "CosAlpha",
  "density": null,
  "density_source": null,
  "density_units": "g/cm^3",
  "density_units_source": null,
  "linear_attenuation_coefficient": null,
  "linear_attenuation_coefficient_source": null,
  "linear_attenuation_coefficient_units": "1/m",
  "linear_attenuation_coefficient_units_source": null,
  "material": null,
  "material_source": null,
  "minimum_cos_alpha": 1e-12,
  "normalize_to_normal_incidence": true,
  "thickness": null,
  "thickness_source": null,
  "thickness_units": "m",
  "thickness_units_source": null,
  "wavelength": null,
  "wavelength_source": null,
  "wavelength_units": "angstrom",
  "wavelength_units_source": null,
  "with_processing_keys": [
    "sample"
  ]
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `beam_energy` | int or float or NoneType | No | - | - | Incident beam energy. |
| `beam_energy_source` | str or NoneType | No | - | - | Optional IoSources key for incident beam energy. |
| `beam_energy_units` | str | No | keV | - | Units for beam_energy. |
| `beam_energy_units_source` | str or NoneType | No | - | - | Optional IoSources key for beam energy units. |
| `chemical_composition` | str or NoneType | No | - | - | Alias for material. |
| `chemical_composition_source` | str or NoneType | No | - | - | Optional IoSources key for chemical_composition. |
| `correction_key` | str | No | detector_efficiency | - | BaseData key used to store the applied relative or absolute efficiency map. |
| `cos_alpha_key` | str | No | CosAlpha | - | BaseData key containing the incidence cosine n dot rhat. |
| `density` | int or float or NoneType | No | - | - | Sensor material density. |
| `density_source` | str or NoneType | No | - | - | Optional IoSources key for sensor density. |
| `density_units` | str | No | g/cm^3 | - | Units for density. |
| `density_units_source` | str or NoneType | No | - | - | Optional IoSources key for density units. |
| `linear_attenuation_coefficient` | int or float or NoneType | No | - | - | Optional direct linear attenuation coefficient. |
| `linear_attenuation_coefficient_source` | str or NoneType | No | - | - | Optional IoSources key for a direct linear attenuation coefficient. |
| `linear_attenuation_coefficient_units` | str | No | 1/m | - | Units for linear_attenuation_coefficient. |
| `linear_attenuation_coefficient_units_source` | str or NoneType | No | - | - | Optional IoSources key for linear attenuation coefficient units. |
| `material` | str or NoneType | No | - | - | Sensor material formula or element symbol, used with xraylib lookup. |
| `material_source` | str or NoneType | No | - | - | Optional IoSources key for sensor material formula or element symbol. |
| `minimum_cos_alpha` | int or float | No | 1e-12 | - | Lower numerical clip for positive cos_alpha values. |
| `normalize_to_normal_incidence` | bool | No | True | - | If true, divide by efficiency(cos_alpha) / efficiency(cos_alpha=1). |
| `thickness` | int or float or NoneType | No | - | - | Sensor thickness. |
| `thickness_source` | str or NoneType | No | - | - | Optional IoSources key for sensor thickness. |
| `thickness_units` | str | No | m | - | Units for thickness. |
| `thickness_units_source` | str or NoneType | No | - | - | Optional IoSources key for thickness units. |
| `wavelength` | int or float or NoneType | No | - | - | Incident wavelength, used to derive beam energy if beam_energy is not configured. |
| `wavelength_source` | str or NoneType | No | - | - | Optional IoSources key for incident wavelength. |
| `wavelength_units` | str | No | angstrom | - | Units for wavelength. |
| `wavelength_units_source` | str or NoneType | No | - | - | Optional IoSources key for wavelength units. |
| `with_processing_keys` | list | Yes | ["sample"] | - | ProcessingData keys whose signal should be corrected. |

## Notes
Material parameters can be provided as direct values or *_source paths. xraylib is used for material lookup unless linear_attenuation_coefficient is configured directly.
