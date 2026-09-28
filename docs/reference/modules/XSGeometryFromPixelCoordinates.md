# Add Q, Psi, TwoTheta, Omega from pixel coordinates

## Summary
Compute Q-vector components and angles from lab-frame pixel coordinates.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.xs_geometry_from_pixel_coordinates.XSGeometryFromPixelCoordinates`
- **Source:** [`src/modacor/modules/technique_modules/scattering/xs_geometry_from_pixel_coordinates.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/xs_geometry_from_pixel_coordinates.py)
- **Module ID:** XSGeometryFromPixelCoordinates
- **Module version:** 20260927.2
- **Keywords:** geometry, Q, Psi, TwoTheta, Solid Angle, Omega, scattering

## Required data keys
- coord_x
- coord_y
- coord_z

## Modifies
- **Q0**: signal, uncertainties
- **Q1**: signal, uncertainties
- **Q2**: signal, uncertainties
- **Q**: signal, uncertainties
- **Psi**: signal
- **TwoTheta**: signal, uncertainties
- **CosAlpha**: signal, uncertainties
- **Omega**: signal, uncertainties

## Required arguments
- sample_z_source
- wavelength_source
- pixel_pitch_slow_source
- pixel_pitch_fast_source

## Default configuration
```json
{
  "detector_frame": null,
  "detector_normal": [
    0.0,
    0.0,
    1.0
  ],
  "pixel_pitch_fast_source": null,
  "pixel_pitch_fast_uncertainties_sources": {},
  "pixel_pitch_fast_units_source": null,
  "pixel_pitch_slow_source": null,
  "pixel_pitch_slow_uncertainties_sources": {},
  "pixel_pitch_slow_units_source": null,
  "sample_z_override": null,
  "sample_z_source": null,
  "sample_z_uncertainties_sources": {},
  "sample_z_units_source": null,
  "wavelength_source": null,
  "wavelength_uncertainties_sources": {},
  "wavelength_units_source": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `detector_frame` | dict or NoneType | No | - | - | Optional detector-frame adapter. Use {'type': 'nexus', 'source': '<ref>', 'detector_path': '/entry1/instrument/detector'} to load pixel pitch and detector normal from a NeXus NXdetector/NXdetector_module transformation chain. |
| `detector_normal` | tuple | No | [0.0, 0.0, 1.0] | - | Detector normal unit vector in lab frame. |
| `pixel_pitch_fast_source` | str or NoneType | Yes | - | - | IoSources key for fast-axis detector element size signal. |
| `pixel_pitch_fast_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for fast-axis detector element size. |
| `pixel_pitch_fast_units_source` | str or NoneType | No | - | - | IoSources key for fast-axis detector element size units; prefer length units such as m or mm. |
| `pixel_pitch_slow_source` | str or NoneType | Yes | - | - | IoSources key for slow-axis detector element size signal. |
| `pixel_pitch_slow_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for slow-axis detector element size. |
| `pixel_pitch_slow_units_source` | str or NoneType | No | - | - | IoSources key for slow-axis detector element size units; prefer length units such as m or mm. |
| `sample_z_override` | dict or int or float or NoneType | No | - | - | Optional sample z-position override. Mapping form accepts either value/units or {'type': 'nexus', 'source': '<ref>', 'transform_path': '<path>'}. |
| `sample_z_source` | str or NoneType | Yes | - | - | IoSources key for sample z-position signal. |
| `sample_z_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for sample z-position. |
| `sample_z_units_source` | str or NoneType | No | - | - | IoSources key for sample z-position units. |
| `wavelength_source` | str or NoneType | Yes | - | - | IoSources key for wavelength signal. |
| `wavelength_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for wavelength. |
| `wavelength_units_source` | str or NoneType | No | - | - | IoSources key for wavelength units. |
