# Add 3D pixel coordinates (generic)

## Summary
Computes 3D detector element center coordinates in lab-frame axes.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.pixel_coordinates_3d.PixelCoordinates3D`
- **Source:** [`src/modacor/modules/technique_modules/scattering/pixel_coordinates_3d.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/pixel_coordinates_3d.py)
- **Module ID:** PixelCoordinates3D
- **Module version:** 20260927.2
- **Keywords:** geometry, coordinates, detector

## Required data keys
- signal

## Modifies
- **coord_x**: signal, uncertainties
- **coord_y**: signal, uncertainties
- **coord_z**: signal, uncertainties

## Required arguments
- det_coord_z_source
- det_coord_x_source
- det_coord_y_source
- pixel_pitch_slow_source
- pixel_pitch_fast_source

## Default configuration
```json
{
  "basis_fast": [
    1.0,
    0.0,
    0.0
  ],
  "basis_normal": [
    0.0,
    0.0,
    1.0
  ],
  "basis_slow": [
    0.0,
    1.0,
    0.0
  ],
  "det_coord_x_source": null,
  "det_coord_x_uncertainties_sources": {},
  "det_coord_x_units_source": null,
  "det_coord_y_source": null,
  "det_coord_y_uncertainties_sources": {},
  "det_coord_y_units_source": null,
  "det_coord_z_source": null,
  "det_coord_z_uncertainties_sources": {},
  "det_coord_z_units_source": null,
  "detector_frame": null,
  "pixel_pitch_fast_source": null,
  "pixel_pitch_fast_uncertainties_sources": {},
  "pixel_pitch_fast_units_source": null,
  "pixel_pitch_slow_source": null,
  "pixel_pitch_slow_uncertainties_sources": {},
  "pixel_pitch_slow_units_source": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `basis_fast` | tuple | No | [1.0, 0.0, 0.0] | - | Basis vector for the fast detector axis. |
| `basis_normal` | tuple | No | [0.0, 0.0, 1.0] | - | Basis vector for the detector normal. |
| `basis_slow` | tuple | No | [0.0, 1.0, 0.0] | - | Basis vector for the slow detector axis. |
| `det_coord_x_source` | str or NoneType | Yes | - | - | IoSources key for detector x-coordinate signal. |
| `det_coord_x_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for detector x-coordinate. |
| `det_coord_x_units_source` | str or NoneType | No | - | - | IoSources key for detector x-coordinate units. |
| `det_coord_y_source` | str or NoneType | Yes | - | - | IoSources key for detector y-coordinate signal. |
| `det_coord_y_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for detector y-coordinate. |
| `det_coord_y_units_source` | str or NoneType | No | - | - | IoSources key for detector y-coordinate units. |
| `det_coord_z_source` | str or NoneType | Yes | - | - | IoSources key for detector z-coordinate signal. |
| `det_coord_z_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for detector z-coordinate. |
| `det_coord_z_units_source` | str or NoneType | No | - | - | IoSources key for detector z-coordinate units. |
| `detector_frame` | dict or NoneType | No | - | - | Optional detector-frame adapter. Use {'type': 'nexus', 'source': '<ref>', 'detector_path': '/entry1/instrument/detector'} to load a NeXus NXdetector/NXdetector_module transformation chain. |
| `pixel_pitch_fast_source` | str or NoneType | Yes | - | - | IoSources key for fast-axis detector element size signal. |
| `pixel_pitch_fast_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for fast-axis detector element size. |
| `pixel_pitch_fast_units_source` | str or NoneType | No | - | - | IoSources key for fast-axis detector element size units; prefer length units such as m or mm. |
| `pixel_pitch_slow_source` | str or NoneType | Yes | - | - | IoSources key for slow-axis detector element size signal. |
| `pixel_pitch_slow_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for slow-axis detector element size. |
| `pixel_pitch_slow_units_source` | str or NoneType | No | - | - | IoSources key for slow-axis detector element size units; prefer length units such as m or mm. |
