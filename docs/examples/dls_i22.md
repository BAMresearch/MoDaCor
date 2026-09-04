# DLS I22 pipeline patterns

The DLS I22 operando workflow is represented by the workspace notebook
`I22_solids_server_operando_preprocessed.ipynb` and the paired SAXS/WAXS
pipeline YAML files `I22_SAXS_solids_operando.yaml` and
`I22_WAXS_solids_operando.yaml`. Use this page as a high-level application
example for Diamond Light Source beamline I22 rather than as a bundled test
fixture.

## Runtime service workflow

The notebook runs the SAXS and WAXS pipelines through the MoDaCor runtime
service. It creates one session per detector pipeline, or optionally one server
process per detector when process-level concurrency is being tested.

The first processed measurement runs in `full` mode to seed the session state.
Later measurements update only the `sample` source and process in `auto` mode,
so unchanged calibration, mask, and background sources can be reused by the
runtime session cache while still falling back to a full rerun after partial-run
failure.

## Preprocessed input shape

The notebook keeps the original NeXus/HDF5 measurement files untouched. For
each measurement it writes a compact MoDaCor-facing file that:

- externally links `/entry1` from the original master file;
- adds broadcast-ready normalization arrays under `/modacor/normalization`;
- stores scalar calibration values under `/modacor/calibration`;
- reduces the beamstop-diode channel over its 2,000-sample axis to per-frame
  mean, standard deviation, SEM, and valid-sample count values;
- reshapes detector count time and transmission arrays to the detector-divisor
  layout used by the pipelines.

Detector geometry is deliberately not precomputed in the notebook. The SAXS and
WAXS YAML files point at NeXus calibration files, and MoDaCor resolves detector
coordinates and scattering geometry through `PixelCoordinates3D` and
`XSGeometryFromPixelCoordinates`.

## Source and sink roles

The server sessions register these HDF source roles:

- `sample`, changed for each operando measurement;
- `background`, held stable across the batch;
- `saxs_calibration` and `saxs_mask`;
- `waxs_calibration` and `waxs_mask`.

The pipelines also use runtime sink roles:

- `plots` for live 1D and 2D Plotly payloads;
- `result_hdf` for selected `ProcessingData` snapshots.

## Correction pattern

The SAXS and WAXS pipelines share the same broad structure:

1. load sample, background, calibration-shape, and mask data;
2. attach Poisson uncertainties;
3. mask invalid raw counts;
4. normalize sample and background by beamstop-diode intensity;
5. normalize by detector count time;
6. average frame stacks with weights;
7. subtract the corrected background;
8. compute static pixel coordinates and scattering geometry from calibration
   NeXus metadata;
9. index pixels for azimuthal integration;
10. attach static maps and combine instrument, sample raw-count, and background
    raw-count masks;
11. apply solid-angle, detector-efficiency, polarization, and absolute-scale
    corrections;
12. publish live 2D and I(Q) plots and write HDF outputs.

The WAXS flow additionally applies an aluminium attenuator-plate correction
before polarization correction. In the current operando YAML, that correction
divides by the angle-dependent aluminium transmission and avoids a second
division by the scalar `/modacor/normalization/transmission` value.

## Detector-specific integration settings

The SAXS pipeline uses azimuthal integration from `0.03` to `3.36 1/nm` with
500 logarithmic Q bins.

The WAXS pipeline uses azimuthal integration from about `4.424` to
`54.717 1/nm` with 1502 linear Q bins.

Both pipelines use the calibration wavelength from the detector-specific NeXus
calibration file, silicon detector-efficiency correction with `0.32 mm`
thickness, linear polarization factor `0.9`, and an absolute intensity factor
stored in the preprocessed sample file.
