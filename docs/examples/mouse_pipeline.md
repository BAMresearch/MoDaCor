# MOUSE solids pipeline

This example documents the current end-to-end workflow for plate-like solid
samples measured on the MOUSE laboratory SAXS/WAXS instrument. It follows the
same runtime-session pattern as the [Diamond I22 example](dls_i22.md), while
keeping MOUSE-specific file discovery and corrections explicit.

The repository includes the corresponding [draft pipeline](MOUSE_solids.yaml).
It has processed the September 2026 Cu example data successfully, but remains a
working example rather than a frozen production recipe. Capillary
self-absorption and displaced-dispersant subtraction are not implemented yet.

## Input files and batch selection

MOUSE preprocessing combines repeated measurements into configuration-specific
stacked NeXus files. A typical MoDaCor-facing filename is
`MOUSE_20260903_2_165_stacked_modacor.nxs`, where `2` is the batch and `165`
is the instrument configuration. Select an inclusive batch range:

```python
SAMPLE_BATCH_START = 2
SAMPLE_BATCH_END = 2

for candidate in sorted(data_root.glob("MOUSE_*_stacked_modacor.nxs")):
    with h5py.File(candidate, "r") as h5:
        batch = int(h5["/entry1/experiment/batchnum"][()])
        if not SAMPLE_BATCH_START <= batch <= SAMPLE_BATCH_END:
            continue
        configuration = int(h5["/entry1/instrument/configuration"][()])
```

Each sample file identifies its background internally at
`/entry1/processing_required_metadata/background_file`. It may also identify
a separately measured dispersant background at
`/entry1/processing_required_metadata/dispersed_background_file`.

Resolve relative references against the sample file's directory. In the
September 2026 converted data, these references still name the pre-conversion
stacked files, so the notebook adds `_modacor` to the referenced stem when it
is absent. This is a transitional input-data convention, not a pipeline rule.

Validate that sample and background configuration numbers match before
processing. The external `mask_file` metadata need not resolve: the usable
beamstop mask is embedded at `/entry1/instrument/mask/Mask`.

## Standard and displaced-dispersant routes

The current YAML is the standard solids route for a sample and its container
background. A measurement requires the future displaced-dispersant route when:

- `dispersed_background_file` is non-empty; and
- `/entry1/sample/matrixfraction` is strictly between zero and one.

The notebook detects this condition and stops with a clear error instead of
silently applying ordinary background subtraction. The eventual route will
need the sample, container background, separate dispersant-in-container
measurement, and displaced-dispersant factor.

## Important stacked-data normalization

`/entry1/instrument/detector00/data` contains the **sum of all frames** in a
stacked acquisition. Divide it by the total detector count time:

```yaml
TI:
  module: Divide
  short_title: Time normalization
  configuration:
    with_processing_keys: [sample]
    divisor_source: sample::/entry1/instrument/detector00/count_time
    divisor_units_source: sample::/entry1/instrument/detector00/count_time@units
```

Do not divide a summed stack by `frame_exposure_time`. That is the duration of
one frame and produces an intensity too high by approximately the number of
summed frames.

## Pipeline structure

The included pipeline performs these operations:

1. Load sample and background signals with their Poisson and SEM estimates.
2. Load the detector pixel mask, embedded beamstop mask, and flatfield matrix.
3. Mask raw counts outside 0 through 1,000,000 and flatfield corrections
   outside 0.96 through 1.04.
4. Normalize summed sample and background data by total `count_time`.
5. Subtract dark current, multiply by the flatfield correction matrix, then
   divide by incident flux and transmission.
6. Average the stacked dimensions and subtract the corrected background.
7. Combine static and dynamic masks and dilate the completed mask by one pixel
   using a square footprint.
8. Calculate pixel coordinates, Q, azimuth, exit-angle cosine, and solid angle.
9. Apply solid-angle, detector-efficiency, polarization, flat-plate
   self-absorption, and sample-thickness corrections.
10. Azimuthally average to I(Q), combine selected uncertainty estimates, and
    publish live corrected-image and I(Q) plots.

The flatfield is a correction matrix and is therefore a multiply operation.
`DetectorEfficiencyCorrection` divides by the calculated absorption
efficiency; its correction multiplier can exceed one although the physical
efficiency itself cannot. Sensor material and thickness come from detector
metadata. The effect is negligible for Cu radiation in the 450 micrometre Si
sensor but becomes relevant for Mo-source measurements.

The laboratory source is treated as unpolarized:

```yaml
PO:
  module: PolarizationCorrection
  configuration:
    with_processing_keys: [sample]
    mode: linear_fraction
    polarization_factor: 0.5
    polarization_angular_offset: 0.0
    polarization_angular_offset_units: degree
```

`FlatPlateSelfAbsorptionCorrection` models a uniform plate normal to the
incident beam. It uses the measured sample transmission and `CosAlpha` from
the scattering geometry. This is not the same correction as a separate
downstream attenuator plate, and it should not be used for capillaries.

## Uncertainty propagation

The draft keeps independent uncertainty sources separate through the
correction chain. It currently reads detector Poisson and SEM estimates,
sample and background transmission SEM, wavelength error, and sample-thickness
SEM. The transmission SEM is also propagated through the angle-dependent
self-absorption factor.

The converted files do not yet provide usable estimates for:

- total detector count time;
- incident flux;
- dark-current correction;
- flatfield correction matrix;
- detector positions and rotations; and
- detector sensor thickness.

These should be added to the source files and wired into the corresponding
`*_uncertainties_sources` configuration when available. Detector readout time
is an acquisition parameter and must not be substituted for count-time
uncertainty.

## Runtime sessions and output

Use one runtime session per MOUSE configuration. Multiple selected batches
with the same configuration may reuse that session and its cached static
state. Register `sample` and `background` HDF sources for each run, using a
full run to seed a new session and `auto` mode for later files after reporting
`sample` and `background` as changed sources.

The runtime `mouse` source profile accepts required `sample` and
`background` sources and optional `defaults`, `intensity_calibration`, and
`intensity_calibration_background` sources. The draft pipeline reads its
correction metadata directly from the sample and background files.

Request HDF output from the process call, for example:

```python
process_payload = {
    "mode": "full",
    "run_name": sample_path.stem,
    "rollback_snapshot": False,
    "write_hdf": {
        "path": str(output_path),
        "data_paths": ["/sample/signal", "/sample/Q"],
    },
}
```

The HDF writer records NeXus `default` attributes through the output
hierarchy, including the terminal `BaseData` group whose default points to
`signal`.

## Live plots are IoSinks

The live plot endpoints use official `IoSink` implementations. A runtime sink
of type `plotly_json` constructs a `PlotlyJSONSink` and registers it in the
session's `IoSinks` collection:

```python
api_request(
    "PUT",
    f"/v1/sessions/{session_id}/sinks",
    payload={"sinks": [
        {"ref": "plots", "type": "plotly_json", "location": "buffer://session"}
    ]},
)
```

The YAML's `Plot1DVisualization` and `Plot2DVisualization` steps write
Plotly payloads to `plots::mouse-1d` and `plots::mouse-2d`. With the local
server at `http://127.0.0.1:8901`, view them at:

```text
/v1/sessions/{session_id}/plots/plots/mouse-1d
/v1/sessions/{session_id}/plots/plots/mouse-2d
```

Install the `plotting` optional dependency for the browser views.

## Notebook-owned server cleanup

A notebook may start a local server when none is already listening. Track the
returned process and stop only that process, leaving independently started
servers untouched:

```python
def stop_server():
    global SERVER_PROCESS
    if SERVER_PROCESS is not None and SERVER_PROCESS.poll() is None:
        SERVER_PROCESS.terminate()
        SERVER_PROCESS.wait(timeout=10)
    SERVER_PROCESS = None
```

An explicit final cleanup cell can call `stop_server()`. Registering the same
function with `atexit` also cleans up a notebook-owned server when the kernel
shuts down normally.

## Current limitations

The checked-in YAML intentionally preserves the working September 2026 draft
so it can be tested and improved alongside MoDaCor. Before treating it as a
general production pipeline, complete and validate:

- displaced-dispersant subtraction;
- capillary self-absorption;
- uncertainty metadata listed above;
- Mo-source measurements and detector-efficiency effects; and
- calibration and absolute-intensity validation over all configurations.
