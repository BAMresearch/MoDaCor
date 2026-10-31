"""Temporary R07 migration snapshot, removed in R07e.

Regenerate from runtime contracts with MODACOR_UPDATE_CONTRACT_SNAPSHOT=1:
    MODACOR_UPDATE_CONTRACT_SNAPSHOT=1 .venv-dev/bin/python -m pytest \
        tests/modules/test_dependency_contract_snapshot.py -q
Never edit the generated JSON expectations by hand.
"""

import json
import os
from pathlib import Path

import modacor.modules as modules
from modacor.io.io_sources import IoSources

SNAPSHOT = Path(__file__).parent / "data" / "dependency_contract_snapshot.json"

# The first case uses default data-key names; both cases select two bundles
# unless a particular output mode accepts only one input.
BASE_CONFIGS = {
    "AppendProcessingData": {"processing_key": "sample", "signal_location": "data::/signal"},
    "CopyDataBundleKeys": {"data_keys": ["signal", "Q"]},
}
CUSTOM_CONFIGS = {
    "AppendProcessingData": {
        "processing_key": "background",
        "databundle_output_key": "counts",
        "signal_location": "data::/counts",
        "units_location": "units::/counts",
        "weights_location": "weights::/counts",
        "uncertainties_sources": {"stat": "errors::/counts"},
    },
    "AppendSource": {"source_identifier": "calibration", "source_location": "calibration.h5"},
    "AppendSink": {"sink_identifier": "result", "sink_location": "result.h5"},
    "AngleToQ": {"angle_zero": 0.5, "angle_key": "theta", "output_key": "q"},
    "ApplyMask": {"mask_key": "bad_pixels", "basedata_to_mask": ["signal", "Q"]},
    "BitwiseOrMasks": {"target_mask_key": "combined", "source_mask_keys": ["bad", "saturated"]},
    "CapillarySampleContainerCorrection": {"empty_mask_key": "empty_mask"},
    "CombineUncertainties": {"target_basedata_key": "counts", "combinations": {"total": ["stat", "sys"]}},
    "CombineUncertaintiesMax": {"target_basedata_key": "counts", "combinations": {"total": ["stat", "sys"]}},
    "ConcatenateDatabundles": {"output_processing_key": "merged", "data_keys": ["counts", "q"]},
    "CopyDataBundleKeys": {"key_map": {"signal": "copied_signal", "Q": "copied_q"}},
    "DilateMask": {"source_mask_key": "bad", "target_mask_key": "dilated"},
    "ReduceMask": {"source_mask_key": "bad", "target_mask_key": "reduced"},
    "DivideDatabundles": {"dividend_data_key": "counts", "divisor_data_key": "normalizer"},
    "MultiplyDatabundles": {"multiplicand_data_key": "counts", "multiplier_data_key": "normalizer"},
    "IndexByCoordinate": {"coordinate_key": "angle", "index_key": "index", "edges_key": "edges"},
    "IndexedAverager": {"with_processing_keys": ["sample"], "output_processing_key": "averaged", "axis_key": None},
    "FindScaleFactor1D": {"fit_background": True, "diagnostic_prefix": "fit"},
    "FindCenterOfMass1D": {"diagnostic_prefix": "custom_center"},
    "Integrate1D": {"output_processing_keys": ["sample_integral", "background_integral"]},
    "Negate": {"data_key": "counts"},
    "PixelCoordinates3D": {"detector_frame": {"type": "nexus", "source": "geometry", "depends_on": "/detector"}},
    "Plot1DVisualization": {
        "x_path": "sample/Q",
        "y_path": "sample/signal",
        "xerr_path": "background/Q",
        "yerr_path": "background/signal",
    },
    "Plot2DVisualization": {"data_path": "sample/signal"},
    "SinkProcessingData": {"data_paths": ["sample", "background/signal"]},
    "SubtractInterpolated1D": {"output_signal_key": "subtracted", "remapped_background_key": "remapped"},
    "ThresholdMask": {"source_basedata_key": "counts", "target_mask_key": "threshold"},
    "UnitsLabelUpdate": {"update_pairs": {"signal": "count", "Q": "1/nm"}},
    "XSGeometryFromPixelCoordinates": {
        "detector_frame": {"type": "nexus", "source": "geometry", "depends_on": "/detector"},
        "sample_z_override": {"type": "nexus", "source": "sample_position", "transform_path": "/sample/z"},
    },
}


def snapshot_cases():
    for name in modules.__all__:
        for label in ("default_keys", "configured"):
            step = getattr(modules, name)(io_sources=IoSources())
            config = {"with_processing_keys": ["sample", "background"]}
            if label == "default_keys":
                config.update(BASE_CONFIGS.get(name, {}))
            else:
                # Source-bearing arguments use distinct refs to detect omissions,
                # including nested uncertainty mappings.
                for key in step.configuration:
                    if key.endswith("_source"):
                        config[key] = f"{key}::/value"
                    elif key.endswith("_uncertainties_sources"):
                        config[key] = {"stat": f"{key}::/sigma"}
                if "mask_key" in step.configuration:
                    config["mask_key"] = "bad_pixels"
                config.update(CUSTOM_CONFIGS.get(name, {}))
            step.modify_config_by_kwargs(**config)
            yield f"{name}[{label}]", step


def test_dependency_contract_snapshot():
    actual = {}
    for label, step in snapshot_cases():
        contract = step.dependency_contract()
        actual[label] = {
            field: sorted(getattr(contract, field))
            for field in ("source_refs", "processing_reads", "processing_writes")
        }
    if os.environ.get("MODACOR_UPDATE_CONTRACT_SNAPSHOT") == "1":
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    expected = json.loads(SNAPSHOT.read_text())
    assert {key.split("[", 1)[0] for key in expected} == set(modules.__all__)
    assert actual == expected
