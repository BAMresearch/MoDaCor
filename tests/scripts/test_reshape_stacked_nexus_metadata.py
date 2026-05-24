from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "reshape_stacked_nexus_metadata.py"
SPEC = importlib.util.spec_from_file_location("reshape_stacked_nexus_metadata", SCRIPT_PATH)
reshape_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = reshape_mod
SPEC.loader.exec_module(reshape_mod)


def _create_stacked_file(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        detector = h5.create_group("entry1/instrument/detector00")
        detector.create_dataset("data", data=np.zeros((3, 1, 4, 5), dtype=np.float32))
        exposure = detector.create_dataset("frame_exposure_time", data=np.array([[1.0], [2.0], [3.0]]))
        exposure.attrs["units"] = "s"
        detector.create_dataset("averaged_number_of_frames", data=np.array([10, 11, 12], dtype=np.int64))

        sample = h5.create_group("entry1/sample")
        transmission = sample.create_dataset("transmission", data=np.array([[0.8], [0.9], [1.0]]))
        transmission.attrs["units"] = "dimensionless"

        beam = sample.create_group("beam")
        beam.create_dataset("flux", data=np.array([[100.0], [110.0], [120.0]]), compression="gzip")

        sample.create_dataset("sample_name", data=np.array([b"a", b"b", b"c"]))
        h5.create_dataset("entry1/pixel_map", data=np.ones((4, 5)))


def test_convert_file_appends_trailing_singleton_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.nxs"
    output = tmp_path / "output.nxs"
    _create_stacked_file(source)

    plans = reshape_mod.convert_file(source, output)

    assert {plan.path for plan in plans} == {
        "entry1/instrument/detector00/averaged_number_of_frames",
        "entry1/instrument/detector00/frame_exposure_time",
        "entry1/sample/beam/flux",
        "entry1/sample/transmission",
    }

    with h5py.File(output, "r") as h5:
        assert h5["entry1/instrument/detector00/data"].shape == (3, 1, 4, 5)
        assert h5["entry1/instrument/detector00/frame_exposure_time"].shape == (3, 1, 1, 1)
        assert h5["entry1/instrument/detector00/averaged_number_of_frames"].shape == (3, 1, 1, 1)
        assert h5["entry1/sample/transmission"].shape == (3, 1, 1, 1)
        assert h5["entry1/sample/beam/flux"].shape == (3, 1, 1, 1)
        assert h5["entry1/sample/beam/flux"].compression == "gzip"
        assert h5["entry1/sample/transmission"].attrs["units"] == "dimensionless"
        assert h5["entry1/sample/sample_name"].shape == (3,)
        assert h5["entry1/pixel_map"].shape == (4, 5)

        np.testing.assert_allclose(
            h5["entry1/sample/transmission"][()],
            np.array([0.8, 0.9, 1.0]).reshape((3, 1, 1, 1)),
        )


def test_dry_run_does_not_write_output_file(tmp_path: Path) -> None:
    source = tmp_path / "source.nxs"
    output = tmp_path / "output.nxs"
    _create_stacked_file(source)

    plans = reshape_mod.convert_file(source, output, dry_run=True)

    assert plans
    assert not output.exists()
    with h5py.File(source, "r") as h5:
        assert h5["entry1/sample/transmission"].shape == (3, 1)


def test_metadata_paths_can_restrict_conversion(tmp_path: Path) -> None:
    source = tmp_path / "source.nxs"
    output = tmp_path / "output.nxs"
    _create_stacked_file(source)

    plans = reshape_mod.convert_file(
        source,
        output,
        metadata_paths=["entry1/sample/transmission", "/entry1/sample/beam/flux"],
    )

    assert [plan.path for plan in plans] == ["entry1/sample/transmission", "entry1/sample/beam/flux"]
    with h5py.File(output, "r") as h5:
        assert h5["entry1/sample/transmission"].shape == (3, 1, 1, 1)
        assert h5["entry1/sample/beam/flux"].shape == (3, 1, 1, 1)
        assert h5["entry1/instrument/detector00/frame_exposure_time"].shape == (3, 1)


def test_exclude_metadata_paths_skips_automatic_discovery(tmp_path: Path) -> None:
    source = tmp_path / "source.nxs"
    output = tmp_path / "output.nxs"
    _create_stacked_file(source)
    with h5py.File(source, "a") as h5:
        h5["entry1/sample"].create_dataset("thickness", data=np.array([[1.0], [1.1], [1.2]]))

    plans = reshape_mod.convert_file(
        source,
        output,
        exclude_metadata_paths=["/entry1/sample/thickness"],
    )

    assert "entry1/sample/thickness" not in {plan.path for plan in plans}
    with h5py.File(output, "r") as h5:
        assert h5["entry1/sample/thickness"].shape == (3, 1)
        assert h5["entry1/sample/transmission"].shape == (3, 1, 1, 1)
