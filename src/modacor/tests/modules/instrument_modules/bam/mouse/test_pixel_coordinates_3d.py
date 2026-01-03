# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_source import IoSource
from modacor.io.io_sources import IoSources
from modacor.modules.instrument_modules.bam.mouse.pixel_coordinates_3d import PixelCoordinates3D


class DictIoSource(IoSource):
    """Minimal IoSource for tests: serves arrays and static metadata from dicts."""

    def __init__(self, *, source_reference: str, arrays: dict[str, np.ndarray], meta: dict[str, object]):
        super().__init__(source_reference=source_reference)
        self._arrays = arrays
        self._meta = meta

    def get_data(self, data_key: str, load_slice=None) -> np.ndarray:
        return np.asarray(self._arrays[data_key])

    def get_data_shape(self, data_key: str):
        return np.asarray(self._arrays[data_key]).shape

    def get_data_dtype(self, data_key: str):
        return np.asarray(self._arrays[data_key]).dtype

    def get_data_attributes(self, data_key: str):
        return {}

    def get_static_metadata(self, data_key: str):
        return self._meta[data_key]


def test_mouse_pixel_coordinates_loads_from_sources():
    ios = IoSources()

    arrays = {
        "origin": np.array([0.0, 0.0, 1.0]),
        "pixel_pitch_slow": np.array(2e-3),
        "pixel_pitch_fast": np.array(1e-3),
        "beam_center_slow_px": np.array(0.0),
        "beam_center_fast_px": np.array(0.0),
    }
    meta = {
        "origin_units": "m",
        "pixel_pitch_slow_units": "m/pixel",
        "pixel_pitch_fast_units": "m/pixel",
        "beam_center_slow_px_units": "pixel",
        "beam_center_fast_px_units": "pixel",
    }

    ios.register_source(DictIoSource(source_reference="dummy", arrays=arrays, meta=meta))

    pd = ProcessingData()
    b = DataBundle()
    b["signal"] = BaseData(signal=np.zeros((2, 3)), units=ureg.dimensionless, rank_of_data=2)
    pd["sample"] = b

    step = PixelCoordinates3D(io_sources=ios)
    step.configuration.update(
        {
            "with_processing_keys": ["sample"],
            "origin_source": "dummy::origin",
            "origin_units_source": "dummy::origin_units",
            "origin_uncertainties_sources": {},
            "pixel_pitch_slow_source": "dummy::pixel_pitch_slow",
            "pixel_pitch_slow_units_source": "dummy::pixel_pitch_slow_units",
            "pixel_pitch_slow_uncertainties_sources": {},
            "pixel_pitch_fast_source": "dummy::pixel_pitch_fast",
            "pixel_pitch_fast_units_source": "dummy::pixel_pitch_fast_units",
            "pixel_pitch_fast_uncertainties_sources": {},
            "beam_center_slow_px_source": "dummy::beam_center_slow_px",
            "beam_center_slow_px_units_source": "dummy::beam_center_slow_px_units",
            "beam_center_slow_px_uncertainties_sources": {},
            "beam_center_fast_px_source": "dummy::beam_center_fast_px",
            "beam_center_fast_px_units_source": "dummy::beam_center_fast_px_units",
            "beam_center_fast_px_uncertainties_sources": {},
            # identity basis
            "basis_fast": (1.0, 0.0, 0.0),
            "basis_slow": (0.0, 1.0, 0.0),
            "basis_normal": (0.0, 0.0, 1.0),
        }
    )

    step.execute(pd)
    out = pd["sample"]

    n_slow, n_fast = out["coord_x"].signal.shape

    fast = (np.arange(n_fast) + 0.5) * 1e-3
    slow = (np.arange(n_slow) + 0.5) * 2e-3

    exp_x = np.broadcast_to(fast[None, :], (n_slow, n_fast))
    exp_y = np.broadcast_to(slow[:, None], (n_slow, n_fast))
    exp_z = np.ones((n_slow, n_fast)) * 1.0

    np.testing.assert_allclose(out["coord_x"].signal, exp_x)
    np.testing.assert_allclose(out["coord_y"].signal, exp_y)
    np.testing.assert_allclose(out["coord_z"].signal, exp_z)

    assert np.all(out["coord_x"].signal[0, :] == out["coord_x"].signal[1, :])
    assert np.all(out["coord_y"].signal[:, 0] == out["coord_y"].signal[:, 1])
