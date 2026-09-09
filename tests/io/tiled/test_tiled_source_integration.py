# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest

from modacor.io.tiled import TiledSource


def test_tiled_source_reads_from_server():
    pytest.importorskip("tiled")
    pytest.importorskip("sqlalchemy", reason="Install modacor[tiled-tests] for server integration tests")
    from tiled.adapters.array import ArrayAdapter
    from tiled.adapters.mapping import MapAdapter
    from tiled.client import Context, from_context
    from tiled.server.app import build_app

    tree = MapAdapter(
        {
            "entry": MapAdapter(
                {
                    "data": ArrayAdapter.from_array(
                        np.arange(6).reshape(2, 3), metadata={"attrs": {"units": "counts", "description": "synthetic"}}
                    ),
                    "scalar": ArrayAdapter.from_array(np.array(42)),
                },
                metadata={"attrs": {"title": "Example"}},
            )
        }
    )
    with Context.from_app(build_app(tree)) as context:
        source = TiledSource(
            source_reference="live",
            root_node=from_context(context),
            iosource_method_kwargs={"base_item_path": "entry"},
        )
        # Query metadata before reading to exercise the actual client structure.
        assert source.get_data_shape("data") == (2, 3)
        assert source.get_data_dtype("data") == np.dtype(int)
        assert source.get_data_attributes("data") == {"units": "counts", "description": "synthetic"}
        assert source.get_static_metadata("data@units") == "counts"
        assert source.get_static_metadata("@title") == "Example"
        np.testing.assert_array_equal(source.get_data("data"), np.arange(6).reshape(2, 3))
        np.testing.assert_array_equal(source.get_data("data", np.s_[1, :]), [3, 4, 5])
        assert source.get_data_shape("scalar") == ()
        np.testing.assert_array_equal(source.get_data("scalar"), np.array(42))
