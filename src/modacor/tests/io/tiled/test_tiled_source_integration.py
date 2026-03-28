# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "22/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import threading
import time
from collections.abc import Iterator
from typing import Optional

import numpy as np
import pytest

from modacor.io.tiled.tiled_source import TiledSource

try:
    from tiled.adapters.array import ArrayAdapter
    from tiled.server.app import build_app
    from tiled.tree import Tree
except ImportError:  # pragma: no cover - dependency optional
    ArrayAdapter = None  # type: ignore[assignment]
    build_app = None  # type: ignore[assignment]
    Tree = None  # type: ignore[assignment]

try:
    import uvicorn
except ImportError:  # pragma: no cover - dependency optional
    uvicorn = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    ArrayAdapter is None or uvicorn is None,
    reason="Tiled integration test requires 'tiled' and 'uvicorn' extras",
)


@pytest.fixture(scope="module")
def tiled_server() -> Iterator[str]:
    """Start a minimal Tiled server in the background and yield its base URL."""
    data = np.arange(6).reshape(2, 3)
    tree = Tree(
        {
            "entry": Tree(
                {
                    "data": ArrayAdapter.from_array(
                        data,
                        metadata={"attrs": {"units": "counts", "description": "synthetic"}},
                    )
                }
            )
        }
    )

    app = build_app(tree)

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)

    def _run_server() -> None:
        server.run()

    thread = threading.Thread(target=_run_server, name="uvicorn-tiled", daemon=True)
    thread.start()

    # Wait for server to start listening and record the assigned port
    timeout_s = 10.0
    start = time.perf_counter()
    bound_address: Optional[tuple[str, int]] = None

    while (time.perf_counter() - start) < timeout_s:
        if server.started and server.servers:
            sockets = server.servers[0].sockets
            if sockets:
                sockname = sockets[0].getsockname()
                host = sockname[0]
                port = sockname[1]
                bound_address = (host, port)
                break
        time.sleep(0.05)

    if bound_address is None:
        server.should_exit = True
        thread.join(timeout=1)
        raise RuntimeError("Timed out while waiting for test Tiled server to start.")

    try:
        yield f"http://{bound_address[0]}:{bound_address[1]}"  # noqa: E231
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("Tiled server thread did not stop cleanly.")


def test_tiled_source_reads_from_live_server(tiled_server: str) -> None:
    source = TiledSource(
        source_reference="live",
        resource_location=tiled_server,
        iosource_method_kwargs={"base_item_path": "entry"},
    )

    data = source.get_data("data")
    np.testing.assert_array_equal(data, np.arange(6).reshape(2, 3))

    attrs = source.get_data_attributes("data")
    assert attrs == {"units": "counts", "description": "synthetic"}

    assert source.get_static_metadata("data@units") == "counts"
    assert source.get_data_shape("data") == (2, 3)
    assert source.get_data_dtype("data") == np.dtype(int)
