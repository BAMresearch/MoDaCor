"""Enable dependency-contract checking for tests, excluding source doctests."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def contract_trace_patches():
    # Delay package imports until collection finishes: tox tests an installed
    # wheel while also collecting src doctests, which must establish imports first.
    from tests._contract_trace import TRACE_ENABLED, install_contract_trace

    with pytest.MonkeyPatch.context() as patch:
        install_contract_trace(patch)
        yield TRACE_ENABLED


@pytest.fixture(autouse=True)
def contract_trace_enabled(request, contract_trace_patches):
    token = contract_trace_patches.set(request.node.get_closest_marker("no_contract_trace") is None)
    try:
        yield
    finally:
        contract_trace_patches.reset(token)
