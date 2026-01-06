# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "30/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import sys
import types
from typing import List, Tuple

import pytest

from modacor.io.io_sources import IoSource, IoSources
from modacor.modules.base_modules.append_source import AppendSource


def _install_dummy_iosource_module(monkeypatch) -> Tuple[str, List[tuple]]:
    """
    Install a dummy loader module into sys.modules and return:

    - the fully qualified loader path string
    - a list that will collect call arguments for assertions
    """
    calls: List[tuple] = []

    module_name = "modacor.tests.dummy_iosource_module"
    mod = types.ModuleType(module_name)

    class DummySource(IoSource):
        """
        Minimal IoSource subclass used only for testing.

        We deliberately override __init__ and *do not* call super().__init__
        to avoid depending on IoSource's real constructor signature.
        We just provide the attributes IoSources.register_source is expected
        to use (source_reference, resource_location).
        """

        def __init__(self, ref: str, loc: str) -> None:
            self.source_reference = ref
            self.resource_location = loc

    def dummy_loader(*, source_reference: str, resource_location: str, **kwargs) -> IoSource:
        """
        Minimal loader that mimics the real loader signature used by AppendSource.
        Returns a DummySource instance and records calls for assertions.
        """
        calls.append((source_reference, resource_location))
        return DummySource(source_reference, resource_location)

    # The attribute name here is what AppendSource._resolve_iosource_callable
    # will try to retrieve from the module.
    mod.DummyLoader = dummy_loader  # type: ignore[attr-defined]

    # Inject into sys.modules so import_module() can find it
    monkeypatch.setitem(sys.modules, module_name, mod)

    loader_path = f"{module_name}.DummyLoader"
    return loader_path, calls


def _make_append_source_instance() -> AppendSource:
    """
    Create an AppendSource instance without going through ProcessStep.__init__.

    We only need 'configuration' and 'io_sources' for these tests, so __new__
    is sufficient and avoids coupling to ProcessStep's constructor signature.
    """
    instance = AppendSource.__new__(AppendSource)
    return instance


def test_append_single_source(monkeypatch):
    loader_path, calls = _install_dummy_iosource_module(monkeypatch)

    step = _make_append_source_instance()
    step.configuration = {
        "source_identifier": "sample_source",
        "source_location": "/tmp/sample.dat",
        "iosource_module": loader_path,
    }
    step.io_sources = IoSources()

    result = step.calculate()

    # No databundles modified
    assert result == {}

    # The new source should be present
    assert "sample_source" in step.io_sources.defined_sources

    # Loader should be called exactly once with the expected args
    assert calls == [("sample_source", "/tmp/sample.dat")]


def test_append_multiple_sources(monkeypatch):
    loader_path, calls = _install_dummy_iosource_module(monkeypatch)

    step = _make_append_source_instance()
    step.configuration = {
        "source_identifier": ["src1", "src2"],
        "source_location": ["/tmp/file1.dat", "/tmp/file2.dat"],
        "iosource_module": loader_path,
    }
    step.io_sources = IoSources()

    result = step.calculate()

    # Still no databundles modified
    assert result == {}

    # Both sources should be present
    assert "src1" in step.io_sources.defined_sources
    assert "src2" in step.io_sources.defined_sources

    # Loader should have been called twice in order
    assert calls == [
        ("src1", "/tmp/file1.dat"),
        ("src2", "/tmp/file2.dat"),
    ]


def test_mismatched_source_lengths_raises(monkeypatch):
    loader_path, _ = _install_dummy_iosource_module(monkeypatch)

    step = _make_append_source_instance()
    step.configuration = {
        "source_identifier": ["src1", "src2"],
        "source_location": ["/tmp/only_one.dat"],
        "iosource_module": loader_path,
    }
    step.io_sources = IoSources()

    with pytest.raises(ValueError, match="counts must match"):
        step.calculate()


def test_existing_source_is_not_overwritten(monkeypatch):
    loader_path, calls = _install_dummy_iosource_module(monkeypatch)

    step = _make_append_source_instance()
    step.io_sources = IoSources()

    class PreExistingSource(IoSource):
        """
        Pre-registered IoSource subclass for testing overwrite behaviour.
        """

        def __init__(self, ref: str, loc: str) -> None:
            self.source_reference = ref
            self.resource_location = loc

    # Pre-register a source with identifier "existing"
    step.io_sources.register_source(PreExistingSource(ref="existing", loc="/tmp/original.dat"))

    # Configuration attempts to append a source with the same identifier
    step.configuration = {
        "source_identifier": "existing",
        "source_location": "/tmp/new_location.dat",
        "iosource_module": loader_path,
    }

    result = step.calculate()

    # No databundles modified
    assert result == {}

    # The identifier should still be present…
    assert "existing" in step.io_sources.defined_sources

    # …but the loader should not have been called at all,
    # since AppendSource checks defined_sources before appending.
    assert calls == []
