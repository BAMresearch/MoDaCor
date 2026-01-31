# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "20/01/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_module_doc.py"


def _load_script_module():
    spec = spec_from_file_location("generate_module_doc", SCRIPT_PATH)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


generate_module_doc = _load_script_module()
build_markdown = generate_module_doc.build_markdown

# Use a well-known ProcessStep for fixtures
TARGET = "modacor.modules.base_modules.divide.Divide"


@pytest.fixture
def divide_docs():
    from modacor.modules.base_modules.divide import Divide

    return Divide, Divide.documentation


def test_build_markdown_contains_core_sections(divide_docs):
    step_cls, documentation = divide_docs

    md = build_markdown(step_cls, documentation)

    assert md.startswith("# ")
    for section in [
        "## Summary",
        "## Metadata",
        "## Required data keys",
        "## Modifies",
        "## Required arguments",
        "## Default configuration",
        "## Argument specification",
    ]:
        assert section in md

    assert "| Argument | Type | Required | Default | Description |" in md
    assert "| `divisor_source` | str | No | - | IoSources key for the divisor signal. |" in md
    assert "Divide" in md


@pytest.mark.parametrize("target", [TARGET])
def test_cli_writes_to_file(tmp_path: Path, monkeypatch, target: str):
    output_path = tmp_path / "module_doc.md"
    args = ["scripts/generate_module_doc.py", target, "-o", str(output_path)]

    completed = subprocess.run(["python3", *args], check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Divide" in content
    assert "| Argument | Type | Required | Default | Description |" in content
    assert completed.stdout == ""


def test_cli_stdout(monkeypatch):
    args = ["python3", "scripts/generate_module_doc.py", TARGET]
    completed = subprocess.run(args, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert completed.stdout.startswith("# Divide")
    assert "## Argument specification" in completed.stdout
    assert "| Argument | Type | Required | Default | Description |" in completed.stdout


def test_cli_error_on_missing_documentation(tmp_path: Path, monkeypatch):
    with pytest.raises(SystemExit):
        generate_module_doc._load_process_step("collections.Counter")
