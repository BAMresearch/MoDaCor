# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Jérome Kieffer"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports


"""Try to import all submodules from the installed modacor package tree."""

import importlib
from pathlib import Path

import modacor


def test_import_all():
    package_dir = Path(modacor.__file__).resolve().parent
    modules = []
    for py_file in package_dir.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue
        module_path = py_file.with_suffix("").relative_to(package_dir.parent)
        modules.append(".".join(module_path.parts))
    cnt = 0
    for i in modules:
        try:
            _ = importlib.import_module(i)
        except Exception as err:
            print(f"{type(err).__name__} in {i}: {err}.")
            cnt += 1
    assert cnt == 0, f"{cnt} submodules could not import properly"


if __name__ == "__main__":
    test_import_all()
