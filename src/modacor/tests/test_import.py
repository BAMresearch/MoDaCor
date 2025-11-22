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


"""try to import all sub-modules from the project"""

import importlib
import os

dirname = os.path.dirname


def test_import_all():
    project_dir = dirname(dirname(os.path.abspath(__file__)))
    start = len(project_dir) - len("modacor")
    modules = []
    for path, dirs, files in os.walk(project_dir):
        for f in files:
            if f.endswith(".py") and not f.startswith("__"):
                modules.append(os.path.join(path[start:], f[:-3]))
    cnt = 0
    for i in modules:
        j = i.replace(os.sep, ".")
        try:
            _ = importlib.import_module(j)
        except Exception as err:
            print(f"{type(err).__name__} in {j}: {err}.")
            cnt += 1
    assert cnt == 0, f"{cnt} submodules could not import properly"


if __name__ == "__main__":
    test_import_all()
