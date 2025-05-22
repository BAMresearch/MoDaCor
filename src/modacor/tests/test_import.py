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
            print(f"{type(err)} in {j}: {err}.")
            cnt += 1
    assert cnt == 0


if __name__ == "__main__":
    test_import_all()
