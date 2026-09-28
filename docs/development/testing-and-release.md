# Testing and release

Create the development environment with `uv`:

```bash
uv venv --python 3.12 .venv-dev
source .venv-dev/bin/activate
uv pip install -e ".[tests,lint,docs,server]"
```

Prefer focused tests for changed behavior:

```bash
.venv-dev/bin/python -m pytest -q tests/path/to/test_file.py
```

Run broader environments when a shared contract changes:

```bash
tox -e lint
tox -e py312
tox -e docs
tox -e linkcheck
```

Documentation changes must pass the README QuickStart test, generated-module
freshness check, Sphinx warnings-as-errors build, and relevant link checks.

Version and changelog updates are produced by the release-preparation workflow
after changes reach `main`. Local version inspection must use
`semantic-release version --print`; do not create release tags from a feature
branch.
