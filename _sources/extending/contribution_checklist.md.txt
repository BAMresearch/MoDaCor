# Contribution checklist

Use this checklist before opening or merging a MoDaCor change.

## Environment

- Work against Python 3.12 or newer.
- Install the local development extras you need:

```bash
pip install -e ".[tests,lint,docs]"
```

## Code and tests

- Run the focused tests for the area you changed.
- Run the full suite before merging:

```bash
python -m pytest -q
```

- Run linting for touched files:

```bash
python -m flake8 src/modacor tests
python -m isort --check-only --diff --filter-files src/modacor tests
```

## Documentation

- Update user-facing docs when CLI, runtime-service, pipeline behavior, or
  extension contracts change.
- If you add or change a public `ProcessStep`, regenerate the module reference
  pages:

```bash
python scripts/generate_module_doc.py --all --output-dir docs/reference/modules --index docs/reference/modules/index.md
```

- Rebuild the docs and confirm there are no warnings:

```bash
python -m sphinx -E -b html docs dist/docs
```

## Review points

- Keep `modacor.modules.__all__` aligned with the intended public step surface.
- Keep CLI and runtime-service paths aligned when they share request or IO
  behavior.
- Preserve docstrings and type hints when simplifying code.
- Avoid leaving unfinished pages or stale backlog state after structural
  refactors.
