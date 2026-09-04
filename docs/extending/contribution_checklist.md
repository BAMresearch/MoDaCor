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
- Use `docs:` as the commit prefix for documentation-only changes that should
  be classified as documentation updates in generated release notes.
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

## Release notes

- Do not manually edit `CHANGELOG.md` for ordinary PRs. The release-preparation
  workflow generates a dedicated release PR after changes reach `main`.
- Use semantic commit prefixes that match the project configuration:
  `fix:`/`perf:` for patch releases, `enh:`/`feat:` for minor releases, and
  `docs:` for documentation entries.
- Review and merge the generated release PR before expecting a new version tag
  or PyPI publication.
