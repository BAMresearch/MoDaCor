# Contribution checklist

Use this checklist before opening or merging a MoDaCor change.

## Environment

- Work against Python 3.12 or newer.
- Install the local development extras you need:

```bash
uv pip install -e ".[tests,lint,docs]"
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
python scripts/generate_module_doc.py --all --output-dir docs/reference/modules --index docs/reference/modules/index.md --check
```

- Rebuild the docs and confirm there are no warnings:

```bash
python -m sphinx -E -W --keep-going -b html docs dist/docs
```

## Review points

- Keep `modacor.modules.__all__` aligned with the intended public step surface.
- Keep package dependencies one-way: modules may use models, geometry, and IO;
  models may use geometry; geometry must remain format- and pipeline-independent.
- Promote reusable mathematics out of `ProcessStep` methods, and do not import
  private helpers from one process-step module into another.
- Keep non-step pipeline adapters under `modacor.modules.helpers`; keep retired
  steps under `deprecated` and out of `modacor.modules.__all__`.
- Keep NeXus/HDF5 traversal in `modacor.io`, even when the values it resolves
  ultimately describe geometry.
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
- **Versioning policy until v2.0:** MoDaCor's pipeline interface is still
  settling, so incompatible changes to step configuration or pipeline YAML
  are released as **minor** versions, not major ones. Each such change must
  add a section to
  [`docs/processing/interface-migrations.md`](../processing/interface-migrations.md)
  telling users exactly what to change, and the PR must link to it. Use
  `feat:` (or `enh:`) as the prefix. Do **not** use `type!:` or a
  `BREAKING CHANGE:` footer: python-semantic-release turns either into a
  major release. v2.0 is reserved for a deliberate milestone at which the
  pipeline interface is declared stable; strict semantic versioning applies
  from then on.
- Squash-merge PRs with a conventional title (it decides the release bump).
  Before confirming the squash, replace GitHub's default body, which
  concatenates every commit message, with a short summary. This keeps stray
  breaking-change footers from branch commits out of `main`.
- Review and merge the generated release PR before expecting a new version tag
  or PyPI publication.
