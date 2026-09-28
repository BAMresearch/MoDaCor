# Documentation contributor guide

The Sphinx source uses MyST Markdown. Documentation is organized by reader
question, not package directory:

- `introduction/`: purpose, scope, and principles;
- `getting-started/`: installation and first run;
- `data-model/`: scientific container semantics;
- `processing/`: pipeline concepts and local operation;
- `modules/`: curated workflow/science guides;
- `server/`: client-server operation;
- `examples/`: small examples and the external instrument catalogue;
- `reference/`: exact generated and hand-written reference;
- `development/`: contribution guidance and design records; and
- `project/`: changelog, authors, citation, and license.

## Authoring rules

- Put tutorials, explanations, how-to guidance, and exact reference in their
  respective sections rather than combining them into one long page.
- Link to generated module pages instead of copying configuration tables.
- Keep instrument notebooks, pipeline YAML, manifests, and data in
  MoDaCor-examples.
- Use repository-relative links and stable Sphinx labels; never publish absolute
  workstation paths.
- Add an old-to-new Rediraffe entry when moving a published page.
- Update the navigation landing page and “where next” links when adding a new
  section.

## Build locally

```bash
uv venv --python 3.12 .venv-docs
source .venv-docs/bin/activate
uv pip install -e ".[docs]"
python scripts/generate_module_doc.py \
  --all --output-dir docs/reference/modules \
  --index docs/reference/modules/index.md --check
sphinx-build -E -W --keep-going -b html docs dist/docs
```

Or run `tox -e docs`. Use `tox -e linkcheck` for external and internal link
validation.

When changing public steps, regenerate the checked-in module pages without
`--check`, review the diff, then run the check form.
