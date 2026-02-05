# Documentation contributor guide

This folder contains the Sphinx documentation source for MoDaCor. The content is written in Markdown and rendered via
MyST, so you can keep editing `.md` files directly.

## Structure overview

- `index.md` – main landing page and toctree.
- `readme.md` – includes the project-level `README.md`.
- `getting_started/` – onboarding and quickstart material.
- `pipeline_operations/` – pipeline authoring and debugging guidance.
- `extending/` – module authoring and IO extension guidance.
- `examples/` – instrument-specific walkthroughs (MOUSE, SAXSess).
- `reference/` – auto-generated API docs.

## Build docs locally

From the repository root:

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
pip install --upgrade pip
pip install -r docs/requirements.txt

sphinx-build -E -b html docs dist/docs
```

Open `dist/docs/index.html` in a browser.

### Using tox

If you already use tox for this project, you can run:

```bash
tox -e docs
```

This uses the same Sphinx command and writes output to `dist/docs`.

## Markdown notes

- Use fenced code blocks and MyST directives where needed.
- `.rst` files are still supported, but Markdown is the preferred authoring format.
