# -*- coding: utf-8 -*-
import subprocess
from os.path import abspath, dirname, join

import pydata_sphinx_theme
import toml

base_path = dirname(dirname(abspath(__file__)))
project_meta = toml.load(join(base_path, "pyproject.toml"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxext.rediraffe",
    "sphinxcontrib.mermaid",
]
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]
myst_heading_anchors = 3
inheritance_edge_attrs = dict(color="gray")  # readable in darkmode too
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_generate_overwrite = True
templates_path = ["_templates"]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
project = "MoDaCor"
year = "2025-2026"
author = (
    "Brian R. Pauw, Malte Storm, Jérôme Kieffer, Ingo Breßler, Anja Hörmann, Glen Smales, Armin Moser, and Tim Snow"
)
copyright = "{0}, {1}".format(year, author)
version = "1.11.0"
release = version
commit_id = None
try:
    commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("ascii")
except subprocess.CalledProcessError as e:
    print(e)


autodoc_mock_imports = [
    "ipykernel",
    "notebook",
    "pandas",
    "ipywidgets",
    "scipy",
    "h5py",
    "pint",
    "sasmodels",
    "chempy",
    "graphviz",
    "mcsas3",
]

pygments_style = "trac"
extlinks = {
    "issue": (join(project_meta["project"]["urls"]["repository"], "issues", "%s"), "#%s"),
    "pr": (join(project_meta["project"]["urls"]["repository"], "pull", "%s"), "PR #%s"),
}
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": project_meta["project"]["urls"]["repository"],
            "icon": "fa-brands fa-github",
        },
    ],
}

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
if commit_id:
    html_last_updated_fmt += f" (git {commit_id})"
html_split_index = False
html_sidebars = {
    "**": [
        "searchbox.html",
        "globaltoc.html",
        "sourcelink.html",
    ],
}
html_short_title = "%s-%s" % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

linkcheck_ignore = [
    join(
        project_meta["project"]["urls"]["documentation"],
        project_meta["tool"]["coverage"]["report"]["path"],
    )
    + r".*",
    # attempted fix of '406 Client Error: Not Acceptable for url'
    # https://github.com/sphinx-doc/sphinx/issues/1331
    join(project_meta["project"]["urls"]["repository"], "commit", r"[0-9a-fA-F]+"),
    # Generated module pages contain one source link per public step. Checking
    # those in a single run quickly trips GitHub's anonymous rate limits; the
    # generator tests verify that every corresponding local source path exists.
    r"https://github\.com/BAMresearch/MoDaCor/blob/main/src/.*",
    # DOI resolvers and the IUCr journal site intermittently reject automated
    # HEAD/GET requests even though these stable literature links work in a
    # browser. Keep them in the prose but exclude them from linkcheck.
    r"https://doi\.org/.*",
    r"https://journals\.iucr\.org/.*",
    # A historical changelog entry mentions ``license.py``; MyST linkification
    # interprets it as this non-existent host when the changelog is included.
    r"http://license\.py/?",
]
linkcheck_anchors_ignore_for_url = [
    r"https://pypi\.org/project/[^/]+",
]

# Preserve published page URLs while the documentation is organized by reader
# journey. Keys and values are Sphinx document names, without ``.html``.
rediraffe_redirects = {
    "installation": "getting-started/installation",
    "getting_started/index": "getting-started/index",
    "getting_started/quickstart": "getting-started/quickstart",
    "getting_started/cli_and_runner": "processing/local-execution",
    "pipeline_operations/index": "processing/index",
    "pipeline_operations/pipeline_basics": "processing/pipeline-graphs",
    "pipeline_operations/configuration_reference": "reference/pipeline-schema",
    "pipeline_operations/tracing_and_debugging": "processing/tracing-and-provenance",
    "pipeline_operations/server_installation": "server/installation-and-deployment",
    "pipeline_operations/advanced_server_use": "server/custom-steps-and-io",
    "pipeline_operations/runtime_service_api": "reference/server-api",
    "pipeline_operations/backlog": "development/design/pipeline-operations-backlog",
    "corrections/index": "modules/scattering-corrections",
    "corrections/capillary_self_absorption": "modules/capillary-self-absorption",
    "examples/mouse_pipeline": "examples/index",
    "examples/saxsess_pipeline": "examples/index",
    "examples/dls_i22": "examples/index",
    "extending/index": "development/index",
    "extending/module_author_guide": "development/module-author-guide",
    "extending/io_source_sink_guide": "development/io-source-sink-guide",
    "extending/contribution_checklist": "development/contribution-checklist",
    "readme": "development/documentation-guide",
    "usage": "processing/local-execution",
    "contributing": "development/contributing",
    "authors": "project/authors",
    "changelog": "project/changelog",
    "design/index": "development/design/index",
    "design/capillary-self-absorption": "development/design/capillary-self-absorption",
    "design/chunked-beamline-validation": "development/design/chunked-beamline-validation",
    "design/chunked-operation": "development/design/chunked-operation",
    "design/chunked-sink-implementation-plan": "development/design/chunked-sink-implementation-plan",
    "design/documentation-architecture": "development/design/documentation-architecture",
    "design/documentation-refactor-implementation-plan": (
        "development/design/documentation-refactor-implementation-plan"
    ),
    "design/external-parallel-runner": "development/design/external-parallel-runner",
    "design/pixel-unit-removal": "development/design/pixel-unit-removal",
    "design/tiled-io-upgrade": "development/design/tiled-io-upgrade",
    "design/completed/index": "development/design/completed/index",
    "design/completed/api-buffer-source-sink": "development/design/completed/api-buffer-source-sink",
    "design/completed/architecture-upgrade-plan": "development/design/completed/architecture-upgrade-plan",
    "design/completed/chunked-processing-handoff": "development/design/completed/chunked-processing-handoff",
    "design/completed/code-coherence": "development/design/completed/code-coherence",
    "design/completed/io-sink-runtime-api": "development/design/completed/io-sink-runtime-api",
    "design/completed/reduce-dimensionality-uncertainty-estimators": (
        "development/design/completed/reduce-dimensionality-uncertainty-estimators"
    ),
}
