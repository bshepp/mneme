"""MkDocs hook: publish the study write-ups under ``studies/`` on the docs site.

MkDocs only serves files under ``docs/``. The studies keep their README (design)
and RESULTS (findings) next to their configs and numbers under ``studies/``, so
this hook adds those files, and the small data files the write-ups link to, to
the build at the same paths (``studies/<name>/RESULTS.md`` and so on). Relative
links between the studies therefore work unchanged. The two kinds of link that
leave ``studies/`` are rewritten: ``../../docs/X.md`` becomes ``../../X.md``,
and a repo-root file that is not on the site (``project_plan.md``) is linked on
GitHub.
"""

from __future__ import annotations

import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ROOT / "studies"
PAGE_NAMES = {"README.md", "RESULTS.md"}
DATA_SUFFIXES = {".json", ".csv"}
SKIP_DIRS = {"__pycache__"}
NO_DATA_DIRS = {"betse_configs"}


def _wanted(path: Path) -> bool:
    parts = set(path.relative_to(STUDIES).parts[:-1])
    if parts & SKIP_DIRS:
        return False
    if path.name in PAGE_NAMES:
        return True
    return path.suffix in DATA_SUFFIXES and not (parts & NO_DATA_DIRS)


def on_files(files, config):
    for path in sorted(STUDIES.rglob("*")):
        if path.is_file() and _wanted(path):
            rel = path.relative_to(ROOT).as_posix()
            files.append(
                File(rel, str(ROOT), config["site_dir"], config["use_directory_urls"])
            )
    return files


_ROOT_MD_LINK = re.compile(r"\]\(\.\./\.\./([A-Za-z0-9_\-]+\.md)\)")


def on_page_markdown(markdown, page, config, files):
    if not page.file.src_uri.startswith("studies/"):
        return markdown
    # Repo-root files first, so that a docs/ link rewritten below is not
    # mistaken for one afterwards.
    repo = config["repo_url"].rstrip("/")
    markdown = _ROOT_MD_LINK.sub(lambda m: f"]({repo}/blob/main/{m.group(1)})", markdown)
    return markdown.replace("](../../docs/", "](../../")
