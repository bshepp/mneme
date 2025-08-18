### Repository audit (2025-08-18)

- **CI/CD**
  - Tests: Now Ubuntu + Python 3.11/3.12, lint passes; mypy temporarily non-blocking; smoke test present.
  - Docs: Markdown bundled and deployed to `gh-pages` with proper permissions.
  - Recommendation: Re-enable strict mypy after type hygiene pass; add real tests to replace smoke.

- **Packaging/metadata**
  - `setup.py` fixed URLs and console scripts; `pyproject.toml` still declared Python >=3.8 and pointed scripts to missing `mneme.scripts.*`.
  - Action: Update `pyproject.toml` to Python >=3.12, correct URLs, and map console scripts to `mneme.cli:main` placeholders.

- **APIs vs code**
  - CLI references `create_bioelectric_pipeline` (missing). Action: Add stub using current config schema.
  - `models/` docs mention `autoencoders.py`, `symbolic.py`, `field_models.py`; only `__init__.py` exists. Action: Add minimal `autoencoders.py` and `symbolic.py` placeholders exporting documented classes.

- **Types/static analysis**
  - mypy reports many errors (Pydantic types used as annotations, missing return types, unions). Also `python_version` set to 3.8.
  - Action: Set mypy `python_version` to 3.12; add `types-PyYAML` to dev deps; later, add annotations/ignores as needed.

- **Docs vs reality**
  - README fixed to `import mneme` and Python 3.12. Markdown docs match structure; notebooks/tests listed in docs exceed current repo. With placeholders added, structure aligns better.

- **NotImplemented/TODOs**
  - Core algorithms in `core/topology.py` and `core/attractors.py` include `NotImplementedError` and TODOs (Rips/Alpha persistence, Lyapunov, clustering, basins). Acceptable for roadmap; ensure API surfaces communicate experimental status.

- **Next steps (implemented now)**
  - Patch `pyproject.toml` (Python 3.12, URLs, scripts, mypy 3.12).
  - Add `create_bioelectric_pipeline` stub.
  - Add `types-PyYAML` to `requirements-dev.txt`.
  - Add `models/autoencoders.py` and `models/symbolic.py` placeholders and update `models/__init__.py`.

- **Future improvements**
  - Add concrete loaders and tests; flesh out topology/attractor features; add docs page summarizing known placeholders and roadmap.
