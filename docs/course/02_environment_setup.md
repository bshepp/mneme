# Module 2: Environment Setup and Sanity Checks

- Objectives
  - Install/activate Mneme in a virtual environment
  - Verify core and optional dependencies
  - Run smoke tests on your machine
- Time: 30–45 minutes

## 2.1 Install and activate
```bash
# From repo root
source venv/bin/activate
pip install -e .
# Optional dev tools
pip install -r requirements-dev.txt
```

## 2.2 Verify toolchain
```bash
mneme info
```
Check:
- Python, NumPy, CUDA availability
- Optional deps: `gudhi`, `pysr`, `h5py`, `scipy`, `sklearn`
- Default topology backend (from config if provided)

## 2.3 Smoke tests
```bash
python -c "import mneme; print('OK')"
pytest -q  # optional dev
```

## 2.4 GPU optionality
- GPU not required for MVP. Neural-field reconstructor is a placeholder; keep CPU for now.

## Exercises
1) Change verbosity: run `mneme info -v` and note any differences in logging
2) Optional: install GUDHI if missing; confirm cubical backend will be used
3) Optional: install PySR; run `mneme info` and confirm Julia availability status

Solutions (outline)
- `mneme info` reports optional deps and default backend; with `-v`, console logging is verbose
- With GUDHI installed, cubical/Rips/Alpha backends are available
- PySR shows ✓; Julia may install lazily on first use