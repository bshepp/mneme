# Mneme Course: From First Principles to Mastery

Welcome to the comprehensive, hands-on course for Mneme: a system for detecting field-like memory structures in biological systems. This program takes you from foundational theory to confident, practitioner-level use of Mneme’s CLI and Python APIs—with exercises, projects, and optional advanced modules.

- Audience: Scientists, ML/DS engineers, biophysicists, and curious generalists
- Prerequisites: Python fundamentals; basic linear algebra and probability; comfort with NumPy; curiosity about fields and topology
- Compute: CPU is sufficient for the MVP; GPU optional (PyTorch, heavy models)
- Duration: ~12–18 hours total (self-paced)

## Learning paths

- Foundations (Modules 1–4): First principles, environment, CLI, pipeline anatomy
- Practitioner (Modules 5–9): Reconstruction, topology, attractors, visualization, experiments
- Advanced (Modules 10–11): Performance/monitoring; optional symbolic regression (PySR)
- Capstone: End-to-end experiment with reporting

## Syllabus

1. [First Principles: Fields, Topology, Attractors](01_first_principles.md)
2. [Environment Setup and Sanity Checks](02_environment_setup.md)
3. [CLI Quickstart: Generate → Analyze → Visualize](03_cli_quickstart.md)
4. [Pipeline Anatomy and Configuration](04_pipeline_anatomy.md)
5. [Field Reconstruction (IFT and GP)](05_field_reconstruction.md)
6. [Topology (Cubical, Rips, Alpha) and Features](06_topology_tda.md)
7. [Attractor Detection (Recurrence, Lyapunov, Clustering)](07_attractor_detection.md)
8. [Visualization and Reporting](08_visualization_reporting.md)
9. [Designing Experiments and Reproducibility](09_experiments_reproducibility.md)
10. [Performance and Monitoring (MVP Tools)](10_performance_monitoring.md)
11. [Optional: Symbolic Regression with PySR](11_symbolic_regression.md)

## How to use this course

- Each module includes learning objectives, short readings, and exercises
- Exercises are designed to run in minutes on CPU
- Solutions are outlined inline after exercises (concealed by headings)

## Reference docs

- Core project docs: [Project Structure](../PROJECT_STRUCTURE.md), [Development Setup](../DEVELOPMENT_SETUP.md), [API Design](../API_DESIGN.md), [Data Pipeline](../DATA_PIPELINE.md)
- Source: `src/mneme/` (see `analysis/`, `core/`, `data/`, `utils/`)

Happy exploring!