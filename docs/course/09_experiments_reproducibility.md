# Module 9: Designing Experiments and Reproducibility

- Objectives
  - Use the `experiment` CLI to structure runs
  - Save configs, results, and plots reproducibly
- Time: 45–60 minutes

## 9.1 Run an experiment
```bash
mneme experiment planarian_demo -d data/synthetic/quickstart.npz -p bioelectric -b experiments
```
Creates a timestamped directory with config, results, and plots.

## 9.2 Configuration discipline
- Keep `config/default.yaml` under version control; override per-run via CLI or copying to an experiment folder

## 9.3 Exercises
1) Compare two experiments with different topology backends; write a 5-sentence summary
2) Change attractor thresholds; capture differences in a simple CSV of metrics (edit `analysis/metrics.py` if desired)

Solutions (outline)
- Record backend, diagram counts, feature vector lengths, and any attractor counts; summarize trade-offs
- Metrics scripts help quantify differences reproducibly