# Mneme Project Plan v2

**Last updated:** 2026-02-12
**Previous version:** [docs/mneme_project_plan_v1_original.md](docs/mneme_project_plan_v1_original.md)

---

## Purpose

Mneme is an exploratory research system designed to detect field-like, emergent memory structures embedded in biological systems. Starting with planarian regeneration and bioelectric data, it seeks to uncover attractor states, regulatory logic, and latent architectures not captured by sequence-based models alone.

## Guiding Premise

Biological systems, particularly those capable of regeneration, may encode memory not just genetically but as persistent spatial or field-based attractors. Mneme aims to identify, model, and interpret these attractor dynamics using machine learning, field theory, and topological methods.

## Core Values

- No prior overfitting to clean systems (Drosophila bias avoided)
- Sensitivity to emergent, nonlinear, and self-correcting behavior
- Respect for biological memory as distributed and dynamic
- Simulation-first: generate rich data through computational biology before requiring wet-lab access
- Interdisciplinary: bridge topology, dynamical systems, and bioelectricity

---

## Phase 1: Synthetic Data Prototyping -- COMPLETE

**Objective:** Build and validate a modular analysis pipeline on synthetic field data inspired by planarian voltage maps and regenerative logic.

| Task | Status | Implementation |
|------|--------|---------------|
| Synthetic 2D field data with noise and attractors | Done | `SyntheticFieldGenerator`, `generate_planarian_bioelectric_sequence()` |
| IFT / field reconstruction | Done | `SparseGPReconstructor` (default, O(nm^2)), `DenseIFTReconstructor`, Neural Fields |
| Dimensionality reduction / latent spaces | Done | Convolutional VAE (`FieldAutoencoder`) with training, encoding, interpolation |
| TDA for persistent structures | Done | Full GUDHI integration: cubical, Rips, Alpha complexes; persistence diagrams/landscapes/images |
| Symbolic regression for field rules | Done | PySR integration with `discover_field_dynamics()` for automatic PDE discovery |
| Validate internal coherence | Partial | Pipeline runs end-to-end; formal cross-method validation report still needed |

**Key validation result:** PhysioNet ECG heart rate variability analysis yielded lambda_1 = +0.12/s, D_KY = 2.35, predictability horizon ~8s -- matching published literature on cardiac chaos.

---

## Phase 2: Real Bioelectric Data -- ACTIVE

**Objective:** Apply Mneme to real and simulated biological data, starting with BETSE bioelectric tissue simulations from published Levin Lab papers, then expanding to self-collected ECG data and experimental datasets.

### 2A. BETSE Simulation Data (Current Focus)

BETSE (BioElectric Tissue Simulation Engine) is a 2D bioelectric tissue simulator used by Levin Lab. It produces spatiotemporal voltage fields, ion concentrations, and current densities -- exactly the kind of data Mneme is built to analyze.

**Data pipeline:** BETSE CSV exports --> `betse_loader.py` (interpolation to regular grid) --> Mneme `Field` objects --> full analysis pipeline

| Task | Status | Notes |
|------|--------|-------|
| Install and validate BETSE | Done | Ran `betse try` locally; confirmed CSV output format |
| Build BETSE data loader | Done | `betse_loader.py`: handles irregular cell data, multi-frame time series, metadata |
| Build BETSE analysis script | Done | `scripts/analyze_betse.py`: topology, attractors, recurrence, Lyapunov |
| Run paper simulation configs on AWS | In progress | c6i.xlarge spot instance running 5 configs in parallel (est. 8-14h) |
| Analyze attractor configs (2016 Frontiers) | Pending | `attractors_2016_1.yaml`, `attractors_2016_2.yaml`, `BETSE_test_sim_3.yaml` |
| Analyze pattern configs (2018 PBMB) | Pending | `patterns_2018.yaml` |
| Analyze physiology configs (2018 PBMB) | Pending | `physiology_2018.yaml` |
| Cross-simulation comparison | Pending | Compare topological signatures, attractor types, and Lyapunov spectra across configs |
| Parameter sweep experiments | Not started | Vary gap junction conductance, ion channel expression; track attractor bifurcations |

**AWS compute details:**
- Instance: c6i.xlarge (4 vCPU, 8 GB), spot pricing ~$0.067/hr
- All 5 paper configs running in parallel
- Results auto-packaged as `betse-results.tar.gz`
- Estimated cost: $0.50-$1.00

### 2B. Self-Collected ECG Data (Next)

Hardware: 2x AD8232 Single-Lead ECG sensor modules (acquired).

| Task | Status | Notes |
|------|--------|-------|
| Design ECG recording protocol | Not started | Duration, sampling rate, electrode placement, activity states |
| Build Arduino/RPi data acquisition script | Not started | Serial read from AD8232, timestamp, save to CSV |
| Build ECG data loader for Mneme | Not started | Adapt PhysioNet loader patterns to raw AD8232 output |
| Baseline recordings (resting, breathing patterns) | Not started | |
| Compare self-data to PhysioNet HRV analysis | Not started | Validate against known lambda_1 ~ +0.12/s result |
| Longitudinal tracking | Not started | Daily recordings to detect attractor drift over time |

### 2C. Published Experimental Data (Parallel Track)

| Source | Data Type | Status |
|--------|-----------|--------|
| PhysioNet MIT-BIH | ECG / HRV time series | Done -- validated Lyapunov pipeline |
| Planform Database | Planarian phenotype images | Not started |
| ModelDB | Computational neuroscience models | Not started |
| Figshare / Dryad | Supplementary data from Levin Lab papers | Not started |
| SmedGD | Planarian gene expression | Not started |

### Phase 2 Deliverables

- [ ] Reproducible analysis of all 5 BETSE paper simulation configs
- [ ] Topological comparison across simulation conditions (attractors vs patterns vs physiology)
- [ ] Self-collected ECG data with Lyapunov/attractor analysis
- [ ] Cross-method validation report: do topology, Lyapunov, recurrence, and VAE latent space tell consistent stories?
- [ ] Cleaned dataset repository with provenance metadata
- [ ] Per-paper validation document (see Phase 2.5 below)

---

## Phase 2.5: JOSS Publication (Software Paper) -- NOT STARTED

**Objective:** Publish Mneme as a peer-reviewed research tool in the [Journal of Open Source Software](https://joss.theoj.org/). JOSS reviews the *software itself* -- code quality, tests, documentation, and demonstrated utility. This is distinct from the Phase 3 whitepaper, which is about the *scientific findings*.

**Why JOSS, why now:**
- JOSS is specifically for research software. The review process evaluates your code, not just your claims.
- A DOI makes the tool citable. Labs won't adopt software they can't cite.
- Peer review of the software catches bugs and design issues before you build Phase 3 on top of it.
- It's the credibility bridge between "GitHub repo" and "research tool" -- and the thing that makes outreach to Levin Lab or other groups a professional conversation rather than a cold pitch.

**JOSS requirements vs current status:**

| Requirement | Status | Gap |
|-------------|--------|-----|
| Open source repo | Done | None |
| Documentation sufficient for new users | Mostly done | API reference docs not yet generated |
| Tests that can be run | Done (~80 tests) | Push coverage from 40% toward 70%+ |
| Community guidelines (CONTRIBUTING, CODE_OF_CONDUCT) | Not done | Straightforward to add |
| Statement of need (why this software matters) | Not written | Draft from validation results |
| Example usage | Partial | Scripts exist; need a clean tutorial/notebook |
| Validation on real data | In progress | BETSE runs + PhysioNet = multi-system validation |

**Tasks:**

| Task | Status | Notes |
|------|--------|-------|
| Write per-paper validation docs | Not started | For each BETSE config: paper citation, their values, your values, percent error, caveats |
| Push test coverage to 70%+ | Not started | Focus on BETSE loader, pipeline orchestration, edge cases |
| Generate API reference docs | Not started | Sphinx or mkdocs from existing docstrings |
| Add CONTRIBUTING.md and CODE_OF_CONDUCT.md | Not started | Standard templates, adapted |
| Create a clean Jupyter notebook walkthrough | Not started | End-to-end: load data, reconstruct field, analyze, visualize |
| Draft JOSS paper (1-2 pages) | Not started | Statement of need, summary, validation highlights, references |
| Submit to JOSS | Not started | After all above; expect 1-3 month review cycle |
| Respond to reviewer feedback | Not started | JOSS reviewers file GitHub issues; fix and respond |

**JOSS paper outline (short -- typically 1,000 words max):**

1. **Summary** -- What Mneme does in two sentences
2. **Statement of need** -- No existing toolkit combines field reconstruction, Lyapunov analysis, TDA, and symbolic regression for bioelectric data. Researchers using BETSE or similar simulators currently write ad-hoc analysis scripts.
3. **Key features** -- Sparse GP reconstruction, Wolf algorithm Lyapunov spectrum, GUDHI TDA integration, PySR equation discovery, BETSE data loader
4. **Validation** -- PhysioNet ECG (lambda_1 matches literature), BETSE paper configs (N papers, quantitative comparison)
5. **References** -- BETSE, GUDHI, PySR, Wolf algorithm, PhysioNet, the 4 papers you validated against

**Post-publication outreach (after JOSS acceptance or submission):**

- Contact Levin Lab with: "I built a validated analysis toolkit for BETSE output; here's the JOSS paper; would any of your students/postdocs find it useful?"
- Post in BETSE GitHub discussions
- Cross-post to relevant communities (computational biology, bioelectric signaling)
- The DOI and peer review convert "random person's GitHub repo" into "published research software"

**Compute-for-collaboration (ongoing after JOSS):**

Troll arXiv and journals for papers with open computational questions Mneme can answer. Target:
- Papers citing BETSE/PLIMBO that lack topology or attractor analysis
- Bioelectric patterning studies that describe "stability" or "robustness" qualitatively but don't quantify (Lyapunov spectrum answers this)
- Regeneration papers with spatiotemporal voltage data and uncharacterized pattern memory
- Gap junction perturbation studies observing bistability without attractor landscape characterization

The pitch: "I ran your published simulation configs through a peer-reviewed analysis toolkit. Here are the topological signatures and Lyapunov spectra. Want to co-author a follow-up?" This converts compute time into co-authorships and builds the collaborative network needed for Phase 3 theory work.

---

## Phase 3: Interpretation + Theory Development -- NOT STARTED

**Objective:** Formalize insights into a model of distributed memory encoding via fields in biological tissue.

| Task | Status | Dependencies |
|------|--------|-------------|
| Identify recurring attractor geometries across datasets | Not started | Phase 2 results |
| Map topological signatures to biological function | Not started | BETSE analysis + literature review |
| Compare field attractors to Hopfield networks | Not started | Phase 2 cross-method validation |
| Compare to morphogenetic field models (Turing, positional information) | Not started | |
| Propose theoretical framework for field memory dynamics | Not started | All above |
| Draft whitepaper or preprint | Not started | Framework + results |
| Design follow-up experiments (perturbation predictions) | Not started | Framework |

**Potential extensions:**
- Cross-organism validation (zebrafish, axolotl bioelectric data if available)
- Inclusion of behavior/fear response patterning in planaria
- Field inheritance modeling across generations
- Collaboration with experimental labs for perturbation testing

---

## Infrastructure

| Component | Status | Details |
|-----------|--------|---------|
| Python package (`pyproject.toml`) | Done | PEP 621, editable install, optional deps for TDA/PySR/GPU |
| CLI | Done | `mneme analyze`, `mneme info` via Click |
| Test suite | Done | pytest with unit + integration tests, coverage reporting |
| CI/CD | Done | GitHub Actions: pytest, mypy, flake8, coverage |
| Code quality | Done | Black, isort, flake8, mypy, pre-commit hooks |
| Documentation | Partial | README, CLAUDE.md, CHANGELOG, course modules; API docs not yet generated |
| Cloud compute | Active | AWS EC2 spot instances for BETSE simulations |
| Data loaders | Active | BETSE loader done; ECG loader needs work; PhysioNet validated |

---

## Current Priorities (February 2026)

1. **Retrieve AWS simulation results** -- Pull `betse-results.tar.gz` when the 5 configs finish, terminate instance
2. **Run full Mneme analysis on each config** -- Topology, Lyapunov spectrum, recurrence, VAE encoding
3. **Cross-simulation comparison** -- Do different biological conditions produce distinguishable topological signatures?
4. **ECG data acquisition setup** -- Get the AD8232 sensors producing clean data
5. **Cross-method validation** -- Formal report on whether topology, Lyapunov, recurrence, and latent space analysis agree

---

## Open Questions

These are the scientific questions driving the current work:

1. **Do BETSE attractor simulations produce topologically distinct persistence diagrams compared to pattern/physiology simulations?** If the topology of the voltage field differs between conditions designed to show attractor behavior vs. pattern formation, that's evidence that TDA can distinguish biologically meaningful states.

2. **Can we discover governing PDEs from BETSE output using symbolic regression?** If `discover_field_dynamics()` recovers equations resembling known bioelectric models (gap junction diffusion, Nernst potentials), that validates the approach for use on experimental data where the equations are unknown.

3. **Do VAE latent spaces organize biologically meaningful states?** When we encode BETSE voltage field time series, do nearby points in latent space correspond to similar biological conditions? Do trajectories through latent space reflect known developmental/regenerative paths?

4. **Is cardiac HRV chaos (from ECG) topologically similar to bioelectric tissue chaos (from BETSE)?** Both are bioelectric systems. If their attractor geometries share topological features, that suggests something universal about bioelectric memory encoding.

5. **Can we predict perturbation outcomes?** Given a trained model of the attractor landscape, can we predict what happens when a gap junction blocker is applied (simulated in BETSE)?

---

## Timeline (Rough)

| Period | Focus |
|--------|-------|
| Feb 2026 | Retrieve and analyze BETSE simulation results; write per-paper validation docs; begin ECG setup |
| Mar 2026 | Cross-simulation comparison report; first self-ECG recordings; push test coverage to 70%+ |
| Apr 2026 | Parameter sweep experiments in BETSE; ECG longitudinal tracking; generate API docs; create tutorial notebook |
| May 2026 | Cross-method validation report; draft JOSS paper; add CONTRIBUTING/CODE_OF_CONDUCT |
| Jun 2026 | Submit JOSS paper; begin Phase 3 interpretation while review is pending |
| Jul-Aug 2026 | Respond to JOSS reviewer feedback; theory development; outreach to Levin Lab / BETSE community |
| Sep-Oct 2026 | JOSS publication (expected); draft science whitepaper |

**Note:** The JOSS paper (Phase 2.5) and the science whitepaper (Phase 3) are different documents with different audiences. JOSS says "here is a tool, it works, it's tested." The whitepaper says "here is what we found using the tool." JOSS should go first -- it establishes the tool's credibility before you make claims with it.
