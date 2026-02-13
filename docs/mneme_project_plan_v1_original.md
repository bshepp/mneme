**Project Title: Mneme**

**Purpose:**
Mneme is an exploratory research system designed to detect field-like, emergent memory structures embedded in biological systems, beginning with planarian regeneration and bioelectric data. It seeks to uncover attractor states, regulatory logic, and latent architectures not captured by sequence-based models alone.

---

**Guiding Premise:**
Biological systems, particularly those capable of regeneration, may encode memory not just genetically but as persistent spatial or field-based attractors. Mneme aims to identify, model, and interpret these attractor dynamics using machine learning, field theory, and topological methods.

---

**Phase 1: Synthetic Data Prototyping (Planarian Focus)**

**Objective:** Develop and validate a modular analysis pipeline on synthetic field-like data inspired by planarian voltage maps and regenerative logic.

**Tasks:**
1. Create synthetic 2D field data with noise and attractor behavior.
2. Apply Information Field Theory (IFT) reconstruction to interpolate continuous fields.
3. Run dimensionality reduction (PCA, autoencoders) to uncover latent spaces.
4. Apply Topological Data Analysis (TDA) to identify persistent structures.
5. Use Symbolic Regression to extract mathematical rules governing local field behaviors.
6. Validate internal coherence across methods.

**Tools:** Python, Jupyter/Colab, NumPy, SciPy, PyTorch/Keras, PySR, GUDHI (for TDA)

---

**Phase 2: Real Bioelectric + Gene Expression Data (Planarian)**

**Objective:** Test Mneme on actual biological data, starting with planarian bioelectric images, gene expression overlays, and regeneration timelines.

**Tasks:**
1. Collect and preprocess datasets from Levin Lab, SmedGD, and published literature.
2. Normalize spatial and temporal scales.
3. Reconstruct bioelectric and expression fields with IFT tools.
4. Embed expression snapshots into latent space (e.g., via VAE).
5. Detect attractors, loops, and bifurcations using TDA.
6. Run symbolic regression on voltage-expression transitions.
7. Correlate findings with known regeneration outcomes and perturbation experiments.

**Deliverables:**
- A reproducible Jupyter notebook pipeline
- Cleaned and annotated dataset repository
- Visualization module for field and attractor mapping

---

**Phase 3: Interpretation + Theory Development**

**Objective:** Formalize insights into a model of distributed memory encoding via fields in biological tissue.

**Tasks:**
1. Identify recurring attractor geometries and recovery behaviors.
2. Propose theoretical framework for field memory dynamics.
3. Compare results to Hopfield networks and morphogenetic models.
4. Draft whitepaper or publication.
5. Design follow-up experiments (e.g., perturbation predictions).

**Potential Extensions:**
- Cross-organism validation (e.g., zebrafish, axolotl)
- Inclusion of behavior/fear response patterning
- Field inheritance modeling across generations

---

**Core Values:**
- No prior overfitting to clean systems (Drosophila bias avoided)
- Sensitivity to emergent, nonlinear, and self-correcting behavior
- Respect for biological memory as distributed and dynamic

---

**Current Status:**
Project name chosen: Mneme
Project plan initialized
Synthetic data prototyping phase in progress
Real planarian datasets to be sourced and prepped

