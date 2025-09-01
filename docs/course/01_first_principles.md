# Module 1: First Principles — Fields, Topology, Attractors

- Objectives
  - Understand why biological memory can be field-like and distributed
  - Grasp core notions: Information Field Theory (IFT), persistent homology, attractors
  - Map concepts to Mneme’s MVP: reconstruction → topology → attractors
- Prereqs: None
- Time: 60–90 minutes

## 1.1 Why fields for biological memory?
Biological tissues exhibit spatially distributed bioelectric patterns. Hypothesis: some memory is encoded as stable, recoverable attractors in such fields, beyond sequence-only models.

Key ideas:
- A field f(x, y, t) over tissue; memory as stable structures/trajectories
- Perturbations relax back to morphology via attractors

## 1.2 Information Field Theory (IFT)
IFT reconstructs continuous fields from sparse/noisy observations with priors on smoothness/correlation. In Mneme:
- Reconstructors: IFT, Gaussian Process (GP), Neural Field (placeholder)
- Goal: obtain a continuous field suitable for topology + further analysis

## 1.3 Topological Data Analysis (TDA)
Persistent homology summarizes shape across scales. In Mneme:
- Cubical complex for 2D fields (default)
- Rips/Alpha for point-clouds extracted from fields (via adapters)
- Outputs: diagrams, features, derived images/landscapes

## 1.4 Attractors in dynamical systems
Attractors are sets toward which trajectories evolve (fixed point, limit cycle, strange). Mneme MVP:
- Recurrence-based detection (default)
- Basic Lyapunov and clustering modes (experimental MVP)

## 1.5 How Mneme ties these together
1) Preprocess + reconstruct continuous fields
2) Compute persistence features (structure across thresholds)
3) Detect attractors from dynamical trajectories when temporal data exist
4) Visualize + report

## Exercises
1) Concept check (short answers)
- Define “field-like memory” in one sentence
- What does persistent homology measure, at a high level?
- Name two attractor types and one biological reason they matter

2) Mini-derivation (paper & pencil)
- Suppose field noise is i.i.d. Gaussian. What prior/likelihood choices make GP inference natural? How does correlation length shape reconstructions?

3) Sanity coding (optional, Python)
- Build a tiny 2D array with two bright blobs. Threshold at multiple values and count connected components by hand; sketch how births/deaths would look.

Solutions (outline)
- Field-like memory: information stored as stable spatial patterns whose dynamics encode state
- PH measures the birth/death of k-dimensional features across thresholds; robustness of structure
- Fixed point: steady patterns; limit cycles: oscillations; important for regenerative stability
- GP prior with RBF kernel + Gaussian likelihood; longer correlation length → smoother reconstructions; smaller → finer detail
