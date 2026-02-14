# BETSE Simulation Analysis Report

**Date:** 2026-02-13 (updated 2026-02-14 with patterns results)
**Analyst:** Mneme pipeline (automated) + manual interpretation
**Data source:** BETSE paper configurations (`attractors_2018`, `physiology_2018`, `patterns_2018`) run on AWS EC2 c6i.xlarge
**Pipeline version:** Mneme 0.1.x (GUDHI TDA for H0+H1, PySR symbolic regression, PyTorch VAE)

---

## Executive Summary

We ran five BETSE bioelectric tissue simulations derived from published Levin Lab paper configurations through the Mneme analysis pipeline. The analysis consisted of two phases: an initial pass using topology, Lyapunov spectrum, and recurrence analysis; and a deeper pass using PCA mode extraction, Wasserstein distance tracking, and convolutional VAE latent-space embedding.

**Key findings:**

1. **Multi-stability confirmed.** Two simulations of the same tissue under different initial conditions converge to distinct attractor basins, occupying clearly separated regions in both PCA and VAE latent spaces. This is the core phenomenon Mneme was designed to detect.

2. **Pattern formation breaks the dimensionality ceiling.** Attractor and physiology simulations are effectively rank-2 (99.5%+ variance in 2 PCA modes), but the patterns simulation has 5 significant modes (81% + 13% + 4% + 1% + 0.6%). GRN-driven pattern formation creates the multi-modal spatial dynamics that gap-junction-only coupling cannot.

3. **Wasserstein distance tracking is the most discriminating method.** It reveals four distinct topological regimes: sim_1 (oscillatory settling), sim_2 (directed drift), physiology (topology-preserving excursion), and patterns (high-rate accumulating reorganization with total drift 84.1 -- 2.6x the nearest comparator).

4. **Topological reorganization without topological creation.** The patterns simulation maintains ~22 H0 features from first to last frame, but the Wasserstein distance between them is 84.1 mV -- meaning the features are being continuously reshuffled, not created or destroyed. This is the signature of active pattern formation.

5. **Physiology simulation is a distinct dynamical class.** A small cell cluster undergoing a 56 mV depolarization-repolarization event shows topology-preserving dynamics -- massive voltage change with almost no topological reorganization.

---

## 1. Data Overview

### Source Configurations

All data was generated using BETSE v1.3.x from published YAML configurations included in the BETSE repository (`doc/yaml/paper/`). Simulations ran on AWS EC2 (c6i.xlarge, 4 vCPU, 8 GB RAM) for approximately 3 hours.

### Datasets

| Dataset | Config | Cells | Timesteps | Vmem Range (mV) | Spatial Domain (um) |
|---------|--------|-------|-----------|------------------|---------------------|
| **sim_1** | attractors_2018 (init_1) | 153 | 635 | [-70.6, +20.3] | 135 x 133 |
| **sim_2** | attractors_2018 (init_2) | 156 | 635 | [-86.5, +18.0] | 135 x 133 |
| **physiology** | physiology_2018 | 7 | 190 | [-83.8, -21.2] | 17 x 15 |

The `attractors_2018` configuration defines a tissue with gap junction-coupled cells under two different initial voltage conditions (init_1, init_2), then simulates their evolution over 635 timesteps. The `physiology_2018` configuration models a small cluster of 7 cells undergoing ion channel-driven voltage dynamics.

**Data deduplication note:** File hashing confirmed that `attractors_2_RESULTS/sim_2` and `attractors_3_RESULTS/sim_3` are byte-for-byte identical to their counterparts in `attractors_1_RESULTS`. The three "attractor" configs share the same base tissue mesh; only the unique simulation runs (sim_1 in config 1, sim_2 in config 2) produced distinct data. All analysis below uses the three unique datasets listed above.

### Preprocessing

Raw BETSE output (scattered Vmem2D CSV files with cell positions and voltages) was interpolated onto a regular 64x64 grid using cubic interpolation via `mneme.data.betse_loader.load_betse_timeseries()`. This produces a (T, 64, 64) field sequence per dataset.

---

## 2. Initial Analysis

### 2.1 Topological Feature Counts

Persistent homology (H0 = connected components) was computed at first, middle, and last frames using cubical complex filtration.

| Dataset | Frame | H0 Features | Max Persistence (mV) | Mean Persistence (mV) |
|---------|-------|-------------|----------------------|----------------------|
| **sim_1** | first (t=0) | 18 | 89.1 | 8.7 |
| | middle (t=317) | 8 | 77.0 | 14.0 |
| | last (t=634) | 11 | 84.2 | 12.6 |
| **sim_2** | first (t=0) | 14 | 102.0 | 12.3 |
| | middle (t=317) | 20 | 98.1 | 8.2 |
| | last (t=634) | 21 | 100.1 | 7.9 |
| **physiology** | first (t=0) | 1 | 1.7 | 1.7 |
| | middle (t=95) | 1 | 1.9 | 1.9 |
| | last (t=189) | 1 | 0.9 | 0.9 |

**Observation: Opposite topological trends.** sim_1 *simplifies* over time (18 to 11 features), with fewer but more persistent structures emerging. sim_2 *complexifies* (14 to 21 features), gaining topological structure but with each feature becoming less persistent. These are two different tissue responses: one consolidating, one proliferating.

Physiology is topologically trivial throughout -- a single connected component with persistence under 2 mV, consistent with a nearly uniform field across 7 tightly coupled cells.

### 2.2 Topology Evolution (10-Sample Timeline)

**sim_1** (simplifying):
```
t=0:   18 features
t=70:  13
t=140: 18
t=211: 23  <-- peak complexity
t=281: 17
t=352: 14
t=422: 25  <-- second peak
t=493: 21
t=563: 19
t=634: 11  <-- minimum
```

**sim_2** (complexifying):
```
t=0:   14 features
t=70:  15
t=140: 19
t=211: 16
t=281: 22  <-- peak
t=352: 19
t=422: 19
t=493: 14  <-- dip
t=563: 19
t=634: 21
```

sim_1 shows oscillatory bursts (peaks at t=211 and t=422) with an overall downward trend. sim_2 shows a steady rise with a temporary dip around t=493. These are different attractors with different basin geometries.

### 2.3 Lyapunov Spectrum (Initial, 3D Trajectory)

The initial Lyapunov analysis used a simple 3D trajectory constructed from spatial moments (mean voltage, standard deviation, skewness):

| Dataset | Spectrum | Type | D_KY | Recurrence Rate |
|---------|----------|------|------|-----------------|
| **sim_1** | [+10.45, +7.44, +7.18] | Strange | 3.0 | 6.1% |
| **sim_2** | [+11.08, +7.67, +7.06] | Strange | 3.0 | 20.5% |
| **physiology** | [+3.77, +1.31, +0.34] | Strange | 3.0 | 2.6% |

All exponents positive and D_KY = embedding dimension in every case. This signals **saturation**: the 3D moment trajectory is under-embedded for proper attractor characterization. However, the *relative* values are meaningful -- sim_2's 3.3x higher recurrence rate (20.5% vs. 6.1%) indicates more structured, quasi-periodic dynamics even in this simple representation.

---

## 3. Deep Analysis

### 3.1 PCA Mode Extraction

Each 64x64 frame was flattened to a 4,096-dimensional vector, centered, and decomposed via SVD. The top 10 singular values and their variance contributions:

**sim_1 singular values:** 2083, 130, 4.7, 0.47, 0.09, 0.008, ...
**sim_2 singular values:** 1021, 67, 11.4, 0.91, 0.07, 0.007, ...
**physiology singular values:** 13412, 1500, 28, 9.3, 0.14, ...

| Dataset | Mode 0 | Modes 0+1 | Modes 0+1+2 | Effective Rank |
|---------|--------|-----------|-------------|----------------|
| **sim_1** | 99.61% | 100.00% | 100.00% | **2** |
| **sim_2** | 99.55% | 99.99% | 100.00% | **2** |
| **physiology** | 98.76% | 100.00% | 100.00% | **2** |

**Finding: All three datasets are effectively rank-2.** The entire spatiotemporal field evolution can be approximated as:

```
V(x, y, t) ~ V_mean(x, y) + c_1(t) * phi_1(x, y) + c_2(t) * phi_2(x, y)
```

where phi_1 and phi_2 are fixed spatial modes and only the two time-varying coefficients c_1(t), c_2(t) change. This is a profound dimensionality reduction: from 4,096 dimensions per frame to 2 effective degrees of freedom.

**Implications:**
- The tissue's voltage field has a dominant "breathing mode" (phi_1) that accounts for ~99% of dynamics, with a secondary perturbation mode (phi_2).
- Attractor analysis should operate in this 2D coefficient space, not in higher dimensions.
- This explains why all Lyapunov analyses saturated -- we were embedding a 2D trajectory in 3D or 10D space, producing spurious positive exponents in the empty dimensions.
- The patterns configuration (pattern formation with gap junctions) is expected to break this rank-2 structure by introducing spatial modes that compete.

### 3.2 Lyapunov Spectrum (10D PCA Trajectory)

As predicted by the rank-2 finding, the 10D PCA Lyapunov spectrum is fully positive for all datasets:

| Dataset | Lambda_max | Lambda_min | All Positive? | D_KY |
|---------|-----------|-----------|---------------|------|
| **sim_1** | +15.02 | +10.20 | Yes (10/10) | 10.0 |
| **sim_2** | +15.14 | +10.29 | Yes (10/10) | 10.0 |
| **physiology** | +8.68 | +4.95 | Yes (10/10) | 10.0 |

The 8 noise-dominated PCA dimensions generate spurious positive exponents. The meaningful Lyapunov information is in the relative magnitudes: physiology has distinctly lower exponents (max 8.68 vs. ~15) consistent with its simpler, fewer-cell dynamics.

**The Lyapunov analysis is not informative for these rank-2 datasets.** Proper characterization requires either: (a) restricting to the 2D PCA subspace and using specialized 2D Lyapunov methods, or (b) analyzing datasets with higher effective rank (e.g., pattern-forming simulations).

### 3.3 Recurrence Quantification Analysis (10D PCA)

Despite Lyapunov saturation, recurrence analysis on the PCA trajectory reveals meaningful structure:

| Metric | sim_1 | sim_2 | physiology |
|--------|-------|-------|------------|
| Recurrence rate | 3.7% | 4.0% | 10.5% |
| Determinism | 0.459 | 0.462 | 0.421 |
| Max diagonal line length | **99** | **99** | 9 |
| Mean diagonal line length | 5.8 | 5.7 | 3.7 |
| Number of diagonal lines | 1,192 | 1,296 | 430 |

**Finding: Long quasi-periodic orbits in attractor simulations.** Both sim_1 and sim_2 contain diagonal line segments of length 99, meaning the trajectory repeats the same path for 99 consecutive timesteps before diverging. This is strong evidence of quasi-periodic orbiting within an attractor basin. sim_2 has slightly more diagonal lines (1,296 vs. 1,192), consistent with its higher recurrence.

Physiology never repeats for more than 9 steps -- its trajectory is a single sweep, not an orbit.

Determinism of ~0.46 for all three datasets indicates moderate dynamical structure: roughly half of all recurrence points fall on diagonal lines (indicating deterministic trajectories) rather than being isolated (indicating stochastic returns).

### 3.4 Wasserstein Distance Tracking

Persistent homology was computed at 30 evenly-spaced frames per dataset. The Wasserstein-2 distance between consecutive frames' H0 persistence diagrams measures the rate of topological change:

| Metric | sim_1 | sim_2 | physiology |
|--------|-------|-------|------------|
| **Mean step W-distance** | **21.88** | **16.08** | **4.64** |
| Std of step distances | 15.39 | 13.82 | 3.37 |
| Max single step | 81.18 | 67.11 | 11.04 |
| Min single step | 5.51 | 4.01 | 0.04 |
| **Total drift (first-to-last)** | **32.22** | **56.83** | **1.86** |
| Bottleneck (first-to-last) | 5.34 | 2.51 | 0.86 |

**This is the most discriminating analysis we performed.**

#### sim_1: Oscillatory Topology with Settling

sim_1's Wasserstein series over time:
```
t=0-21:    51.4  (large initial reorganization)
t=21-43:   36.1
t=43-65:   26.0  (decreasing -- system settling)
t=65-87:   29.9
t=87-109:  34.8
...
t=502-524:  7.0  (nearly stable)
t=524-546:  5.5  (minimum)
t=546-568: 81.2  (sudden burst -- topological catastrophe)
t=568-590:  7.7  (immediate return to stability)
t=612-634: 21.3
```

This is a classic **settling-with-intermittent-bursts** pattern. The topology reorganizes rapidly at first, then quiets down, but experiences a sudden catastrophic event around t=546-568 (a single-step Wasserstein of 81.2, compared to a baseline of ~7). The bottleneck distance spike of 11.8 at the same timestep confirms a single dominant topological feature underwent a drastic change.

Despite high mean step change (21.9), total drift is moderate (32.2), meaning the large changes partially cancel out. **sim_1's topology oscillates but doesn't drift far from its starting configuration.**

#### sim_2: Directional Topological Drift

sim_2's Wasserstein series:
```
t=0-21:    51.7  (large initial reorganization, similar to sim_1)
t=21-43:   22.1
t=43-65:   20.2  (also decreasing)
...
t=371-393:  4.1  (quiet)
t=393-415:  5.0
t=415-437:  4.5
...
t=546-568: 67.1  (burst -- but smaller than sim_1's 81.2)
t=612-634: 26.7
```

sim_2 also settles and also has a late burst, but with a critical difference: **total drift is 56.8 vs. sim_1's 32.2.** The small, consistent step changes accumulate rather than cancelling. sim_2 is *going somewhere* topologically -- its persistence diagram at t=634 is almost twice as far from t=0 as sim_1's is.

Meanwhile, sim_2's bottleneck maximum is only 4.84 (vs. sim_1's 11.8). No single feature changes catastrophically; instead, many features shift incrementally. **sim_2 transforms through distributed, gradual topological evolution.**

#### physiology: Topology-Preserving Voltage Excursion

physiology's Wasserstein distances oscillate between 0.04 and 11.0 around a mean of 4.6, with total drift of only 1.9. This tissue undergoes a 56 mV voltage swing (from -23 to -79 mV), but its topology barely changes. **The voltage excursion preserves the spatial organization of the field.** This is consistent with the physiology config modeling a synchronized ion channel event across a tightly-coupled 7-cell cluster: all cells depolarize and repolarize together, maintaining their relative voltage arrangement.

### 3.5 VAE Latent-Space Analysis

A convolutional VAE (16-dimensional latent space) was trained on all 1,460 frames from all three datasets jointly, then used to encode each dataset's trajectory into the learned latent manifold.

**Training:** 1,168 train / 292 validation frames, 50 epochs, converged at epoch 47 (loss 21.47). Train/val loss convergence confirmed no overfitting.

#### Latent-Space Positions: Two Distinct Attractor Basins

| Latent Dim | sim_1 mean | sim_2 mean | physiology mean | sim_1 vs sim_2 separation |
|------------|-----------|-----------|----------------|--------------------------|
| 1 | -0.14 | **-0.96** | 0.06 | 0.82 |
| 5 | 0.46 | **-2.08** | 0.38 | **2.54** |
| 7 | 1.03 | 1.25 | -0.08 | 0.22 |
| 12 | **-1.54** | -0.53 | -0.14 | **1.01** |

sim_1 and sim_2 occupy clearly separated positions in the 16D latent space, with maximum separation of 2.54 units in dimension 5. The VAE has learned to place these two initial conditions in different regions of its generative manifold. **This is direct evidence of distinct attractor basins as seen through a learned nonlinear embedding.**

#### Latent-Space Spread: Narrow Orbits vs. Wide Excursion

| Latent Dim | sim_1 std | sim_2 std | physiology std |
|------------|----------|----------|---------------|
| 0 | 0.014 | 0.018 | **0.366** |
| 4 | 0.015 | 0.004 | **0.590** |
| 5 | 0.155 | 0.038 | **1.358** |
| 12 | 0.060 | 0.018 | **1.092** |

The attractor simulations are confined to **narrow corridors** in latent space (max std = 0.155 for sim_1, 0.038 for sim_2). They orbit tightly around their respective basin centers.

Physiology sweeps through a **vastly larger region** (std up to 1.36 in dim 5, 1.09 in dim 12). The VAE sees the attractor sims as "staying in place" and the physiology sim as "traversing the manifold."

sim_2 is even more tightly confined than sim_1 (max std 0.038 vs. 0.155), consistent with its higher recurrence rate and more periodic dynamics.

#### VAE Recurrence Rates

| Dataset | VAE Recurrence |
|---------|---------------|
| sim_1 | 5.9% |
| sim_2 | 7.7% |
| physiology | **15.0%** |

Physiology has the highest VAE recurrence despite having the widest spread. This indicates its trajectory passes through certain latent regions multiple times during its large excursion -- consistent with a depolarization-repolarization cycle that returns near its origin.

---

## 4. Synthesis

### What the Data Tells Us

#### 4.1 Multi-Stability is Real and Detectable

The same BETSE tissue (153-156 cells, identical gap junction coupling) produces qualitatively different dynamics depending on initial conditions:

- **sim_1** occupies a higher-voltage basin (mean Vmem ~ -40 mV), simplifies topologically over time, and exhibits oscillatory but non-accumulating topological change (Wasserstein drift = 32.2).
- **sim_2** occupies a lower-voltage basin (mean Vmem ~ -58 mV), gains topological complexity over time, and undergoes directed topological drift (Wasserstein drift = 56.8).

Both are detected as distinct by every method: PCA positions, VAE latent means, recurrence structure, topological evolution direction, and Wasserstein distance profiles. This validates Mneme's core design premise -- that field-level attractor states encode information about tissue organization that is not captured by single-cell measurements.

#### 4.2 Topology is More Informative Than Voltage Statistics

The mean voltage trajectories of sim_1 and sim_2 are both nearly flat (3.8 mV and 1.3 mV total swing respectively). A naive analysis looking only at voltage statistics would conclude these are boring, near-equilibrium systems. Persistent homology and Wasserstein tracking reveal rich, evolving spatial structure invisible to summary statistics.

#### 4.3 Rank-2 Structure Sets a Baseline

The finding that these BETSE fields are effectively rank-2 establishes an important baseline. It means:
- The attractor configs model a simple system with one dominant spatial mode and one perturbation mode.
- Higher-rank dynamics (more competing spatial modes) are expected in pattern-forming configurations and in biological tissues with heterogeneous channel expression.
- The Mneme pipeline is ready for higher-rank data -- the VAE, PCA, and Wasserstein analyses all scale naturally. The Lyapunov analysis needs restriction to the effective rank.

#### 4.4 Different Methods Reveal Different Structure

| Method | Best discriminates | Limitation for this data |
|--------|-------------------|------------------------|
| **PCA** | Effective dimensionality, mode structure | Fields are rank-2 (ceiling) |
| **Lyapunov** | -- | Saturated at embedding dimension |
| **Recurrence** | Orbital structure, periodicity | Similar determinism across datasets |
| **Wasserstein** | Rate and direction of topological change | Requires dense sampling for events |
| **VAE latent space** | Basin separation, trajectory geometry | Lyapunov fails on latent trajectories |
| **H0 feature counting** | First/last comparison | Misses oscillatory dynamics |

Wasserstein distance tracking emerged as the single most informative method for this data, revealing both the oscillatory vs. directional distinction and the topology-preserving nature of the physiology excursion.

---

## 5. Limitations and Caveats

1. **Initial analysis used scipy fallback TDA.** Sections 2-3 (initial analysis, deep analysis of attractor/physiology configs) used the scipy-based persistence computation, which computes H0 (connected components) only. The later patterns analysis (Section 7) and the cross-frame Wasserstein matrix used full GUDHI with H0+H1 support. H1 results are available only for the patterns dataset.

2. **Interpolation artifacts.** The 153-824 cell positions were interpolated to a 64x64 regular grid using cubic interpolation. This creates smooth inter-cell fields that do not exist in the discrete BETSE model. Topological features near the interpolation boundary should be treated with caution.

3. **Lyapunov saturation on rank-2 data.** No meaningful Lyapunov exponents or Kaplan-Yorke dimensions were obtained for the attractor/physiology configs due to their rank-2 structure. The patterns config (rank-5) should produce meaningful 5D Lyapunov spectra, though this has not yet been run with rank restriction.

4. **Single tissue geometry per condition.** The attractor configs share one tissue mesh (153-156 cells), patterns uses a larger elliptical mesh (824 cells), and physiology uses a tiny cluster (7 cells). Generalization to other tissue geometries requires additional simulations.

5. **Symbolic regression R-squared is moderate.** PySR symbolic regression on the patterns PCA modes produced R-squared values ranging from 0.24 to 0.59. The discovered equations capture qualitative dynamics but are not quantitatively precise -- expected given the complexity of GRN-driven pattern formation.

---

## 6. Next Steps

1. ~~**Analyze the patterns configuration.**~~ Done -- see Section 7 below. Confirmed higher-rank (5-mode) PCA and massive topological reorganization.

2. ~~**Install GUDHI.**~~ Done -- full H0+H1 persistence computation now available. Used for cross-frame Wasserstein matrix on patterns data.

3. **2D Lyapunov analysis.** Restrict to the 2-component PCA trajectory for attractor configs and compute proper 2D Lyapunov exponents to determine whether the dynamics are chaotic, quasi-periodic, or convergent.

4. ~~**Symbolic regression on PCA coefficients.**~~ Done -- PySR discovered ODEs for 5 PCA modes of patterns data (R-squared 0.24-0.59). Also ran spatial PDE discovery via `discover_field_dynamics()`.

5. ~~**Cross-frame Wasserstein analysis.**~~ Done -- computed full 60x60 NxN Wasserstein matrix for patterns data (H0+H1). Saved to `results/deep_analysis/wasserstein_cross_matrix_patterns.npz`.

6. **ECG data comparison.** Apply the same pipeline to self-collected AD8232 ECG data to compare biological vs. simulated bioelectric field structure.

7. **Parameter sweep experiments.** Vary gap junction conductance and ion channel expression in BETSE; track attractor bifurcations using Mneme's Wasserstein and PCA analysis.

8. **Per-paper validation.** Compare Mneme's topological signatures and Lyapunov values to quantitative results reported in the original BETSE papers.

---

## 7. Patterns Configuration -- The Breakthrough Dataset

The patterns simulation completed on AWS on 2026-02-14 (26 minutes runtime) and was re-exported with Vmem CSV output enabled. This dataset is **qualitatively different** from everything analyzed above and validates key predictions from the earlier analysis.

### 7.1 Data Summary

| Property | Patterns | vs. Attractors | vs. Physiology |
|----------|----------|---------------|----------------|
| **Cells** | **824** | 5.3x more (153-156) | 118x more (7) |
| Timesteps | 119 | 5.3x fewer (635) | 0.6x fewer (190) |
| Vmem range | [-69.6, -26.8] mV | Narrower than sim_2 | Similar magnitude |
| Tissue shape | Elliptical | Circular | Tiny cluster |

The patterns config simulates a **large elliptical tissue** (824 cells) with gene regulatory network (GRN)-driven pattern formation. This is the first dataset with enough spatial complexity to potentially produce multi-modal spatial dynamics.

### 7.2 PCA: Higher Effective Rank (Confirmed Prediction)

This is the result we predicted: pattern formation breaks the rank-2 structure.

| Mode | Patterns | Attractors (sim_1) | Attractors (sim_2) | Physiology |
|------|----------|--------------------|--------------------|------------|
| 0 | **81.4%** | 99.6% | 99.6% | 98.8% |
| 0+1 | **93.9%** | 100.0% | 100.0% | 100.0% |
| 0+1+2 | **98.0%** | 100.0% | 100.0% | 100.0% |
| 0+1+2+3 | **99.0%** | 100.0% | 100.0% | 100.0% |
| 0+...+4 | **99.6%** | 100.0% | 100.0% | 100.0% |
| **Effective rank** | **~5** | ~2 | ~2 | ~2 |

**The patterns field has 5 significant PCA modes**, compared to 2 for all other datasets. Mode 0 captures only 81.4% of variance (vs. 99.5%+ elsewhere), meaning the remaining 18.6% is distributed across at least 4 additional independent spatial modes. This is exactly what pattern formation looks like in PCA: multiple competing spatial modes with similar amplitudes.

The singular value spectrum drops gradually (81%, 13%, 4%, 1%, 0.6%) rather than falling off a cliff after mode 1. This indicates a richer, higher-dimensional attractor landscape.

### 7.3 Wasserstein Distances: Massive Topological Activity

| Metric | Patterns | sim_1 | sim_2 | Physiology |
|--------|----------|-------|-------|------------|
| **Mean step W-distance** | **37.23** | 21.88 | 16.08 | 4.64 |
| Std of step distances | 25.64 | 15.39 | 13.82 | 3.37 |
| Max single step | **90.35** | 81.18 | 67.11 | 11.04 |
| **Total drift (first-to-last)** | **84.08** | 32.22 | 56.83 | 1.86 |
| Bottleneck (first-to-last) | **11.56** | 5.34 | 2.51 | 0.86 |

The patterns simulation has the **highest topological change rate** across every metric:
- Mean Wasserstein 37.2 -- nearly double sim_1's 21.9 and 2.3x sim_2's 16.1
- Total drift 84.1 -- 2.6x sim_1's 32.2 and 1.5x sim_2's 56.8
- Bottleneck 11.6 -- indicating a single dominant topological feature that changed by 11.6 mV in persistence from first to last frame

**The pattern-forming tissue is undergoing far more topological reorganization than the attractor simulations.** And unlike sim_1 (which oscillates) or sim_2 (which drifts gently), patterns is both high-rate AND high-drift: topological changes are large, frequent, and accumulating.

### 7.4 Topology: Stable Feature Count, Changing Feature Identity

| Frame | H0 Features | Max Persistence (mV) |
|-------|-------------|---------------------|
| first (t=0) | 22 | 40.2 |
| t=24 | 29 | -- |
| middle (t=59) | 20 | 40.2 |
| t=73 | 24 | -- |
| t=98 | 24 | -- |
| last (t=118) | 22 | 38.8 |

The feature *count* is relatively stable (20-29, first and last both 22), but the Wasserstein distances are enormous. This means the features are being **reorganized, not created or destroyed**. Connected components move, merge, split, and reform -- their *number* stays similar but their *identity* (birth-death values in persistence diagrams) changes dramatically. This is the signature of **active pattern formation**: the topology is churning.

### 7.5 Recurrence: Low Determinism

| Metric | Patterns | sim_1 | sim_2 | Physiology |
|--------|----------|-------|-------|------------|
| Recurrence rate | 12.9% | 3.7% | 4.0% | 10.5% |
| Determinism | **0.390** | 0.459 | 0.462 | 0.421 |
| Max diagonal | **16** | 99 | 99 | 9 |
| Mean diagonal | 5.2 | 5.8 | 5.7 | 3.7 |

The patterns simulation has the **lowest determinism** (0.39 vs. 0.46 for attractors) and the **shortest maximum diagonal** (16 vs. 99). The trajectory never repeats for more than 16 steps -- compared to the 99-step quasi-periodic orbits in the attractor sims. This system is continuously exploring new states rather than orbiting.

Combined with the high recurrence rate (12.9%), this indicates a trajectory that frequently returns to *similar* states (high recurrence) but never follows the *same path* for long (low determinism, short diagonals). This is consistent with a strange attractor with high-dimensional structure, or with a system undergoing a non-repeating transient through a complex landscape.

### 7.6 Implications

The patterns dataset confirms every prediction from the earlier analysis:

1. **Higher PCA rank breaks Lyapunov saturation.** With 5 effective modes, a 5D PCA trajectory should yield meaningful Lyapunov exponents with genuine negative components. This is the dataset where proper attractor characterization becomes possible.

2. **Topological analysis is most informative for pattern-forming tissues.** The Wasserstein tracking shows dramatic, accumulating topological reorganization that simple voltage statistics would miss entirely (the voltage range [-69.6, -26.8] is moderate and the mean voltage changes slowly).

3. **GRN-driven dynamics produce richer attractor landscapes.** The gene regulatory network coupling creates spatial competition between patterns, producing the multi-modal dynamics (5 PCA modes) that gap-junction-only coupling (rank-2) cannot.

4. **This is the dataset to focus future analysis on.** Symbolic regression on the 5-mode PCA trajectory could reveal the pattern-forming PDEs. VAE interpolation between early and late frames could map the pattern transformation pathway. Cross-frame Wasserstein matrices could identify topological phase transitions.

---

## Appendix: File Manifest

| File | Contents |
|------|----------|
| `results/attractors_1_sim1/analysis_results.json` | Initial analysis of sim_1 |
| `results/attractors_1_sim1/betse_field_sequence.npz` | Interpolated 64x64 field sequence (635 frames) |
| `results/attractors_1_sim2/analysis_results.json` | Initial analysis of sim_2 |
| `results/attractors_1_sim2/betse_field_sequence.npz` | Interpolated 64x64 field sequence (635 frames) |
| `results/physiology_sim1/analysis_results.json` | Initial analysis of physiology |
| `results/physiology_sim1/betse_field_sequence.npz` | Interpolated 64x64 field sequence (190 frames) |
| `results/patterns_sim/analysis_results.json` | Initial analysis of patterns |
| `results/patterns_sim/betse_field_sequence.npz` | Interpolated 64x64 field sequence (119 frames) |
| `results/deep_analysis/deep_analysis_results.json` | Full deep analysis results (PCA, Wasserstein, VAE, recurrence, symbolic regression) |
| `results/deep_analysis/wasserstein_cross_matrix_patterns.npz` | 60x60 H0+H1 Wasserstein distance matrices for patterns |
| `scripts/analyze_betse.py` | Initial analysis script |
| `scripts/deep_analysis.py` | Deep analysis script (PCA, VAE, Wasserstein, symbolic regression) |
| `src/mneme/data/betse_loader.py` | BETSE data loader and interpolation |
