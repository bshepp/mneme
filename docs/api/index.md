# API Reference

Auto-generated reference documentation for all public Mneme modules.

Components are sorted into core, frozen and experimental tiers; see [Scope and Support Status](../SCOPE.md). Experimental components emit `mneme.ExperimentalWarning` when constructed.

## Core Modules

The foundation of Mneme's analysis capabilities:

- [**Field Theory**](core/field_theory.md) -- Field reconstruction from sparse observations (subset GP, Wiener filter, standard GP; experimental neural field)
- [**Topology**](core/topology.md) -- Persistent homology, persistence diagrams, Wasserstein/bottleneck distances
- [**Attractors**](core/attractors.md) -- Recurrence- and clustering-based detection of dense regions (experimental; reports `UNDETERMINED` type)
- [**Lyapunov**](core/lyapunov.md) -- Largest Lyapunov exponent (Rosenstein 1993) and exploratory spectrum (frozen)
- [**Surrogates**](core/surrogates.md) -- IAAFT surrogate-data significance testing (frozen)
- [**Classification**](core/classify.md) -- Surrogate-gated attractor classification, Kaplan-Yorke dimension (frozen)
- [**Embedding**](core/embedding.md) -- Delay/dimension selection (MI delay, Cao 1997, Theiler window) (frozen)

## Models

Machine learning models for field analysis (experimental):

- [**Autoencoders**](models/autoencoders.md) -- Convolutional VAE for learning compressed field representations
- [**Symbolic Regression**](models/symbolic.md) -- PySR integration for searching for governing equations

## Data

Data loading, generation, and preprocessing:

- [**Generators**](data/generators.md) -- Synthetic field data generation (Gaussian blobs, bioelectric sequences)
- [**Preprocessors**](data/preprocessors.md) -- Denoising, normalization, interpolation
- [**BETSE Loader**](data/betse_loader.md) -- Load BETSE bioelectric tissue simulation output

## Analysis

Pipeline orchestration and visualization:

- [**Pipeline**](analysis/pipeline.md) -- End-to-end analysis pipeline with configurable stages
- [**Steady State**](analysis/steady_state.md) -- Has a run settled, and how many distinct end states do several runs reach
- [**Visualization**](analysis/visualization.md) -- Field visualization, dashboards, and plotting utilities
