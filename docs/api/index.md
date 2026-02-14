# API Reference

Auto-generated reference documentation for all public Mneme modules.

## Core Modules

The foundation of Mneme's analysis capabilities:

- [**Field Theory**](core/field_theory.md) -- Field reconstruction from sparse observations (Sparse GP, Dense IFT, Neural Fields)
- [**Topology**](core/topology.md) -- Persistent homology, persistence diagrams, Wasserstein/bottleneck distances
- [**Attractors**](core/attractors.md) -- Recurrence analysis, Lyapunov spectrum, clustering-based attractor detection

## Models

Machine learning models for field analysis:

- [**Autoencoders**](models/autoencoders.md) -- Convolutional VAE for learning compressed field representations
- [**Symbolic Regression**](models/symbolic.md) -- PySR integration for discovering governing equations

## Data

Data loading, generation, and preprocessing:

- [**Generators**](data/generators.md) -- Synthetic field data generation (Gaussian blobs, bioelectric sequences)
- [**Preprocessors**](data/preprocessors.md) -- Denoising, normalization, interpolation
- [**BETSE Loader**](data/betse_loader.md) -- Load BETSE bioelectric tissue simulation output

## Analysis

Pipeline orchestration and visualization:

- [**Pipeline**](analysis/pipeline.md) -- End-to-end analysis pipeline with configurable stages
- [**Visualization**](analysis/visualization.md) -- Field visualization, dashboards, and plotting utilities
