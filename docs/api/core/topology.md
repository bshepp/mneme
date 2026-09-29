# Topology

Persistent homology computation and topological distance metrics for field data.

`sublevel` filtration tracks pits as the threshold rises; `superlevel` tracks peaks, with diagrams expressed in units of the negated field. Without GUDHI only H0 is computed (exact union-find) and a `RuntimeWarning` is emitted; the Wasserstein and bottleneck fallbacks also warn. `compute_cycles=True` raises `NotImplementedError`.

## Primary Interface

::: mneme.core.topology.PersistentHomology

## Complex Types

::: mneme.core.topology.RipsComplex

::: mneme.core.topology.AlphaComplex

## Distance Metrics

::: mneme.core.topology.compute_wasserstein_distance

::: mneme.core.topology.compute_bottleneck_distance

## Topological Descriptors

::: mneme.core.topology.compute_betti_curve

## Utilities

::: mneme.core.topology.filter_persistence_diagram

::: mneme.core.topology.field_to_point_cloud
