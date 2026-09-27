# Attractors

Recurrence- and clustering-based detection of dense regions in a trajectory.

!!! warning "Experimental"
    These detectors locate regions. They do not determine attractor type, and report `UNDETERMINED`. See [Scope and Support Status](../../SCOPE.md).

(For the corrected Lyapunov estimator and surrogate-significance gate, see
[Lyapunov](lyapunov.md), [Surrogates](surrogates.md), and
[Classification](classify.md). For embedding-parameter selection see
[Embedding](embedding.md).)

## Recurrence Analysis

::: mneme.core.attractors.RecurrenceAnalysis

## Lyapunov Analysis (detector)

::: mneme.core.attractors.LyapunovAnalysis

## Correlation Dimension

::: mneme.core.attractors.compute_correlation_dimension

## Clustering-Based Detection

::: mneme.core.attractors.ClusteringDetector

## Dispatcher

::: mneme.core.attractors.AttractorDetector
