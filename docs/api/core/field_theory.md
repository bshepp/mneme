# Field Theory

Field reconstruction from sparse observations: a Gaussian process on a random subset of the data (default), a dense Wiener filter, a standard Gaussian process, and an experimental neural field.

The names `SparseGPReconstructor`, `IFTReconstructor` and `DenseIFTReconstructor` are deprecated aliases of the classes below, as are the method names `ift`, `sparse_gp` and `dense_ift`.

!!! warning "Experimental"
    `NeuralFieldReconstructor` has no accuracy test and its `uncertainty()` raises `NotImplementedError`. See [Scope and Support Status](../../SCOPE.md).

## Factory Function

::: mneme.core.field_theory.create_reconstructor

## Reconstructors

::: mneme.core.field_theory.SubsetGPReconstructor

::: mneme.core.field_theory.WienerFilterReconstructor

::: mneme.core.field_theory.GaussianProcessReconstructor

::: mneme.core.field_theory.NeuralFieldReconstructor

## Base Class

::: mneme.core.field_theory.BaseFieldReconstructor

## Dispatcher

::: mneme.core.field_theory.FieldReconstructor
