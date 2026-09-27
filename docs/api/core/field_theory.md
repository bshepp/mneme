# Field Theory

Field reconstruction from sparse observations: a Gaussian process on a random subset of the data (default), a dense Wiener filter, a standard Gaussian process, and an experimental neural field.

The names `SparseGPReconstructor`, `IFTReconstructor` and `DenseIFTReconstructor` are deprecated aliases of the classes below.

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
