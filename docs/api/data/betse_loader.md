# BETSE Loader

Load and preprocess output from BETSE (BioElectric Tissue Simulation Engine) simulations.

Frames are read in numeric time order. Prefer `load_betse_cells()` unless a regular grid is required: grid values outside the convex hull of the cells are nearest-neighbour fill, not simulation output, and the `inside_hull` mask in the metadata marks them. Grid interpolation defaults to linear.

## Preferred Entry Point

::: mneme.data.betse_loader.load_betse_cells

## Grid Loading

::: mneme.data.betse_loader.betse_to_field

::: mneme.data.betse_loader.load_betse_timeseries

## Single-File Loading

::: mneme.data.betse_loader.load_betse_vmem_csv

::: mneme.data.betse_loader.load_betse_exported_data

## Grid Interpolation

::: mneme.data.betse_loader.interpolate_to_grid
