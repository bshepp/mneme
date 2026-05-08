# Mneme — Future Ideas / Deferred Implementations

This file collects implementation notes for features that were intentionally
removed from the codebase rather than left as stubs. Use it as a starting
point if a future contributor (human or AI) asks "what could we try next?"

---

## Basin of Attraction Estimation

**Removed:** `mneme.core.attractors.compute_basin_of_attraction()` was
removed in v0.1.x. It previously raised `NotImplementedError`.

### Why it was removed

A meaningful basin-of-attraction map requires knowing the underlying dynamical
system so you can simulate trajectories from a grid of initial conditions and
record which attractor each trajectory converges to. From a single
*observational* time series — which is what Mneme typically receives — the
ground-truth basin is not recoverable: you only see one trajectory and
therefore one basin (by definition).

The previous signature `(attractor, trajectory, grid_resolution)` could not
deliver that, so it was removed to avoid misleading downstream users.

### When it would make sense to add it back

Add basin estimation only once Mneme can either

1. Operate on a *learned* surrogate model of the dynamics (e.g. the symbolic
   regressor in `mneme.models.symbolic` or the VAE-decoded vector field), or
2. Accept an explicit user-supplied `dynamics_fn(state) -> dstate/dt`.

### Suggested approach (when prerequisites exist)

```python
def compute_basin_of_attraction(
    attractors: list[Attractor],
    dynamics_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    grid_resolution: int = 50,
    *,
    t_max: float = 1000.0,
    dt: float = 0.01,
    convergence_radius: float | None = None,
) -> np.ndarray:
    """
    Returns an integer array of shape (grid_resolution,) * len(bounds)
    where each cell holds the index of the attractor that the trajectory
    starting at that cell converges to (-1 = no convergence within t_max).
    """
```

Implementation sketch:

1. Build a meshgrid over `bounds` at `grid_resolution` per axis.
2. For each grid point, integrate `dynamics_fn` (RK4 or `scipy.integrate.solve_ivp`).
3. After `t_max`, classify the endpoint by nearest attractor center
   (within `convergence_radius`, defaulting to `0.05 * mean(bound_widths)`).
4. Parallelize over grid cells with `joblib.Parallel` — each integration is
   independent.
5. For 2D systems, expose a quick `plot_basin(basin, bounds, attractors)`
   helper that uses `matplotlib.pcolormesh` with a discrete colormap.

### Open design questions

- **Stochastic systems**: should we report basin *probabilities* via Monte
  Carlo over noise realizations rather than a single integer label?
- **High-dimensional fields**: basins in >3D are not visualizable; consider
  reporting only basin *volume fractions* and a t-SNE / UMAP projection of
  the labeled grid.
- **Riddled / fractal basins**: provide a refinement option that adaptively
  subdivides cells whose neighbours disagree.

### Useful references

- Nusse & Yorke, *Dynamics: Numerical Explorations* (Springer) — Ch. on basins.
- Aguirre, Viana & Sanjuán, "Fractal structures in nonlinear dynamics", *RMP* 2009.
- `scipy.integrate.solve_ivp` with `events=` for early-termination on convergence.

---

## (template for future deferred features)

When removing another stub, add a section here with:

- **Removed:** the symbol and version.
- **Why it was removed.**
- **When it would make sense to add back.**
- **Suggested approach** (signature + sketch).
- **Open design questions.**
- **Useful references.**
