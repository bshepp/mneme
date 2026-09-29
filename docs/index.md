# Mneme

**Detecting field-like memory structures in biological systems**

Mneme is an exploratory research toolkit for studying field-like memory in biological tissue, starting with simulated bioelectric data.

!!! warning "Validation status"
    No scientific result produced with Mneme is currently asserted. Earlier results were withdrawn after a review found defects in the code that produced them. See [Scope and Support Status](SCOPE.md).

## Capabilities

| Tier | Components |
|---|---|
| **Core** (tested against known answers) | BETSE loading, cubical persistent homology, Wasserstein and bottleneck distances, Gaussian-process and Wiener-filter reconstruction |
| **Frozen** (documented operating range) | Largest Lyapunov exponent, surrogate significance test, gated attractor classification |
| **Experimental** (not validated) | Attractor detectors, symbolic regression, variational autoencoder, neural field reconstruction |

## Quick Start

```python
import numpy as np
from mneme.core import create_reconstructor
from mneme.analysis.pipeline import create_bioelectric_pipeline
from mneme.data.generators import generate_planarian_bioelectric_sequence

# Generate synthetic bioelectric data
data = generate_planarian_bioelectric_sequence(shape=(64, 64), timesteps=30, seed=42)

# Run analysis pipeline
pipe = create_bioelectric_pipeline()
result = pipe.run({'field': data})
print(f"Pipeline completed in {result.execution_time:.2f}s")

# Reconstruct field from sparse observations
positions = np.random.rand(100, 2)
observations = np.sin(4 * np.pi * positions[:, 0])
rec = create_reconstructor('gp_subset', resolution=(128, 128))
rec.fit(observations, positions)
field = rec.reconstruct()
```

## Installation

```bash
git clone https://github.com/bshepp/mneme.git
cd mneme
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[tda]"    # core, plus GUDHI and POT for topology

# Optional: symbolic regression (needs Julia)
pip install -e ".[pysr]"
```

## Documentation

- **[Getting Started](DEVELOPMENT_SETUP.md)** -- Environment setup and dependencies
- **[Project Structure](PROJECT_STRUCTURE.md)** -- Code organization and architecture
- **[API Reference](api/index.md)** -- Auto-generated reference for all modules
- **[Data Pipeline](DATA_PIPELINE.md)** -- Pipeline architecture and stages
- **[Scope and Support Status](SCOPE.md)** -- What is core, frozen and experimental
- **[Lyapunov Operating Range](LYAPUNOV_OPERATING_RANGE.md)** -- Measured accuracy and limits
- **[Course](course/README.md)** -- 11-module learning course

## License

MIT License. See [LICENSE](https://github.com/bshepp/mneme/blob/main/LICENSE) for details.
