# Mneme

**Detecting field-like memory structures in biological systems**

Mneme is an exploratory research system for uncovering attractor states, regulatory logic, and latent architectures in biological tissue -- structures not captured by sequence-based models alone. The project employs Information Field Theory (IFT), Topological Data Analysis (TDA), and machine learning to identify and model distributed memory encoding via fields.

## Key Capabilities

- **Field Reconstruction** -- Scalable Sparse GP (default), Dense IFT, Standard GP, and Neural Field backends. Handles 256x256 fields in sub-second time.
- **Topology Analysis** -- Full GUDHI integration for cubical, Rips, and Alpha complexes. Persistence diagrams, landscapes, images, Wasserstein/bottleneck distances.
- **Attractor Detection** -- Recurrence-based, Lyapunov, and clustering detectors for identifying stable states in temporal field data.
- **Lyapunov Spectrum** -- Full Wolf algorithm for computing Lyapunov exponents. Kaplan-Yorke dimension and automatic attractor classification.
- **Symbolic Regression** -- PySR integration for discovering governing PDEs from field dynamics.
- **Latent Space Analysis** -- Convolutional VAE for learning compressed field representations with interpolation and sampling.
- **BETSE Integration** -- Direct ingestion of BETSE bioelectric tissue simulation output.

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
rec = create_reconstructor('ift', resolution=(128, 128))
rec.fit(observations, positions)
field = rec.reconstruct()
```

## Installation

```bash
git clone https://github.com/bshepp/mneme.git
cd mneme
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Optional: TDA and symbolic regression
pip install gudhi pysr
```

## Documentation

- **[Getting Started](DEVELOPMENT_SETUP.md)** -- Environment setup and dependencies
- **[Project Structure](PROJECT_STRUCTURE.md)** -- Code organization and architecture
- **[API Reference](api/index.md)** -- Auto-generated reference for all modules
- **[Data Pipeline](DATA_PIPELINE.md)** -- Pipeline architecture and stages
- **[BETSE Analysis](BETSE_ANALYSIS_REPORT.md)** -- Results from BETSE simulation analysis
- **[Course](course/README.md)** -- 11-module learning course

## License

MIT License. See [LICENSE](https://github.com/bshepp/mneme/blob/main/LICENSE) for details.
