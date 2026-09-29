# Mneme Testing Strategy

> Status: The suite has 372 tests in a flat `tests/` directory (one file per module) and takes about 5 minutes. Coverage is about 70%; CI fails below 60%. CI installs GUDHI and POT so the real persistence and Wasserstein paths are exercised, not only the fallbacks. Several sections below (property-based, performance, data-validation examples) describe an intended approach rather than tests that exist.

## Testing Philosophy

The Mneme project employs comprehensive testing to ensure:
- **Correctness**: Mathematical and algorithmic accuracy
- **Robustness**: Handling edge cases and invalid inputs
- **Performance**: Efficient processing of large datasets
- **Reproducibility**: Deterministic results with fixed seeds

## Test Categories

### 1. Unit Tests

Test individual functions and classes in isolation. A component is core only when a test checks its output against an answer known independently of the code (a closed-form field, a union-find H0 count, a known time constant); see [SCOPE.md](SCOPE.md).

```python
# tests/test_field_theory.py
import pytest
import numpy as np
from mneme.core.field_theory import FieldReconstructor

class TestFieldReconstructor:
    def test_initialization(self):
        reconstructor = FieldReconstructor(method='gaussian_process')
        assert reconstructor.method == 'gaussian_process'
        assert reconstructor.resolution == (256, 256)
    
    def test_fit_with_valid_data(self):
        # Generate test data
        observations = np.random.randn(100)
        positions = np.random.rand(100, 2)
        
        reconstructor = FieldReconstructor()
        reconstructor.fit(observations, positions)
        
        assert reconstructor.is_fitted
        assert reconstructor.observations.shape == (100,)
    
    def test_reconstruct_shape(self):
        # Setup
        observations = np.random.randn(50)
        positions = np.random.rand(50, 2)
        
        reconstructor = FieldReconstructor(resolution=(128, 128))
        reconstructor.fit(observations, positions)
        
        # Test
        field = reconstructor.reconstruct()
        
        assert field.shape == (128, 128)
        assert not np.any(np.isnan(field))
    
    @pytest.mark.parametrize("method", ['gp_subset', 'gaussian_process', 'neural_field'])
    def test_different_methods(self, method):
        reconstructor = FieldReconstructor(method=method)
        # Test method-specific behavior
```

### 2. Integration Tests

Test interactions between components.

```python
# tests/test_pipeline.py
import numpy as np
import pytest
from mneme.analysis.pipeline import MnemePipeline, default_config, merge_config

class TestPipelineIntegration:
    def test_full_pipeline_execution(self, gaussian_blob_field, sparse_observations):
        observations, positions = sparse_observations
        config = merge_config(default_config('bioelectric'),
                              {'reconstruction': {'resolution': (16, 16)},
                               'topology': {'max_dimension': 1}})
        pipeline = MnemePipeline(config)

        result = pipeline.run({'field': gaussian_blob_field,
                               'observations': observations,
                               'positions': positions})

        assert result.success
        assert result.failed_stages == []
        assert result.stage_results['reconstruction']['status'] == 'completed'
        assert result.analysis_result.reconstruction.field.data.shape == (16, 16)
        assert result.analysis_result.topology is not None

    def test_reconstruction_skipped_without_observations(self, gaussian_blob_field):
        pipeline = MnemePipeline(default_config('bioelectric'))
        result = pipeline.run({'field': gaussian_blob_field})

        # Skipped, not faked: no reconstruction result, and the run still succeeds
        assert result.success
        assert result.stage_results['reconstruction']['status'] == 'skipped'
        assert result.analysis_result.reconstruction is None

    def test_failed_stage_is_reported(self, gaussian_blob_field):
        pipeline = MnemePipeline({'topology': {'max_dimension': 1}})
        field = gaussian_blob_field.copy()
        field[0, 0] = np.nan          # persistence of a NaN field raises

        result = pipeline.run({'field': field})

        assert not result.success
        assert 'topology' in result.failed_stages
        assert result.errors
```

### 3. Property-Based Tests (not yet in the suite)

Hypothesis is not currently a dependency. If added, properties of persistence diagrams are a natural target.

```python
# tests/test_topology_properties.py (illustrative)
import numpy as np
import hypothesis as hp
from hypothesis import strategies as st
from mneme.core.topology import PersistentHomology

class TestTopologyProperties:
    @hp.given(
        field=st.lists(
            st.lists(st.floats(min_value=-100, max_value=100), 
                    min_size=10, max_size=10),
            min_size=10, max_size=100
        )
    )
    def test_persistence_diagram_properties(self, field):
        field_array = np.array(field)
        
        ph = PersistentHomology(max_dimension=1, persistence_threshold=0.0)
        diagrams = ph.compute_persistence(field_array)
        
        # Property: birth <= death for every bar
        for diagram in diagrams:
            pts = diagram.points
            assert np.all(pts[:, 0] <= pts[:, 1])
        
        # Property: exactly one infinite H0 bar (one connected field)
        assert np.sum(np.isinf(diagrams[0].points[:, 1])) == 1
```

### 4. Performance Tests (not yet in the suite)

Ensure operations meet performance requirements. The `performance` marker is registered in `pyproject.toml`.

```python
# tests/test_reconstruction_performance.py (illustrative)
import pytest
import time
from mneme.core.field_theory import FieldReconstructor

class TestReconstructionPerformance:
    @pytest.mark.performance
    def test_reconstruction_speed(self, benchmark):
        # Setup
        observations = np.random.randn(1000)
        positions = np.random.rand(1000, 2)
        
        reconstructor = FieldReconstructor(method='gaussian_process')
        reconstructor.fit(observations, positions)
        
        # Benchmark reconstruction
        result = benchmark(reconstructor.reconstruct)
        
        # Assert performance threshold
        assert benchmark.stats['mean'] < 1.0  # Should complete in < 1 second
    
    @pytest.mark.performance
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_scaling_behavior(self, size):
        observations = np.random.randn(size)
        positions = np.random.rand(size, 2)
        
        reconstructor = FieldReconstructor()
        
        start = time.time()
        reconstructor.fit(observations, positions)
        reconstructor.reconstruct()
        duration = time.time() - start
        
        # Log-linear scaling expected
        expected_max_time = 0.001 * size * np.log(size)
        assert duration < expected_max_time
```

### 5. Data Validation Tests

Test data loading and validation.

```python
# tests/test_data_validation.py (illustrative)
import numpy as np
import pytest
from mneme.types import FieldDataSchema
from mneme.data.validation import DataValidator

class TestDataValidation:
    def test_valid_field_data(self):
        schema = FieldDataSchema(
            shape=(None, 256, 256),
            dtype=np.float32,
            value_range=(-100, 100)
        )
        
        # Valid data
        valid_data = np.random.uniform(-50, 50, (10, 256, 256)).astype(np.float32)
        validator = DataValidator(schema)
        
        result = validator.validate(valid_data)   # ValidationResult
        assert result.is_valid
        assert result.errors == []
    
    def test_invalid_shape(self):
        schema = FieldDataSchema(shape=(None, 256, 256))
        invalid_data = np.zeros((10, 128, 128))  # Wrong spatial dimensions
        
        validator = DataValidator(schema)
        result = validator.validate(invalid_data)
        
        assert not result.is_valid
        assert 'shape' in result.errors[0]
```

### 6. Fixtures

`tests/conftest.py` provides shared fixtures: closed-form 32x32 fields (`gaussian_blob_field`, `two_peak_field`, `sinusoidal_field`), a `temporal_field_sequence`, RK4-integrated `lorenz_rk4` and `rossler_rk4` trajectories with known λ₁ (0.906 and 0.071), `sparse_observations` for reconstruction, and a `minimal_pipeline_config`.

```python
# Usage in tests
def test_two_peaks_give_two_components(two_peak_field):
    from mneme.core.topology import PersistentHomology
    ph = PersistentHomology(max_dimension=0, filtration='superlevel', persistence_threshold=0.1)
    h0, = ph.compute_persistence(two_peak_field)
    assert len(h0.points) == 2
```

## Test Organization

The suite is flat: one file per module, plus shared fixtures.

```
tests/
├── conftest.py                    # Shared fixtures
├── test_field_theory.py           # Reconstructors against a known field
├── test_topology.py
├── test_topology_correctness.py   # Closed-form fields, union-find H0, sublevel/superlevel
├── test_attractors.py
├── test_embedding.py
├── test_lyapunov.py               # Lorenz / Rössler
├── test_surrogates.py
├── test_classify.py
├── test_steady_state.py           # Known time constants, double-well states
├── test_betse_loader.py           # Frame order, cell counts, value ranges
├── test_pipeline.py               # Success semantics, skipped reconstruction
├── test_cli.py
├── test_status.py                 # ExperimentalWarning
├── test_models.py
└── ...                            # config, io, loaders, logging, metrics, preprocessors, ...
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests (about 5 minutes)
pytest

# Run specific test file
pytest tests/test_field_theory.py

# Run tests matching pattern
pytest -k "reconstruction"

# Run with coverage
pytest --cov=src/mneme --cov-report=html

# Run only marked tests
pytest -m "not slow"
```

### Test Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_large_dataset_processing():
    # Test that takes > 1 second
    pass

# Mark tests requiring GPU
@pytest.mark.gpu
def test_neural_field_cuda():
    # Test requiring CUDA
    pass

# Mark integration tests
@pytest.mark.integration
def test_full_pipeline():
    # Cross-component test
    pass
```

### Continuous Integration

`.github/workflows/tests.yml` runs on pushes and pull requests to `main`, on Ubuntu with Python 3.12:

1. Install `libhdf5-dev`, then `pip install -e ".[dev,tda]"` (GUDHI and POT included, so the real persistence and Wasserstein paths are tested)
2. `flake8` for syntax errors and undefined names (fails the build); style warnings are reported but do not fail
3. `mypy src/mneme --ignore-missing-imports` (advisory)
4. `pytest tests -v --tb=short --cov=src/mneme --cov-report=term-missing --cov-report=xml`
5. `coverage report --fail-under=60`

A separate `docs.yml` workflow builds the mkdocs site and deploys it to GitHub Pages on pushes to `main`.

## Test-Driven Development Guidelines

1. **Write Tests First**: Define expected behavior before implementation
2. **Test Edge Cases**: Empty inputs, extreme values, invalid parameters
3. **Mock External Dependencies**: Use mocks for file I/O, network calls
4. **Keep Tests Fast**: Mock expensive operations, use small test data
5. **Clear Test Names**: `test_<what>_<condition>_<expected_result>`
6. **One Assertion Per Test**: Make failures easy to diagnose
7. **Use Fixtures**: Share setup code, ensure cleanup

## Coverage Requirements

- CI floor: 60% (`coverage report --fail-under=60`; `fail_under = 60` in `pyproject.toml`)
- Current: about 70%
- Coverage is a floor, not the goal. What moves a component to the core tier is a test against an independently known answer, not line coverage.
- Excluded from coverage: `__init__.py` files and tests (see `[tool.coverage.run]` in `pyproject.toml`)

## Debugging Tests

```python
# Use pytest debugging
pytest --pdb  # Drop into debugger on failure

# Capture print statements
pytest -s  # No capture, show prints

# Verbose output
pytest -vv  # Very verbose

# Run specific test with debugging
pytest tests/test_field_theory.py::TestFieldReconstructor::test_fit_with_valid_data --pdb -vv
```