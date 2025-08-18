# Mneme Testing Strategy

## Testing Philosophy

The Mneme project employs comprehensive testing to ensure:
- **Correctness**: Mathematical and algorithmic accuracy
- **Robustness**: Handling edge cases and invalid inputs
- **Performance**: Efficient processing of large datasets
- **Reproducibility**: Deterministic results with fixed seeds

## Test Categories

### 1. Unit Tests

Test individual functions and classes in isolation.

```python
# tests/unit/test_field_theory.py
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
    
    @pytest.mark.parametrize("method", ['gaussian_process', 'ift', 'neural_field'])
    def test_different_methods(self, method):
        reconstructor = FieldReconstructor(method=method)
        # Test method-specific behavior
```

### 2. Integration Tests

Test interactions between components.

```python
# tests/integration/test_pipeline.py
import pytest
from mneme.analysis.pipeline import MnemePipeline
from mneme.data.generators import SyntheticFieldGenerator

class TestPipelineIntegration:
    def test_full_pipeline_execution(self):
        # Generate synthetic data
        generator = SyntheticFieldGenerator(seed=42)
        field_data = generator.generate_dynamic(
            shape=(64, 64), 
            timesteps=10,
            parameters={'noise_level': 0.1}
        )
        
        # Create and run pipeline
        pipeline = MnemePipeline({
            'preprocessing': {'normalize': True},
            'reconstruction': {'method': 'ift'},
            'analysis': {'compute_topology': True}
        })
        
        results = pipeline.run({'field': field_data})
        
        # Verify all expected outputs
        assert 'reconstructed_field' in results
        assert 'persistence_diagrams' in results
        assert results['reconstructed_field'].shape[1:] == (256, 256)
    
    def test_pipeline_stage_dependencies(self):
        # Test that stages execute in correct order
        pipeline = MnemePipeline({})
        
        # Add stages with dependencies
        pipeline.add_stage('preprocess', lambda x: x, ['raw'], ['processed'])
        pipeline.add_stage('analyze', lambda x: x, ['processed'], ['results'])
        
        # Verify dependency resolution
        order = pipeline._resolve_execution_order()
        assert order.index('preprocess') < order.index('analyze')
```

### 3. Property-Based Tests

Use hypothesis for generative testing.

```python
# tests/unit/test_topology_properties.py
import hypothesis as hp
from hypothesis import strategies as st
from mneme.core.topology import PersistentHomology

class TestTopologyProperties:
    @hp.given(
        field=st.lists(
            st.lists(st.floats(min_value=-100, max_value=100), 
                    min_size=10, max_size=100),
            min_size=10, max_size=100
        )
    )
    def test_persistence_diagram_properties(self, field):
        # Convert to numpy array
        field_array = np.array(field)
        
        ph = PersistentHomology()
        diagrams = ph.compute_persistence(field_array)
        
        # Property: Birth times <= Death times
        for diagram in diagrams:
            assert np.all(diagram[:, 0] <= diagram[:, 1])
        
        # Property: Finite persistence
        assert np.all(np.isfinite(diagrams[0]))
```

### 4. Performance Tests

Ensure operations meet performance requirements.

```python
# tests/performance/test_reconstruction_performance.py
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
# tests/unit/test_data_validation.py
import pytest
from mneme.data.validation import FieldDataSchema, DataValidator

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
        
        is_valid, errors = validator.validate(valid_data)
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_shape(self):
        schema = FieldDataSchema(shape=(None, 256, 256))
        invalid_data = np.zeros((10, 128, 128))  # Wrong spatial dimensions
        
        validator = DataValidator(schema)
        is_valid, errors = validator.validate(invalid_data)
        
        assert not is_valid
        assert 'shape' in errors[0]
    
    def test_out_of_range_values(self):
        schema = FieldDataSchema(value_range=(0, 1))
        invalid_data = np.array([[-1, 2, 0.5]])  # Values outside range
        
        validator = DataValidator(schema)
        is_valid, errors = validator.validate(invalid_data)
        
        assert not is_valid
        assert 'value_range' in errors[0]
```

### 6. Fixture and Mock Tests

```python
# tests/conftest.py
import pytest
import numpy as np
from mneme.data.generators import SyntheticFieldGenerator

@pytest.fixture
def sample_field():
    """Generate a sample field for testing."""
    generator = SyntheticFieldGenerator(seed=42)
    return generator.generate_static(
        shape=(64, 64),
        parameters={'pattern': 'gaussian_blob', 'noise': 0.1}
    )

@pytest.fixture
def mock_bioelectric_data(tmp_path):
    """Create mock bioelectric data file."""
    import h5py
    
    data_file = tmp_path / "mock_data.h5"
    with h5py.File(data_file, 'w') as f:
        f.create_dataset('voltage_fields', data=np.random.randn(10, 64, 64))
        f.create_dataset('timestamps', data=np.arange(10))
        f.attrs['sampling_rate_hz'] = 10.0
    
    return data_file

# Usage in tests
def test_with_fixture(sample_field):
    assert sample_field.shape == (64, 64)
    assert sample_field.dtype == np.float64
```

## Test Organization

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_field_theory.py
│   ├── test_topology.py
│   ├── test_attractors.py
│   ├── test_data_loaders.py
│   └── test_models.py
│
├── integration/               # Integration tests
│   ├── test_pipeline.py
│   ├── test_data_flow.py
│   └── test_model_training.py
│
├── performance/               # Performance benchmarks
│   ├── test_reconstruction_performance.py
│   └── test_topology_performance.py
│
├── fixtures/                  # Test data and fixtures
│   ├── synthetic_fields.npz
│   └── test_config.yaml
│
└── conftest.py               # Shared fixtures and configuration
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_field_theory.py

# Run tests matching pattern
pytest -k "reconstruction"

# Run with coverage
pytest --cov=mneme --cov-report=html

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

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=mneme --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Test-Driven Development Guidelines

1. **Write Tests First**: Define expected behavior before implementation
2. **Test Edge Cases**: Empty inputs, extreme values, invalid parameters
3. **Mock External Dependencies**: Use mocks for file I/O, network calls
4. **Keep Tests Fast**: Mock expensive operations, use small test data
5. **Clear Test Names**: `test_<what>_<condition>_<expected_result>`
6. **One Assertion Per Test**: Make failures easy to diagnose
7. **Use Fixtures**: Share setup code, ensure cleanup

## Coverage Requirements

- Minimum overall coverage: 80%
- Core modules (`field_theory`, `topology`): 90%
- Critical paths: 95%
- Exclude from coverage: Visualization code, scripts

## Debugging Tests

```python
# Use pytest debugging
pytest --pdb  # Drop into debugger on failure

# Capture print statements
pytest -s  # No capture, show prints

# Verbose output
pytest -vv  # Very verbose

# Run specific test with debugging
pytest tests/unit/test_field_theory.py::TestFieldReconstructor::test_fit_with_valid_data --pdb -vv
```