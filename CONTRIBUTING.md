# Contributing to Mneme

Thank you for your interest in contributing to Mneme! This document provides guidelines and information for contributors.

## Code of Conduct

This project is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to bshepp@gmail.com.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/mneme.git`
3. Add upstream remote: `git remote add upstream https://github.com/bshepp/mneme.git`
4. Create a feature branch: `git checkout -b feature/your-feature-name`
5. Set up development environment (see `docs/DEVELOPMENT_SETUP.md`)

## Development Process

### 1. Before You Start

- Check existing issues and pull requests
- Discuss major changes in an issue first
- Ensure your idea aligns with project goals

### 2. Making Changes

#### Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names

```python
# Good
def reconstruct_field(
    observations: np.ndarray, 
    positions: np.ndarray,
    method: str = "gaussian_process"
) -> np.ndarray:
    """Reconstruct continuous field from discrete observations."""
    
# Bad
def recon(obs, pos, m="gp"):
    """recon field"""
```

#### Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Examples:
```
feat(topology): add persistent homology computation

fix(data): handle missing values in bioelectric loader

docs(api): update field reconstruction examples

test(models): add autoencoder integration tests
```

### 3. Testing

- Install with `pip install -e ".[dev,tda]"` so GUDHI and POT are present, as in CI
- Write tests for new functionality
- Ensure all tests pass: `pytest` (about 5 minutes)
- Maintain or improve code coverage (CI fails below 60%)
- Add integration tests for complex features

### 4. Documentation

- Update docstrings for new/modified functions
- Update relevant documentation in `docs/`
- Add examples for new features
- Update CLAUDE.md if adding new development commands

### 5. Pull Request Process

1. Update your branch with latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Run quality checks:
   ```bash
   # Format code
   black src/ tests/
   
   # Lint
   flake8 src/ tests/
   
   # Type check
   mypy src/
   
   # Run tests
   pytest
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create pull request with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes
   - Test results

## Project-Specific Guidelines

### Component Tiers

Every component is core, frozen or experimental ([docs/SCOPE.md](docs/SCOPE.md)). A new component is experimental until it has a test that checks its output against an answer known independently of the code, and that test runs in CI. Until then its constructor should call `mneme._status.warn_experimental("Name")`, and its documentation should say so.

Do not add a scientific result to the documentation unless the analysis that produced it is reproducible from the repository and its inputs are recorded. Results that are later found to rest on defective code are withdrawn in `CHANGELOG.md`, not silently edited.

### Adding New Analysis Methods

When adding new analysis methods:

1. Create module in appropriate subpackage
2. Implement base functionality with clear API
3. Add tests, including at least one against a known answer
4. Create example notebook
5. Update pipeline integration and the tier table in `docs/SCOPE.md`

Example structure:
```python
# src/mneme/core/new_method.py
class NewAnalysisMethod:
    """One-line description.
    
    Longer description explaining the method,
    its purpose, and theoretical background.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2
        
    Examples
    --------
    >>> method = NewAnalysisMethod(param1=value)
    >>> result = method.analyze(data)
    """
```

### Data Format Standards

When working with data:

- Use HDF5 for large datasets
- Include comprehensive metadata
- Follow established schema (see `docs/DATA_PIPELINE.md`)
- Validate data types and ranges

### Performance Considerations

- Profile code for bottlenecks
- Use NumPy operations over Python loops
- Consider memory usage for large fields
- Add benchmarks for critical paths

## Areas for Contribution

### High Priority

- [ ] Move experimental components to the core tier by testing them against known answers (symbolic regression on a system with known equations; the VAE on data with known latent structure; the quality checker's thresholds)
- [ ] Re-run the withdrawn BETSE and PhysioNet analyses under the corrected code
- [ ] Improve visualization tools
- [ ] Optimize memory usage for large datasets
- [ ] Add support for 3D field data

### Good First Issues

- [ ] Add unit tests for uncovered functions
- [ ] Improve error messages
- [ ] Add type hints to older code
- [ ] Create example notebooks
- [ ] Fix documentation typos

### Research Contributions

- Propose new analysis methods
- Test on additional biological systems where an answer is known independently
- Improve theoretical foundations
- Contribute experimental data (with proper permissions)

## Review Process

Pull requests are reviewed for:

1. **Correctness**: Does the code work as intended?
2. **Tests**: Are changes adequately tested?
3. **Documentation**: Is the code documented?
4. **Style**: Does it follow project conventions?
5. **Performance**: No significant regressions?
6. **Security**: No security vulnerabilities?

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for general questions
- Check documentation first
- Be patient - maintainers are volunteers

## Recognition

Contributors are recognized in:
- Git history
- Release notes
- Academic publications (for significant contributions)

Thank you for contributing to advancing our understanding of biological memory systems!