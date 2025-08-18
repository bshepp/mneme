# Contributing to Mneme

Thank you for your interest in contributing to Mneme! This document provides guidelines and information for contributors.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/mneme.git`
3. Add upstream remote: `git remote add upstream https://github.com/original/mneme.git`
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

- Write tests for new functionality
- Ensure all tests pass: `pytest`
- Maintain or improve code coverage
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

### Adding New Analysis Methods

When adding new analysis methods:

1. Create module in appropriate subpackage
2. Implement base functionality with clear API
3. Add comprehensive tests
4. Create example notebook
5. Update pipeline integration

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

- [ ] Implement additional IFT reconstruction methods
- [ ] Add more symbolic regression backends
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
- Validate on additional biological systems
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
- CONTRIBUTORS.md file
- Release notes
- Academic publications (for significant contributions)

Thank you for contributing to advancing our understanding of biological memory systems!