# Contributing to Mneme

Thank you for your interest in contributing to Mneme! 

For full contributing guidelines, see [CONTRIBUTING.md](https://github.com/bshepp/mneme/blob/main/CONTRIBUTING.md) in the repository root.

## Quick Summary

1. Fork the repository and create a feature branch
2. Follow PEP 8 style with Black formatting (88 char line length)
3. Use [conventional commits](https://www.conventionalcommits.org/) format
4. Write tests for new functionality
5. Update docstrings (NumPy style) for new/modified functions
6. Run quality checks before submitting:

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
pytest
```

## Areas for Contribution

**High Priority:**

- Additional IFT reconstruction methods
- More symbolic regression backends
- Visualization improvements
- Memory optimization for large datasets
- 3D field data support

**Good First Issues:**

- Add unit tests for uncovered functions
- Improve error messages
- Add type hints to older code
- Create example notebooks

See the [full guidelines](https://github.com/bshepp/mneme/blob/main/CONTRIBUTING.md) for detailed information on code style, pull request process, and project-specific guidelines.
