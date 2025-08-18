"""Setup script for Mneme package."""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read development requirements
with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
    dev_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mneme",
    version="0.1.0",
    author="Mneme Development Team",
    author_email="mneme@example.com",
    description="Detecting field-like memory structures in biological systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bshepp/mneme",
    project_urls={
        "Bug Tracker": "https://github.com/bshepp/mneme/issues",
        "Source Code": "https://github.com/bshepp/mneme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "nbsphinx>=0.8.0",
        ],
        "gpu": [
            "cupy>=9.0.0",  # GPU acceleration
        ],
    },
    entry_points={
        "console_scripts": [
            "mneme=mneme.cli:main",
            # Temporary CLI wrappers to top-level scripts until moved under package
            "mneme-generate=mneme.cli:main",  # placeholder
            "mneme-pipeline=mneme.cli:main",  # placeholder
            "mneme-visualize=mneme.cli:main",  # placeholder
        ],
    },
    include_package_data=True,
    package_data={
        "mneme": [
            "config/*.yaml",
            "data/examples/*.npz",
        ],
    },
    zip_safe=False,
)