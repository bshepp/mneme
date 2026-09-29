#!/bin/bash
# Setup development environment for Mneme

set -e  # Exit on error

echo "Setting up Mneme development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
echo "✓ Core requirements installed"

echo "Installing development requirements..."
pip install -r pyproject.toml
echo "✓ Development requirements installed"

# Install package in development mode
echo "Installing Mneme in development mode..."
pip install -e .
echo "✓ Mneme installed"

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install
echo "✓ Pre-commit hooks installed"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/{raw,processed,synthetic}
mkdir -p experiments/{configs,results}
mkdir -p logs
mkdir -p cache
echo "✓ Directories created"

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created (please update with your values)"
else
    echo "✓ .env file already exists"
fi

# Install Jupyter kernel
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name mneme --display-name "Mneme"
echo "✓ Jupyter kernel installed"

# Run validation
echo "Running installation validation..."
python -c "import mneme; print(f'✓ Mneme {mneme.__version__} successfully imported')"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Update .env file with your configuration"
echo "3. Run tests: pytest"
echo "4. Start Jupyter: jupyter notebook"
echo ""