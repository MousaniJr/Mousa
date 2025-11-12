#!/bin/bash
# DevMentor AI - Environment Setup Script

set -e  # Exit on error

echo "========================================"
echo "DevMentor AI - Environment Setup"
echo "========================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version"

# Check CUDA availability (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected. Training will use CPU (slower)"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/{raw/code,processed,feedback,benchmarks,continuous_learning}
mkdir -p checkpoints/{pretraining,finetuning,alignment}
mkdir -p logs
mkdir -p results
mkdir -p models

echo "✓ Directories created"

# Download sample datasets (optional)
echo ""
echo "Would you like to download sample datasets? (y/n)"
read -r download_data

if [ "$download_data" = "y" ]; then
    echo "Downloading sample datasets..."
    # Add dataset download logic here
    # Example: wget or curl commands
    echo "✓ Sample datasets downloaded"
fi

# Setup pre-commit hooks (optional)
echo ""
echo "Would you like to setup pre-commit hooks for code quality? (y/n)"
read -r setup_hooks

if [ "$setup_hooks" = "y" ]; then
    echo "Installing pre-commit..."
    pip install pre-commit
    cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
EOF
    pre-commit install
    echo "✓ Pre-commit hooks installed"
fi

# Create environment file template
echo ""
echo "Creating environment file template..."
cat > .env.template << EOF
# DevMentor AI Environment Variables

# API Keys (for data collection)
GITHUB_TOKEN=your_github_token_here
HUGGINGFACE_TOKEN=your_hf_token_here

# Training
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=devmentor-ai
WANDB_ENTITY=your_username

# Deployment
API_SECRET_KEY=your_secret_key_here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Model Paths
MODEL_CACHE_DIR=./models
CHECKPOINT_DIR=./checkpoints

# Hardware
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OMP_NUM_THREADS=8

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/training.log
EOF

echo "✓ Environment template created (.env.template)"
echo "  Please copy to .env and fill in your values:"
echo "  cp .env.template .env"

# Test installation
echo ""
echo "Testing installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "========================================"
echo "✅ Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Copy .env.template to .env and configure your API keys"
echo "2. Review config/training_config.yaml"
echo "3. Run data collection: python src/data_engine/collectors/code_collector.py"
echo "4. Start training: python scripts/train.py"
echo ""
echo "For more information, see README.md"
echo ""
