# DevMentor AI - Quick Start Guide

Get up and running with DevMentor AI in 15 minutes.

## Prerequisites

- **Python 3.8+**
- **100GB+ free disk space** (for datasets and checkpoints)
- **CUDA-capable GPU** (recommended: 8x A100 80GB for full training)
  - Minimum: 1x GPU with 16GB VRAM for development/testing
  - CPU-only mode supported but much slower

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/devmentor-ai.git
cd devmentor-ai

# Run setup script
bash scripts/setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.template .env

# Edit .env and add your API keys
nano .env
```

Required environment variables:
- `GITHUB_TOKEN` - For collecting code from GitHub
- `WANDB_API_KEY` - For experiment tracking (optional)

## Quick Training (Development Mode)

For testing the pipeline without full-scale training:

### Step 1: Collect Sample Data

```bash
# Collect a small sample of Python code
python src/data_engine/collectors/code_collector.py
```

### Step 2: Preprocess Data

```bash
# Clean and validate collected data
python src/data_engine/processors/preprocess.py
```

### Step 3: Train Small Model

```bash
# Train a small model (350M parameters)
python scripts/train.py --config config/training_config.yaml --stage all
```

This will run through all three training stages:
1. Pre-training (~1-2 hours on single GPU)
2. Fine-tuning (~30 minutes)
3. Constitutional alignment (~20 minutes)

## Production Training

For full-scale production training:

### 1. Data Collection

Collect large-scale code datasets:

```bash
# Configure data collection
# Edit config/data_collection.yaml

# Run collectors for all supported languages
python src/data_engine/collectors/code_collector.py --languages all --max-repos 10000
```

Expected: ~500GB of raw code data

### 2. Data Processing

```bash
# Run full preprocessing pipeline
python src/data_engine/processors/preprocess.py --input data/raw/code --output data/processed

# Validate data quality
python src/data_engine/validators/security_validator.py --input data/processed
```

### 3. Pre-training

```bash
# Start distributed pre-training (8x GPUs)
python scripts/train.py --stage pretraining --config config/training_config.yaml
```

Expected duration: 3-4 weeks on 8x A100 GPUs

Monitor training:
```bash
# View logs
tail -f logs/training.log

# Open TensorBoard
tensorboard --logdir runs/

# View in Weights & Biases
# Check your W&B dashboard
```

### 4. Fine-tuning

```bash
# Fine-tune on development tasks
python scripts/train.py --stage finetuning --config config/training_config.yaml
```

Expected duration: 3-5 days

### 5. Constitutional Alignment

```bash
# Apply constitutional principles and RLHF
python scripts/train.py --stage alignment --config config/training_config.yaml
```

Expected duration: 5-7 days

## Running Inference

### Local CLI

```bash
# Interactive CLI
python src/deployment/inference/cli.py --model-path checkpoints/alignment/aligned_model.pt
```

### API Server

```bash
# Start FastAPI server
python src/deployment/inference/server.py --model-path checkpoints/alignment/aligned_model.pt --port 8000
```

Access API:
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/

### Example API Usage

```python
import requests

# Completion request
response = requests.post(
    "http://localhost:8000/v1/completions",
    headers={"Authorization": "Bearer dev-key-123"},
    json={
        "prompt": "def fibonacci(n):",
        "max_tokens": 256,
        "temperature": 0.7
    }
)

print(response.json())
```

## Evaluation

### Run Benchmarks

```bash
# Evaluate on HumanEval
python src/evaluation/benchmarks/code_evaluation.py --model-path checkpoints/alignment/aligned_model.pt
```

### Expected Performance Targets

| Benchmark | Target | Production |
|-----------|--------|------------|
| HumanEval Pass@1 | > 50% | 65%+ |
| MBPP Pass@1 | > 60% | 70%+ |
| Code Quality | > 0.8 | 0.85+ |

## Continuous Learning

Enable continuous learning to improve from user feedback:

```bash
# Start continuous learning pipeline
python src/training/continuous_learning.py
```

This will:
1. Collect user feedback
2. Validate quality
3. Retrain weekly with new data
4. Deploy improved model

## Common Issues

### Out of Memory (OOM)

```bash
# Reduce batch size in config
micro_batch_size: 2  # Instead of 4

# Enable gradient checkpointing
gradient_checkpointing: true

# Use smaller model size
model.size: "small"
```

### Slow Training

```bash
# Enable mixed precision
mixed_precision: "bf16"

# Increase number of workers
num_workers: 8

# Use Flash Attention
use_flash_attention: true
```

### Data Collection Timeout

```bash
# Increase timeout in code_collector.py
timeout: 30  # seconds

# Use fewer max repos
max_repos: 1000
```

## Next Steps

1. **Read Full Documentation**: See `docs/ARCHITECTURE.md` for system details
2. **Review Constitution**: Understand alignment principles in `docs/CONSTITUTION.md`
3. **Customize Configuration**: Edit `config/training_config.yaml` for your needs
4. **Join Community**: (Add your community links)

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions on GitHub Discussions

## Monitoring Training

### Weights & Biases

```bash
# View experiments
wandb login
# Then visit: https://wandb.ai/your-username/devmentor-ai
```

### TensorBoard

```bash
tensorboard --logdir runs/
# Visit: http://localhost:6006
```

### System Metrics

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Disk usage
df -h

# Memory usage
free -h
```

## Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t devmentor-ai:latest .

# Run container
docker run -p 8000:8000 --gpus all devmentor-ai:latest
```

### Kubernetes Deployment

```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods
```

---

**Ready to build the future of developer assistance! 🚀**

For detailed information, see the full [README.md](README.md) and [docs/](docs/) directory.
