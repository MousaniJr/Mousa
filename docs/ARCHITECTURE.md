# DevMentor AI - System Architecture

## Table of Contents
1. [Overview](#overview)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Model Architecture](#model-architecture)
5. [Training Pipeline](#training-pipeline)
6. [Deployment Architecture](#deployment-architecture)
7. [Scalability & Performance](#scalability--performance)

---

## Overview

DevMentor AI is built on a modular, pipeline-based architecture that separates concerns across data processing, training, evaluation, and deployment. The system is designed for:

- **Modularity**: Each component can be developed, tested, and scaled independently
- **Extensibility**: Easy integration of new data sources, training methods, or evaluation metrics
- **Observability**: Comprehensive logging, metrics, and monitoring at every stage
- **Safety**: Built-in guardrails, validation, and constitutional alignment

### Design Philosophy

```
Data Quality > Model Size > Training Time > Inference Speed
```

We prioritize high-quality, diverse training data over massive parameter counts, ensuring the model learns correct patterns rather than memorizing noise.

---

## System Components

### 1. Data Engine

**Purpose**: Collect, validate, and prepare training data from ethical, open-source origins.

#### Sub-components:

**a. Collectors** (`src/data_engine/collectors/`)
- `code_collector.py` - Scrapes open-source repositories (GitHub, GitLab, Bitbucket)
- `documentation_collector.py` - Collects official language docs, API references
- `qa_collector.py` - Gathers Stack Overflow, Reddit programming threads
- `tutorial_collector.py` - Curates educational content, blog posts

**b. Processors** (`src/data_engine/processors/`)
- `deduplicator.py` - Removes exact and near-duplicate code samples
- `tokenizer.py` - Custom tokenizer optimized for code (preserves indentation, special chars)
- `augmenter.py` - Creates variations (refactoring, style changes) for robustness
- `quality_filter.py` - Filters low-quality, obfuscated, or malicious code

**c. Validators** (`src/data_engine/validators/`)
- `syntax_validator.py` - Ensures syntactic correctness across languages
- `license_validator.py` - Checks license compatibility (MIT, Apache, BSD, etc.)
- `security_validator.py` - Scans for hardcoded secrets, vulnerabilities
- `bias_detector.py` - Identifies potentially problematic content

#### Data Flow:
```
Raw Sources → Collectors → Validators → Processors → Storage
                               ↓
                         (Rejected data logged)
```

#### Storage Format:
- **Format**: Parquet files for efficient columnar access
- **Schema**:
  ```python
  {
    "id": str,              # Unique identifier
    "content": str,         # Code or text content
    "language": str,        # Programming language
    "source": str,          # Origin (repo, docs, etc.)
    "license": str,         # License type
    "quality_score": float, # 0.0-1.0 quality rating
    "metadata": dict        # Additional context
  }
  ```

---

### 2. Training Engine

**Purpose**: Multi-stage training pipeline from pre-training to aligned production model.

#### Stage 1: Pre-training (`src/training/pretraining/`)

**Objective**: Learn general programming patterns, syntax, and semantic understanding.

**Architecture**: Transformer-based decoder-only model (similar to GPT architecture)
- **Model Size Options**:
  - Small: 350M parameters (development/testing)
  - Medium: 1.3B parameters (production baseline)
  - Large: 7B parameters (high-performance)
  - XL: 13B+ parameters (research-grade)

**Training Configuration**:
```yaml
model:
  hidden_size: 2048
  num_layers: 24
  num_heads: 16
  intermediate_size: 8192
  vocab_size: 50000  # Code-optimized vocabulary
  max_position_embeddings: 4096
  dropout: 0.1

training:
  batch_size: 512  # Global batch size
  micro_batch_size: 4  # Per-device batch size
  gradient_accumulation_steps: 128
  learning_rate: 3e-4
  warmup_steps: 2000
  total_steps: 500000
  weight_decay: 0.1
  gradient_clipping: 1.0
```

**Optimization**:
- **Mixed Precision**: BF16 for stability with large models
- **Gradient Checkpointing**: Reduces memory usage by 40%
- **Flash Attention**: 2-3x speedup for attention computation
- **ZeRO Stage 3**: Distributed optimizer state sharding

**Loss Function**: Causal language modeling (next-token prediction)
```python
loss = -log P(token_t | tokens_0..t-1)
```

#### Stage 2: Domain Fine-tuning (`src/training/finetuning/`)

**Objective**: Specialize model for software development tasks.

**Datasets**:
1. **Code Completion**: Partial functions → complete implementation
2. **Bug Fixing**: Buggy code → corrected version
3. **Refactoring**: Legacy code → modernized version
4. **Documentation**: Code → explanatory docstrings
5. **Code Review**: PR diffs → review comments

**Training Approach**:
- **Supervised Fine-tuning (SFT)**: Learn from high-quality examples
- **Instruction Tuning**: Format as instruction-response pairs
  ```
  Instruction: "Refactor this Python function to use list comprehension"
  Input: <code>
  Output: <refactored_code>
  ```

**Hyperparameters**:
```yaml
learning_rate: 1e-5  # Lower than pre-training
epochs: 3-5
batch_size: 128
evaluation_steps: 500
save_steps: 1000
```

#### Stage 3: Constitutional Alignment (`src/training/alignment/`)

**Objective**: Align model behavior with developer needs and safety principles.

**Method 1: Principle-Based Self-Critique**
1. Model generates response
2. Model critiques its own response against constitutional principles
3. Model revises response based on critique
4. Supervised training on (original → revised) pairs

**Method 2: Reinforcement Learning from Human Feedback (RLHF)**
1. **Reward Model Training**:
   - Collect comparison data: (prompt, response_A, response_B, preference)
   - Train reward model to predict human preferences

2. **PPO Training**:
   - Use reward model to score generated responses
   - Optimize policy via Proximal Policy Optimization
   - Add KL penalty to prevent drift from fine-tuned model

**Constitutional Principles** (see [CONSTITUTION.md](CONSTITUTION.md)):
- Helpfulness, clarity, security, best practices, etc.

**Training Configuration**:
```yaml
rlhf:
  reward_model:
    hidden_size: 1024
    num_layers: 12
  ppo:
    learning_rate: 1e-6
    kl_penalty: 0.1
    epochs: 1
    batch_size: 64
```

---

### 3. Evaluation Engine

**Purpose**: Rigorously test model capabilities and safety.

#### Benchmarks (`src/evaluation/benchmarks/`)

**Code Generation**:
- **HumanEval**: 164 programming problems with unit tests
- **MBPP**: 1000+ Python programming problems
- **CodeContests**: Competition-level problems
- **Custom DevTasks**: Real-world development scenarios

**Code Understanding**:
- **CodeSearchNet**: Code search and documentation
- **Bug Detection**: Identify bugs in code samples
- **Code Summarization**: Generate accurate descriptions

**Reasoning**:
- **APPS**: 10,000 coding problems with difficulty levels
- **Codeforces**: Competitive programming challenges

**Metrics**:
- **Pass@k**: Percentage of problems solved in k attempts
- **BLEU/CodeBLEU**: Similarity to reference solutions
- **Cyclomatic Complexity**: Quality of generated code
- **Execution Success Rate**: Functional correctness

#### Safety Tests (`src/evaluation/safety_tests/`)

**Security**:
- SQL injection patterns
- XSS vulnerability detection
- Hardcoded credential checks
- Insecure crypto usage

**Bias & Fairness**:
- Gender/race bias in variable names
- Inclusive language checks
- Stereotype detection

**Adversarial Robustness**:
- Prompt injection resistance
- Jailbreak attempt detection
- Malicious code generation prevention

**Evaluation Pipeline**:
```python
# Automated evaluation runs after each training checkpoint
python src/evaluation/run_benchmarks.py \
  --model-path checkpoints/step-10000 \
  --benchmarks humaneval,mbpp,safety \
  --output results/step-10000.json
```

---

### 4. Deployment Stack

#### Inference Engine (`src/deployment/inference/`)

**Optimization Techniques**:
- **Quantization**: INT8/INT4 for faster inference (via GPTQ, AWQ)
- **KV-Cache Optimization**: Reduces redundant computation
- **Batching**: Dynamic batching for throughput
- **Speculative Decoding**: 2-3x speedup for long generation

**Deployment Modes**:
1. **Local CLI**: Interactive terminal interface
2. **API Server**: RESTful API with OpenAI-compatible endpoints
3. **Edge Deployment**: ONNX export for low-latency serving
4. **IDE Plugins**: VS Code, IntelliJ, Vim integrations

**Example API Endpoint**:
```python
POST /v1/completions
{
  "prompt": "def fibonacci(n):",
  "max_tokens": 256,
  "temperature": 0.7,
  "stop": ["\n\n"]
}
```

#### API Server (`src/deployment/api/`)

**Framework**: FastAPI for high-performance async handling

**Features**:
- Rate limiting
- API key authentication
- Request logging
- Prometheus metrics
- Health checks

**Scalability**:
- Horizontal scaling via load balancer
- Model replica management
- Auto-scaling based on queue depth

---

### 5. Monitoring & Governance

#### Metrics (`src/monitoring/metrics/`)

**Training Metrics**:
- Loss curves (training & validation)
- Learning rate schedule
- Gradient norms
- Hardware utilization (GPU, memory)

**Inference Metrics**:
- Latency (p50, p95, p99)
- Throughput (tokens/second)
- Error rates
- User satisfaction scores

**Logging**:
- Structured JSON logs
- Distributed tracing (Jaeger)
- Experiment tracking (Weights & Biases, MLflow)

#### Governance (`src/monitoring/governance/`)

**Compliance Auditing**:
- Data lineage tracking
- Model version control
- Access control logs
- Incident response procedures

**Safety Monitoring**:
- Real-time output filtering
- Anomaly detection
- User report handling
- Model behavior drift detection

---

## Data Flow

### End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                        Data Collection                        │
│  GitHub, GitLab → Documentation → Stack Overflow → Blogs     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      Data Validation                          │
│    License Check → Security Scan → Quality Filter            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                     Data Preprocessing                        │
│  Deduplication → Tokenization → Augmentation → Storage       │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                       Pre-training                            │
│  Next-Token Prediction on Massive Code Corpus                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      Fine-tuning                              │
│  Task-Specific Training on Curated Datasets                  │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   Constitutional Alignment                    │
│  Self-Critique + RLHF → Safety & Helpfulness                 │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                       Evaluation                              │
│  Benchmarks → Safety Tests → Human Review                    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                       Deployment                              │
│  Optimization → Serving → Monitoring                          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    Continuous Learning                        │
│  Feedback Collection → Validation → Retraining               │
└──────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

### Transformer Configuration

**Base Architecture**: Decoder-only Transformer with causal attention

```
Input Embeddings (Token + Position)
         ↓
[Transformer Block] × N
  - Multi-Head Self-Attention (causal)
  - Layer Normalization
  - Feed-Forward Network (SwiGLU activation)
  - Residual Connections
         ↓
Final Layer Norm
         ↓
Language Modeling Head
         ↓
Output Logits (Vocabulary Distribution)
```

**Key Modifications for Code**:

1. **Extended Vocabulary**:
   - Standard words: 30k tokens
   - Code-specific tokens: 15k tokens (identifiers, operators, keywords)
   - Special tokens: 5k tokens (indentation levels, brackets, etc.)

2. **Positional Encoding**:
   - Rotary Position Embeddings (RoPE) for better length generalization
   - Supports up to 16k context length with RoPE scaling

3. **Attention Pattern**:
   - Standard causal attention for autoregressive generation
   - Optional: Sliding window attention for very long contexts

4. **Activation Function**:
   - SwiGLU for better gradient flow and performance

**Parameter Count Breakdown** (7B model example):
```
Embedding Layer: 100M parameters
24 Transformer Layers: 6.8B parameters
Output Head: 100M parameters
Total: ~7B parameters
```

---

## Training Pipeline

### Distributed Training Strategy

**Framework**: PyTorch + DeepSpeed/FSDP

**Configuration for 8x A100 GPUs**:
```yaml
distributed:
  backend: nccl
  zero_optimization:
    stage: 3  # Shard optimizer states, gradients, and parameters
    offload_optimizer:
      device: cpu
      pin_memory: true
  gradient_accumulation_steps: 16
  fp16:
    enabled: false
  bf16:
    enabled: true
```

**Training Timeline** (7B model):
- **Pre-training**: 500k steps × 8 GPUs ≈ 3-4 weeks
- **Fine-tuning**: 10k steps ≈ 3-5 days
- **Alignment**: 5k steps ≈ 2-3 days
- **Total**: ~5-6 weeks for complete training

**Checkpoint Strategy**:
- Save every 1000 steps
- Keep last 5 checkpoints
- Best checkpoint based on validation loss
- Separate evaluation checkpoints

---

## Deployment Architecture

### Production Infrastructure

```
┌─────────────────────────────────────────────────────────────┐
│                       Load Balancer                          │
│                     (Nginx/HAProxy)                          │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│ API    │ │ API    │ │ API    │
│ Server │ │ Server │ │ Server │
│   1    │ │   2    │ │   N    │
└───┬────┘ └───┬────┘ └───┬────┘
    │          │          │
    └──────────┼──────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌──────────┐         ┌──────────┐
│  Model   │         │  Model   │
│ Instance │         │ Instance │
│  (GPU)   │         │  (GPU)   │
└──────────┘         └──────────┘
```

**Auto-Scaling Policy**:
- Scale up when average queue time > 5s
- Scale down when GPU utilization < 30%
- Min replicas: 2, Max replicas: 10

---

## Scalability & Performance

### Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference Latency (p95) | < 2s | TBD |
| Throughput | 100 req/s | TBD |
| Code Pass@1 (HumanEval) | > 50% | TBD |
| MBPP Pass@1 | > 60% | TBD |
| Uptime | 99.9% | TBD |

### Optimization Roadmap

**Phase 1: Baseline**
- Standard FP16 inference
- Single GPU per request

**Phase 2: Optimization**
- INT8 quantization (1.5-2x speedup)
- KV-cache optimization
- Dynamic batching

**Phase 3: Advanced**
- Speculative decoding
- Model distillation (13B → 7B)
- Custom CUDA kernels

---

## Security Considerations

### Threat Model

**Potential Threats**:
1. Prompt injection attacks
2. Data poisoning during training
3. Model extraction via API
4. PII leakage from training data
5. Malicious code generation

**Mitigations**:
1. Input sanitization and validation
2. Rigorous data validation pipeline
3. Rate limiting, output watermarking
4. PII detection and removal
5. Safety classifiers, output filtering

### Access Control

- API key authentication
- Role-based access control (RBAC)
- Audit logging for all requests
- Encrypted communication (TLS 1.3)

---

## Future Enhancements

1. **Multi-modal Support**: Code + diagrams, UI screenshots → code generation
2. **Agent Capabilities**: Multi-step planning, tool usage, debugging
3. **Personalization**: User-specific coding style adaptation
4. **Collaborative Features**: Multi-user code sessions
5. **Hardware Acceleration**: TPU support, custom ASICs

---

**Document Version**: 1.0
**Last Updated**: 2025-11-12
**Authors**: DevMentor AI Team
