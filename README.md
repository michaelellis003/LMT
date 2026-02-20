# Language Modeling using Transformers (LMT)

[![CI](https://github.com/michaelellis003/LMT/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/michaelellis003/LMT/actions/workflows/ci-cd.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A PyTorch implementation of transformer-based language models including GPT architecture for pretraining and fine-tuning. This project is designed for educational and research purposes to help users understand how the attention mechanism and Transformer architecture work in Large Language Models (LLMs).

## 🚀 Features

- **GPT Architecture**: Complete implementation of decoder-only transformer models
- **Attention Mechanisms**: Multi-head self-attention with causal masking
- **Tokenization**: Multiple tokenizer implementations (BPE, Naive)
- **Training Pipeline**: Comprehensive trainer with pretraining and fine-tuning support
- **Educational Focus**: Well-documented code for learning transformer internals
- **Modern Stack**: Built with PyTorch 2.7+, Python 3.11+

## 📦 Installation

### Prerequisites

- Python 3.11 or 3.12
- PyTorch 2.7+

### Install from PyPI

```bash
pip install language-modeling-transformers
```

### Install from GitHub

```bash
pip install git+https://github.com/michaelellis003/LMT.git
```

## 🏃‍♂️ Quick Start

### Basic Model Usage

```python
from lmt import GPT, ModelConfig
from lmt.models.config import ModelConfigPresets
import torch

# Create a small GPT model
config = ModelConfigPresets.small_gpt()
model = GPT(config)

# Generate some text
input_ids = torch.randint(0, config.vocab_size, (1, 10))
with torch.no_grad():
    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")  # (1, 10, vocab_size)
```

### Training a Model

```python
from lmt import Trainer, GPT
from lmt.training import BaseTrainingConfig
from lmt.models.config import ModelConfigPresets

# Configure model and training
model_config = ModelConfigPresets.small_gpt()
training_config = BaseTrainingConfig(
    num_epochs=10,
    batch_size=4,
    learning_rate=1e-4
)

# Initialize model and trainer
model = GPT(model_config)
trainer = Trainer(
    model=model,
    train_loader=your_train_loader,
    val_loader=your_val_loader,
    config=training_config
)

# Start training
trainer.train()
```

### Using the Training Script

```bash
# Pretraining
python scripts/train.py --task pretraining --num_epochs 20 --batch_size 4

# Classification fine-tuning
python scripts/train.py --task classification --download_model --learning_rate 1e-5
```

## 📚 Documentation

### Model Components

- **GPT**: Main model class implementing decoder-only transformer
- **TransformerBlock**: Individual transformer layer with attention and feed-forward
- **MultiHeadAttention**: Multi-head self-attention mechanism
- **CausalAttention**: Attention with causal masking for autoregressive generation

### Tokenizers

- **BPETokenizer**: Byte-Pair Encoding tokenizer
- **NaiveTokenizer**: Simple character-level tokenizer
- **BaseTokenizer**: Abstract base class for custom tokenizers

### Training

- **Trainer**: Main training orchestrator with support for pretraining and fine-tuning
- **BaseTrainingConfig**: Configuration class for training parameters
- **Custom datasets and dataloaders**: Support for various text datasets

## 🗂️ Project Structure

```
src/lmt/
├── __init__.py              # Main package exports
├── models/                  # Model architectures
│   ├── gpt/                # GPT implementation
│   ├── config.py           # Model configuration
│   └── utils.py            # Model utilities
├── layers/                  # Neural network layers
│   ├── attention/          # Attention mechanisms
│   └── transformers/       # Transformer blocks
├── tokenizer/              # Tokenization implementations
├── training/               # Training pipeline
└── generate.py             # Text generation utilities

scripts/
├── train.py                # Main training script
└── utils.py                # Training utilities

tests/                      # Comprehensive test suite
notebooks/                  # Educational Jupyter notebooks
docs/                       # Sphinx documentation
```

## 📊 Examples and Notebooks

Explore the interactive notebooks in the `notebooks/` directory:

- `attention.ipynb`: Understanding attention mechanisms
- `pretraining_gpt.ipynb`: GPT pretraining walkthrough
- `tokenizer.ipynb`: Tokenization techniques

## 🔧 Configuration

### Model Configuration

```python
from lmt.models.config import ModelConfig

config = ModelConfig(
    vocab_size=50257,
    embed_dim=768,
    context_length=1024,
    num_layers=12,
    num_heads=12,
    dropout=0.1
)
```

### Training Configuration

```python
from lmt.training.config import BaseTrainingConfig

training_config = BaseTrainingConfig(
    num_epochs=10,
    batch_size=8,
    learning_rate=3e-4,
    weight_decay=0.1,
    print_every=100,
    eval_every=500
)
```

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
