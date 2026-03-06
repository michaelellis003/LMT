# Getting Started

## Installation

### From PyPI

```bash
pip install pylmt
```

### From Source (for development)

```bash
git clone https://github.com/michaelellis003/LMT.git
cd LMT
pip install uv
uv sync
```

## Your First Model

### 1. Define a Configuration

Every model in LMT starts with a `ModelConfig`:

```python
from lmt.models.config import ModelConfig

config = ModelConfig(
    vocab_size=32000,
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    context_length=512,
    dropout=0.1,
)
```

### 2. Build a Model

```python
from lmt.models.llama import LLaMA

model = LLaMA(config)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
```

### 3. Run a Forward Pass

```python
import torch

x = torch.randint(0, config.vocab_size, (1, 64))
logits = model(x)
print(f'Input: {x.shape}')   # [1, 64]
print(f'Output: {logits.shape}')  # [1, 64, 32000]
```

### 4. Use Individual Layers

LMT's layers are designed to work independently:

```python
from lmt.layers.attention import GroupedQueryAttention
from lmt.layers.ffn import SwiGLU
from lmt.layers.normalization import RMSNorm

# Each layer works on [batch, seq_len, d_model] tensors
norm = RMSNorm(d_model=256)
attn = GroupedQueryAttention(config)
ffn = SwiGLU(d_model=256)

x = torch.randn(1, 64, 256)
x = x + attn(norm(x))  # Pre-norm attention
x = x + ffn(norm(x))   # Pre-norm FFN
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
src/lmt/
  layers/
    attention/     # MHA, GQA, Sliding Window, MLA
    ffn/           # SwiGLU, MoE
    normalization/ # RMSNorm
    positional/    # RoPE
  models/
    gpt/           # Original GPT architecture
    llama/         # LLaMA (RMSNorm + RoPE + SwiGLU + GQA)
    mixtral/       # Mixtral (LLaMA + MoE + sliding window)
  training/        # Trainer, configs, dataloaders
  tokenizer/       # BPE, naive tokenizers
```
