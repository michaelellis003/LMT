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
from lmt import ModelConfig

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

LMT provides several model architectures:

```python
from lmt import LLaMA, GPT, Mamba

# LLaMA: RMSNorm + RoPE + SwiGLU + GQA
model = LLaMA(config)

# GPT: LayerNorm + learned positional embeddings
model = GPT(config)

# Mamba: Selective State Space Model (no attention!)
model = Mamba(config)

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

### 4. Build Custom Architectures

LMT's `BaseModel` + `BlockConfig` lets you mix any attention, FFN, and
normalization:

```python
from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.models.base import BaseModel

# GQA + SwiGLU + RMSNorm (LLaMA-style)
block = BlockConfig(attention='gqa', ffn='swiglu', norm='rmsnorm')
model = BaseModel(config, block_config=block)

# Gated Delta Networks + SwiGLU (linear attention)
block = BlockConfig(attention='gated_delta_net', ffn='swiglu', norm='rmsnorm')
model = BaseModel(config, block_config=block)

# SSD + SwiGLU (Mamba-2 style duality)
block = BlockConfig(attention='ssd', ffn='swiglu', norm='rmsnorm')
model = BaseModel(config, block_config=block)
```

### 5. Use Individual Layers

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

## Train a Model

LMT includes a `CharTokenizer` for quick experiments -- no external
dependencies needed:

```python
from lmt import ModelConfig, LLaMA, CharTokenizer
from lmt.training.datasets import GPTDataset
from torch.utils.data import DataLoader
import torch

# Prepare data
text = open('your_text.txt').read()
tokenizer = CharTokenizer.from_text(text)

config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128, num_heads=4, num_layers=4,
    context_length=128, dropout=0.1,
)
model = LLaMA(config)

# Create dataset and loader
dataset = GPTDataset(text, tokenizer, config.context_length, stride=64)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for inp, tgt in loader:
    logits = model(inp)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), tgt.flatten()
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

For a complete training recipe with data download, evaluation, and text
generation, see `examples/train_shakespeare.py`.

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
src/lmt/
  layers/
    attention/     # MHA, GQA, Sliding Window, MLA, Flash, GDN, SSD
    blocks/        # ConfigurableBlock + BlockConfig
    ffn/           # SwiGLU, MoE (with shared experts)
    normalization/ # RMSNorm
    positional/    # RoPE, YaRN, ALiBi
  models/
    base/          # BaseModel (composable architecture)
    gpt/           # GPT (learned pos + MHA)
    llama/         # LLaMA (RoPE + GQA + SwiGLU + RMSNorm)
    mixtral/       # Mixtral (LLaMA + MoE + sliding window)
    mamba/         # Mamba (Selective SSM, no attention)
  training/        # Trainer, configs, dataloaders, loss functions
  tokenizer/       # CharTokenizer, BPETokenizer, NaiveTokenizer
  generate.py      # Text generation with temperature + top-k sampling
```
