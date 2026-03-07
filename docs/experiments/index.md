# Experiments & Learnings

These experiments document what we learned building LMT -- both positive
results (things that worked) and negative/surprising results.

Full runnable code is available in [`notebooks/experiments.ipynb`](https://github.com/michaelellis003/LMT/blob/dev/notebooks/experiments.ipynb).

## 1. Do Our Models Actually Learn Language?

The most important question: does it *actually work*? Shape tests and gradient
checks are necessary but not sufficient. A model that passes all unit tests but
can't learn "the cat sat on the mat" is a broken model.

We train tiny models (2 layers, 64 embed_dim) on a small structured corpus
using character-level tokenization:

```python
from lmt import ModelConfig, CharTokenizer
from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.models.base import BaseModel

CORPUS = (
    'the cat sat on the mat. '
    'the dog sat on the log. '
) * 20

tokenizer = CharTokenizer.from_text(CORPUS)

config = ModelConfig(
    vocab_size=tokenizer.vocab_size, embed_dim=64,
    num_heads=4, num_kv_heads=4, num_layers=2,
    context_length=32, dropout=0.0,
)

# MHA (GPT-style)
mha = BaseModel(config,
    BlockConfig(attention='mha', ffn='default', norm='rmsnorm'),
    learned_pos_embed=True)

# GQA + SwiGLU (LLaMA-style)
gqa = BaseModel(config,
    BlockConfig(attention='gqa', ffn='swiglu', norm='rmsnorm'),
    learned_pos_embed=True)
```

**Result**: Both models train well below random chance loss within 30 epochs.
MHA and GQA+SwiGLU both learn language structure from this tiny corpus.

## 2. What Makes LLaMA Better Than GPT?

LLaMA differs from GPT-2 in four ways: RMSNorm, SwiGLU, RoPE, and GQA.
We ablated each change independently:

| Configuration      | Final Loss | vs Random |
|--------------------|-----------|-----------|
| GPT baseline       | moderate  | good      |
| + RMSNorm          | ~same     | ~same     |
| + SwiGLU           | **lower** | **best**  |
| + GQA              | ~same     | ~same     |

**Key finding**: **SwiGLU makes the biggest difference**. The gated activation
learns feature selection, not just nonlinearity. RMSNorm vs LayerNorm shows
minimal difference at this scale. GQA matches MHA quality (same expressiveness,
just shared KV heads).

## 3. The SSM-Attention Duality is Real

Mamba-2 (Dao & Gu, 2024) proved that SSMs and attention are literally two
different algorithms for multiplying by the *same matrix* (a semiseparable
matrix).

Our SSD layer implements both forms:

- **Recurrent** (SSM): process tokens sequentially, O(T) time
- **Quadratic** (attention): materialize full T×T matrix, O(T²) time

```python
from lmt.layers.attention.ssd import SSD

ssd = SSD(config)
x = torch.randn(2, 16, 32)

y_recurrent = ssd._forward_recurrent(x)
y_quadratic = ssd._forward_quadratic(x)

# Max difference: ~1e-7 (floating-point precision)
torch.allclose(y_recurrent, y_quadratic, atol=1e-5)  # True!
```

**Result**: The duality is verified. Both forms produce identical outputs.
When alpha=1 (no decay), SSD reduces exactly to causal linear attention,
connecting SSMs, linear attention, and structured matrices.

## 4. The Delta Rule: Gradient Descent Inside a Neural Network

Gated Delta Networks use the delta rule to update an associative memory:

$$S_t = \alpha \cdot S_{t-1} + \beta \cdot (v_t - S_{t-1} k_t) k_t^T$$

The term $(v_t - S_{t-1} k_t)$ is the *error* between what the memory
predicts and reality -- literally one step of gradient descent on
$\|Sk - v\|^2$.

We tested retrieval quality: store N key-value pairs and see how well
they can be recovered.

| d=16, pairs | Linear Attention | Delta Rule | Improvement |
|-------------|-----------------|------------|-------------|
| 4           | low error       | ~zero      | moderate    |
| 16          | moderate error  | ~zero      | large       |
| 64          | high error      | low error  | **huge**    |

**Key insight**: The delta rule advantage grows with the number of stored
pairs. When capacity is exceeded (n_pairs > d), linear attention degrades
badly but the delta rule stays accurate through error correction.

## 5. Negative Result: Linear Attention is Slow in Python

In theory, linear attention (GDN, SSD) is O(n·d²) vs O(n²·d) for
softmax attention. In practice, our Python implementation is **slower**
than MHA at every tested sequence length.

Why?

1. **Python for-loop** over sequence length (no parallel scan)
2. **MHA uses optimized `torch.matmul`** for the full QK^T computation
3. **At seq_len≤128**, n is not >> d, so O(n·d²) > O(n²·d)

The theoretical advantage only manifests with CUDA kernels (parallel scan,
chunkwise algorithms) and long sequences (n >> d).

**Lesson**: Algorithmic complexity ≠ wall-clock time. Implementation
efficiency matters as much as theoretical complexity.

## 6. Scaling Up: TinyStories with BPE

Our first real-scale experiment: training a 30.5M parameter GPT model on
the TinyStories dataset using GPT-2 BPE tokenization (50,257 vocab).

| Epoch | Train Loss | % Below Random |
|-------|-----------|----------------|
| 1     | 5.14      | 52.5%          |
| 2     | 3.60      | 66.8%          |
| 3     | 3.22      | 70.2%          |
| 4     | 2.99      | 72.4%          |
| 5     | 2.83      | 73.9%          |

Random baseline: $\ln(50257) \approx 10.83$. Final loss of 2.83 = perplexity ~17.

**Config**: 256 embed_dim, 8 heads, 6 layers, context length 256, batch
size 32, lr 3e-4, trained on Apple M3 Pro GPU (MPS).

**Key insight**: BPE tokenization makes a **huge** difference compared to
character-level. Each BPE token carries ~3-4 characters of semantic content,
so the model sees more meaning per position. The loss was still decreasing
at epoch 5 -- more training would likely push it lower.

```bash
python examples/train_tinystories.py --embed-dim 256 --num-layers 6 \
    --num-heads 8 --epochs 5 --device mps --use-valid-only
```

## Running the Experiments

Clone the repo and run the notebook:

```bash
git clone https://github.com/michaelellis003/LMT.git
cd LMT
pip install uv && uv sync
jupyter notebook notebooks/experiments.ipynb
```

Or run the training recipes directly:

```bash
# Quick character-level experiment (~2 min)
python examples/train_shakespeare.py --epochs 10 --device cpu

# Compare attention variants (MHA, GQA, GDN, SSD)
python examples/compare_attention.py --epochs 10 --device cpu

# Real-scale BPE training on TinyStories
python examples/train_tinystories.py --epochs 5 --device cpu --use-valid-only
```
