# Experiments & Results

These experiments document what we learned building LMT -- both positive
results and negative/surprising results. Each experiment includes
runnable code, reproducible results, and analysis.

## Architecture Studies

### [Architecture Comparison](architecture-comparison.md)

Does architecture choice matter? We compared GPT, LLaMA, Qwen3, Gemma,
and Mixtral under controlled conditions on both random and real data.

**Key finding**: On random data, all architectures are identical (0.03%
spread). On WikiText-2, architecture matters -- Mixtral leads at 3.01 BPB,
while GPT trails at 3.56 BPB (16.6% spread).

### [Architecture Ablation](architecture-ablation.md)

LLaMA beats GPT by ~11% on WikiText-2. Which of its four innovations
(RoPE, SwiGLU, GQA, RMSNorm) drive the improvement? We isolate each
feature using the `ConfigurableBlock` system.

## Pretraining Recipes

### [BabyLM Pretraining](babylm-pretraining.md)

End-to-end pretraining pipeline for the BabyLM Challenge: data loading,
BPE tokenizer training, model creation, training, and BPB evaluation.
Achieves 76% improvement over random with just 50K words and a 1M
parameter model.

## Classic Experiments

### Do Our Models Actually Learn Language?

The most important question: does it *actually work*? We train tiny models
(2 layers, 64 embed_dim) on a small structured corpus using character-level
tokenization.

**Result**: Both MHA (GPT-style) and GQA+SwiGLU (LLaMA-style) train well
below random chance loss within 30 epochs.

### The SSM-Attention Duality

Mamba-2 (Dao & Gu, 2024) proved that SSMs and attention are two different
algorithms for multiplying by the same matrix (a semiseparable matrix).
Our SSD layer implements both forms and verifies they produce identical
outputs (max difference ~1e-7).

### The Delta Rule: Gradient Descent Inside a Neural Network

Gated Delta Networks use the delta rule to update an associative memory.
We tested retrieval quality and found the delta rule advantage grows with
the number of stored pairs -- when capacity is exceeded, linear attention
degrades but the delta rule stays accurate through error correction.

### Negative Result: Linear Attention is Slow in Python

In theory, linear attention is O(n*d^2) vs O(n^2*d) for softmax. In
practice, our Python implementation is slower at every tested length due
to sequential for-loops vs optimized `torch.matmul`.

**Lesson**: Algorithmic complexity != wall-clock time.

### Scaling Up: TinyStories with BPE

A 30.5M parameter GPT trained on TinyStories with GPT-2 BPE tokenization
reaches loss 2.83 (perplexity ~17) after 5 epochs -- 74% below random
baseline. BPE tokenization makes a huge difference vs character-level.

## Running Experiments

```bash
# Architecture comparison (random data, ~2 min)
uv run python experiments/arch_comparison.py

# Architecture comparison (WikiText-2, ~5 min)
uv run python experiments/arch_comparison_wikitext.py

# Architecture ablation (~10 min)
uv run python experiments/arch_ablation.py

# BabyLM pretraining (quick test, ~2 min)
uv run python recipes/babylm_pretrain.py --max-words 100000
```

All results are saved to `experiments/results/` as JSONL registry files
for reproducibility.
