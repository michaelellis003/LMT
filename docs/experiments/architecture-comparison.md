# Architecture Comparison

Does architecture choice matter for language modeling? We tested five
architectures -- GPT, LLaMA, Qwen3, Gemma, and Mixtral -- under controlled
conditions to find out.

## Experiment Design

All models share identical hyperparameters:

| Parameter      | Value |
|----------------|-------|
| Embedding dim  | 64    |
| Layers         | 4     |
| Heads          | 4     |
| KV heads       | 2     |
| FFN hidden dim | 128   |
| Context length | 128   |
| Learning rate  | 1e-3  |
| Training steps | 500   |
| Batch size     | 32    |

The only difference is the **architecture template** used by
`create_model(arch, config)`, which determines:

- **Attention type**: MHA vs GQA
- **FFN type**: GELU vs SwiGLU
- **Normalization**: LayerNorm vs RMSNorm
- **Positional encoding**: Learned vs RoPE

## Experiment 1: Random Data (Control)

**Hypothesis**: On random (incompressible) data, all architectures should
perform identically since there are no patterns to exploit.

**Result**: Confirmed. All five architectures converge to BPB ~8.03,
which is the theoretical optimum for uniform random bytes
($\log_2(256) = 8.0$).

| Architecture | Best BPB | Spread from mean |
|-------------|----------|-----------------|
| LLaMA       | 8.0300   | -0.0012         |
| Qwen3       | 8.0300   | -0.0012         |
| Gemma       | 8.0309   | -0.0003         |
| Mixtral     | 8.0309   | -0.0003         |
| GPT         | 8.0322   | +0.0010         |

**Total spread: 0.0023 BPB (<0.03%)** -- effectively zero.

<div class="exp-chart" data-experiment="arch-comparison-random"></div>

!!! info "Why this matters"
    This is the **control experiment**. It establishes that our training
    infrastructure treats all architectures fairly. Any differences we see
    on real data are genuine architectural effects, not training artifacts.

## Experiment 2: WikiText-2 (Real Language)

**Hypothesis**: On real text with learnable structure, modern architectures
(LLaMA, Qwen3, Gemma) with RoPE and SwiGLU should outperform vanilla GPT.

We use character-level tokenization on WikiText-2 to keep the comparison
simple (no tokenizer effects).

**Result**: Architecture matters -- a lot.

| Architecture | Best BPB | vs GPT   | Final Loss |
|-------------|----------|----------|------------|
| **Mixtral** | **3.014**| **-15.2%** | 2.060  |
| LLaMA       | 3.167    | -10.9%   | 2.171      |
| Qwen3       | 3.167    | -10.9%   | 2.171      |
| Gemma       | 3.376    | -5.1%    | 2.315      |
| GPT         | 3.556    | baseline | 2.446      |

**BPB spread: 0.542 (16.6%)** -- architecture choice causes a >15%
difference in compression quality on real data.

<div class="exp-chart" data-experiment="arch-comparison-wikitext"></div>

### Key Findings

1. **Mixtral wins convincingly** (-15.2% vs GPT). The Mixture-of-Experts
   architecture provides extra capacity without proportional parameter
   increase, allowing better specialization.

2. **LLaMA and Qwen3 are identical**. This makes sense -- Qwen3 uses the
   same building blocks as LLaMA (GQA + SwiGLU + RMSNorm + RoPE). The
   factory creates equivalent models.

3. **Gemma is an intermediate step** (-5.1% vs GPT). It uses RoPE and
   RMSNorm but with GeGLU instead of SwiGLU, and has a different
   embedding/logit structure.

4. **GPT is the weakest**. Learned positional embeddings + LayerNorm +
   GELU + MHA is the baseline -- every modern feature improves on it.

!!! note "Random vs Real Data"
    The contrast between experiments is striking:

    - **Random data**: 0.03% spread (architecture doesn't matter)
    - **Real data**: 16.6% spread (architecture matters a lot)

    This confirms that architectural innovations help models learn
    *structure*, not just memorize. Random data has no structure to exploit.

## Running the Experiments

```bash
# Random data control (fast, ~2 min)
uv run python experiments/arch_comparison.py

# WikiText-2 real data (~5 min)
uv run python experiments/arch_comparison_wikitext.py
```

Results are saved to `experiments/results/` as JSONL registry files.

## Source Code

- [`experiments/arch_comparison.py`](https://github.com/michaelellis003/LMT/blob/main/experiments/arch_comparison.py)
  -- Random data experiment
- [`experiments/arch_comparison_wikitext.py`](https://github.com/michaelellis003/LMT/blob/main/experiments/arch_comparison_wikitext.py)
  -- WikiText-2 experiment
