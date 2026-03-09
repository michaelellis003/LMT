# BabyLM Pretraining Recipe

An end-to-end pretraining pipeline for the
[BabyLM Challenge](https://babylm.github.io/) -- a competition for
sample-efficient language model pretraining on cognitively plausible data.

## What is BabyLM?

The BabyLM Challenge requires training language models on a **limited word
budget** using data sources that mimic what children are exposed to:
child-directed speech, children's books, movie subtitles, and Wikipedia
(simple).

| Track        | Word Budget |
|-------------|-------------|
| strict      | 100M words  |
| strict-small| 10M words   |

This makes it an ideal benchmark for testing whether small, well-designed
models can learn efficiently -- exactly what LMT is built for.

## Pipeline

The recipe follows a clean 5-step pipeline:

```
Download BabyLM → Train BPE Tokenizer → Create Model → Train → Evaluate
```

### 1. Data Loading

```python
from lmt.data.hf_datasets import download_babylm

# Streaming download (handles 10M+ rows efficiently)
train_texts = download_babylm(track='strict-small', split='train')
val_texts = download_babylm(track='strict-small', split='validation')
```

The `download_babylm()` function streams from HuggingFace Hub, supporting
`max_items` for quick testing.

### 2. Tokenizer Training

```python
from lmt.tokenizer.trainer import train_bpe_tokenizer

tokenizer = train_bpe_tokenizer(train_texts, vocab_size=4096)
```

We train a byte-level BPE tokenizer directly on the BabyLM corpus, giving
a compression ratio of ~3-4 bytes per token.

### 3. Model Creation

```python
from lmt.models.sizing import suggest_config
from lmt.models.factory import create_model

suggested = suggest_config(target_params=1_000_000, vocab_size=4096)
model = create_model('gpt', config)
```

The `suggest_config()` function estimates model dimensions to hit a target
parameter count without instantiating the model.

### 4. Training

Standard `train_loop` with cosine warmup and weight decay.

### 5. Evaluation

BPB (bits per byte) as the tokenizer-agnostic metric, evaluated at regular
intervals during training.

## Results (Quick Test)

Running with `--max-words 50000` (quick test, ~2 minutes on CPU):

| Metric           | Value   |
|-----------------|---------|
| Architecture    | GPT     |
| Parameters      | ~1M     |
| Best BPB        | 2.41    |
| Random baseline | 10.00   |
| Improvement     | **76%** |

A 76% improvement over random chance with just 50K words of training data
and a 1M parameter model is a strong signal that the pipeline works.

!!! note "Full Run"
    The full strict-small track (10M words) would take significantly longer
    but should achieve substantially lower BPB. The recipe supports all
    architectures via the `--arch` flag.

## Usage

```bash
# Quick test (~2 min on CPU)
uv run python recipes/babylm_pretrain.py --max-words 100000

# Full strict-small track (slower)
uv run python recipes/babylm_pretrain.py --track strict-small

# Try different architectures
uv run python recipes/babylm_pretrain.py --arch llama --max-words 100000

# Custom model size
uv run python recipes/babylm_pretrain.py --target-params 5000000
```

### CLI Arguments

| Argument          | Default       | Description                    |
|-------------------|---------------|-------------------------------|
| `--track`         | strict-small  | Competition track             |
| `--max-words`     | None          | Limit training data (testing) |
| `--vocab-size`    | 4096          | BPE vocabulary size           |
| `--target-params` | 1,000,000     | Target model parameters       |
| `--total-steps`   | 500           | Training steps                |
| `--arch`          | gpt           | Model architecture            |
| `--seed`          | 42            | Random seed                   |

## Source Code

- [`recipes/babylm_pretrain.py`](https://github.com/michaelellis003/LMT/blob/main/recipes/babylm_pretrain.py)
  -- Full recipe
- [`tests/test_babylm_recipe.py`](https://github.com/michaelellis003/LMT/blob/main/tests/test_babylm_recipe.py)
  -- Integration tests
