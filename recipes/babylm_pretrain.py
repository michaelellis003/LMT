"""BabyLM Challenge pretraining recipe.

End-to-end pretraining pipeline for the BabyLM Challenge
(sample-efficient language model pretraining). Demonstrates:

- Data: BabyLM corpus from HuggingFace Hub (strict-small: 10M words)
- Tokenizer: BPE trained on the BabyLM corpus itself
- Model: GPT-style, sized to fit the data budget
- Training: train_loop with cosine warmup, weight decay
- Evaluation: val_bpb at regular intervals

The BabyLM Challenge requires training on a limited word budget
(100M for strict, 10M for strict-small) with cognitively plausible
data (child-directed speech, children's books, subtitles, etc.).

Usage::

    # Quick test with limited data (runs in ~2 minutes on CPU)
    uv run python recipes/babylm_pretrain.py --max-words 100000

    # Full strict-small track (10M words, slower)
    uv run python recipes/babylm_pretrain.py --track strict-small

See https://babylm.github.io/ for challenge details.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.data.hf_datasets import download_babylm
from lmt.data.text_dataset import TextDataset
from lmt.eval.bpb import compute_bpb
from lmt.models.config import ModelConfig
from lmt.models.factory import create_model
from lmt.models.sizing import suggest_config
from lmt.tokenizer.trainer import train_bpe_tokenizer
from lmt.training.profiler import profile_model
from lmt.training.reproducibility import set_seed
from lmt.training.train_config import TrainConfig
from lmt.training.train_loop import train_loop

# --- Defaults ---

SEED = 42
CONTEXT_LENGTH = 128
VOCAB_SIZE = 4096
BATCH_SIZE = 32
EVAL_INTERVAL = 100
OUTPUT_DIR = Path('experiments/results/babylm')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='BabyLM Challenge pretraining recipe'
    )
    parser.add_argument(
        '--track',
        choices=['strict', 'strict-small'],
        default='strict-small',
        help='Competition track (default: strict-small, 10M words)',
    )
    parser.add_argument(
        '--max-words',
        type=int,
        default=None,
        help='Limit training data to this many words (for quick testing)',
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=VOCAB_SIZE,
        help=f'Tokenizer vocabulary size (default: {VOCAB_SIZE})',
    )
    parser.add_argument(
        '--target-params',
        type=int,
        default=1_000_000,
        help='Target model parameter count (default: 1M)',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=500,
        help='Total training steps (default: 500)',
    )
    parser.add_argument(
        '--arch',
        default='gpt',
        help='Model architecture (default: gpt)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=SEED,
        help=f'Random seed (default: {SEED})',
    )
    return parser.parse_args()


def load_babylm_data(
    track: str, max_words: int | None
) -> tuple[list[str], list[str]]:
    """Load BabyLM train and validation data.

    Args:
        track: Competition track name.
        max_words: Optional word budget limit.

    Returns:
        Tuple of (train_texts, val_texts).
    """
    print(f'  Downloading BabyLM ({track})...')

    # For strict-small (10M words), load with streaming
    # Use max_items to limit download for testing
    max_items = None
    if max_words is not None:
        # Rough estimate: ~8 words per line on average
        max_items = max_words // 8

    train_texts = download_babylm(
        track=track, split='train', max_items=max_items
    )
    val_texts = download_babylm(
        track=track,
        split='validation',
        max_items=max_items // 10 if max_items else 10000,
    )

    train_words = sum(len(t.split()) for t in train_texts)
    val_words = sum(len(t.split()) for t in val_texts)

    print(f'  Train: {len(train_texts):,} lines, {train_words:,} words')
    print(f'  Val:   {len(val_texts):,} lines, {val_words:,} words')

    # Verify word budget
    budget = 100_000_000 if track == 'strict' else 10_000_000
    if train_words > budget:
        print(f'  WARNING: Training data exceeds {budget:,} word budget!')

    return train_texts, val_texts


def loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Next-token prediction loss."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    return f.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def main() -> None:
    """Run BabyLM pretraining recipe."""
    args = parse_args()
    set_seed(args.seed)

    print('=' * 60)
    print('BabyLM Pretraining Recipe')
    print('=' * 60)
    print(f'Track: {args.track}')
    print(f'Arch:  {args.arch}')
    print(f'Seed:  {args.seed}')

    # 1. Load data
    print('\n1. Loading data...')
    train_texts, val_texts = load_babylm_data(args.track, args.max_words)

    # 2. Train tokenizer
    print('\n2. Training BPE tokenizer...')
    start = time.time()
    tokenizer = train_bpe_tokenizer(train_texts, vocab_size=args.vocab_size)
    tok_time = time.time() - start
    print(f'  Vocab size: {tokenizer.vocab_size}')
    print(f'  Time: {tok_time:.1f}s')

    # Test compression ratio
    sample = ' '.join(train_texts[:100])
    sample_bytes = len(sample.encode('utf-8'))
    sample_tokens = len(tokenizer.encode(sample))
    bytes_per_token = sample_bytes / max(sample_tokens, 1)
    print(f'  Bytes/token: {bytes_per_token:.2f}')

    # 3. Tokenize and pack
    print(f'\n3. Tokenizing (context_length={CONTEXT_LENGTH})...')
    train_dataset = TextDataset(
        train_texts, tokenizer, context_length=CONTEXT_LENGTH
    )
    val_dataset = TextDataset(
        val_texts, tokenizer, context_length=CONTEXT_LENGTH
    )
    print(f'  Train sequences: {len(train_dataset):,}')
    print(f'  Val sequences:   {len(val_dataset):,}')

    # Convert to tensors
    train_data = torch.stack(
        [train_dataset[i] for i in range(len(train_dataset))]
    )
    val_limit = min(500, len(val_dataset))
    val_data = torch.stack([val_dataset[i] for i in range(val_limit)])

    # 4. Create model
    print('\n4. Creating model...')
    suggested = suggest_config(
        target_params=args.target_params,
        vocab_size=tokenizer.vocab_size,
    )
    config = ModelConfig(
        embed_dim=suggested['embed_dim'],
        num_heads=suggested['num_heads'],
        num_kv_heads=max(1, suggested['num_heads'] // 2),
        num_layers=suggested['num_layers'],
        ffn_hidden_dim=suggested['ffn_hidden_dim'],
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        tie_weights=True,
        dropout=0.0,
    )
    model = create_model(args.arch, config)
    profile = profile_model(model)
    print(f'  Architecture: {args.arch}')
    print(
        f'  Config: embed={config.embed_dim}, '
        f'layers={config.num_layers}, '
        f'heads={config.num_heads}'
    )
    print(f'  Parameters: {profile.human_readable_params()}')

    # 5. Train
    print('\n5. Training...')
    train_config = TrainConfig(
        lr=1e-3,
        total_steps=args.total_steps,
        warmup_steps=args.total_steps // 10,
        weight_decay=0.1,
        max_grad_norm=1.0,
    )
    print(f'  Steps: {train_config.total_steps}')
    print(f'  LR: {train_config.lr}')
    print(f'  Batch size: {BATCH_SIZE}')

    # Eval callback
    best_bpb = float('inf')

    def eval_fn(m: nn.Module, step: int) -> dict[str, float]:
        nonlocal best_bpb
        m.eval()
        bpb = compute_bpb(m, val_data, bytes_per_token=bytes_per_token)
        marker = ''
        if bpb < best_bpb:
            best_bpb = bpb
            marker = ' *'
        print(f'  Step {step:4d} | val_bpb={bpb:.4f}{marker}')
        return {'val_bpb': bpb}

    start = time.time()
    state = train_loop(
        model=model,
        train_data=train_data,
        loss_fn=loss_fn,
        config=train_config,
        batch_size=BATCH_SIZE,
        eval_fn=eval_fn,
        eval_interval=EVAL_INTERVAL,
    )
    elapsed = time.time() - start

    # 6. Results
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'  Track:          {args.track}')
    print(f'  Architecture:   {args.arch}')
    print(f'  Parameters:     {profile.human_readable_params()}')
    print(f'  Final loss:     {state.final_loss:.4f}')
    print(f'  Best val_bpb:   {best_bpb:.4f}')
    print(f'  Training time:  {elapsed:.1f}s')

    # Random baseline for reference
    random_bpb = math.log2(tokenizer.vocab_size)
    improvement = (1 - best_bpb / random_bpb) * 100
    print(f'  Random baseline: {random_bpb:.4f}')
    print(f'  Improvement:    {improvement:.1f}%')

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / 'results.txt'
    with open(results_path, 'w') as fp:
        fp.write(f'track: {args.track}\n')
        fp.write(f'arch: {args.arch}\n')
        fp.write(f'params: {profile.total_params}\n')
        fp.write(f'final_loss: {state.final_loss:.6f}\n')
        fp.write(f'best_bpb: {best_bpb:.6f}\n')
        fp.write(f'random_bpb: {random_bpb:.6f}\n')
        fp.write(f'improvement_pct: {improvement:.2f}\n')
        fp.write(f'training_time_s: {elapsed:.1f}\n')
        fp.write(f'seed: {args.seed}\n')
    print(f'\n  Results saved to: {results_path}')


if __name__ == '__main__':
    main()
