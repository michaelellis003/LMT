"""Autoresearch demo: automated experiment iteration.

Demonstrates the full autoresearch pipeline:
1. Define a search space (learning rates)
2. Train small models with grid search
3. Evaluate BPB on validation data
4. Identify the best configuration

This runs entirely on CPU with tiny models — designed to
validate the pipeline, not produce useful models.

Usage::

    uv run python recipes/autoresearch_demo.py

References:
    Karpathy, A. (2025). "autoresearch" (GitHub: karpathy/autoresearch)
"""

import torch

from lmt.data.token_dataset import TokenDataset
from lmt.eval.bpb import compute_bpb
from lmt.models.config import ModelConfig
from lmt.models.qwen3.qwen3 import Qwen3
from lmt.research.experiment import (
    ExperimentRunConfig,
)
from lmt.research.search import SearchSpace, grid_search
from lmt.training.reproducibility import set_seed


def main() -> None:
    """Run the autoresearch demo."""
    print('=== Autoresearch Demo ===')
    print('Demonstrates: grid search → train → BPB eval → best model')
    print()

    set_seed(42)

    # 1. Create tiny model config
    config = ModelConfig(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ffn_hidden_dim=128,
        vocab_size=256,
        context_length=32,
        tie_weights=True,
        qk_norm=True,
        dropout=0.0,
    )
    param_count = sum(p.numel() for p in Qwen3(config).parameters())
    print(f'Model: Qwen3 ({param_count:,} params)')

    # 2. Create training + validation data
    # Use the same token distribution for train/val so BPB can improve
    all_tokens = torch.randint(0, 256, (3000,))
    train_tokens = all_tokens[:2400]
    val_tokens = all_tokens[2400:]
    train_ds = TokenDataset(train_tokens, context_length=32)
    val_ds = TokenDataset(val_tokens, context_length=32)

    train_data = torch.stack([train_ds[i] for i in range(len(train_ds))])
    val_data = torch.stack([val_ds[i] for i in range(len(val_ds))])

    print(f'Train: {len(train_ds)} sequences')
    print(f'Val: {len(val_ds)} sequences')

    # 3. Measure baseline BPB (untrained model)
    baseline_model = Qwen3(config)
    baseline_model.eval()
    baseline_bpb = compute_bpb(baseline_model, val_data, bytes_per_token=1.0)
    print(f'Baseline BPB (untrained): {baseline_bpb:.4f}')
    print()

    # 4. Grid search over learning rates
    search_space = SearchSpace(
        lr=[1e-2, 5e-3, 1e-3, 5e-4],
    )
    base_config = ExperimentRunConfig(
        name='autoresearch',
        train_steps=100,
        eval_interval=100,
        batch_size=4,
    )

    def model_fn() -> Qwen3:
        set_seed(42)  # Same init for fair comparison
        return Qwen3(config)

    print(f'Running grid search: {len(search_space.grid())} experiments...')
    results = grid_search(
        model_fn=model_fn,
        train_data=train_data,
        search_space=search_space,
        base_config=base_config,
        output_dir='/tmp/autoresearch_demo',
        val_data=val_data,
    )

    # 5. Display results
    print()
    print('Results:')
    print(f'{"LR":<12} {"Train Loss":<14} {"Val BPB":<12}')
    print('-' * 38)

    for result in results:
        lr = result.config.lr
        loss = result.final_loss or 0
        bpb = result.metrics.get_best('val_bpb') or 0
        print(f'{lr:<12.0e} {loss:<14.4f} {bpb:<12.4f}')

    # 6. Identify best
    best = min(
        results,
        key=lambda r: r.metrics.get_best('val_bpb') or float('inf'),
    )
    best_bpb = best.metrics.get_best('val_bpb') or 0
    print()
    print(f'Best LR: {best.config.lr:.0e}')
    print(f'Best BPB: {best_bpb:.4f}')
    if best_bpb < baseline_bpb:
        print(f'Improvement over baseline: {baseline_bpb - best_bpb:.4f} BPB')
    else:
        print(
            'Note: No improvement over baseline. This is expected '
            'with random data (max entropy = no learnable structure).'
        )
        print('With real text data, training reduces BPB significantly.')
    print()
    print('=== Demo Complete ===')


if __name__ == '__main__':
    main()
