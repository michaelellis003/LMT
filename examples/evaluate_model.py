#!/usr/bin/env python3
"""Evaluate an LMT model using lm-eval-harness benchmarks.

Usage:
    # Evaluate a random (untrained) small GPT on a few benchmarks:
    python examples/evaluate_model.py

    # Evaluate a checkpoint:
    python examples/evaluate_model.py --checkpoint path/to/model.pt

    # Specific benchmarks with more samples:
    python examples/evaluate_model.py \
        --tasks hellaswag piqa arc_easy --limit 100

Requires: uv sync --group eval
"""

import argparse

import torch

from lmt.models.config import ModelConfig, ModelConfigPresets
from lmt.models.gpt import GPT
from lmt.tokenizer import BPETokenizer
from lmt.training.lm_eval_adapter import LMTEvalModel


def load_model(
    checkpoint: str | None = None,
) -> tuple[torch.nn.Module, ModelConfig]:
    """Load an LMT model, optionally from a checkpoint."""
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config_dict = state.get('config', {})
        config = ModelConfig(
            embed_dim=config_dict.get('embed_dim', 384),
            num_heads=config_dict.get('num_heads', 6),
            num_layers=config_dict.get('num_layers', 6),
            vocab_size=config_dict.get('vocab_size', 50257),
            context_length=config_dict.get('context_length', 256),
            dropout=0.0,
        )
        model = GPT(config)
        model.load_state_dict(state['model_state_dict'])
        print(f'Loaded checkpoint: {checkpoint}')
    else:
        config = ModelConfigPresets.small_gpt(context_length=256)
        model = GPT(config)
        print('Using random (untrained) small GPT model')

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {param_count:,}')
    return model, config


def main():
    """Run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate LMT model')
    parser.add_argument(
        '--checkpoint', type=str, default=None, help='Path to model checkpoint'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['hellaswag', 'piqa', 'arc_easy', 'lambada_openai'],
        help='Benchmark tasks to run',
    )
    parser.add_argument(
        '--limit', type=int, default=None, help='Limit samples per task'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', help='Device (cpu/cuda/mps)'
    )
    parser.add_argument(
        '--num_fewshot',
        type=int,
        default=0,
        help='Number of few-shot examples',
    )
    args = parser.parse_args()

    # Lazy import -- lm_eval stubs have incorrect signatures
    from typing import Any

    _evaluate: Any = __import__('lm_eval').simple_evaluate

    model, config = load_model(args.checkpoint)
    tokenizer = BPETokenizer()
    adapter = LMTEvalModel(model, tokenizer, device=args.device)

    print(f'\nRunning benchmarks: {", ".join(args.tasks)}')
    if args.limit:
        print(f'Limiting to {args.limit} samples per task')
    print()

    results = _evaluate(
        model=adapter,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=False,
    )
    assert results is not None

    # Print results table
    print('\n' + '=' * 60)
    print(f'{"Benchmark":<25} {"Metric":<15} {"Score":>10}')
    print('=' * 60)
    for task_name, metrics in results['results'].items():
        for key, value in sorted(metrics.items()):
            if isinstance(value, float) and 'stderr' not in key:
                print(f'{task_name:<25} {key:<15} {value:>10.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
