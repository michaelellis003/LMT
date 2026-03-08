r"""Modern GPT pretraining recipe.

Karpathy-inspired training recipe combining modern architectural
tricks with automated experiment iteration. Trains a small GPT-style
model and evaluates using bits-per-byte (BPB).

Key design choices (from Karpathy's autoresearch):
- RoPE positional encoding (already in LMT)
- QK-norm for training stability (already in LMT)
- SwiGLU feed-forward (already in LMT)
- val_bpb as the primary evaluation metric
- Short training runs (configurable), evaluate, iterate

Usage::

    # Dry run (tiny model, no GPU needed):
    uv run python recipes/modern_gpt.py --dry_run

    # Full training with custom config:
    uv run python recipes/modern_gpt.py \
        --output_dir runs/modern_gpt \
        --train_steps 500 \
        --context_length 256

References:
    Karpathy, A. (2025). "autoresearch" (GitHub: karpathy/autoresearch)
    -- AI agent modifies train.py, trains 5 min, evaluates val_bpb,
    keeps/discards. ~600M GPT with modern tricks.
"""

import argparse
import json
from pathlib import Path

import torch

from lmt.data.token_dataset import TokenDataset
from lmt.eval.bpb import compute_bpb
from lmt.models.config import ModelConfig
from lmt.research.experiment import (
    ExperimentRunConfig,
    ExperimentRunner,
)
from lmt.training.reproducibility import set_seed


def create_tiny_config() -> ModelConfig:
    """Create a tiny model config for dry-run validation.

    Returns:
        ModelConfig suitable for CPU testing.
    """
    return ModelConfig(
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


def create_small_config(
    context_length: int = 256,
    vocab_size: int = 32000,
) -> ModelConfig:
    """Create a small model config for real training (~10M params).

    Args:
        context_length: Context window size.
        vocab_size: Vocabulary size.

    Returns:
        ModelConfig for a ~10M parameter model.
    """
    return ModelConfig(
        embed_dim=256,
        num_heads=8,
        num_kv_heads=4,
        num_layers=6,
        ffn_hidden_dim=512,
        vocab_size=vocab_size,
        context_length=context_length,
        tie_weights=True,
        qk_norm=True,
        dropout=0.0,
    )


def run_dry_run() -> None:
    """Run a minimal pretraining loop on random data.

    Validates the full pipeline: model creation, data loading,
    training via ExperimentRunner, and BPB evaluation.
    """
    print('=== DRY RUN: Validating pretraining pipeline ===')

    set_seed(42)

    config = create_tiny_config()

    from lmt.models.qwen3.qwen3 import Qwen3

    model = Qwen3(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    # Random training data (would be tokenized corpus in real run)
    tokens = torch.randint(0, config.vocab_size, (2000,))
    dataset = TokenDataset(tokens, context_length=config.context_length)
    ctx = config.context_length
    print(f'Dataset: {len(dataset)} sequences of length {ctx}')

    # Prepare data as a 2D tensor for ExperimentRunner
    train_data = torch.stack([dataset[i] for i in range(len(dataset))])

    # Run experiment
    exp_config = ExperimentRunConfig(
        name='dry_run',
        train_steps=10,
        eval_interval=5,
        lr=1e-3,
        batch_size=4,
    )

    runner = ExperimentRunner(output_dir='/tmp/lmt_dry_run')
    result = runner.run(model, train_data, exp_config)

    print(f'Final train loss: {result.final_loss:.4f}')

    # Evaluate BPB
    seq_len = config.context_length + 1
    eval_tokens = torch.randint(0, config.vocab_size, (5, seq_len))
    bpb = compute_bpb(
        model,
        eval_tokens,
        bytes_per_token=1.0,  # byte-level for random data
    )
    print(f'val_bpb: {bpb:.4f}')

    print('=== DRY RUN COMPLETE ===')
    print('Pipeline validated. Ready for real training with data + GPU.')


def run_training(args: argparse.Namespace) -> None:
    """Set up and run pretraining experiment.

    .. note::

        This is currently a **placeholder** for full pretraining.
        It creates the model and config but requires a real tokenized
        dataset. Use ``--dry_run`` to validate the pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = create_small_config(
        context_length=args.context_length,
    )

    config_dict = {
        'model': {
            'embed_dim': model_config.embed_dim,
            'num_heads': model_config.num_heads,
            'num_layers': model_config.num_layers,
            'vocab_size': model_config.vocab_size,
            'context_length': model_config.context_length,
        },
        'training': {
            'train_steps': args.train_steps,
            'lr': args.lr,
            'seed': args.seed,
        },
    }

    config_path = output_dir / 'config.json'
    config_path.write_text(json.dumps(config_dict, indent=2) + '\n')

    print(f'Config saved to {config_path}')
    print(f'Model: ~10M params, context_length={args.context_length}')
    print(f'Training: {args.train_steps} steps, lr={args.lr}')
    print(
        '\nFull training requires a tokenized dataset.'
        '\nUse --dry_run to validate the pipeline.'
    )


def main() -> None:
    """Entry point for modern GPT recipe."""
    parser = argparse.ArgumentParser(
        description='Modern GPT Pretraining Recipe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='runs/modern_gpt',
        help='Output directory for checkpoints and logs',
    )
    parser.add_argument(
        '--train_steps',
        type=int,
        default=500,
        help='Number of training steps',
    )
    parser.add_argument(
        '--context_length',
        type=int,
        default=256,
        help='Context window size',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run minimal pipeline validation on CPU',
    )

    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
    else:
        run_training(args)


if __name__ == '__main__':
    main()
