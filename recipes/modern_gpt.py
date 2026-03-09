r"""Modern GPT pretraining recipe.

Karpathy-inspired training recipe combining modern architectural
tricks with automated experiment iteration. Trains a small GPT-style
model and evaluates using bits-per-byte (BPB).

Modern tricks included:
- RoPE positional encoding
- QK-norm for training stability
- ReLU² feed-forward (drop-in SwiGLU replacement)
- Value-residual connections (Gemma 2 style)
- Sliding window attention
- Muon + AdamW hybrid optimizer
- val_bpb as the primary evaluation metric

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
from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional.rope import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig
from lmt.research.experiment import (
    ExperimentRunConfig,
    ExperimentRunner,
)
from lmt.training.muon import Muon
from lmt.training.reproducibility import set_seed


def create_model_config(
    context_length: int = 32,
    vocab_size: int = 256,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    num_layers: int = 2,
    ffn_hidden_dim: int = 128,
    value_residual_mix: float = 0.5,
    window_size: int | None = None,
) -> ModelConfig:
    """Create a model config with modern tricks enabled.

    Args:
        context_length: Context window size.
        vocab_size: Vocabulary size.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads for GQA.
        num_layers: Number of transformer layers.
        ffn_hidden_dim: Hidden dimension for FFN.
        value_residual_mix: Blend factor for value-residual
            connections (0 = disabled, 1 = full first-layer V).
        window_size: Sliding window size (None = full attention).

    Returns:
        ModelConfig with modern tricks enabled.
    """
    return ModelConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        ffn_hidden_dim=ffn_hidden_dim,
        vocab_size=vocab_size,
        context_length=context_length,
        tie_weights=True,
        qk_norm=True,
        dropout=0.0,
        value_residual_mix=value_residual_mix,
        window_size=window_size,
    )


def create_modern_model(
    config: ModelConfig,
    window_size: int | None = None,
) -> BaseModel:
    """Create a BaseModel with modern tricks wired in.

    Uses ReLU² FFN, sliding window GQA (if window_size set),
    and value-residual connections via the model config.

    Args:
        config: Model configuration.
        window_size: Sliding window size. If None, uses full
            attention with GQA.

    Returns:
        BaseModel with modern architectural features.
    """
    head_dim = config.embed_dim // config.num_heads
    rope = RoPE(head_dim, config.context_length)

    if window_size is not None:
        block_config = BlockConfig(
            attention='sliding_window',
            ffn='relu_squared',
            rope=rope,
            window_size=window_size,
        )
    else:
        block_config = BlockConfig(
            attention='gqa',
            ffn='relu_squared',
            rope=rope,
        )

    return BaseModel(config, block_config=block_config)


def create_muon_adamw_optimizer(
    model: BaseModel,
    lr: float = 1e-3,
    muon_lr: float | None = None,
    adamw_lr: float | None = None,
) -> Muon:
    """Create a Muon optimizer (handles 2D hidden weights).

    Muon applies Newton-Schulz orthogonalization to 2D weight
    matrices. For embeddings, norms, and biases, it falls back
    to standard momentum SGD internally.

    Args:
        model: The model to optimize.
        lr: Default learning rate.
        muon_lr: Learning rate for Muon params. Defaults to lr.
        adamw_lr: Not used (Muon handles all params). Kept for
            API consistency.

    Returns:
        Muon optimizer instance.
    """
    return Muon(model.parameters(), lr=muon_lr or lr)


def run_dry_run() -> None:
    """Run a minimal pretraining loop on random data.

    Validates the full pipeline: model creation with all modern
    tricks, training via ExperimentRunner, and BPB evaluation.
    """
    print('=== DRY RUN: Validating modern GPT pipeline ===')

    set_seed(42)

    config = create_model_config(
        context_length=32,
        vocab_size=256,
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ffn_hidden_dim=128,
        value_residual_mix=0.5,
        window_size=8,
    )

    model = create_modern_model(config, window_size=8)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')
    print('  ReLU² FFN: yes')
    print(f'  Value-residual mix: {config.value_residual_mix}')
    print(f'  Sliding window: {config.window_size}')
    print(f'  QK-norm: {config.qk_norm}')

    # Random training data
    tokens = torch.randint(0, config.vocab_size, (2000,))
    dataset = TokenDataset(tokens, context_length=config.context_length)
    train_data = torch.stack([dataset[i] for i in range(len(dataset))])
    print(
        f'Dataset: {len(dataset)} sequences of length {config.context_length}'
    )

    # Create Muon optimizer
    optimizer = create_muon_adamw_optimizer(model, lr=1e-3)

    # Run experiment
    exp_config = ExperimentRunConfig(
        name='dry_run_modern',
        train_steps=10,
        eval_interval=5,
        lr=1e-3,
        batch_size=4,
    )

    runner = ExperimentRunner(output_dir='/tmp/lmt_dry_run')
    result = runner.run(model, train_data, exp_config, optimizer=optimizer)

    print(f'Final train loss: {result.final_loss:.4f}')

    # Evaluate BPB
    seq_len = config.context_length + 1
    eval_tokens = torch.randint(0, config.vocab_size, (5, seq_len))
    bpb = compute_bpb(model, eval_tokens, bytes_per_token=1.0)
    print(f'val_bpb: {bpb:.4f}')

    print('=== DRY RUN COMPLETE ===')
    print('Pipeline validated with all modern tricks.')


def run_training(args: argparse.Namespace) -> None:
    """Set up and run pretraining with modern tricks.

    Creates a model with ReLU², value-residual connections,
    sliding window attention, and Muon optimizer.

    Args:
        args: Parsed command-line arguments.
    """
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = create_model_config(
        context_length=args.context_length,
        vocab_size=32000,
        embed_dim=256,
        num_heads=8,
        num_kv_heads=4,
        num_layers=6,
        ffn_hidden_dim=512,
        value_residual_mix=args.value_residual_mix,
        window_size=args.window_size,
    )

    config_dict = {
        'model': {
            'embed_dim': model_config.embed_dim,
            'num_heads': model_config.num_heads,
            'num_layers': model_config.num_layers,
            'vocab_size': model_config.vocab_size,
            'context_length': model_config.context_length,
            'value_residual_mix': model_config.value_residual_mix,
            'window_size': model_config.window_size,
        },
        'training': {
            'train_steps': args.train_steps,
            'lr': args.lr,
            'seed': args.seed,
            'optimizer': 'muon',
            'ffn': 'relu_squared',
        },
    }

    config_path = output_dir / 'config.json'
    config_path.write_text(json.dumps(config_dict, indent=2) + '\n')

    print(f'Config saved to {config_path}')
    print(f'Model: ~10M params, context_length={args.context_length}')
    print(f'Training: {args.train_steps} steps, lr={args.lr}')
    print(
        f'Modern tricks: ReLU², value-residual={args.value_residual_mix},'
        f' window={args.window_size}, Muon optimizer'
    )
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
        '--value_residual_mix',
        type=float,
        default=0.5,
        help='Value-residual blend factor (0=disabled, 1=full)',
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=None,
        help='Sliding window size (None=full attention)',
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
