#!/usr/bin/env python3
"""Train a character-level language model on TinyShakespeare.

This is an end-to-end training recipe demonstrating how to use LMT
to train a small language model on real text. It downloads the
TinyShakespeare dataset (~1MB), trains a model, generates samples
during training, and reports final metrics.

Supports multiple architectures: GPT, LLaMA, and Mamba.

Usage:
    # Train a LLaMA model (default) for 10 epochs:
    python examples/train_shakespeare.py

    # Train a GPT model for 20 epochs:
    python examples/train_shakespeare.py --arch gpt --epochs 20

    # Train a Mamba model on GPU:
    python examples/train_shakespeare.py --arch mamba --device cuda

    # Customize model size:
    python examples/train_shakespeare.py --embed-dim 128 --num-layers 4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lmt.generate import generate
from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig
from lmt.models.mamba import Mamba
from lmt.tokenizer.char import CharTokenizer
from lmt.training.datasets import GPTDataset

DATA_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/'
    'master/data/tinyshakespeare/input.txt'
)
DATA_DIR = Path('data')
DATA_FILE = DATA_DIR / 'tinyshakespeare.txt'


def download_data() -> str:
    """Download TinyShakespeare if not already cached.

    Returns:
        The full text of the dataset.
    """
    if DATA_FILE.exists():
        print(f'Using cached data: {DATA_FILE}')
        return DATA_FILE.read_text()

    print(f'Downloading TinyShakespeare from {DATA_URL}...')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    text = DATA_FILE.read_text()
    print(f'Downloaded {len(text):,} characters.')
    return text


def build_model(
    arch: str,
    config: ModelConfig,
) -> nn.Module:
    """Build a model from the architecture name.

    Args:
        arch: Architecture name ('gpt', 'llama', or 'mamba').
        config: Model configuration.

    Returns:
        The constructed model.
    """
    head_dim = config.embed_dim // config.num_heads

    if arch == 'gpt':
        block_config = BlockConfig(
            attention='mha', ffn='default', norm='layernorm'
        )
        return BaseModel(
            config, block_config=block_config, learned_pos_embed=True
        )

    elif arch == 'llama':
        rope = RoPE(d_model=head_dim, max_seq_len=config.context_length)
        block_config = BlockConfig(
            attention='gqa', ffn='swiglu', norm='rmsnorm', rope=rope
        )
        return BaseModel(config, block_config=block_config)

    elif arch == 'mamba':
        return Mamba(config, d_state=16, expand=2, d_conv=4)

    else:
        raise ValueError(f'Unknown architecture: {arch!r}')


def generate_sample(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str,
    context_length: int,
    device: torch.device,
    max_tokens: int = 200,
    temperature: float = 0.8,
) -> str:
    """Generate a text sample from the model.

    Args:
        model: The language model.
        tokenizer: Character tokenizer.
        prompt: Starting text.
        context_length: Model's context window.
        device: Device to run on.
        max_tokens: Number of tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The generated text (including prompt).
    """
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_tokens,
            context_size=context_length,
            temperature=temperature,
        )

    model.train()
    return tokenizer.decode(output_ids[0].tolist())


def count_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> int:
    """Run the training recipe."""
    parser = argparse.ArgumentParser(
        description='Train a character-level LM on TinyShakespeare'
    )
    parser.add_argument(
        '--arch',
        choices=['gpt', 'llama', 'mamba'],
        default='llama',
        help='Model architecture (default: llama)',
    )
    parser.add_argument(
        '--epochs', type=int, default=10, help='Training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32, help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=128,
        help='Embedding dimension (default: 128)',
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=4,
        help='Number of layers (default: 4)',
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=4,
        help='Number of attention heads (default: 4)',
    )
    parser.add_argument(
        '--context-len',
        type=int,
        default=128,
        help='Context length (default: 128)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device: cpu, cuda, mps (default: cpu)',
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed (default: 42)'
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Data ---
    text = download_data()
    tokenizer = CharTokenizer.from_text(text)
    vocab_size = tokenizer.vocab_size

    # Train/val split (90/10)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_ds = GPTDataset(
        train_text, tokenizer, args.context_len, stride=args.context_len
    )
    val_ds = GPTDataset(
        val_text, tokenizer, args.context_len, stride=args.context_len
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    # --- Model ---
    config = ModelConfig(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_heads,
        num_layers=args.num_layers,
        context_length=args.context_len,
        dropout=0.1,
    )

    model = build_model(args.arch, config)
    model.to(device)
    params = count_params(model)

    random_loss = math.log(vocab_size)

    print('=' * 60)
    print('LMT Training Recipe: TinyShakespeare')
    print('=' * 60)
    print(f'Architecture:   {args.arch.upper()}')
    print(f'Parameters:     {params:,}')
    print(f'Vocab size:     {vocab_size} characters')
    print(f'Context length: {args.context_len}')
    print(f'Embed dim:      {args.embed_dim}')
    print(f'Layers:         {args.num_layers}')
    print(f'Heads:          {args.num_heads}')
    print(f'Train samples:  {len(train_ds):,}')
    print(f'Val samples:    {len(val_ds):,}')
    print(f'Device:         {device}')
    print(f'Random loss:    {random_loss:.3f} (ln({vocab_size}))')
    print('=' * 60)

    # --- Training ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), target_batch.flatten()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches

        # Validation
        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                logits = model(input_batch)
                val_loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), target_batch.flatten()
                )
                val_total += val_loss.item()
                val_batches += 1

        val_loss_avg = val_total / val_batches if val_batches > 0 else 0.0

        elapsed = time.time() - start_time
        print(
            f'Epoch {epoch + 1:3d}/{args.epochs} | '
            f'train: {train_loss:.3f} | val: {val_loss_avg:.3f} | '
            f'time: {elapsed:.0f}s'
        )

        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg

        # Generate a sample every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            sample = generate_sample(
                model,
                tokenizer,
                prompt='ROMEO:\n',
                context_length=args.context_len,
                device=device,
            )
            print(f'\n--- Sample (epoch {epoch + 1}) ---')
            print(sample[:300])
            print('---\n')

    # --- Final report ---
    total_time = time.time() - start_time
    print('=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'Best val loss:  {best_val_loss:.3f}')
    print(f'Random chance:  {random_loss:.3f}')
    print(
        f'Improvement:    {(1 - best_val_loss / random_loss) * 100:.1f}% '
        f'below random'
    )
    print(f'Total time:     {total_time:.1f}s ({total_time / 60:.1f} min)')
    print()

    # Final generation
    print('--- Final generation ---')
    for prompt in ['ROMEO:\n', 'To be, or ', 'KING HENRY:\n']:
        sample = generate_sample(
            model, tokenizer, prompt, args.context_len, device
        )
        print(f'Prompt: {prompt!r}')
        print(sample[:400])
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
