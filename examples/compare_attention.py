#!/usr/bin/env python3
"""Compare attention variants on TinyShakespeare.

Trains the same base model architecture with different attention
mechanisms (MHA, GQA, GatedDeltaNet, SSD) and compares learning
curves, final loss, and generation quality.

This is the key experiment: do different attention mechanisms actually
learn differently on real text?

Usage:
    python examples/compare_attention.py
    python examples/compare_attention.py --epochs 20 --device mps
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lmt.generate import generate
from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig
from lmt.tokenizer.char import CharTokenizer
from lmt.training.datasets import GPTDataset

DATA_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/'
    'master/data/tinyshakespeare/input.txt'
)
DATA_DIR = Path('data')
DATA_FILE = DATA_DIR / 'tinyshakespeare.txt'


def download_data() -> str:
    """Download TinyShakespeare if not already cached."""
    if DATA_FILE.exists():
        return DATA_FILE.read_text()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    return DATA_FILE.read_text()


ATTENTION_VARIANTS: dict[str, BlockConfig] = {
    'MHA': BlockConfig(attention='mha', ffn='default', norm='rmsnorm'),
    'GQA+SwiGLU': BlockConfig(attention='gqa', ffn='swiglu', norm='rmsnorm'),
    'GatedDeltaNet': BlockConfig(
        attention='gated_delta_net', ffn='swiglu', norm='rmsnorm'
    ),
    'SSD': BlockConfig(attention='ssd', ffn='swiglu', norm='rmsnorm'),
}


def train_one(
    name: str,
    block_config: BlockConfig,
    model_config: ModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: CharTokenizer,
    num_epochs: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Train a single model variant and return results.

    Args:
        name: Human-readable name for this variant.
        block_config: Block configuration (attention + FFN + norm).
        model_config: Model configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        tokenizer: Character tokenizer for generation.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        device: Device to train on.

    Returns:
        Dictionary with training metrics and final generation.
    """
    # MHA and GQA with no RoPE need learned positional embeddings
    needs_pos_embed = block_config.attention in ('mha',)
    model = BaseModel(
        model_config,
        block_config=block_config,
        learned_pos_embed=needs_pos_embed,
    )
    model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n{"=" * 50}')
    print(f'{name} ({params:,} params)')
    print(f'{"=" * 50}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    epoch_train_losses: list[float] = []
    epoch_val_losses: list[float] = []
    start = time.time()

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0.0
        n = 0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(inp)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), tgt.flatten()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        epoch_train_losses.append(total_loss / n)

        # Eval
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                logits = model(inp)
                vl += torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), tgt.flatten()
                ).item()
                vn += 1
        epoch_val_losses.append(vl / vn if vn else 0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f'  Epoch {epoch + 1:3d} | '
                f'train: {epoch_train_losses[-1]:.3f} | '
                f'val: {epoch_val_losses[-1]:.3f}'
            )

    elapsed = time.time() - start

    # Generate a sample
    model.eval()
    prompt_ids = tokenizer.encode('ROMEO:\n')
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = generate(
            model,
            input_ids,
            max_new_tokens=150,
            context_size=model_config.context_length,
            temperature=0.8,
        )
    sample = tokenizer.decode(out[0].tolist())

    return {
        'name': name,
        'params': params,
        'train_losses': epoch_train_losses,
        'val_losses': epoch_val_losses,
        'time_s': elapsed,
        'sample': sample[:300],
    }


def main() -> int:
    """Run the attention comparison experiment."""
    parser = argparse.ArgumentParser(
        description='Compare attention variants on TinyShakespeare'
    )
    parser.add_argument(
        '--epochs', type=int, default=10, help='Epochs per model'
    )
    parser.add_argument(
        '--embed-dim', type=int, default=128, help='Embedding dim'
    )
    parser.add_argument('--num-layers', type=int, default=4, help='Layers')
    parser.add_argument(
        '--context-len', type=int, default=128, help='Context length'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32, help='Batch size'
    )
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Data
    text = download_data()
    tokenizer = CharTokenizer.from_text(text)
    vocab_size = tokenizer.vocab_size

    split = int(len(text) * 0.9)
    train_ds = GPTDataset(
        text[:split], tokenizer, args.context_len, stride=args.context_len
    )
    val_ds = GPTDataset(
        text[split:], tokenizer, args.context_len, stride=args.context_len
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    config = ModelConfig(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=4,
        num_kv_heads=4,
        num_layers=args.num_layers,
        context_length=args.context_len,
        dropout=0.1,
    )

    random_loss = math.log(vocab_size)
    print('=' * 60)
    print('Attention Variant Comparison on TinyShakespeare')
    print('=' * 60)
    print(
        f'Vocab: {vocab_size} | Context: {args.context_len} | '
        f'Embed: {args.embed_dim} | Layers: {args.num_layers}'
    )
    print(
        f'Epochs: {args.epochs} | LR: {args.lr} | '
        f'Batch: {args.batch_size} | Device: {device}'
    )
    print(f'Random loss: {random_loss:.3f} (ln({vocab_size}))')
    print(f'Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}')

    # Train each variant
    results = []
    for name, block_config in ATTENTION_VARIANTS.items():
        torch.manual_seed(args.seed)
        result = train_one(
            name,
            block_config,
            config,
            train_loader,
            val_loader,
            tokenizer,
            args.epochs,
            args.lr,
            device,
        )
        results.append(result)

    # Summary table
    print(f'\n{"=" * 60}')
    print('COMPARISON SUMMARY')
    print(f'{"=" * 60}')
    print(
        f'{"Variant":<18} {"Params":>8} {"Train":>8} '
        f'{"Val":>8} {"vs Rand":>8} {"Time":>8}'
    )
    print('-' * 60)
    for r in results:
        tl = r['train_losses'][-1]
        vl = r['val_losses'][-1]
        improvement = (1 - vl / random_loss) * 100
        print(
            f'{r["name"]:<18} {r["params"]:>8,} {tl:>8.3f} '
            f'{vl:>8.3f} {improvement:>7.1f}% '
            f'{r["time_s"]:>7.1f}s'
        )

    # Show samples
    print(f'\n{"=" * 60}')
    print('GENERATED SAMPLES')
    print(f'{"=" * 60}')
    for r in results:
        print(f'\n--- {r["name"]} ---')
        print(r['sample'])

    return 0


if __name__ == '__main__':
    sys.exit(main())
