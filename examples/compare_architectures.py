#!/usr/bin/env python3
# Copyright 2025 Michael Ellis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare training dynamics across LMT model architectures.

This script trains GPT, LLaMA, Qwen3, Gemma, Mixtral, DeepSeek-V2,
Kimi, and Mamba side-by-side on synthetic next-token prediction data
and logs results to MLflow for visual comparison.

Usage:
    python examples/compare_architectures.py [--epochs 5] [--device cpu]

    # Then visualize:
    mlflow ui --backend-store-uri file://./runs/architecture_comparison/mlruns
"""

import argparse
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from lmt.models.config import ModelConfig
from lmt.models.deepseek import DeepSeekV2
from lmt.models.gemma import Gemma
from lmt.models.gpt import GPT
from lmt.models.kimi import Kimi
from lmt.models.llama import LLaMA
from lmt.models.mamba import Mamba
from lmt.models.mixtral import Mixtral
from lmt.models.qwen3 import Qwen3
from lmt.training.config import BaseTrainingConfig
from lmt.training.trainer import Trainer


def make_synthetic_data(
    vocab_size: int,
    seq_len: int,
    num_samples: int = 256,
    batch_size: int = 8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create synthetic next-token prediction data.

    Generates random token sequences split 80/20 into train/val.

    Args:
        vocab_size: Size of the token vocabulary.
        seq_len: Sequence length for each sample.
        num_samples: Total number of samples to generate.
        batch_size: Batch size for data loaders.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    gen = torch.Generator().manual_seed(seed)
    size = (num_samples, seq_len)
    inputs = torch.randint(0, vocab_size, size, generator=gen)
    targets = torch.randint(0, vocab_size, size, generator=gen)

    split = int(0.8 * num_samples)
    train_ds = TensorDataset(inputs[:split], targets[:split])
    val_ds = TensorDataset(inputs[split:], targets[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def build_models(
    vocab_size: int, seq_len: int, embed_dim: int, num_layers: int
) -> dict[str, torch.nn.Module]:
    """Build comparable versions of each architecture.

    Args:
        vocab_size: Vocabulary size.
        seq_len: Context length.
        embed_dim: Embedding dimension.
        num_layers: Number of blocks/layers.

    Returns:
        Dictionary mapping model names to model instances.
    """
    config = ModelConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=4,
        num_kv_heads=4,
        num_layers=num_layers,
        context_length=seq_len,
        dropout=0.0,
    )

    models: dict[str, torch.nn.Module] = {}

    # GPT: classic transformer with MHA + GELU FFN
    models['GPT'] = GPT(config)

    # LLaMA: RMSNorm + GQA + SwiGLU + RoPE
    models['LLaMA'] = LLaMA(config)

    # Qwen3: LLaMA + QK-Norm + weight tying
    models['Qwen3'] = Qwen3(config)

    # Gemma: LLaMA + QK-Norm + interleaved local/global attention
    models['Gemma'] = Gemma(config, local_window=seq_len, global_every=2)

    # Mixtral: LLaMA + MoE (4 experts, top-2)
    models['Mixtral'] = Mixtral(config, num_experts=4, top_k=2)

    # DeepSeek-V2: MLA + MoE
    models['DeepSeek-V2'] = DeepSeekV2(
        config,
        kv_compress_dim=embed_dim // 4,
        q_compress_dim=embed_dim // 4,
        rope_dim=embed_dim // 8,
        num_experts=4,
        top_k=2,
    )

    # Kimi: DeepSeek-V2 variant (MLA + MoE)
    models['Kimi'] = Kimi(
        config,
        kv_compress_dim=embed_dim // 4,
        q_compress_dim=embed_dim // 4,
        rope_dim=embed_dim // 8,
        num_experts=4,
        top_k=2,
    )

    # Mamba: SSM (no attention at all!)
    models['Mamba'] = Mamba(config, d_state=16, expand=2, d_conv=4)

    return models


def count_params(model: torch.nn.Module) -> int:
    """Count total parameters in a model.

    Args:
        model: The PyTorch model.

    Returns:
        Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    """Run the architecture comparison experiment."""
    parser = argparse.ArgumentParser(
        description='Compare LMT model architectures'
    )
    parser.add_argument(
        '--epochs', type=int, default=5, help='Number of epochs'
    )
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument(
        '--embed-dim', type=int, default=64, help='Embed dimension'
    )
    parser.add_argument(
        '--num-layers', type=int, default=2, help='Number of layers'
    )
    parser.add_argument(
        '--seq-len', type=int, default=32, help='Sequence length'
    )
    parser.add_argument(
        '--num-samples', type=int, default=256, help='Data samples'
    )
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    vocab_size = 256
    save_dir = 'runs/architecture_comparison'

    print('=' * 60)
    print('LMT Architecture Comparison Experiment')
    print('=' * 60)
    print(f'Epochs: {args.epochs}')
    print(f'Device: {args.device}')
    print(f'Embed dim: {args.embed_dim}, Layers: {args.num_layers}')
    print(f'Seq len: {args.seq_len}, Samples: {args.num_samples}')
    print()

    train_loader, val_loader = make_synthetic_data(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    models = build_models(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
    )

    print('Model Parameter Counts:')
    print('-' * 40)
    for name, model in models.items():
        params = count_params(model)
        print(f'  {name:15s} {params:>10,} params')
    print()

    results = {}
    for name, model in models.items():
        print(f'\n{"=" * 60}')
        print(f'Training: {name}')
        print(f'{"=" * 60}')

        is_moe = name in ('Mixtral', 'DeepSeek-V2', 'Kimi')
        aux_coeff = 0.01 if is_moe else 0.0

        config = BaseTrainingConfig(
            num_epochs=args.epochs,
            eval_freq=5,
            eval_iter=2,
            learning_rate=args.lr,
            weight_decay=0.01,
            batch_size=args.batch_size,
            device=args.device,
            save_dir=save_dir,
            run_name=name,
            aux_loss_coeff=aux_coeff,
        )

        trainer = Trainer(model, train_loader, val_loader, config)
        result = trainer.train()
        results[name] = result

        tl = result['train_losses']
        vl = result['val_losses']
        final_train = tl[-1] if tl else float('nan')
        final_val = vl[-1] if vl else float('nan')
        print(
            f'{name} final -- train: {final_train:.4f}, val: {final_val:.4f}'
        )
        print(f'{name} time: {result["execution_time"]:.2f} min')

    # Summary
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(
        f'{"Model":15s} {"Params":>10s} {"Train Loss":>12s} '
        f'{"Val Loss":>12s} {"Time (min)":>12s}'
    )
    print('-' * 65)
    for name in models:
        params = count_params(models[name])
        r = results[name]
        tl = r['train_losses'][-1] if r['train_losses'] else float('nan')
        vl = r['val_losses'][-1] if r['val_losses'] else float('nan')
        print(
            f'{name:15s} {params:>10,} {tl:>12.4f} {vl:>12.4f} '
            f'{r["execution_time"]:>12.2f}'
        )

    print(f'\nMLflow logs saved to: {save_dir}/mlruns/')
    print(f'Run: mlflow ui --backend-store-uri file://./{save_dir}/mlruns')

    return


if __name__ == '__main__':
    sys.exit(main() or 0)
