"""Architecture comparison experiment.

Controlled comparison of GPT, LLaMA, Qwen3, Gemma, and Mixtral
on identical synthetic data using the same hyperparameters.

Tests whether architecture choice matters at toy scale (~50K params).

Hypothesis: At this scale, all architectures should converge to
similar BPB values because the structural differences (GQA vs MHA,
SwiGLU vs GELU, RoPE variants) only matter when models are large
enough to exploit them.

Usage::

    uv run python experiments/arch_comparison.py

Results are saved to experiments/results/arch_comparison/registry.jsonl
and a summary table is printed at the end.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.eval.bpb import compute_bpb
from lmt.models.config import ModelConfig
from lmt.models.factory import create_model
from lmt.research.experiment import (
    ExperimentResult,
    ExperimentRunConfig,
    MetricTracker,
)
from lmt.research.registry import ExperimentRegistry
from lmt.training.profiler import profile_model
from lmt.training.reproducibility import set_seed
from lmt.training.train_config import TrainConfig
from lmt.training.train_loop import train_loop

# --- Configuration ---

MODEL_CONFIG = ModelConfig(
    embed_dim=64,
    num_heads=4,
    num_kv_heads=2,
    num_layers=4,
    ffn_hidden_dim=128,
    vocab_size=256,
    context_length=64,
    tie_weights=True,
    qk_norm=True,
    dropout=0.0,
)

TRAIN_CONFIG = TrainConfig(
    lr=1e-3,
    total_steps=200,
    warmup_steps=20,
    weight_decay=0.1,
    max_grad_norm=1.0,
)

ARCHITECTURES = ['gpt', 'llama', 'qwen3', 'gemma', 'mixtral']
BATCH_SIZE = 32
EVAL_INTERVAL = 50
SEED = 42
OUTPUT_DIR = Path('experiments/results/arch_comparison')


def make_data(
    vocab_size: int, num_samples: int, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic train/val data."""
    total = num_samples + num_samples // 4
    tokens = torch.randint(0, vocab_size, (total, seq_len + 1))
    train_data = tokens[:num_samples]
    val_data = tokens[num_samples:]
    return train_data, val_data


def loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Next-token prediction loss."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    return f.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def run_single(
    arch: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> dict:
    """Train one architecture and return results."""
    set_seed(SEED)

    model = create_model(arch, MODEL_CONFIG)
    profile = profile_model(model)

    print(f'\n  {arch.upper()}')
    print(f'    Params: {profile.human_readable_params()}')

    # Define eval callback
    eval_results: list[tuple[int, float]] = []

    def eval_fn(m: nn.Module, step: int) -> dict[str, float]:
        m.eval()
        bpb = compute_bpb(m, val_data, bytes_per_token=1.0)
        eval_results.append((step, bpb))
        print(f'    Step {step:4d} | val_bpb={bpb:.4f}')
        return {'val_bpb': bpb}

    start = time.time()
    state = train_loop(
        model=model,
        train_data=train_data,
        loss_fn=loss_fn,
        config=TRAIN_CONFIG,
        batch_size=BATCH_SIZE,
        eval_fn=eval_fn,
        eval_interval=EVAL_INTERVAL,
    )
    elapsed = time.time() - start

    final_bpb = eval_results[-1][1] if eval_results else None
    best_bpb = min(v for _, v in eval_results) if eval_results else None

    print(f'    Final loss: {state.final_loss:.4f}')
    print(f'    Best BPB:   {best_bpb:.4f}' if best_bpb else '')
    print(f'    Time:       {elapsed:.1f}s')

    return {
        'arch': arch,
        'params': profile.total_params,
        'final_loss': state.final_loss,
        'final_bpb': final_bpb,
        'best_bpb': best_bpb,
        'elapsed_s': elapsed,
        'losses': state.losses,
        'eval_results': eval_results,
    }


def main() -> None:
    """Run architecture comparison experiment."""
    print('=' * 60)
    print('Architecture Comparison Experiment')
    print('=' * 60)
    print(
        f'Config: embed_dim={MODEL_CONFIG.embed_dim}, '
        f'layers={MODEL_CONFIG.num_layers}, '
        f'steps={TRAIN_CONFIG.total_steps}'
    )
    print(f'Architectures: {", ".join(ARCHITECTURES)}')

    # Create data
    train_data, val_data = make_data(
        vocab_size=MODEL_CONFIG.vocab_size,
        num_samples=512,
        seq_len=MODEL_CONFIG.context_length,
    )
    print(
        f'Data: {train_data.shape[0]} train, {val_data.shape[0]} val sequences'
    )

    # Run experiments
    results = []
    registry = ExperimentRegistry(OUTPUT_DIR / 'registry.jsonl')

    for arch in ARCHITECTURES:
        result = run_single(arch, train_data, val_data)
        results.append(result)

        # Register in experiment registry
        tracker = MetricTracker()
        for step, bpb in result['eval_results']:
            tracker.log('val_bpb', bpb, step=step)
        for i, loss in enumerate(result['losses']):
            tracker.log('train_loss', loss, step=i)

        exp_config = ExperimentRunConfig(
            name=f'arch_{arch}',
            lr=TRAIN_CONFIG.lr,
            train_steps=TRAIN_CONFIG.total_steps,
            batch_size=BATCH_SIZE,
        )
        exp_result = ExperimentResult(
            config=exp_config,
            metrics=tracker,
            final_loss=result['final_loss'],
        )
        registry.add(exp_result)

    # Print summary table
    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    header = (
        f'{"Arch":<10} {"Params":>8} {"Loss":>8} '
        f'{"BPB":>8} {"Best BPB":>10} {"Time":>6}'
    )
    print(header)
    print('-' * 60)

    # Sort by best BPB
    results.sort(key=lambda r: r['best_bpb'] or float('inf'))

    for r in results:
        bpb_str = f'{r["final_bpb"]:.4f}' if r['final_bpb'] else '-'
        best_str = f'{r["best_bpb"]:.4f}' if r['best_bpb'] else '-'
        print(
            f'{r["arch"]:<10} {r["params"]:>8,} '
            f'{r["final_loss"]:>8.4f} {bpb_str:>8} '
            f'{best_str:>10} {r["elapsed_s"]:>5.1f}s'
        )

    # Analysis
    bpbs = [r['best_bpb'] for r in results if r['best_bpb']]
    if bpbs:
        spread = max(bpbs) - min(bpbs)
        mean_bpb = sum(bpbs) / len(bpbs)
        print(
            f'\nBPB range: {min(bpbs):.4f} - {max(bpbs):.4f} '
            f'(spread: {spread:.4f}, {spread / mean_bpb * 100:.1f}%)'
        )

        if spread / mean_bpb < 0.05:
            print('=> Architecture choice has <5% effect at this scale')
            print('   (as hypothesized)')
        else:
            print('=> Architecture choice matters even at toy scale!')
            print(f'   Winner: {results[0]["arch"]}')

    print(f'\nResults saved to: {OUTPUT_DIR / "registry.jsonl"}')
    print(registry.summary(metric='val_bpb'))


if __name__ == '__main__':
    main()
