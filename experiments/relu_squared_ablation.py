r"""Ablation: ReLU² GLU vs SwiGLU on tiny language modeling.

Hypothesis: ReLU² produces sparser activations, which may improve
sample efficiency on tiny models by forcing sharper feature selection.

Experimental design:
- Two models: identical architecture, only FFN activation differs
- Same data, same seed, same optimizer, same LR schedule
- Metric: final train loss and BPB after N steps
- Run 3 seeds for statistical validity

This is a controlled ablation — the only variable is the activation
function. Everything else is held constant.

Usage::

    uv run python experiments/relu_squared_ablation.py
"""

import statistics

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.eval.bpb import compute_bpb
from lmt.layers.ffn.relu_squared import ReluSquaredGLU
from lmt.layers.ffn.swiglu import SwiGLU
from lmt.models.config import ModelConfig
from lmt.training.lr_schedule import cosine_warmup_lr
from lmt.training.reproducibility import set_seed


def create_config() -> ModelConfig:
    """Create model config for ablation."""
    return ModelConfig(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=3,
        ffn_hidden_dim=128,
        vocab_size=256,
        context_length=32,
        tie_weights=True,
        qk_norm=True,
        dropout=0.0,
    )


def train_model(
    model: nn.Module,
    data: torch.Tensor,
    steps: int = 200,
    lr: float = 3e-4,
    batch_size: int = 8,
    warmup_steps: int = 20,
) -> dict:
    """Train a model and return metrics.

    Args:
        model: Model to train.
        data: Token IDs [num_samples, seq_len].
        steps: Number of training steps.
        lr: Peak learning rate.
        batch_size: Training batch size.
        warmup_steps: LR warmup steps.

    Returns:
        Dict with train_losses, final_loss, bpb.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_samples = data.shape[0]

    model.train()
    losses = []

    for step in range(steps):
        # Cosine warmup LR
        current_lr = cosine_warmup_lr(step, lr, warmup_steps, steps)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        # Sample batch
        indices = torch.randint(0, num_samples, (batch_size,))
        batch = data[indices]
        inputs, targets = batch[:, :-1], batch[:, 1:]

        logits = model(inputs)
        loss = f.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Evaluate BPB
    model.eval()
    bpb = compute_bpb(model, data[:32], bytes_per_token=1.0)

    return {
        'train_losses': losses,
        'final_loss': losses[-1],
        'bpb': bpb,
    }


def swap_ffn(
    model: nn.Module,
    ffn_class: type,
    d_model: int,
    hidden_dim: int,
) -> None:
    """Replace all FFN layers in a model with a different type.

    Args:
        model: Model to modify (in-place).
        ffn_class: New FFN class to use.
        d_model: Model embedding dimension.
        hidden_dim: FFN hidden dimension.
    """
    for name, module in model.named_modules():
        if isinstance(module, (SwiGLU, ReluSquaredGLU)):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, attr_name, ffn_class(d_model, hidden_dim))


def main() -> None:
    """Run the ablation study."""
    print('=' * 60)
    print('ABLATION: ReLU² GLU vs SwiGLU')
    print('=' * 60)

    config = create_config()
    seeds = [42, 123, 456]
    steps = 300

    # Generate fixed dataset
    set_seed(0)
    seq_len = config.context_length + 1
    data = torch.randint(0, config.vocab_size, (200, seq_len))

    swiglu_results = []
    relu_sq_results = []

    for seed in seeds:
        print(f'\n--- Seed {seed} ---')

        # Train SwiGLU model
        set_seed(seed)
        from lmt.models.qwen3.qwen3 import Qwen3

        swiglu_model = Qwen3(config)  # Default uses SwiGLU
        n_params = sum(p.numel() for p in swiglu_model.parameters())
        result = train_model(swiglu_model, data, steps=steps)
        swiglu_results.append(result)
        print(
            f'  SwiGLU:  loss={result["final_loss"]:.4f}  '
            f'bpb={result["bpb"]:.4f}'
        )

        # Train ReLU² model (same seed → same init except FFN)
        set_seed(seed)
        relu_sq_model = Qwen3(config)
        swap_ffn(
            relu_sq_model,
            ReluSquaredGLU,
            config.embed_dim,
            config.ffn_hidden_dim,
        )
        result = train_model(relu_sq_model, data, steps=steps)
        relu_sq_results.append(result)
        print(
            f'  ReLU²:   loss={result["final_loss"]:.4f}  '
            f'bpb={result["bpb"]:.4f}'
        )

    # Summary statistics
    print('\n' + '=' * 60)
    print('RESULTS (mean ± std over 3 seeds)')
    print('=' * 60)

    swiglu_losses = [r['final_loss'] for r in swiglu_results]
    relu_sq_losses = [r['final_loss'] for r in relu_sq_results]
    swiglu_bpbs = [r['bpb'] for r in swiglu_results]
    relu_sq_bpbs = [r['bpb'] for r in relu_sq_results]

    print(f'\nParams: {n_params:,}')
    print(f'Steps:  {steps}')
    ctx = config.context_length
    print(f'Data:   {data.shape[0]} sequences, context_length={ctx}')
    print()
    print(
        f'SwiGLU  loss: '
        f'{statistics.mean(swiglu_losses):.4f} ± '
        f'{statistics.stdev(swiglu_losses):.4f}'
    )
    print(
        f'ReLU²   loss: '
        f'{statistics.mean(relu_sq_losses):.4f} ± '
        f'{statistics.stdev(relu_sq_losses):.4f}'
    )
    print(
        f'SwiGLU  bpb:  '
        f'{statistics.mean(swiglu_bpbs):.4f} ± '
        f'{statistics.stdev(swiglu_bpbs):.4f}'
    )
    print(
        f'ReLU²   bpb:  '
        f'{statistics.mean(relu_sq_bpbs):.4f} ± '
        f'{statistics.stdev(relu_sq_bpbs):.4f}'
    )

    # Winner
    swiglu_mean = statistics.mean(swiglu_bpbs)
    relu_sq_mean = statistics.mean(relu_sq_bpbs)
    diff_pct = (swiglu_mean - relu_sq_mean) / swiglu_mean * 100

    winner = 'ReLU²' if relu_sq_mean < swiglu_mean else 'SwiGLU'
    print(f'\nBPB difference: {abs(diff_pct):.1f}% ({winner} lower)')

    if abs(diff_pct) < 2.0:
        print('Conclusion: No meaningful difference at this scale.')
    else:
        print(f'Conclusion: {winner} shows a {abs(diff_pct):.1f}% advantage.')


if __name__ == '__main__':
    main()
