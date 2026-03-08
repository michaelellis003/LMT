r"""Learning rate schedules for transformer training.

Implements cosine annealing with linear warmup, the standard schedule
for modern LLM pretraining (GPT-3, LLaMA, Chinchilla, etc.).

.. math::

    \text{lr}(t) = \begin{cases}
        \text{max\_lr} \cdot \frac{t}{\text{warmup}}
            & \text{if } t < \text{warmup} \\
        \text{min\_lr} + \frac{\text{max\_lr} - \text{min\_lr}}{2}
        \left(1 + \cos\!\left(\pi \cdot
        \frac{t - \text{warmup}}{\text{total} - \text{warmup}}
        \right)\right)
            & \text{otherwise}
    \end{cases}

Usage::

    from lmt.training.lr_schedule import get_cosine_warmup_scheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_cosine_warmup_scheduler(
        optimizer, warmup_steps=2000, total_steps=100000
    )
"""

import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_lr(
    step: int,
    max_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    r"""Compute learning rate at a given step.

    Linear warmup from 0 to ``max_lr`` over ``warmup_steps``, then
    cosine decay to ``min_lr`` over the remaining steps.

    Args:
        step: Current training step (0-indexed).
        max_lr: Peak learning rate (reached at end of warmup).
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate at end of decay.

    Returns:
        Learning rate as a float. Always non-negative.
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / max(warmup_steps, 1)

    if step >= total_steps:
        return min_lr

    # Cosine decay
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_factor


def get_cosine_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> LambdaLR:
    """Create a PyTorch LR scheduler with cosine warmup.

    Wraps :func:`cosine_warmup_lr` in a ``LambdaLR`` scheduler
    compatible with PyTorch's training loop.

    Args:
        optimizer: PyTorch optimizer with ``lr`` set to ``max_lr``.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate at end of decay.

    Returns:
        LambdaLR scheduler. Call ``scheduler.step()`` after each
        ``optimizer.step()``.
    """
    max_lr = optimizer.param_groups[0]['lr']

    def lr_lambda(step: int) -> float:
        """Compute LR multiplier for LambdaLR."""
        return (
            cosine_warmup_lr(step, max_lr, warmup_steps, total_steps, min_lr)
            / max_lr
        )

    return LambdaLR(optimizer, lr_lambda)
