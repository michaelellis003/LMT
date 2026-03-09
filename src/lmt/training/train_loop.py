"""Composable training loop for research recipes.

Provides a single ``train_loop()`` function that handles the complete
training pipeline: optimizer setup, gradient accumulation, gradient
clipping, LR scheduling, periodic evaluation, and metric tracking.

Designed to be the core building block for all training recipes,
eliminating boilerplate while remaining flexible through callbacks.

Usage::

    from lmt.training.train_loop import train_loop
    from lmt.training.train_config import TrainConfig


    def loss_fn(model, batch):
        logits = model(batch[:, :-1])
        return F.cross_entropy(logits.view(-1, V), batch[:, 1:].view(-1))


    state = train_loop(
        model=model,
        train_data=data,  # [N, seq_len] tensor
        loss_fn=loss_fn,
        config=TrainConfig(total_steps=500, lr=1e-3, warmup_steps=50),
        batch_size=32,
        eval_fn=lambda m, s: {'val_bpb': compute_bpb(m, val_data)},
        eval_interval=50,
    )
    print(f'Final loss: {state.final_loss:.4f}')
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from lmt.training.train_config import configure_training


@dataclass
class TrainState:
    """Result of a training run.

    Attributes:
        step: Final step number.
        losses: Per-step training loss values.
        lrs: Per-step learning rate values.
        eval_metrics: Dict mapping metric names to lists of
            ``(step, value)`` tuples from eval callbacks.
    """

    step: int
    losses: list[float]
    lrs: list[float]
    eval_metrics: dict[str, list[tuple[int, float]]] = field(
        default_factory=dict
    )

    @property
    def final_loss(self) -> float | None:
        """Return the last training loss, or None if empty."""
        if not self.losses:
            return None
        return self.losses[-1]


def train_loop(
    model: nn.Module,
    train_data: torch.Tensor,
    loss_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    config: Any,
    batch_size: int = 32,
    accumulation_steps: int = 1,
    eval_fn: Callable[[nn.Module, int], dict[str, float]] | None = None,
    eval_interval: int = 100,
) -> TrainState:
    """Run a complete training loop.

    Args:
        model: Model to train.
        train_data: Training data tensor of shape ``[N, seq_len]``.
        loss_fn: Function ``(model, batch) -> loss`` that computes
            the training loss for a batch.
        config: Training config (``TrainConfig`` or compatible object
            with ``lr``, ``total_steps``, ``warmup_steps``, etc.).
        batch_size: Micro-batch size for each forward pass.
        accumulation_steps: Number of micro-batches per optimizer
            step. Effective batch = ``batch_size * accumulation_steps``.
        eval_fn: Optional callback ``(model, step) -> metrics_dict``
            called at ``eval_interval`` steps. Return dict maps
            metric names to float values.
        eval_interval: Steps between eval callbacks.

    Returns:
        ``TrainState`` with full training history.
    """
    optimizer, scheduler = configure_training(model, config)

    num_samples = train_data.shape[0]
    losses: list[float] = []
    lrs: list[float] = []
    eval_metrics: dict[str, list[tuple[int, float]]] = {}

    model.train()
    optimizer.zero_grad()
    micro_step = 0

    for step in range(1, config.total_steps + 1):
        # Sample a random micro-batch
        indices = torch.randint(0, num_samples, (batch_size,))
        batch = train_data[indices]

        # Forward + backward
        loss = loss_fn(model, batch)
        if accumulation_steps > 1:
            loss = loss / accumulation_steps
        loss.backward()
        micro_step += 1

        # Step optimizer after accumulation_steps micro-batches
        if micro_step >= accumulation_steps:
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            micro_step = 0

        # Track metrics
        losses.append(loss.item() * accumulation_steps)
        lrs.append(optimizer.param_groups[0]['lr'])

        # Periodic evaluation
        if eval_fn is not None and step % eval_interval == 0:
            model.eval()
            metrics = eval_fn(model, step)
            model.train()
            for name, value in metrics.items():
                if name not in eval_metrics:
                    eval_metrics[name] = []
                eval_metrics[name].append((step, value))

    return TrainState(
        step=config.total_steps,
        losses=losses,
        lrs=lrs,
        eval_metrics=eval_metrics,
    )
