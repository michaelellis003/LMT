"""Gradient accumulation for memory-efficient training.

Enables training with larger effective batch sizes by accumulating
gradients over multiple micro-batches before performing an optimizer
step. The loss is scaled by ``1 / accumulation_steps`` so that the
accumulated gradient equals the gradient from a single large batch.

Usage::

    acc = GradientAccumulator(accumulation_steps=4, max_grad_norm=1.0)

    for micro_batch in data_loader:
        loss = model(micro_batch)
        acc.scale_loss(loss).backward()
        acc.micro_step += 1

        if acc.should_step():
            acc.step(optimizer, scheduler=scheduler)

The numerical result is equivalent to training with
``batch_size * accumulation_steps`` samples per optimizer step,
but uses only ``batch_size`` memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradientAccumulator:
    """Manages gradient accumulation across micro-batches.

    Tracks micro-step progress and handles loss scaling, gradient
    clipping, optimizer stepping, and gradient zeroing.

    Args:
        accumulation_steps: Number of micro-batches per optimizer step.
        max_grad_norm: Maximum gradient norm for clipping. None to skip.
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
    ) -> None:
        """Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of micro-batches to accumulate
                before stepping the optimizer. Must be >= 1.
            max_grad_norm: If set, clip gradient norm to this value
                before each optimizer step.

        Raises:
            ValueError: If accumulation_steps < 1.
        """
        if accumulation_steps < 1:
            raise ValueError(
                f'accumulation_steps must be >= 1, got {accumulation_steps}'
            )
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.micro_step = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation.

        Divides the loss by ``accumulation_steps`` so that accumulated
        gradients match a single large-batch gradient.

        Args:
            loss: Raw loss value from the model.

        Returns:
            Scaled loss (``loss / accumulation_steps``).
        """
        if self.accumulation_steps == 1:
            return loss
        return loss / self.accumulation_steps

    def should_step(self) -> bool:
        """Check if it's time to step the optimizer.

        Returns:
            True if ``micro_step`` has reached ``accumulation_steps``.
        """
        return self.micro_step >= self.accumulation_steps

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        model: nn.Module | None = None,
    ) -> None:
        """Perform optimizer step with optional clipping and scheduling.

        Clips gradients (if ``max_grad_norm`` is set), steps the
        optimizer, optionally steps the scheduler, zeros gradients,
        and resets the micro-step counter.

        Args:
            optimizer: The optimizer to step.
            scheduler: Optional LR scheduler to step after the optimizer.
            model: Model whose parameters to clip. Required if
                ``max_grad_norm`` is set. If None and clipping is
                requested, parameters are extracted from the optimizer.
        """
        if self.max_grad_norm is not None:
            params = _get_params(optimizer, model)
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        self.micro_step = 0


def _get_params(
    optimizer: torch.optim.Optimizer,
    model: nn.Module | None = None,
) -> list[torch.Tensor]:
    """Extract parameters for gradient clipping.

    Args:
        optimizer: Optimizer containing parameter groups.
        model: Optional model to get parameters from directly.

    Returns:
        List of parameter tensors.
    """
    if model is not None:
        return [p for p in model.parameters() if p.requires_grad]
    params: list[torch.Tensor] = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                params.append(p)
    return params
