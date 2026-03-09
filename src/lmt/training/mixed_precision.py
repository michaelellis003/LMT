"""Mixed precision (AMP) training utility.

Wraps PyTorch's automatic mixed precision (AMP) for clean integration
with the LMT training pipeline. Supports FP16 (with GradScaler) and
BF16 (no scaler needed) modes, with a graceful no-op fallback when
disabled.

Usage::

    amp = AMPContext(enabled=True, dtype=torch.bfloat16, device='cpu')

    with amp.autocast():
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

    scaled_loss = amp.scale_loss(loss)
    scaled_loss.backward()
    amp.unscale_and_step(optimizer, max_grad_norm=1.0, model=model)
    amp.update_scaler()
    optimizer.zero_grad()

BF16 is preferred when hardware supports it (Ampere+ GPUs, Apple
Silicon) because it has the same exponent range as FP32, avoiding
the need for loss scaling.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.nn as nn


class AMPContext:
    """Manages automatic mixed precision for training.

    Encapsulates autocast context, GradScaler (for FP16), and
    gradient unscaling into a single interface.

    Args:
        enabled: Whether to enable mixed precision.
        dtype: Target dtype for autocast (float16 or bfloat16).
        device: Device type for autocast ('cuda', 'cpu').
    """

    def __init__(
        self,
        enabled: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: str = 'cpu',
    ) -> None:
        """Initialize AMP context.

        Args:
            enabled: Whether to enable mixed precision.
            dtype: Target dtype. Use ``torch.bfloat16`` (no scaler) or
                ``torch.float16`` (requires scaler for stability).
            device: Device type string ('cuda', 'cpu', 'mps').
        """
        self.enabled = enabled
        self.dtype = dtype
        self.device = device

        # FP16 needs GradScaler to prevent underflow; BF16 does not
        if enabled and dtype == torch.float16:
            self.scaler: torch.amp.GradScaler | None = (  # type: ignore[attr-defined]
                torch.amp.GradScaler(device)  # type: ignore[attr-defined]
            )
        else:
            self.scaler = None

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """Context manager for automatic mixed precision.

        Yields:
            None. Operations within the block use reduced precision.
        """
        if not self.enabled:
            yield
            return

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for FP16 stability.

        For BF16 or disabled mode, returns loss unchanged.

        Args:
            loss: Raw loss tensor.

        Returns:
            Scaled loss (or unchanged if no scaler).
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def unscale_and_step(
        self,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float | None = None,
        model: nn.Module | None = None,
    ) -> None:
        """Unscale gradients, clip, and step optimizer.

        For FP16: unscales gradients, clips, and uses scaler.step().
        For BF16/disabled: clips and steps directly.

        Args:
            optimizer: Optimizer to step.
            max_grad_norm: Maximum gradient norm for clipping.
            model: Model for gradient clipping. If None, extracts
                parameters from optimizer.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                params = _get_params(optimizer, model)
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            self.scaler.step(optimizer)
        else:
            if max_grad_norm is not None:
                params = _get_params(optimizer, model)
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

    def update_scaler(self) -> None:
        """Update GradScaler after optimizer step.

        No-op when scaler is not active (BF16 or disabled).
        """
        if self.scaler is not None:
            self.scaler.update()


def _get_params(
    optimizer: torch.optim.Optimizer,
    model: nn.Module | None = None,
) -> list[torch.Tensor]:
    """Extract parameters for gradient clipping.

    Args:
        optimizer: Optimizer with parameter groups.
        model: Optional model for direct parameter access.

    Returns:
        List of parameter tensors requiring gradients.
    """
    if model is not None:
        return [p for p in model.parameters() if p.requires_grad]
    params: list[torch.Tensor] = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                params.append(p)
    return params
