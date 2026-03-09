"""Model profiler for parameter counts and memory estimation.

Provides per-layer and total parameter statistics for any PyTorch
model. Useful for comparing architectures and estimating memory.

Usage::

    from lmt.training.profiler import profile_model, format_profile

    profile = profile_model(model)
    print(format_profile(profile))
    print(f'Total: {profile.human_readable_params()}')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


@dataclass
class ModelProfile:
    """Profile of a model's parameters.

    Attributes:
        total_params: Total number of parameters.
        trainable_params: Number of trainable parameters.
        param_bytes: Total memory for parameters in bytes.
        layers: Per-module parameter breakdown.
    """

    total_params: int
    trainable_params: int
    param_bytes: int
    layers: list[dict[str, Any]] = field(default_factory=list)

    def human_readable_params(self) -> str:
        """Format total params in human-readable form.

        Returns:
            String like ``'1.50M'``, ``'2.50B'``, or ``'500'``.
        """
        n = self.total_params
        if n >= 1_000_000_000:
            return f'{n / 1_000_000_000:.2f}B'
        if n >= 1_000_000:
            return f'{n / 1_000_000:.2f}M'
        if n >= 1_000:
            return f'{n / 1_000:.2f}K'
        return str(n)


def profile_model(model: nn.Module) -> ModelProfile:
    """Profile a model's parameters.

    Args:
        model: PyTorch model to profile.

    Returns:
        ModelProfile with total counts and per-layer breakdown.
    """
    total = 0
    trainable = 0
    total_bytes = 0
    layers: list[dict[str, Any]] = []

    for name, module in model.named_children():
        mod_params = 0
        mod_trainable = 0
        mod_bytes = 0

        for p in module.parameters():
            n = p.numel()
            mod_params += n
            mod_bytes += n * p.element_size()
            if p.requires_grad:
                mod_trainable += n

        if mod_params > 0:
            layers.append(
                {
                    'name': name,
                    'type': type(module).__name__,
                    'params': mod_params,
                    'trainable': mod_trainable,
                    'bytes': mod_bytes,
                }
            )

    # Also count total directly (handles shared params correctly)
    for p in model.parameters():
        total += p.numel()
        total_bytes += p.numel() * p.element_size()
        if p.requires_grad:
            trainable += p.numel()

    return ModelProfile(
        total_params=total,
        trainable_params=trainable,
        param_bytes=total_bytes,
        layers=layers,
    )


def format_profile(profile: ModelProfile) -> str:
    """Format a model profile as a readable string.

    Args:
        profile: ModelProfile to format.

    Returns:
        Multi-line string with layer breakdown and totals.
    """
    lines = ['Model Profile']
    lines.append('=' * 60)

    if profile.layers:
        header = f'{"Name":<20} {"Type":<15} {"Params":>10}'
        lines.append(header)
        lines.append('-' * 60)
        for layer in profile.layers:
            line = (
                f'{layer["name"]:<20} '
                f'{layer["type"]:<15} '
                f'{layer["params"]:>10,}'
            )
            lines.append(line)
        lines.append('-' * 60)

    lines.append(
        f'Total params:     {profile.total_params:>10,} '
        f'({profile.human_readable_params()})'
    )
    lines.append(f'Trainable params: {profile.trainable_params:>10,}')
    lines.append(
        f'Param memory:     {profile.param_bytes / 1024 / 1024:>10.2f} MB'
    )

    return '\n'.join(lines)
