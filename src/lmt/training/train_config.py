"""Unified training configuration.

Bundles all training hyperparameters into a single config and provides
a helper to set up optimizer + LR scheduler with proper weight decay.

Usage::

    from lmt.training.train_config import TrainConfig, configure_training

    cfg = TrainConfig(lr=3e-4, total_steps=10000, warmup_steps=1000)
    optimizer, scheduler = configure_training(model, cfg)

    for step in range(cfg.total_steps):
        loss = train_step(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from lmt.training.lr_schedule import get_cosine_warmup_scheduler
from lmt.training.param_groups import get_param_groups


@dataclass
class TrainConfig:
    """Configuration for a training run.

    Attributes:
        lr: Peak learning rate.
        weight_decay: Weight decay for AdamW (applied to 2D+ tensors).
        max_grad_norm: Maximum gradient norm for clipping.
        warmup_steps: Linear warmup steps. 0 = no warmup/scheduler.
        total_steps: Total training steps.
        min_lr: Minimum LR at end of cosine decay.
        beta1: AdamW beta1.
        beta2: AdamW beta2.
    """

    lr: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    total_steps: int = 1000
    min_lr: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        """Deserialize from dict, ignoring unknown keys."""
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


def configure_training(
    model: nn.Module,
    config: TrainConfig,
) -> tuple[AdamW, LambdaLR | None]:
    """Set up optimizer and LR scheduler from config.

    Creates AdamW with proper weight decay parameter groups
    (no decay on biases, norms, embeddings) and an optional
    cosine warmup scheduler.

    Args:
        model: Model to optimize.
        config: Training configuration.

    Returns:
        Tuple of (optimizer, scheduler). Scheduler is None if
        ``warmup_steps == 0``.
    """
    param_groups = get_param_groups(
        model,
        weight_decay=config.weight_decay,
        lr=config.lr,
    )

    optimizer = AdamW(
        param_groups,
        betas=(config.beta1, config.beta2),
    )

    scheduler: LambdaLR | None = None
    if config.warmup_steps > 0:
        scheduler = get_cosine_warmup_scheduler(
            optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            min_lr=config.min_lr,
        )

    return optimizer, scheduler
