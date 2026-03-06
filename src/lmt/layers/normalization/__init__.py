"""Normalization layers for language models."""

import torch.nn as nn

from .rms_norm import RMSNorm

NORM_REGISTRY: dict[str, type] = {
    'rmsnorm': RMSNorm,
    'layernorm': nn.LayerNorm,
}

__all__ = ['RMSNorm', 'NORM_REGISTRY']
