"""Normalization layers for language models."""

from .rms_norm import RMSNorm

NORM_REGISTRY: dict[str, type] = {
    'rmsnorm': RMSNorm,
}

__all__ = ['RMSNorm', 'NORM_REGISTRY']
