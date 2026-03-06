"""Positional encoding layers for language models."""

from .alibi import ALiBi
from .rope import RoPE

POSITIONAL_REGISTRY: dict[str, type] = {
    'rope': RoPE,
    'alibi': ALiBi,
}

__all__ = ['RoPE', 'ALiBi', 'POSITIONAL_REGISTRY']
