"""Normalization layer implementations.

Registry mapping string keys to normalization classes for use in
configurable blocks.
"""

from .layer_norm import LayerNorm

__all__ = ['LayerNorm']

NORM_REGISTRY: dict[str, type] = {
    'layernorm': LayerNorm,
}
