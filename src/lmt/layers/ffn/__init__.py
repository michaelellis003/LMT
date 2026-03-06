"""Feed-forward network layers for language models."""

from lmt.layers.transformers.ff import DefaultFeedForward

from .moe import MoEFeedForward
from .moe_router import TopKRouter
from .swiglu import SwiGLU

FFN_REGISTRY: dict[str, type] = {
    'default': DefaultFeedForward,
    'swiglu': SwiGLU,
    'moe': MoEFeedForward,
}

__all__ = [
    'DefaultFeedForward',
    'SwiGLU',
    'TopKRouter',
    'MoEFeedForward',
    'FFN_REGISTRY',
]
