"""Feed-forward network layers for language models."""

from .moe import MoEFeedForward
from .moe_router import TopKRouter
from .swiglu import SwiGLU

FFN_REGISTRY: dict[str, type] = {
    'swiglu': SwiGLU,
    'moe': MoEFeedForward,
}

__all__ = [
    'SwiGLU',
    'TopKRouter',
    'MoEFeedForward',
    'FFN_REGISTRY',
]
