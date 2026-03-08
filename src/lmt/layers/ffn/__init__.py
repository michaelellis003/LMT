"""Feed-forward network layers for language models."""

from lmt.layers.transformers.ff import DefaultFeedForward

from .moe import MoEFeedForward
from .moe_router import TopKRouter
from .relu_squared import ReluSquaredGLU
from .swiglu import SwiGLU

FFN_REGISTRY: dict[str, type] = {
    'default': DefaultFeedForward,
    'swiglu': SwiGLU,
    'relu_squared': ReluSquaredGLU,
    'moe': MoEFeedForward,
}

__all__ = [
    'DefaultFeedForward',
    'SwiGLU',
    'ReluSquaredGLU',
    'TopKRouter',
    'MoEFeedForward',
    'FFN_REGISTRY',
]
