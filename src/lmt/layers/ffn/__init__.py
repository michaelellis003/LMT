"""Feed-forward network layers for language models."""

from .swiglu import SwiGLU

FFN_REGISTRY: dict[str, type] = {
    'swiglu': SwiGLU,
}

__all__ = ['SwiGLU', 'FFN_REGISTRY']
