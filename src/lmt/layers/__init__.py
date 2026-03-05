"""Sub-package for various layers used in language models.

Organized by component type:

- ``attention/`` -- Attention mechanisms (MHA, GQA, MLA, etc.)
- ``ffn/`` -- Feed-forward networks (Default, SwiGLU, MoE, etc.)
- ``normalization/`` -- Normalization layers (LayerNorm, RMSNorm, etc.)
- ``positional/`` -- Positional encodings (Learned, RoPE, etc.)
- ``blocks/`` -- Transformer block compositions
- ``transformers/`` -- Legacy transformer block (GPT-2 style)
"""

from .attention import MultiHeadAttention
from .ffn import DefaultFeedForward
from .transformers import TransformerBlock

__all__ = [
    'MultiHeadAttention',
    'TransformerBlock',
    'DefaultFeedForward',
]
