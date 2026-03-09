r"""Qwen3 architecture.

Implements the decoder-only transformer from:
    Qwen Team, 2025 -- "Qwen3 Technical Report"

Qwen3 is architecturally very close to LLaMA with two key additions:
- **QK-Norm**: RMSNorm applied to Q and K per-head before the dot
  product, preventing attention logit explosion at scale
- **Weight Tying**: input and output embeddings share weights,
  reducing parameter count

The rest is standard modern LLaMA-family:
- **RMSNorm** (pre-norm)
- **RoPE** (rotary positional embeddings)
- **SwiGLU** feed-forward network
- **GQA** (grouped query attention)
- **No bias** in linear layers

Implemented as a thin wrapper around ``BaseModel`` that forces
``qk_norm=True`` and ``tie_weights=True`` in the config.
"""

import dataclasses

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig


class Qwen3(BaseModel):
    """Qwen3 decoder-only transformer model.

    LLaMA architecture with QK-Norm and weight tying.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize Qwen3 model.

        Args:
            config: Model configuration. ``qk_norm`` and
                ``tie_weights`` are forced True regardless of
                the config values.
        """
        # Copy to avoid mutating caller's config
        config = dataclasses.replace(config, qk_norm=True, tie_weights=True)

        head_dim = config.head_dim or config.embed_dim // config.num_heads
        rope = RoPE(
            d_model=head_dim,
            max_seq_len=config.context_length,
        )

        block_config = BlockConfig(
            attention='gqa',
            ffn='swiglu',
            norm='rmsnorm',
            rope=rope,
        )

        super().__init__(config, block_config=block_config)
