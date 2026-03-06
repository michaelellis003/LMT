r"""LLaMA (Large Language Model Meta AI) architecture.

Implements the decoder-only transformer from:
    Touvron et al., 2023 -- "LLaMA: Open and Efficient Foundation
    Language Models"

Key differences from GPT-2:
- **RMSNorm** instead of LayerNorm (simpler, ~same quality)
- **RoPE** instead of learned positional embeddings (relative pos)
- **SwiGLU** instead of GELU FFN (gated activation)
- **GQA** instead of standard MHA (efficient KV cache)
- **No bias** in linear layers

LLaMA is implemented as a thin wrapper around ``BaseModel`` with
the appropriate ``BlockConfig`` (GQA + SwiGLU + RMSNorm + RoPE).
"""

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig


class LLaMA(BaseModel):
    """LLaMA decoder-only transformer model.

    Composes GQA with RoPE, SwiGLU FFN, and RMSNorm via
    the shared BaseModel scaffold.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize LLaMA model.

        Args:
            config: Model configuration with embed_dim, num_heads,
                num_kv_heads, context_length, vocab_size, num_layers.
        """
        head_dim = config.embed_dim // config.num_heads
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
