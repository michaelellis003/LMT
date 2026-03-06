r"""Mixtral architecture (Mixture of Experts).

Implements the sparse MoE transformer from:
    Jiang et al., 2024 -- "Mixtral of Experts"

Mixtral is LLaMA with two key modifications:
- **MoE FFN**: each layer's dense SwiGLU is replaced by a
  Mixture of Experts layer with top-k routing
- **Sliding Window Attention**: each token attends to the
  previous ``w`` tokens only, with effective receptive field
  growing linearly with depth

Implemented as a thin wrapper around ``BaseModel`` with the
appropriate ``BlockConfig`` (GQA + RoPE + sliding window + MoE).
"""

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig


class Mixtral(BaseModel):
    """Mixtral sparse MoE transformer model.

    LLaMA architecture with MoE FFN and sliding window attention.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        """Initialize Mixtral model.

        Args:
            config: Model configuration.
            num_experts: Experts per MoE layer.
            top_k: Experts activated per token.
        """
        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(
            d_model=head_dim,
            max_seq_len=config.context_length,
        )

        block_config = BlockConfig(
            attention='gqa',
            ffn='moe',
            norm='rmsnorm',
            rope=rope,
            window_size=config.window_size,
            moe_num_experts=num_experts,
            moe_top_k=top_k,
        )

        super().__init__(config, block_config=block_config)
