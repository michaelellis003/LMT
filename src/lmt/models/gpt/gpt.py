"""GPT (Generative Pre-trained Transformer) architecture.

Implements a decoder-only transformer inspired by GPT-2:
- **Multi-Head Attention** with fused QKV projection
- **GELU Feed-Forward Network** (expand 4x, project back)
- **LayerNorm** (pre-norm)
- **Learned positional embeddings**

Implemented as a thin wrapper around ``BaseModel`` with
the appropriate ``BlockConfig`` (MHA + default FFN + LayerNorm).
"""

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig


class GPT(BaseModel):
    """A decoder-only Transformer model inspired by the GPT architecture.

    Composes Multi-Head Attention with GELU FFN, LayerNorm, and
    learned positional embeddings via the shared BaseModel scaffold.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize GPT model.

        Args:
            model_config: Configuration object containing model
                hyperparameters such as vocabulary size, embedding
                dimension, context length, number of layers, and
                dropout rate.
        """
        block_config = BlockConfig(
            attention='mha',
            ffn='default',
            norm='layernorm',
        )

        super().__init__(
            model_config,
            block_config=block_config,
            learned_pos_embed=True,
        )
