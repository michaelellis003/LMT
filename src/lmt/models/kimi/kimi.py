r"""Kimi K2 architecture.

Implements the MoE transformer from:
    Kimi Team, Moonshot AI, 2025 -- "Kimi K2: From Exploration
    to Mastery"

Kimi K2 is architecturally a DeepSeek-V2/V3-family model:
- **Multi-Head Latent Attention (MLA)**: compressed KV cache
  with decoupled RoPE, same as DeepSeek-V2
- **MoE FFN**: sparse mixture-of-experts with shared experts
- **Dense early layers**: first few layers use standard SwiGLU
  instead of MoE for stable representations
- **RMSNorm** (pre-norm)

The main innovation of Kimi K2 is on the training side
(MuonClip optimizer, multi-stage training), not architecture.
This wrapper provides an easy entry point for users who want
to instantiate a Kimi-style model.

Since the architecture is DeepSeek-V2, this is a thin re-export
with Kimi-appropriate defaults and docstrings.
"""

from torch import Tensor

from lmt.models.config import ModelConfig
from lmt.models.deepseek import DeepSeekV2


class Kimi(DeepSeekV2):
    """Kimi K2 MoE transformer model.

    DeepSeek-V2 architecture (MLA + MoE) with Kimi-appropriate
    defaults: smaller KV/Q compression dims and fewer experts
    for the sub-1B scale we target.
    """

    def __init__(
        self,
        config: ModelConfig,
        kv_compress_dim: int = 32,
        q_compress_dim: int = 48,
        rope_dim: int = 16,
        num_experts: int = 4,
        top_k: int = 2,
        num_shared_experts: int = 1,
        num_dense_layers: int = 1,
    ) -> None:
        """Initialize Kimi model.

        Args:
            config: Model configuration.
            kv_compress_dim: KV latent dimension for MLA.
            q_compress_dim: Q latent dimension for MLA.
            rope_dim: Decoupled RoPE dimension.
            num_experts: Experts per MoE layer.
            top_k: Experts activated per token.
            num_shared_experts: Always-active shared experts.
            num_dense_layers: Dense layers before MoE layers.
        """
        super().__init__(
            config,
            kv_compress_dim=kv_compress_dim,
            q_compress_dim=q_compress_dim,
            rope_dim=rope_dim,
            num_experts=num_experts,
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            num_dense_layers=num_dense_layers,
        )

    def forward(self, in_idx: Tensor) -> Tensor:
        """Forward pass of Kimi K2.

        Args:
            in_idx: Token indices ``[batch, seq_len]``.

        Returns:
            Logits ``[batch, seq_len, vocab_size]``.
        """
        return super().forward(in_idx)
