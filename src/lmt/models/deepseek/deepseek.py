r"""DeepSeek-V2 architecture.

Implements the MoE transformer from:
    DeepSeek-AI, 2024 -- "DeepSeek-V2: A Strong, Economical, and
    Efficient Mixture-of-Experts Language Model"

Key innovations over Mixtral:
- **Multi-Head Latent Attention (MLA)**: compresses KV into a
  low-rank latent space with decoupled RoPE, dramatically
  reducing KV cache size
- **DeepSeekMoE**: fine-grained expert segmentation with
  optional shared experts for stable base representations
- **Dense early layers**: first ``num_dense_layers`` use
  standard SwiGLU FFN instead of MoE

The aux_loss from all MoE layers is summed and stored as
``self.aux_loss`` for the training loop.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from lmt.layers.attention.multi_head_latent_attention import (
    MultiHeadLatentAttention,
)
from lmt.layers.ffn import SwiGLU
from lmt.layers.ffn.moe import MoEFeedForward
from lmt.layers.normalization import RMSNorm
from lmt.models.config import ModelConfig


class DeepSeekBlock(nn.Module):
    """A single DeepSeek-V2 block: RMSNorm + MLA + MoE/SwiGLU.

    Args:
        config: Model configuration.
        kv_compress_dim: KV latent dimension for MLA.
        q_compress_dim: Q latent dimension for MLA.
        rope_dim: Dimension for decoupled RoPE (0 = disabled).
        num_experts: Number of MoE experts (0 = dense SwiGLU).
        top_k: Experts activated per token.
        num_shared_experts: Always-active shared experts.
    """

    def __init__(
        self,
        config: ModelConfig,
        kv_compress_dim: int,
        q_compress_dim: int,
        rope_dim: int,
        num_experts: int = 0,
        top_k: int = 0,
        num_shared_experts: int = 0,
    ) -> None:
        """Initialize DeepSeekBlock.

        Args:
            config: Model configuration.
            kv_compress_dim: KV latent dimension for MLA.
            q_compress_dim: Q latent dimension for MLA.
            rope_dim: Decoupled RoPE dimension (0 = disabled).
            num_experts: MoE experts (0 = dense SwiGLU FFN).
            top_k: Experts per token.
            num_shared_experts: Shared experts count.
        """
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim)
        self.attn = MultiHeadLatentAttention(
            config,
            kv_compress_dim=kv_compress_dim,
            q_compress_dim=q_compress_dim,
            rope_dim=rope_dim,
        )
        self.ffn_norm = RMSNorm(config.embed_dim)

        self.use_moe = num_experts > 0
        if self.use_moe:
            self.ffn = MoEFeedForward(
                d_model=config.embed_dim,
                num_experts=num_experts,
                top_k=top_k,
                num_shared_experts=num_shared_experts,
            )
        else:
            self.ffn = SwiGLU(config.embed_dim)  # type: ignore[assignment]

        self.aux_loss: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply DeepSeek-V2 block.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output with same shape.
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))

        if self.use_moe:
            self.aux_loss = self.ffn.aux_loss  # type: ignore[union-attr]
        else:
            self.aux_loss = None

        return x


class DeepSeekV2(nn.Module):
    """DeepSeek-V2 sparse MoE transformer with MLA.

    Combines Multi-Head Latent Attention for efficient KV cache
    with DeepSeekMoE for sparse expert routing. Early layers can
    use dense SwiGLU FFN via ``num_dense_layers``.
    """

    def __init__(
        self,
        config: ModelConfig,
        kv_compress_dim: int,
        q_compress_dim: int,
        rope_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        num_shared_experts: int = 0,
        num_dense_layers: int = 0,
    ) -> None:
        """Initialize DeepSeek-V2 model.

        Args:
            config: Model configuration.
            kv_compress_dim: KV latent dimension for MLA.
            q_compress_dim: Q latent dimension for MLA.
            rope_dim: Decoupled RoPE dimension (0 = disabled).
            num_experts: Experts per MoE layer.
            top_k: Experts activated per token.
            num_shared_experts: Always-active shared experts.
            num_dense_layers: Number of early layers using dense
                SwiGLU instead of MoE.
        """
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        blocks = []
        for i in range(config.num_layers):
            if i < num_dense_layers:
                # Dense layer: no MoE
                block = DeepSeekBlock(
                    config,
                    kv_compress_dim=kv_compress_dim,
                    q_compress_dim=q_compress_dim,
                    rope_dim=rope_dim,
                    num_experts=0,
                    top_k=0,
                )
            else:
                # MoE layer
                block = DeepSeekBlock(
                    config,
                    kv_compress_dim=kv_compress_dim,
                    q_compress_dim=q_compress_dim,
                    rope_dim=rope_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    num_shared_experts=num_shared_experts,
                )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.embed_dim)
        self.out_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False
        )
        self.aux_loss = torch.tensor(0.0)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with scaled residual init."""
        num_layers = len(self.blocks)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scaled residual init for residual projections
        std = 0.02 / math.sqrt(2 * num_layers)
        for module in self.blocks:
            block: DeepSeekBlock = module  # type: ignore[assignment]
            # Scale attention output projection
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            # Scale FFN output projection (w2 for SwiGLU / each expert)
            ffn = block.ffn
            if isinstance(ffn, MoEFeedForward):
                for expert in ffn.experts:
                    nn.init.normal_(expert.w2.weight, mean=0.0, std=std)
                for expert in ffn.shared_experts:
                    nn.init.normal_(expert.w2.weight, mean=0.0, std=std)
            elif isinstance(ffn, SwiGLU):
                nn.init.normal_(ffn.w2.weight, mean=0.0, std=std)

    def forward(self, in_idx: Tensor) -> Tensor:
        """Forward pass of DeepSeek-V2.

        Args:
            in_idx: Token indices ``[batch, seq_len]``.

        Returns:
            Logits ``[batch, seq_len, vocab_size]``.
        """
        x = self.tok_embed(in_idx)
        total_aux = torch.tensor(0.0, device=x.device)
        for module in self.blocks:
            blk: DeepSeekBlock = module  # type: ignore[assignment]
            x = blk(x)
            if blk.aux_loss is not None:
                total_aux = total_aux + blk.aux_loss
        self.aux_loss = total_aux
        x = self.final_norm(x)
        return self.out_head(x)
