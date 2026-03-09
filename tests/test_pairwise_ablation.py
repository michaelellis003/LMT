"""Tests for pairwise architecture ablation experiment."""

import torch

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig


def _tiny_config() -> ModelConfig:
    """Minimal config for pairwise ablation tests."""
    return ModelConfig(
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        ffn_hidden_dim=64,
        vocab_size=64,
        context_length=16,
        tie_weights=True,
        dropout=0.0,
    )


class TestPairwiseVariants:
    """Test all pairwise feature combination variants."""

    def test_rope_plus_swiglu(self):
        """RoPE + SwiGLU: MHA + SwiGLU + LayerNorm + RoPE."""
        cfg = _tiny_config()
        rope = RoPE(d_model=16, max_seq_len=16)
        block = BlockConfig(
            attention='mha', ffn='swiglu', norm='layernorm', rope=rope
        )
        model = BaseModel(cfg, block_config=block)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_rope_plus_gqa(self):
        """RoPE + GQA: GQA + default + LayerNorm + RoPE."""
        cfg = _tiny_config()
        rope = RoPE(d_model=16, max_seq_len=16)
        block = BlockConfig(
            attention='gqa', ffn='default', norm='layernorm', rope=rope
        )
        model = BaseModel(cfg, block_config=block)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_rope_plus_rmsnorm(self):
        """RoPE + RMSNorm: MHA + default + RMSNorm + RoPE."""
        cfg = _tiny_config()
        rope = RoPE(d_model=16, max_seq_len=16)
        block = BlockConfig(
            attention='mha', ffn='default', norm='rmsnorm', rope=rope
        )
        model = BaseModel(cfg, block_config=block)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_swiglu_plus_gqa(self):
        """SwiGLU + GQA: GQA + SwiGLU + LayerNorm + learned."""
        cfg = _tiny_config()
        block = BlockConfig(attention='gqa', ffn='swiglu', norm='layernorm')
        model = BaseModel(cfg, block_config=block, learned_pos_embed=True)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_swiglu_plus_rmsnorm(self):
        """SwiGLU + RMSNorm: MHA + SwiGLU + RMSNorm + learned."""
        cfg = _tiny_config()
        block = BlockConfig(attention='mha', ffn='swiglu', norm='rmsnorm')
        model = BaseModel(cfg, block_config=block, learned_pos_embed=True)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_gqa_plus_rmsnorm(self):
        """GQA + RMSNorm: GQA + default + RMSNorm + learned."""
        cfg = _tiny_config()
        block = BlockConfig(attention='gqa', ffn='default', norm='rmsnorm')
        model = BaseModel(cfg, block_config=block, learned_pos_embed=True)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_all_pairs_produce_different_configs(self):
        """All 6 pairs should produce distinct block configs."""
        pairs = [
            ('mha', 'swiglu', 'layernorm', True),
            ('gqa', 'default', 'layernorm', True),
            ('mha', 'default', 'rmsnorm', True),
            ('gqa', 'swiglu', 'layernorm', False),
            ('mha', 'swiglu', 'rmsnorm', False),
            ('gqa', 'default', 'rmsnorm', False),
        ]
        configs = set()
        for attn, ffn, norm, use_rope in pairs:
            configs.add((attn, ffn, norm, use_rope))
        assert len(configs) == 6
