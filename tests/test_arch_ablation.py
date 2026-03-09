"""Tests for architecture ablation experiment."""

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig
from lmt.training.train_config import TrainConfig
from lmt.training.train_loop import TrainState, train_loop


def _tiny_config() -> ModelConfig:
    """Minimal config for ablation tests."""
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


def _loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    return f.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


class TestAblationVariants:
    """Test that all ablation variant configs produce trainable models."""

    def test_gpt_baseline(self):
        """GPT baseline: MHA + default + layernorm + learned pos."""
        cfg = _tiny_config()
        block = BlockConfig(attention='mha', ffn='default', norm='layernorm')
        model = BaseModel(cfg, block_config=block, learned_pos_embed=True)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_gpt_plus_rope(self):
        """GPT + RoPE: MHA + default + layernorm + RoPE."""
        cfg = _tiny_config()
        rope = RoPE(d_model=16, max_seq_len=16)
        block = BlockConfig(
            attention='mha', ffn='default', norm='layernorm', rope=rope
        )
        model = BaseModel(cfg, block_config=block)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_gpt_plus_swiglu(self):
        """GPT + SwiGLU: MHA + swiglu + layernorm + learned pos."""
        cfg = _tiny_config()
        block = BlockConfig(attention='mha', ffn='swiglu', norm='layernorm')
        model = BaseModel(cfg, block_config=block, learned_pos_embed=True)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_gpt_plus_gqa(self):
        """GPT + GQA: GQA + default + layernorm + learned pos."""
        cfg = _tiny_config()
        block = BlockConfig(attention='gqa', ffn='default', norm='layernorm')
        model = BaseModel(cfg, block_config=block, learned_pos_embed=True)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_gpt_plus_rmsnorm(self):
        """GPT + RMSNorm: MHA + default + rmsnorm + learned pos."""
        cfg = _tiny_config()
        block = BlockConfig(attention='mha', ffn='default', norm='rmsnorm')
        model = BaseModel(cfg, block_config=block, learned_pos_embed=True)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_llama_full(self):
        """Full LLaMA: GQA + swiglu + rmsnorm + RoPE."""
        cfg = _tiny_config()
        rope = RoPE(d_model=16, max_seq_len=16)
        block = BlockConfig(
            attention='gqa', ffn='swiglu', norm='rmsnorm', rope=rope
        )
        model = BaseModel(cfg, block_config=block)
        x = torch.randint(0, 64, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 64)

    def test_all_variants_train(self):
        """All 6 variants should train successfully."""
        torch.manual_seed(42)
        data = torch.randint(0, 64, (32, 17))
        config = TrainConfig(total_steps=3, lr=1e-3)
        cfg = _tiny_config()

        variants = [
            (
                BlockConfig(attention='mha', ffn='default', norm='layernorm'),
                True,
            ),
            (
                BlockConfig(attention='mha', ffn='swiglu', norm='layernorm'),
                True,
            ),
            (
                BlockConfig(attention='gqa', ffn='default', norm='layernorm'),
                True,
            ),
            (
                BlockConfig(attention='mha', ffn='default', norm='rmsnorm'),
                True,
            ),
        ]

        for block_config, learned_pos in variants:
            model = BaseModel(
                cfg,
                block_config=block_config,
                learned_pos_embed=learned_pos,
            )
            state = train_loop(
                model=model,
                train_data=data,
                loss_fn=_loss_fn,
                config=config,
                batch_size=8,
            )
            assert isinstance(state, TrainState)
            assert state.step == 3
