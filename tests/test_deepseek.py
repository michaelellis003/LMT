"""Tests for DeepSeek-V2 model architecture."""

import torch

from lmt.models.config import ModelConfig


def _make_config(**kwargs: object) -> ModelConfig:
    """Create a small config for testing."""
    defaults = {
        'vocab_size': 256,
        'embed_dim': 64,
        'num_heads': 4,
        'num_kv_heads': 4,
        'num_layers': 2,
        'context_length': 128,
        'dropout': 0.0,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


class TestDeepSeekBlock:
    """Test DeepSeek-V2 transformer block."""

    def test_block_forward_shape(self) -> None:
        """Block preserves input shape."""
        from lmt.models.deepseek.deepseek import DeepSeekBlock

        config = _make_config()
        block = DeepSeekBlock(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
        )
        block.eval()
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_block_gradient_flow(self) -> None:
        """Gradients flow through the block."""
        from lmt.models.deepseek.deepseek import DeepSeekBlock

        config = _make_config()
        block = DeepSeekBlock(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
        )
        x = torch.randn(1, 4, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_block_without_moe(self) -> None:
        """Block works with dense FFN (num_experts=0)."""
        from lmt.models.deepseek.deepseek import DeepSeekBlock

        config = _make_config()
        block = DeepSeekBlock(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=0,
            top_k=0,
        )
        block.eval()
        x = torch.randn(1, 6, 64)
        out = block(x)
        assert out.shape == (1, 6, 64)

    def test_block_aux_loss(self) -> None:
        """Block produces aux_loss when using MoE."""
        from lmt.models.deepseek.deepseek import DeepSeekBlock

        config = _make_config()
        block = DeepSeekBlock(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
        )
        x = torch.randn(1, 4, 64)
        block(x)
        assert block.aux_loss is not None
        assert block.aux_loss.ndim == 0  # scalar


class TestDeepSeekModel:
    """Test full DeepSeek-V2 model."""

    def test_model_forward_shape(self) -> None:
        """Model produces logits with correct shape."""
        from lmt.models.deepseek import DeepSeekV2

        config = _make_config()
        model = DeepSeekV2(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
        )
        model.eval()
        tokens = torch.randint(0, 256, (2, 10))
        logits = model(tokens)
        assert logits.shape == (2, 10, 256)

    def test_model_aux_loss_aggregation(self) -> None:
        """Model aggregates aux_loss from all MoE layers."""
        from lmt.models.deepseek import DeepSeekV2

        config = _make_config()
        model = DeepSeekV2(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
        )
        tokens = torch.randint(0, 256, (1, 6))
        model(tokens)
        assert model.aux_loss.item() >= 0.0

    def test_model_gradient_flow(self) -> None:
        """Gradients flow through entire model."""
        from lmt.models.deepseek import DeepSeekV2

        config = _make_config()
        model = DeepSeekV2(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
        )
        tokens = torch.randint(0, 256, (1, 4))
        logits = model(tokens)
        logits.sum().backward()
        # Check gradients on embedding
        assert model.tok_embed.weight.grad is not None

    def test_model_with_shared_experts(self) -> None:
        """Model works with shared experts in MoE."""
        from lmt.models.deepseek import DeepSeekV2

        config = _make_config()
        model = DeepSeekV2(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
            num_shared_experts=1,
        )
        model.eval()
        tokens = torch.randint(0, 256, (1, 6))
        logits = model(tokens)
        assert logits.shape == (1, 6, 256)

    def test_model_dense_layers(self) -> None:
        """Model supports dense (non-MoE) layers for early layers."""
        from lmt.models.deepseek import DeepSeekV2

        config = _make_config(num_layers=4)
        model = DeepSeekV2(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
            num_dense_layers=2,
        )
        model.eval()
        tokens = torch.randint(0, 256, (1, 6))
        logits = model(tokens)
        assert logits.shape == (1, 6, 256)

    def test_model_no_rope(self) -> None:
        """Model works with rope_dim=0 (pure content attention)."""
        from lmt.models.deepseek import DeepSeekV2

        config = _make_config()
        model = DeepSeekV2(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=0,
            num_experts=4,
            top_k=2,
        )
        model.eval()
        tokens = torch.randint(0, 256, (1, 6))
        logits = model(tokens)
        assert logits.shape == (1, 6, 256)

    def test_weight_init(self) -> None:
        """Scaled residual init is applied."""
        from lmt.models.deepseek import DeepSeekV2

        config = _make_config(num_layers=4)
        model = DeepSeekV2(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
        )
        # out_proj weights should have smaller std than default 0.02
        # due to scaled residual init: 0.02 / sqrt(2 * num_layers)
        from lmt.models.deepseek.deepseek import DeepSeekBlock

        for block in model.blocks:
            blk: DeepSeekBlock = block  # type: ignore[assignment]
            std = blk.attn.out_proj.weight.std().item()
            assert std < 0.02, f'out_proj std {std} should be < 0.02'

    def test_parameter_count(self) -> None:
        """Verify model has reasonable parameter count."""
        from lmt.models.deepseek import DeepSeekV2

        config = _make_config()
        model = DeepSeekV2(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
        )
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        # Should be more than LLaMA equivalent due to MoE experts
        # but manageable for testing
        assert num_params < 10_000_000  # sanity check
