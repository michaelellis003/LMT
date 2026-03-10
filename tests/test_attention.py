"""Unit tests for attention mechanisms."""

import math

import pytest
import torch
import torch.nn as nn

from lmt.layers.attention import MultiHeadAttention
from lmt.layers.attention.grouped_query_attention import GroupedQueryAttention
from lmt.layers.positional import ALiBi, RoPE
from lmt.models.config import ModelConfig, ModelConfigPresets


class TestMultiHeadAttention:
    """Test suite for the MultiHeadAttention layer."""

    def test_multihead_attention_initialization(self):
        """Test MultiHeadAttention initializes correctly."""
        config = ModelConfigPresets.small_gpt()
        attn = MultiHeadAttention(config)

        # Check basic attributes
        assert attn.embed_dim == config.embed_dim
        assert attn.num_heads == config.num_heads
        assert attn.head_dim == config.embed_dim // config.num_heads

        # Check layers
        assert hasattr(attn, 'qkv_proj')
        assert hasattr(attn, 'out_proj')
        assert hasattr(attn, 'causal_mask')

        # Check dimensions
        assert attn.qkv_proj.in_features == config.embed_dim
        assert attn.qkv_proj.out_features == 3 * config.embed_dim
        assert attn.out_proj.in_features == config.embed_dim
        assert attn.out_proj.out_features == config.embed_dim

    def test_multihead_attention_invalid_config(self):
        """Test that invalid configurations raise errors."""
        # embed_dim not divisible by num_heads
        with pytest.raises(
            AssertionError, match='embed_dim must be divisible by num_heads'
        ):
            config = ModelConfig(
                embed_dim=100,  # Not divisible by 3
                num_heads=3,
                context_length=8,
                vocab_size=1000,
                num_layers=1,
            )
            MultiHeadAttention(config)

    def test_multihead_attention_forward_shape(self):
        """Test that MultiHeadAttention forward pass preserves input shape."""
        config = ModelConfigPresets.small_gpt()
        attn = MultiHeadAttention(config)

        batch_size = 2
        seq_length = 8
        input_tensor = torch.randn(batch_size, seq_length, config.embed_dim)

        output = attn(input_tensor)

        assert output.shape == input_tensor.shape

    def test_multihead_attention_causal_mask(self):
        """Test that causal mask is properly applied."""
        config = ModelConfig(
            embed_dim=64,
            num_heads=4,
            context_length=4,
            vocab_size=1000,
            num_layers=1,
            dropout=0.0,  # No dropout for cleaner testing
        )
        attn = MultiHeadAttention(config)
        attn.eval()

        batch_size = 1
        seq_length = 4

        # Create input where each position has a unique pattern
        input_tensor = torch.zeros(batch_size, seq_length, config.embed_dim)
        for i in range(seq_length):
            input_tensor[0, i, :] = (
                i + 1
            )  # Position 0 has 1s, position 1 has 2s, etc.

        with torch.no_grad():
            # Get attention weights by modifying the forward pass temporarily
            x = input_tensor
            b, seq_len, _ = x.shape

            qkv = attn.qkv_proj(x)
            queries, keys, values = qkv.chunk(3, dim=-1)

            queries = queries.view(
                b, seq_len, attn.num_heads, attn.head_dim
            ).transpose(1, 2)
            keys = keys.view(
                b, seq_len, attn.num_heads, attn.head_dim
            ).transpose(1, 2)

            attn_scores = queries @ keys.transpose(-2, -1) * attn.scale
            causal_mask = attn.causal_mask[:seq_len, :seq_len]
            attn_scores = attn_scores + causal_mask
            attn_weights = nn.functional.softmax(attn_scores, dim=-1)

            # Check that attention weights respect causal mask
            # (no attention to future positions)
            for head in range(attn.num_heads):
                weights = attn_weights[0, head]  # [seq_len, seq_len]
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        assert weights[i, j].item() == 0.0, (
                            f'Attention to future at pos ({i}, {j})'
                        )

    def test_multihead_attention_scaling(self):
        """Test that attention scores are properly scaled."""
        config = ModelConfigPresets.small_gpt()
        attn = MultiHeadAttention(config)

        expected_scale = 1.0 / math.sqrt(attn.head_dim)
        assert abs(attn.scale - expected_scale) < 1e-6

    def test_multihead_attention_qkv_bias(self):
        """Test MultiHeadAttention with and without QKV bias."""
        # Test with bias
        config_with_bias = ModelConfig(
            embed_dim=64,
            num_heads=4,
            context_length=8,
            vocab_size=1000,
            num_layers=1,
            qkv_bias=True,
        )
        attn_with_bias = MultiHeadAttention(config_with_bias)
        assert attn_with_bias.qkv_proj.bias is not None

        # Test without bias
        config_without_bias = ModelConfig(
            embed_dim=64,
            num_heads=4,
            context_length=8,
            vocab_size=1000,
            num_layers=1,
            qkv_bias=False,
        )
        attn_without_bias = MultiHeadAttention(config_without_bias)
        assert attn_without_bias.qkv_proj.bias is None

    def test_multihead_attention_different_sequence_lengths(self):
        """Test MultiHeadAttention with different sequence lengths."""
        config = ModelConfig(
            embed_dim=64,
            num_heads=4,
            context_length=16,
            vocab_size=1000,
            num_layers=1,
        )
        attn = MultiHeadAttention(config)

        batch_size = 1

        for seq_length in [1, 4, 8, 16]:
            input_tensor = torch.randn(
                batch_size, seq_length, config.embed_dim
            )
            output = attn(input_tensor)
            assert output.shape == input_tensor.shape

    def test_multihead_attention_gradient_flow(self):
        """Test that gradients flow properly through MultiHeadAttention."""
        config = ModelConfigPresets.small_gpt()
        attn = MultiHeadAttention(config)

        batch_size = 2
        seq_length = 4
        input_tensor = torch.randn(
            batch_size, seq_length, config.embed_dim, requires_grad=True
        )

        output = attn(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for all parameters
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f'No gradient for {name}'
                assert not torch.isnan(param.grad).any(), (
                    f'NaN gradient for {name}'
                )

        # Check input gradient
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()

    def test_multihead_attention_deterministic_inference(self):
        """Test that MultiHeadAttention produces deterministic outputs in eval mode."""
        config = ModelConfigPresets.small_gpt()
        attn = MultiHeadAttention(config)
        attn.eval()

        batch_size = 1
        seq_length = 4
        input_tensor = torch.randn(batch_size, seq_length, config.embed_dim)

        with torch.no_grad():
            output1 = attn(input_tensor)
            output2 = attn(input_tensor)

        torch.testing.assert_close(output1, output2)

    def test_multihead_attention_different_head_counts(self):
        """Test MultiHeadAttention with different numbers of heads."""
        embed_dim = 128

        for num_heads in [1, 2, 4, 8]:
            config = ModelConfig(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_length=8,
                vocab_size=1000,
                num_layers=1,
            )
            attn = MultiHeadAttention(config)

            batch_size = 1
            seq_length = 4
            input_tensor = torch.randn(batch_size, seq_length, embed_dim)

            output = attn(input_tensor)
            assert output.shape == input_tensor.shape
            assert attn.head_dim == embed_dim // num_heads

    def test_multihead_attention_parameter_count(self):
        """Test that parameter count is as expected."""
        config = ModelConfig(
            embed_dim=128,
            num_heads=4,
            context_length=8,
            vocab_size=1000,
            num_layers=1,
            qkv_bias=False,
        )
        attn = MultiHeadAttention(config)

        # Calculate expected parameters
        # qkv_proj: embed_dim * (3 * embed_dim) = 128 * 384
        # out_proj: embed_dim * embed_dim = 128 * 128
        expected_params = (128 * 384) + (128 * 128)

        actual_params = sum(p.numel() for p in attn.parameters())
        assert actual_params == expected_params


class TestMHARoPE:
    """Test MHA with RoPE positional encoding."""

    @pytest.fixture()
    def config(self) -> ModelConfig:
        """Small config for MHA + RoPE tests."""
        return ModelConfig(
            embed_dim=64,
            num_heads=4,
            context_length=32,
            vocab_size=100,
            num_layers=2,
            qkv_bias=False,
        )

    @pytest.fixture()
    def rope(self, config: ModelConfig) -> RoPE:
        """RoPE instance matching config."""
        head_dim = config.embed_dim // config.num_heads
        return RoPE(d_model=head_dim, max_seq_len=config.context_length)

    def test_mha_accepts_rope(self, config: ModelConfig, rope: RoPE) -> None:
        """MHA should accept an optional rope parameter."""
        attn = MultiHeadAttention(config, rope=rope)
        assert attn.rope is rope

    def test_mha_rope_forward_shape(
        self, config: ModelConfig, rope: RoPE
    ) -> None:
        """MHA with RoPE preserves input shape."""
        attn = MultiHeadAttention(config, rope=rope)
        x = torch.randn(2, 8, config.embed_dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_mha_rope_changes_output(
        self, config: ModelConfig, rope: RoPE
    ) -> None:
        """MHA with vs without RoPE should produce different outputs."""
        torch.manual_seed(42)
        attn_no_rope = MultiHeadAttention(config)
        torch.manual_seed(42)
        attn_rope = MultiHeadAttention(config, rope=rope)

        x = torch.randn(1, 8, config.embed_dim)
        with torch.no_grad():
            out_no = attn_no_rope(x)
            out_yes = attn_rope(x)

        # Outputs should differ (RoPE rotates Q/K)
        assert not torch.allclose(out_no, out_yes, atol=1e-5)

    def test_mha_rope_gradient_flow(
        self, config: ModelConfig, rope: RoPE
    ) -> None:
        """Gradients flow through MHA + RoPE."""
        attn = MultiHeadAttention(config, rope=rope)
        x = torch.randn(2, 8, config.embed_dim, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        for name, p in attn.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No gradient for {name}'

    def test_mha_rope_none_is_default(self, config: ModelConfig) -> None:
        """MHA without rope argument behaves as before."""
        attn = MultiHeadAttention(config)
        assert attn.rope is None


class TestMHAALiBi:
    """Test MHA with ALiBi positional encoding."""

    @pytest.fixture()
    def config(self) -> ModelConfig:
        """Small config for MHA + ALiBi tests."""
        return ModelConfig(
            embed_dim=64,
            num_heads=4,
            context_length=32,
            vocab_size=100,
            num_layers=2,
            qkv_bias=False,
        )

    @pytest.fixture()
    def alibi(self, config: ModelConfig) -> ALiBi:
        """ALiBi instance matching config."""
        return ALiBi(
            num_heads=config.num_heads,
            max_seq_len=config.context_length,
        )

    def test_mha_accepts_alibi(
        self, config: ModelConfig, alibi: ALiBi
    ) -> None:
        """MHA should accept an optional alibi parameter."""
        attn = MultiHeadAttention(config, alibi=alibi)
        assert attn.alibi is alibi

    def test_mha_alibi_forward_shape(
        self, config: ModelConfig, alibi: ALiBi
    ) -> None:
        """MHA with ALiBi preserves input shape."""
        attn = MultiHeadAttention(config, alibi=alibi)
        x = torch.randn(2, 8, config.embed_dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_mha_alibi_changes_output(
        self, config: ModelConfig, alibi: ALiBi
    ) -> None:
        """MHA with vs without ALiBi should differ."""
        torch.manual_seed(42)
        attn_no = MultiHeadAttention(config)
        torch.manual_seed(42)
        attn_ali = MultiHeadAttention(config, alibi=alibi)

        x = torch.randn(1, 8, config.embed_dim)
        with torch.no_grad():
            out_no = attn_no(x)
            out_ali = attn_ali(x)
        assert not torch.allclose(out_no, out_ali, atol=1e-5)

    def test_mha_alibi_gradient_flow(
        self, config: ModelConfig, alibi: ALiBi
    ) -> None:
        """Gradients flow through MHA + ALiBi."""
        attn = MultiHeadAttention(config, alibi=alibi)
        x = torch.randn(2, 8, config.embed_dim, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        for name, p in attn.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No gradient for {name}'


class TestGQAALiBi:
    """Test GQA with ALiBi positional encoding."""

    @pytest.fixture()
    def config(self) -> ModelConfig:
        """Small config for GQA + ALiBi tests."""
        return ModelConfig(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=2,
            context_length=32,
            vocab_size=100,
            num_layers=2,
        )

    @pytest.fixture()
    def alibi(self, config: ModelConfig) -> ALiBi:
        """ALiBi instance matching config."""
        return ALiBi(
            num_heads=config.num_heads,
            max_seq_len=config.context_length,
        )

    def test_gqa_accepts_alibi(
        self, config: ModelConfig, alibi: ALiBi
    ) -> None:
        """GQA should accept an optional alibi parameter."""
        attn = GroupedQueryAttention(config, alibi=alibi)
        assert attn.alibi is alibi

    def test_gqa_alibi_forward_shape(
        self, config: ModelConfig, alibi: ALiBi
    ) -> None:
        """GQA with ALiBi preserves input shape."""
        attn = GroupedQueryAttention(config, alibi=alibi)
        x = torch.randn(2, 8, config.embed_dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_alibi_changes_output(
        self, config: ModelConfig, alibi: ALiBi
    ) -> None:
        """GQA with vs without ALiBi should differ."""
        torch.manual_seed(42)
        attn_no = GroupedQueryAttention(config)
        torch.manual_seed(42)
        attn_ali = GroupedQueryAttention(config, alibi=alibi)

        x = torch.randn(1, 8, config.embed_dim)
        with torch.no_grad():
            out_no = attn_no(x)
            out_ali = attn_ali(x)
        assert not torch.allclose(out_no, out_ali, atol=1e-5)

    def test_gqa_rope_and_alibi_exclusive(
        self, config: ModelConfig, alibi: ALiBi
    ) -> None:
        """GQA should reject both rope and alibi simultaneously."""
        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(d_model=head_dim, max_seq_len=config.context_length)
        with pytest.raises(ValueError, match='rope.*alibi'):
            GroupedQueryAttention(config, rope=rope, alibi=alibi)

    def test_gqa_alibi_gradient_flow(
        self, config: ModelConfig, alibi: ALiBi
    ) -> None:
        """Gradients flow through GQA + ALiBi."""
        attn = GroupedQueryAttention(config, alibi=alibi)
        x = torch.randn(2, 8, config.embed_dim, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
