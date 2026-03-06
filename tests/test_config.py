"""Tests for ModelConfig and ModelConfigPresets."""

from types import SimpleNamespace

from lmt.models.config import ModelConfig, ModelConfigPresets


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config has sensible values."""
        cfg = ModelConfig()
        assert cfg.context_length == 4
        assert cfg.vocab_size == 50257
        assert cfg.num_layers == 12
        assert cfg.num_heads == 12
        assert cfg.embed_dim == 768
        assert cfg.dropout == 0.1
        assert cfg.qkv_bias is False
        assert cfg.ff_network is None
        assert cfg.num_kv_heads is None
        assert cfg.window_size is None

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        cfg = ModelConfig(
            context_length=512,
            vocab_size=32000,
            num_layers=4,
            num_heads=8,
            embed_dim=256,
            dropout=0.0,
            qkv_bias=True,
            num_kv_heads=2,
            window_size=128,
        )
        assert cfg.context_length == 512
        assert cfg.vocab_size == 32000
        assert cfg.num_kv_heads == 2
        assert cfg.window_size == 128

    def test_num_kv_heads_none_means_mha(self) -> None:
        """num_kv_heads=None signals standard MHA."""
        cfg = ModelConfig(num_heads=8)
        assert cfg.num_kv_heads is None


class TestModelConfigPresets:
    """Test preset configurations."""

    def test_gpt2_124m(self) -> None:
        """GPT-2 124M preset has correct architecture."""
        cfg = ModelConfigPresets.gpt2_124m()
        assert cfg.vocab_size == 50257
        assert cfg.num_layers == 12
        assert cfg.num_heads == 12
        assert cfg.embed_dim == 768
        assert cfg.qkv_bias is True
        # embed_dim must be divisible by num_heads
        assert cfg.embed_dim % cfg.num_heads == 0

    def test_gpt2_355m(self) -> None:
        """GPT-2 355M preset has correct architecture."""
        cfg = ModelConfigPresets.gpt2_355m()
        assert cfg.vocab_size == 50257
        assert cfg.num_layers == 24
        assert cfg.num_heads == 16
        assert cfg.embed_dim == 1024
        assert cfg.embed_dim % cfg.num_heads == 0

    def test_small_gpt(self) -> None:
        """Small GPT preset is suitable for testing."""
        cfg = ModelConfigPresets.small_gpt()
        assert cfg.num_layers == 6
        assert cfg.num_heads == 6
        assert cfg.embed_dim == 384
        assert cfg.embed_dim % cfg.num_heads == 0
        # Should be smaller than GPT-2 124M
        assert cfg.embed_dim < 768

    def test_preset_custom_context_length(self) -> None:
        """Presets accept custom context_length."""
        cfg = ModelConfigPresets.gpt2_124m(context_length=2048)
        assert cfg.context_length == 2048

    def test_preset_custom_dropout(self) -> None:
        """Presets accept custom dropout."""
        cfg = ModelConfigPresets.small_gpt(dropout=0.0)
        assert cfg.dropout == 0.0

    def test_from_args(self) -> None:
        """from_args creates config from namespace-like object."""
        args = SimpleNamespace(
            context_length=128,
            vocab_size=1000,
            num_layers=2,
            num_heads=4,
            embed_dim=64,
            dropout=0.0,
        )
        cfg = ModelConfigPresets.from_args(args)
        assert cfg.context_length == 128
        assert cfg.vocab_size == 1000
        assert cfg.num_layers == 2
        assert cfg.num_heads == 4
        assert cfg.embed_dim == 64
        assert cfg.dropout == 0.0

    def test_from_args_with_qkv_bias(self) -> None:
        """from_args picks up qkv_bias when present."""
        args = SimpleNamespace(
            context_length=128,
            vocab_size=1000,
            num_layers=2,
            num_heads=4,
            embed_dim=64,
            dropout=0.0,
            qkv_bias=True,
        )
        cfg = ModelConfigPresets.from_args(args)
        assert cfg.qkv_bias is True

    def test_from_args_without_qkv_bias(self) -> None:
        """from_args defaults qkv_bias to False when absent."""
        args = SimpleNamespace(
            context_length=128,
            vocab_size=1000,
            num_layers=2,
            num_heads=4,
            embed_dim=64,
            dropout=0.0,
        )
        cfg = ModelConfigPresets.from_args(args)
        assert cfg.qkv_bias is False

    def test_all_presets_embed_divisible_by_heads(self) -> None:
        """All presets have embed_dim divisible by num_heads."""
        presets = [
            ModelConfigPresets.gpt2_124m(),
            ModelConfigPresets.gpt2_355m(),
            ModelConfigPresets.small_gpt(),
        ]
        for cfg in presets:
            assert cfg.embed_dim % cfg.num_heads == 0, (
                f'embed_dim={cfg.embed_dim} not divisible '
                f'by num_heads={cfg.num_heads}'
            )
