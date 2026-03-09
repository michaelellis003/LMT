"""Tests for downloading and loading models from HuggingFace Hub.

These tests require internet access and the huggingface_hub package.
They are marked slow since they download model files.
"""

import pytest

try:
    import huggingface_hub as _hf_hub  # noqa: F401

    _has_hf_hub = True
except ImportError:
    _has_hf_hub = False

try:
    import safetensors as _safetensors  # noqa: F401

    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_can_download = _has_hf_hub and _has_safetensors


@pytest.mark.skipif(
    not _can_download, reason='huggingface_hub or safetensors not installed'
)
class TestDownloadHFConfig:
    """Test downloading and parsing HF model configs."""

    @pytest.mark.slow
    def test_download_qwen3_config(self):
        """Download Qwen3-0.6B config from Hub and parse to ModelConfig."""
        from lmt.models.hf_loader import download_hf_config

        config = download_hf_config('Qwen/Qwen3-0.6B')
        assert config.embed_dim == 1024
        assert config.num_heads == 16
        assert config.num_layers == 28
        assert config.vocab_size == 151936
        assert config.qk_norm is True
        assert config.tie_weights is True

    @pytest.mark.slow
    def test_download_returns_model_type(self):
        """download_hf_config should also return the model type string."""
        from lmt.models.hf_loader import download_hf_config

        config, model_type = download_hf_config(
            'Qwen/Qwen3-0.6B', return_model_type=True
        )
        assert model_type == 'qwen3'
        assert config.embed_dim == 1024


@pytest.mark.skipif(
    not _can_download, reason='huggingface_hub or safetensors not installed'
)
class TestLoadFromHub:
    """Test full model loading from HuggingFace Hub."""

    @pytest.mark.slow
    def test_load_qwen3_from_hub(self):
        """Load Qwen3-0.6B from Hub into an LMT model."""
        import torch

        from lmt.models.hf_loader import load_from_hub

        # Use bfloat16 and short context to reduce memory
        model = load_from_hub(
            'Qwen/Qwen3-0.6B',
            dtype=torch.bfloat16,
            context_length=512,
        )

        # Should be a valid model
        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        # Qwen3-0.6B has ~600M params
        assert n_params > 500_000_000
        assert n_params < 700_000_000

        # Should produce finite logits
        x = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 4, 151936)
        assert torch.isfinite(logits).all()
