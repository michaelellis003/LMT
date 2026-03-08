"""Tests for HuggingFace weight export.

Exporting LMT models to HF-compatible format (safetensors + config.json)
is required for leaderboard submission and HuggingFace Hub upload.

The critical property: round-trip correctness.
export(model) → load(exported) should produce identical outputs.
"""

import json
from pathlib import Path

import torch

from lmt.models.config import ModelConfig


def _make_tiny_config(*, tie_weights: bool = True) -> ModelConfig:
    """Create a minimal config for testing."""
    return ModelConfig(
        embed_dim=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ffn_hidden_dim=64,
        vocab_size=128,
        context_length=16,
        tie_weights=tie_weights,
        qk_norm=True,
        dropout=0.0,
    )


class TestConfigExport:
    """Test LMT config → HF config.json conversion."""

    def test_qwen3_config_has_required_fields(self):
        """HF config.json must have model_type and architectures."""
        from lmt.models.hf_exporter import lmt_config_to_hf

        config = _make_tiny_config()
        hf_config = lmt_config_to_hf(config, 'qwen3')

        assert hf_config['model_type'] == 'qwen3'
        assert 'architectures' in hf_config
        assert isinstance(hf_config['architectures'], list)

    def test_qwen3_config_dimensions(self):
        """Dimension fields should match LMT config."""
        from lmt.models.hf_exporter import lmt_config_to_hf

        config = _make_tiny_config()
        hf_config = lmt_config_to_hf(config, 'qwen3')

        assert hf_config['hidden_size'] == 32
        assert hf_config['num_attention_heads'] == 4
        assert hf_config['num_key_value_heads'] == 2
        assert hf_config['num_hidden_layers'] == 2
        assert hf_config['intermediate_size'] == 64
        assert hf_config['vocab_size'] == 128

    def test_llama_config_export(self):
        """LLaMA config export with untied weights."""
        from lmt.models.hf_exporter import lmt_config_to_hf

        config = _make_tiny_config(tie_weights=False)
        hf_config = lmt_config_to_hf(config, 'llama')

        assert hf_config['model_type'] == 'llama'
        assert hf_config['tie_word_embeddings'] is False

    def test_config_round_trip(self):
        """Export config → import config should preserve all values."""
        from lmt.models.hf_exporter import lmt_config_to_hf
        from lmt.models.hf_loader import config_from_hf

        original = _make_tiny_config()
        hf_config = lmt_config_to_hf(original, 'qwen3')
        restored = config_from_hf(hf_config)

        assert restored.embed_dim == original.embed_dim
        assert restored.num_heads == original.num_heads
        assert restored.num_kv_heads == original.num_kv_heads
        assert restored.num_layers == original.num_layers
        assert restored.ffn_hidden_dim == original.ffn_hidden_dim
        assert restored.vocab_size == original.vocab_size
        assert restored.tie_weights == original.tie_weights

    def test_unsupported_model_type(self):
        """Should raise ValueError for unsupported model types."""
        import pytest

        from lmt.models.hf_exporter import lmt_config_to_hf

        config = _make_tiny_config()
        with pytest.raises(ValueError, match='Unsupported'):
            lmt_config_to_hf(config, 'unsupported_model')


class TestKeyReversal:
    """Test LMT → HF key mapping (reverse of loading)."""

    def test_reverse_mapping_covers_all_params(self):
        """Every LMT learnable param should map to an HF key."""
        from lmt.models.hf_exporter import build_reverse_key_mapping

        config = _make_tiny_config()
        reverse_map = build_reverse_key_mapping('qwen3', config)

        # Check critical keys exist
        assert 'tok_embed.weight' in reverse_map
        assert 'final_norm.weight' in reverse_map
        assert 'blocks.0.attn.q_proj.weight' in reverse_map
        assert 'blocks.0.ffn.wg.weight' in reverse_map

    def test_reverse_mapping_produces_hf_keys(self):
        """Reversed keys should look like HF state dict keys."""
        from lmt.models.hf_exporter import build_reverse_key_mapping

        config = _make_tiny_config()
        reverse_map = build_reverse_key_mapping('qwen3', config)

        # SwiGLU reverse mapping
        assert reverse_map['blocks.0.ffn.wg.weight'] == (
            'model.layers.0.mlp.gate_proj.weight'
        )
        assert reverse_map['blocks.0.ffn.w1.weight'] == (
            'model.layers.0.mlp.up_proj.weight'
        )
        assert reverse_map['blocks.0.ffn.w2.weight'] == (
            'model.layers.0.mlp.down_proj.weight'
        )

    def test_reverse_is_inverse_of_forward(self):
        """reverse_map[forward_map[hf_key]] == hf_key for all keys."""
        from lmt.models.hf_exporter import build_reverse_key_mapping
        from lmt.models.hf_loader import build_key_mapping

        config = _make_tiny_config()
        forward_map = build_key_mapping('qwen3', config)
        reverse_map = build_reverse_key_mapping('qwen3', config)

        for hf_key, lmt_key in forward_map.items():
            if lmt_key is None:
                continue  # Skipped keys (tied weights)
            assert lmt_key in reverse_map, (
                f'LMT key {lmt_key!r} not in reverse map'
            )
            assert reverse_map[lmt_key] == hf_key


class TestFullExport:
    """Test exporting a complete model to HF format."""

    def test_export_creates_files(self, tmp_path: Path):
        """Export should create model.safetensors and config.json."""
        from lmt.models.hf_exporter import export_to_hf
        from lmt.models.qwen3.qwen3 import Qwen3

        config = _make_tiny_config()
        model = Qwen3(config)

        export_to_hf(model, str(tmp_path), 'qwen3')

        assert (tmp_path / 'model.safetensors').exists()
        assert (tmp_path / 'config.json').exists()

    def test_exported_config_is_valid_json(self, tmp_path: Path):
        """config.json should be valid, parseable JSON."""
        from lmt.models.hf_exporter import export_to_hf
        from lmt.models.qwen3.qwen3 import Qwen3

        config = _make_tiny_config()
        model = Qwen3(config)

        export_to_hf(model, str(tmp_path), 'qwen3')

        with open(tmp_path / 'config.json') as f:
            hf_config = json.load(f)

        assert hf_config['model_type'] == 'qwen3'
        assert hf_config['hidden_size'] == 32

    def test_exported_weights_have_hf_keys(self, tmp_path: Path):
        """Safetensors file should have HF-format keys."""
        from safetensors.torch import load_file

        from lmt.models.hf_exporter import export_to_hf
        from lmt.models.qwen3.qwen3 import Qwen3

        config = _make_tiny_config()
        model = Qwen3(config)

        export_to_hf(model, str(tmp_path), 'qwen3')

        state_dict = load_file(str(tmp_path / 'model.safetensors'))

        # Should have HF keys, not LMT keys
        assert 'model.embed_tokens.weight' in state_dict
        assert 'model.layers.0.self_attn.q_proj.weight' in state_dict
        assert 'model.layers.0.mlp.gate_proj.weight' in state_dict
        # Should NOT have LMT keys
        assert 'tok_embed.weight' not in state_dict
        assert 'blocks.0.attn.q_proj.weight' not in state_dict

    def test_round_trip_weights(self, tmp_path: Path):
        """Export → load should produce identical model weights."""
        from lmt.models.hf_exporter import export_to_hf
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3.qwen3 import Qwen3

        config = _make_tiny_config()
        original = Qwen3(config)

        # Export
        export_to_hf(original, str(tmp_path), 'qwen3')

        # Load into a fresh model
        from safetensors.torch import load_file

        hf_state = load_file(str(tmp_path / 'model.safetensors'))
        restored = Qwen3(config)
        load_hf_state_dict(restored, hf_state, 'qwen3')

        # Compare all parameters
        for name, param in original.named_parameters():
            restored_param = dict(restored.named_parameters())[name]
            assert torch.equal(param, restored_param), (
                f'Parameter {name!r} differs after round-trip'
            )

    def test_round_trip_outputs(self, tmp_path: Path):
        """Export → load should produce identical forward pass outputs."""
        from lmt.models.hf_exporter import export_to_hf
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3.qwen3 import Qwen3

        config = _make_tiny_config()
        original = Qwen3(config)
        original.eval()

        # Export and reload
        export_to_hf(original, str(tmp_path), 'qwen3')

        from safetensors.torch import load_file

        hf_state = load_file(str(tmp_path / 'model.safetensors'))
        restored = Qwen3(config)
        load_hf_state_dict(restored, hf_state, 'qwen3')
        restored.eval()

        # Forward pass comparison
        x = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            out_orig = original(x)
            out_rest = restored(x)

        assert torch.allclose(out_orig, out_rest, atol=1e-6)

    def test_export_llama_untied(self, tmp_path: Path):
        """Export LLaMA with untied weights includes lm_head."""
        from safetensors.torch import load_file

        from lmt.models.hf_exporter import export_to_hf
        from lmt.models.llama.llama import LLaMA

        config = _make_tiny_config(tie_weights=False)
        model = LLaMA(config)

        export_to_hf(model, str(tmp_path), 'llama')

        state_dict = load_file(str(tmp_path / 'model.safetensors'))
        assert 'lm_head.weight' in state_dict
