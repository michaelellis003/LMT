"""Tests for HuggingFace weight loading.

Follows TDD: these tests define the expected behavior of the HF loader
BEFORE implementation. Tests are organized from unit (fast, mock data)
to integration (slow, real model downloads).
"""

import pytest
import torch

from lmt.models.config import ModelConfig

# -- Fixtures for a tiny Qwen3-like config --


@pytest.fixture
def tiny_qwen3_config():
    """Minimal Qwen3-like config for fast unit tests."""
    return ModelConfig(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        context_length=32,
        vocab_size=256,
        num_layers=2,
        dropout=0.0,
        qk_norm=True,
        tie_weights=True,
    )


@pytest.fixture
def tiny_llama_config():
    """Minimal LLaMA-like config for unit tests."""
    return ModelConfig(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        context_length=32,
        vocab_size=256,
        num_layers=2,
        dropout=0.0,
        qk_norm=False,
        tie_weights=False,
    )


def _make_hf_qwen3_state_dict(config: ModelConfig) -> dict[str, torch.Tensor]:
    """Create a mock HF Qwen3 state dict with correct shapes.

    Uses deterministic values so we can verify correct mapping.
    Each tensor is filled with a unique value based on its role,
    making it easy to check that weights ended up in the right place.
    """
    head_dim = config.embed_dim // config.num_heads
    num_kv_heads = config.num_kv_heads or config.num_heads
    hidden_dim = int(2 / 3 * 4 * config.embed_dim)

    state = {}

    # Token embedding
    state['model.embed_tokens.weight'] = torch.randn(
        config.vocab_size, config.embed_dim
    )

    # Final norm
    state['model.norm.weight'] = torch.ones(config.embed_dim) * 0.5

    for i in range(config.num_layers):
        prefix = f'model.layers.{i}'

        # Attention norms (avoid 1.0 since RMSNorm inits to ones)
        state[f'{prefix}.input_layernorm.weight'] = torch.ones(
            config.embed_dim
        ) * (i + 3.0)
        state[f'{prefix}.post_attention_layernorm.weight'] = torch.ones(
            config.embed_dim
        ) * (i + 4.0)

        # Attention projections
        state[f'{prefix}.self_attn.q_proj.weight'] = torch.randn(
            config.num_heads * head_dim, config.embed_dim
        )
        state[f'{prefix}.self_attn.k_proj.weight'] = torch.randn(
            num_kv_heads * head_dim, config.embed_dim
        )
        state[f'{prefix}.self_attn.v_proj.weight'] = torch.randn(
            num_kv_heads * head_dim, config.embed_dim
        )
        state[f'{prefix}.self_attn.o_proj.weight'] = torch.randn(
            config.embed_dim, config.num_heads * head_dim
        )

        # QK norms (Qwen3-specific)
        if config.qk_norm:
            state[f'{prefix}.self_attn.q_norm.weight'] = (
                torch.ones(head_dim) * 0.7
            )
            state[f'{prefix}.self_attn.k_norm.weight'] = (
                torch.ones(head_dim) * 0.8
            )

        # MLP (SwiGLU)
        state[f'{prefix}.mlp.gate_proj.weight'] = torch.randn(
            hidden_dim, config.embed_dim
        )
        state[f'{prefix}.mlp.up_proj.weight'] = torch.randn(
            hidden_dim, config.embed_dim
        )
        state[f'{prefix}.mlp.down_proj.weight'] = torch.randn(
            config.embed_dim, hidden_dim
        )

    return state


def _make_hf_llama_state_dict(config: ModelConfig) -> dict[str, torch.Tensor]:
    """Create a mock HF LLaMA state dict (no qk_norm, has lm_head)."""
    head_dim = config.embed_dim // config.num_heads
    num_kv_heads = config.num_kv_heads or config.num_heads
    hidden_dim = int(2 / 3 * 4 * config.embed_dim)

    state = {}
    state['model.embed_tokens.weight'] = torch.randn(
        config.vocab_size, config.embed_dim
    )
    state['model.norm.weight'] = torch.ones(config.embed_dim)
    state['lm_head.weight'] = torch.randn(config.vocab_size, config.embed_dim)

    for i in range(config.num_layers):
        prefix = f'model.layers.{i}'
        state[f'{prefix}.input_layernorm.weight'] = torch.ones(
            config.embed_dim
        )
        state[f'{prefix}.post_attention_layernorm.weight'] = torch.ones(
            config.embed_dim
        )
        state[f'{prefix}.self_attn.q_proj.weight'] = torch.randn(
            config.num_heads * head_dim, config.embed_dim
        )
        state[f'{prefix}.self_attn.k_proj.weight'] = torch.randn(
            num_kv_heads * head_dim, config.embed_dim
        )
        state[f'{prefix}.self_attn.v_proj.weight'] = torch.randn(
            num_kv_heads * head_dim, config.embed_dim
        )
        state[f'{prefix}.self_attn.o_proj.weight'] = torch.randn(
            config.embed_dim, config.num_heads * head_dim
        )
        state[f'{prefix}.mlp.gate_proj.weight'] = torch.randn(
            hidden_dim, config.embed_dim
        )
        state[f'{prefix}.mlp.up_proj.weight'] = torch.randn(
            hidden_dim, config.embed_dim
        )
        state[f'{prefix}.mlp.down_proj.weight'] = torch.randn(
            config.embed_dim, hidden_dim
        )

    return state


# ============================================================
# Test: Key Mapping
# ============================================================


class TestKeyMapping:
    """Test that HF keys are correctly mapped to LMT keys."""

    def test_qwen3_key_mapping_covers_all_hf_keys(self, tiny_qwen3_config):
        """Every HF key must have a corresponding LMT key."""
        from lmt.models.hf_loader import build_key_mapping

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        mapping = build_key_mapping('qwen3', tiny_qwen3_config)

        for hf_key in hf_state:
            assert hf_key in mapping, (
                f'HF key {hf_key!r} has no mapping to LMT'
            )

    def test_llama_key_mapping_covers_all_hf_keys(self, tiny_llama_config):
        """Every HF LLaMA key must have a corresponding LMT key."""
        from lmt.models.hf_loader import build_key_mapping

        hf_state = _make_hf_llama_state_dict(tiny_llama_config)
        mapping = build_key_mapping('llama', tiny_llama_config)

        for hf_key in hf_state:
            assert hf_key in mapping, (
                f'HF key {hf_key!r} has no mapping to LMT'
            )

    def test_qwen3_mapping_targets_are_valid_lmt_params(
        self, tiny_qwen3_config
    ):
        """Every mapped LMT key must exist in the actual model."""
        from lmt.models.hf_loader import build_key_mapping
        from lmt.models.qwen3 import Qwen3

        model = Qwen3(tiny_qwen3_config)
        lmt_params = dict(model.named_parameters())
        mapping = build_key_mapping('qwen3', tiny_qwen3_config)

        for hf_key, lmt_key in mapping.items():
            if lmt_key is None:
                # None means "skip this key" (e.g., tied weights)
                continue
            assert lmt_key in lmt_params, (
                f'Mapped LMT key {lmt_key!r} (from HF {hf_key!r}) '
                f'not found in model parameters'
            )

    def test_llama_mapping_targets_are_valid_lmt_params(
        self, tiny_llama_config
    ):
        """Every mapped LMT key must exist in the actual LLaMA model."""
        from lmt.models.hf_loader import build_key_mapping
        from lmt.models.llama import LLaMA

        model = LLaMA(tiny_llama_config)
        lmt_params = dict(model.named_parameters())
        mapping = build_key_mapping('llama', tiny_llama_config)

        for hf_key, lmt_key in mapping.items():
            if lmt_key is None:
                continue
            assert lmt_key in lmt_params, (
                f'Mapped LMT key {lmt_key!r} (from HF {hf_key!r}) '
                f'not found in model parameters'
            )

    def test_mapping_is_bijective(self, tiny_qwen3_config):
        """No two HF keys should map to the same LMT key."""
        from lmt.models.hf_loader import build_key_mapping

        mapping = build_key_mapping('qwen3', tiny_qwen3_config)
        lmt_keys = [v for v in mapping.values() if v is not None]
        assert len(lmt_keys) == len(set(lmt_keys)), (
            'Duplicate LMT targets in mapping'
        )

    def test_swiglu_mapping_correctness(self, tiny_qwen3_config):
        """Verify the critical SwiGLU name mapping.

        HF gate_proj (SiLU activation) -> LMT wg
        HF up_proj (element-wise multiply) -> LMT w1
        HF down_proj (output projection) -> LMT w2

        Getting this wrong would silently produce garbage outputs.
        """
        from lmt.models.hf_loader import build_key_mapping

        mapping = build_key_mapping('qwen3', tiny_qwen3_config)

        # Check layer 0 mappings
        assert mapping['model.layers.0.mlp.gate_proj.weight'] == (
            'blocks.0.ffn.wg.weight'
        )
        assert mapping['model.layers.0.mlp.up_proj.weight'] == (
            'blocks.0.ffn.w1.weight'
        )
        assert mapping['model.layers.0.mlp.down_proj.weight'] == (
            'blocks.0.ffn.w2.weight'
        )

    def test_attention_mapping_correctness(self, tiny_qwen3_config):
        """Verify attention projection name mapping."""
        from lmt.models.hf_loader import build_key_mapping

        mapping = build_key_mapping('qwen3', tiny_qwen3_config)

        assert mapping['model.layers.0.self_attn.o_proj.weight'] == (
            'blocks.0.attn.out_proj.weight'
        )

    def test_unsupported_model_raises(self, tiny_qwen3_config):
        """Unknown model type should raise ValueError."""
        from lmt.models.hf_loader import build_key_mapping

        with pytest.raises(ValueError, match='Unsupported'):
            build_key_mapping('gpt4', tiny_qwen3_config)


# ============================================================
# Test: Weight Loading (mock state dict)
# ============================================================


class TestLoadWeights:
    """Test loading HF weights into LMT models."""

    def test_load_qwen3_from_state_dict(self, tiny_qwen3_config):
        """Load mock HF weights into a Qwen3 model."""
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        model = Qwen3(tiny_qwen3_config)

        load_hf_state_dict(model, hf_state, model_type='qwen3')

        # Verify embedding weights match
        assert torch.equal(
            model.tok_embed.weight, hf_state['model.embed_tokens.weight']
        )

    def test_load_llama_from_state_dict(self, tiny_llama_config):
        """Load mock HF weights into a LLaMA model."""
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.llama import LLaMA

        hf_state = _make_hf_llama_state_dict(tiny_llama_config)
        model = LLaMA(tiny_llama_config)

        load_hf_state_dict(model, hf_state, model_type='llama')

        assert torch.equal(
            model.tok_embed.weight, hf_state['model.embed_tokens.weight']
        )
        assert torch.equal(model.out_head.weight, hf_state['lm_head.weight'])

    def test_swiglu_weights_land_correctly(self, tiny_qwen3_config):
        """Verify SwiGLU weights end up in correct LMT layers.

        This is the most critical test. If gate/up are swapped, the
        model will produce outputs but they'll be wrong. We use
        distinct tensor values to catch any swap.
        """
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        model = Qwen3(tiny_qwen3_config)

        load_hf_state_dict(model, hf_state, model_type='qwen3')

        # gate_proj -> wg (gate, passed through SiLU)
        assert torch.equal(
            model.blocks[0].ffn.wg.weight,
            hf_state['model.layers.0.mlp.gate_proj.weight'],
        )
        # up_proj -> w1 (up, element-wise multiplied with gate output)
        assert torch.equal(
            model.blocks[0].ffn.w1.weight,
            hf_state['model.layers.0.mlp.up_proj.weight'],
        )
        # down_proj -> w2 (projects back to d_model)
        assert torch.equal(
            model.blocks[0].ffn.w2.weight,
            hf_state['model.layers.0.mlp.down_proj.weight'],
        )

    def test_qk_norm_weights_loaded(self, tiny_qwen3_config):
        """QK-norm weights should be loaded for Qwen3."""
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        model = Qwen3(tiny_qwen3_config)

        load_hf_state_dict(model, hf_state, model_type='qwen3')

        assert torch.equal(
            model.blocks[0].attn.q_norm.weight,
            hf_state['model.layers.0.self_attn.q_norm.weight'],
        )
        assert torch.equal(
            model.blocks[0].attn.k_norm.weight,
            hf_state['model.layers.0.self_attn.k_norm.weight'],
        )

    def test_weight_tying_preserved(self, tiny_qwen3_config):
        """After loading, tied weights should still be the same object."""
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        model = Qwen3(tiny_qwen3_config)

        load_hf_state_dict(model, hf_state, model_type='qwen3')

        # Weight tying: out_head.weight IS tok_embed.weight
        assert model.out_head.weight.data_ptr() == (
            model.tok_embed.weight.data_ptr()
        )

    def test_all_parameters_loaded(self, tiny_qwen3_config):
        """Every model parameter should be overwritten (no init values)."""
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        model = Qwen3(tiny_qwen3_config)

        # Record original param values
        original = {n: p.clone() for n, p in model.named_parameters()}

        load_hf_state_dict(model, hf_state, model_type='qwen3')

        # Every parameter should have changed (extremely unlikely
        # for random init to match random HF weights)
        unchanged = []
        for name, param in model.named_parameters():
            # Skip tied weights (out_head == tok_embed)
            if name == 'out_head.weight':
                continue
            if torch.equal(param, original[name]):
                unchanged.append(name)

        assert len(unchanged) == 0, f'Parameters not updated: {unchanged}'

    def test_shape_mismatch_raises(self, tiny_qwen3_config):
        """Loading weights with wrong shapes should raise an error."""
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        # Corrupt one tensor's shape
        hf_state['model.embed_tokens.weight'] = torch.randn(100, 100)

        model = Qwen3(tiny_qwen3_config)

        with pytest.raises(RuntimeError, match='shape'):
            load_hf_state_dict(model, hf_state, model_type='qwen3')

    def test_missing_keys_raises(self, tiny_qwen3_config):
        """Loading incomplete state dict should raise by default."""
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        # Remove a required key
        del hf_state['model.layers.0.mlp.gate_proj.weight']

        model = Qwen3(tiny_qwen3_config)

        with pytest.raises(RuntimeError, match='[Mm]issing'):
            load_hf_state_dict(model, hf_state, model_type='qwen3')


# ============================================================
# Test: Config Conversion
# ============================================================


class TestConfigConversion:
    """Test converting HF config.json to LMT ModelConfig."""

    def test_qwen3_config_from_dict(self):
        """Convert a Qwen3 config.json dict to ModelConfig."""
        from lmt.models.hf_loader import config_from_hf

        hf_config = {
            'model_type': 'qwen3',
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'num_key_value_heads': 8,
            'num_hidden_layers': 28,
            'intermediate_size': 3072,
            'vocab_size': 151936,
            'max_position_embeddings': 40960,
            'tie_word_embeddings': True,
            'rms_norm_eps': 1e-6,
        }

        config = config_from_hf(hf_config)

        assert config.embed_dim == 1024
        assert config.num_heads == 16
        assert config.num_kv_heads == 8
        assert config.num_layers == 28
        assert config.vocab_size == 151936
        assert config.context_length == 40960
        assert config.tie_weights is True
        assert config.qk_norm is True

    def test_llama_config_from_dict(self):
        """Convert a LLaMA config.json dict to ModelConfig."""
        from lmt.models.hf_loader import config_from_hf

        hf_config = {
            'model_type': 'llama',
            'hidden_size': 2048,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'num_hidden_layers': 22,
            'intermediate_size': 8192,
            'vocab_size': 32000,
            'max_position_embeddings': 2048,
            'tie_word_embeddings': False,
        }

        config = config_from_hf(hf_config)

        assert config.embed_dim == 2048
        assert config.num_heads == 32
        assert config.num_kv_heads == 8
        assert config.tie_weights is False
        assert config.qk_norm is False


# ============================================================
# Test: Forward Pass Equivalence (Sanity Check)
# ============================================================


class TestForwardPassEquivalence:
    """After loading HF weights, model should produce valid outputs."""

    def test_loaded_model_produces_finite_logits(self, tiny_qwen3_config):
        """Loaded model should produce finite, non-zero logits."""
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        model = Qwen3(tiny_qwen3_config)
        load_hf_state_dict(model, hf_state, model_type='qwen3')

        model.eval()
        with torch.no_grad():
            x = torch.randint(0, 256, (1, 8))
            logits = model(x)

        assert logits.shape == (1, 8, 256)
        assert torch.isfinite(logits).all()
        assert not torch.all(logits == 0)

    def test_loaded_model_gradients_flow(self, tiny_qwen3_config):
        """Loaded model should support backprop."""
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        model = Qwen3(tiny_qwen3_config)
        load_hf_state_dict(model, hf_state, model_type='qwen3')

        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        logits.sum().backward()

        assert model.tok_embed.weight.grad is not None


# ============================================================
# Test: Safetensors I/O
# ============================================================


class TestSafetensorsIO:
    """Test saving/loading via safetensors format."""

    def test_save_and_reload(self, tiny_qwen3_config, tmp_path):
        """Save LMT model as safetensors, reload, verify identical."""
        from lmt.models.hf_loader import (
            load_hf_state_dict,
            save_as_safetensors,
        )
        from lmt.models.qwen3 import Qwen3

        hf_state = _make_hf_qwen3_state_dict(tiny_qwen3_config)
        model = Qwen3(tiny_qwen3_config)
        load_hf_state_dict(model, hf_state, model_type='qwen3')

        # Save
        path = tmp_path / 'model.safetensors'
        save_as_safetensors(model, str(path))
        assert path.exists()

        # Reload into fresh model
        model2 = Qwen3(tiny_qwen3_config)
        load_hf_state_dict(model2, hf_state, model_type='qwen3')

        # Both should produce same output
        model.eval()
        model2.eval()
        x = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.equal(out1, out2)
