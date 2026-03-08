"""HuggingFace weight loading and export for LMT models.

Converts between HuggingFace checkpoint formats (safetensors) and
LMT model parameter names. Supports loading pretrained weights from
the HuggingFace Hub into LMT models for fine-tuning and evaluation.

Supported model families:
- ``qwen3``: Qwen3 (GQA + QK-Norm + SwiGLU + weight tying)
- ``llama``: LLaMA 3.x (GQA + SwiGLU)

Key mapping conventions (HF → LMT):
- ``model.embed_tokens`` → ``tok_embed``
- ``model.layers.{i}`` → ``blocks.{i}``
- ``self_attn.o_proj`` → ``attn.out_proj``
- ``mlp.gate_proj`` → ``ffn.wg`` (SiLU gate)
- ``mlp.up_proj`` → ``ffn.w1`` (element-wise multiply)
- ``mlp.down_proj`` → ``ffn.w2`` (output projection)
"""

import torch
import torch.nn as nn

from lmt.models.config import ModelConfig


def build_key_mapping(
    model_type: str,
    config: ModelConfig,
) -> dict[str, str | None]:
    """Build a mapping from HF state dict keys to LMT parameter names.

    Args:
        model_type: Model family (``'qwen3'``, ``'llama'``).
        config: LMT model config (needed for layer count, qk_norm, etc).

    Returns:
        Dict mapping HF key → LMT key. Value of ``None`` means the
        HF key should be skipped (e.g., tied weights not stored).

    Raises:
        ValueError: If ``model_type`` is not supported.
    """
    if model_type not in _KEY_MAP_BUILDERS:
        supported = ', '.join(sorted(_KEY_MAP_BUILDERS))
        raise ValueError(
            f'Unsupported model type {model_type!r}. Supported: {supported}'
        )

    return _KEY_MAP_BUILDERS[model_type](config)


def _build_qwen3_mapping(config: ModelConfig) -> dict[str, str | None]:
    """Build HF→LMT key mapping for Qwen3."""
    mapping: dict[str, str | None] = {}

    # Top-level
    mapping['model.embed_tokens.weight'] = 'tok_embed.weight'
    mapping['model.norm.weight'] = 'final_norm.weight'
    # Qwen3 ties weights -- lm_head.weight is aliased to embed_tokens.
    # If present in the checkpoint, skip it (mapped to None).
    mapping['lm_head.weight'] = None

    for i in range(config.num_layers):
        hf = f'model.layers.{i}'
        lmt = f'blocks.{i}'

        # Norms
        mapping[f'{hf}.input_layernorm.weight'] = f'{lmt}.attn_norm.weight'
        mapping[f'{hf}.post_attention_layernorm.weight'] = (
            f'{lmt}.ffn_norm.weight'
        )

        # Attention projections
        mapping[f'{hf}.self_attn.q_proj.weight'] = f'{lmt}.attn.q_proj.weight'
        mapping[f'{hf}.self_attn.k_proj.weight'] = f'{lmt}.attn.k_proj.weight'
        mapping[f'{hf}.self_attn.v_proj.weight'] = f'{lmt}.attn.v_proj.weight'
        mapping[f'{hf}.self_attn.o_proj.weight'] = (
            f'{lmt}.attn.out_proj.weight'
        )

        # QK norms (Qwen3-specific)
        if config.qk_norm:
            mapping[f'{hf}.self_attn.q_norm.weight'] = (
                f'{lmt}.attn.q_norm.weight'
            )
            mapping[f'{hf}.self_attn.k_norm.weight'] = (
                f'{lmt}.attn.k_norm.weight'
            )

        # SwiGLU MLP
        # HF gate_proj = SiLU gate = LMT wg
        # HF up_proj = element-wise multiply = LMT w1
        # HF down_proj = output projection = LMT w2
        mapping[f'{hf}.mlp.gate_proj.weight'] = f'{lmt}.ffn.wg.weight'
        mapping[f'{hf}.mlp.up_proj.weight'] = f'{lmt}.ffn.w1.weight'
        mapping[f'{hf}.mlp.down_proj.weight'] = f'{lmt}.ffn.w2.weight'

    return mapping


def _build_llama_mapping(config: ModelConfig) -> dict[str, str | None]:
    """Build HF→LMT key mapping for LLaMA."""
    mapping: dict[str, str | None] = {}

    # Top-level
    mapping['model.embed_tokens.weight'] = 'tok_embed.weight'
    mapping['model.norm.weight'] = 'final_norm.weight'

    # LLaMA may or may not tie weights
    if not config.tie_weights:
        mapping['lm_head.weight'] = 'out_head.weight'

    for i in range(config.num_layers):
        hf = f'model.layers.{i}'
        lmt = f'blocks.{i}'

        # Norms
        mapping[f'{hf}.input_layernorm.weight'] = f'{lmt}.attn_norm.weight'
        mapping[f'{hf}.post_attention_layernorm.weight'] = (
            f'{lmt}.ffn_norm.weight'
        )

        # Attention projections
        mapping[f'{hf}.self_attn.q_proj.weight'] = f'{lmt}.attn.q_proj.weight'
        mapping[f'{hf}.self_attn.k_proj.weight'] = f'{lmt}.attn.k_proj.weight'
        mapping[f'{hf}.self_attn.v_proj.weight'] = f'{lmt}.attn.v_proj.weight'
        mapping[f'{hf}.self_attn.o_proj.weight'] = (
            f'{lmt}.attn.out_proj.weight'
        )

        # QK norms (if model uses them)
        if config.qk_norm:
            mapping[f'{hf}.self_attn.q_norm.weight'] = (
                f'{lmt}.attn.q_norm.weight'
            )
            mapping[f'{hf}.self_attn.k_norm.weight'] = (
                f'{lmt}.attn.k_norm.weight'
            )

        # SwiGLU MLP (same convention as Qwen3)
        mapping[f'{hf}.mlp.gate_proj.weight'] = f'{lmt}.ffn.wg.weight'
        mapping[f'{hf}.mlp.up_proj.weight'] = f'{lmt}.ffn.w1.weight'
        mapping[f'{hf}.mlp.down_proj.weight'] = f'{lmt}.ffn.w2.weight'

    return mapping


# Registry of key mapping builders
_KEY_MAP_BUILDERS: dict = {
    'qwen3': _build_qwen3_mapping,
    'llama': _build_llama_mapping,
}


def config_from_hf(hf_config: dict) -> ModelConfig:
    """Convert a HuggingFace config.json dict to LMT ModelConfig.

    Args:
        hf_config: Parsed config.json from a HF model.

    Returns:
        Equivalent LMT ModelConfig.
    """
    model_type = hf_config.get('model_type', '')

    # Qwen3 always uses QK-norm
    qk_norm = model_type == 'qwen3'

    return ModelConfig(
        embed_dim=hf_config['hidden_size'],
        num_heads=hf_config['num_attention_heads'],
        num_kv_heads=hf_config.get('num_key_value_heads'),
        num_layers=hf_config['num_hidden_layers'],
        vocab_size=hf_config['vocab_size'],
        context_length=hf_config.get('max_position_embeddings', 2048),
        tie_weights=hf_config.get('tie_word_embeddings', False),
        qk_norm=qk_norm,
        dropout=0.0,
    )


def load_hf_state_dict(
    model: nn.Module,
    hf_state_dict: dict[str, torch.Tensor],
    model_type: str,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load HuggingFace weights into an LMT model.

    Args:
        model: An LMT model instance (e.g., Qwen3, LLaMA).
        hf_state_dict: State dict with HF-format keys.
        model_type: Model family (``'qwen3'``, ``'llama'``).
        strict: If True, raise on missing or unexpected keys.

    Returns:
        Tuple of (missing_keys, unexpected_keys).

    Raises:
        RuntimeError: If strict=True and there are missing/unexpected
            keys, or if tensor shapes don't match.
    """
    # Infer config from model
    config: ModelConfig = model.config  # type: ignore[attr-defined]
    mapping = build_key_mapping(model_type, config)

    # Remap HF state dict to LMT keys
    lmt_state: dict[str, torch.Tensor] = {}
    unmapped_hf_keys: list[str] = []

    for hf_key, tensor in hf_state_dict.items():
        if hf_key in mapping:
            lmt_key = mapping[hf_key]
            if lmt_key is not None:
                lmt_state[lmt_key] = tensor
        else:
            unmapped_hf_keys.append(hf_key)

    # Check for shape mismatches before loading
    model_state = model.state_dict()
    for key, tensor in lmt_state.items():
        if key in model_state:
            expected = model_state[key].shape
            actual = tensor.shape
            if expected != actual:
                raise RuntimeError(
                    f'shape mismatch for {key!r}: '
                    f'model expects {list(expected)}, '
                    f'got {list(actual)}'
                )

    # Load with strict=False to get missing/unexpected info,
    # then enforce strictness ourselves
    result = model.load_state_dict(lmt_state, strict=False)

    missing = result.missing_keys
    unexpected = result.unexpected_keys

    # Filter out non-learnable buffers that are computed, not loaded.
    # These are registered buffers in LMT (causal_mask, RoPE caches)
    # that HF models compute on-the-fly.
    non_weight_buffers = {'causal_mask', 'cos_cache', 'sin_cache'}
    missing = [
        k for k in missing if k.split('.')[-1] not in non_weight_buffers
    ]

    # For tied weights, out_head.weight being "missing" is expected
    if config.tie_weights:  # type: ignore[union-attr]
        missing = [k for k in missing if k != 'out_head.weight']

    if strict and missing:
        raise RuntimeError(f'Missing keys when loading HF weights: {missing}')
    if strict and unmapped_hf_keys:
        raise RuntimeError(
            f'Missing key mapping for HF keys: {unmapped_hf_keys}'
        )

    return missing, unexpected


def save_as_safetensors(
    model: nn.Module,
    path: str,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save an LMT model's weights in safetensors format.

    Args:
        model: The LMT model to save.
        path: Output file path (should end in ``.safetensors``).
        metadata: Optional metadata dict to embed in the file.
    """
    from safetensors.torch import save_model

    save_model(model, path, metadata=metadata)
