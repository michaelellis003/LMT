"""Export LMT models to HuggingFace-compatible format.

Converts LMT model weights and config to the format expected by
HuggingFace's ``AutoClass`` system, enabling:

1. Upload to HuggingFace Hub (``huggingface_hub.upload_folder``)
2. Leaderboard submission (Open LLM, BabyLM)
3. Interoperability with HF ecosystem (transformers, vLLM, etc.)

Output structure::

    output_dir/
        model.safetensors  -- weights with HF-format keys
        config.json        -- HF-compatible model configuration

Usage::

    from lmt.models.hf_exporter import export_to_hf
    from lmt.models.qwen3.qwen3 import Qwen3

    model = Qwen3(config)
    # ... train ...
    export_to_hf(model, 'output/', 'qwen3')
"""

import json
from pathlib import Path

import torch

from lmt.models.config import ModelConfig
from lmt.models.hf_loader import build_key_mapping

# HF architectures registry: model_type → class name
_HF_ARCHITECTURES: dict[str, list[str]] = {
    'qwen3': ['Qwen3ForCausalLM'],
    'llama': ['LlamaForCausalLM'],
}


def lmt_config_to_hf(
    config: ModelConfig,
    model_type: str,
) -> dict:
    """Convert an LMT ModelConfig to a HuggingFace config.json dict.

    Args:
        config: LMT model configuration.
        model_type: Model family (``'qwen3'``, ``'llama'``).

    Returns:
        Dict suitable for writing as HF ``config.json``.

    Raises:
        ValueError: If ``model_type`` is not supported.
    """
    if model_type not in _HF_ARCHITECTURES:
        supported = ', '.join(sorted(_HF_ARCHITECTURES))
        raise ValueError(
            f'Unsupported model type {model_type!r}. '
            f'Supported: {supported}'
        )

    hf_config: dict = {
        'model_type': model_type,
        'architectures': _HF_ARCHITECTURES[model_type],
        'hidden_size': config.embed_dim,
        'num_attention_heads': config.num_heads,
        'num_key_value_heads': config.num_kv_heads,
        'num_hidden_layers': config.num_layers,
        'intermediate_size': config.ffn_hidden_dim,
        'vocab_size': config.vocab_size,
        'max_position_embeddings': config.context_length,
        'tie_word_embeddings': config.tie_weights,
        'torch_dtype': 'float32',
    }

    return hf_config


def build_reverse_key_mapping(
    model_type: str,
    config: ModelConfig,
) -> dict[str, str]:
    """Build LMT → HF key mapping (reverse of loading).

    Args:
        model_type: Model family (``'qwen3'``, ``'llama'``).
        config: LMT model configuration.

    Returns:
        Dict mapping LMT parameter name → HF state dict key.
    """
    forward_map = build_key_mapping(model_type, config)

    reverse: dict[str, str] = {}
    for hf_key, lmt_key in forward_map.items():
        if lmt_key is not None:
            reverse[lmt_key] = hf_key

    # For models with untied weights, add out_head → lm_head
    if not config.tie_weights and 'out_head.weight' not in reverse:
        reverse['out_head.weight'] = 'lm_head.weight'

    return reverse


def export_to_hf(
    model: torch.nn.Module,
    output_dir: str,
    model_type: str,
) -> None:
    """Export an LMT model to HuggingFace-compatible format.

    Creates ``model.safetensors`` (weights with HF keys) and
    ``config.json`` in the output directory.

    Args:
        model: Trained LMT model.
        output_dir: Directory to write output files.
        model_type: Model family (``'qwen3'``, ``'llama'``).
    """
    from safetensors.torch import save_file

    config: ModelConfig = model.config  # type: ignore[attr-defined]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build reverse key mapping
    reverse_map = build_reverse_key_mapping(model_type, config)

    # Convert state dict keys from LMT → HF format
    lmt_state = model.state_dict()
    hf_state: dict[str, torch.Tensor] = {}

    # Non-weight buffers that should not be exported
    non_weight_buffers = {'causal_mask', 'cos_cache', 'sin_cache'}

    for lmt_key, tensor in lmt_state.items():
        # Skip computed buffers
        if lmt_key.split('.')[-1] in non_weight_buffers:
            continue

        if lmt_key in reverse_map:
            hf_state[reverse_map[lmt_key]] = tensor
        # For tied weights, tok_embed maps to both embed_tokens
        # and lm_head — skip the duplicate

    # Save weights
    save_file(hf_state, str(out_path / 'model.safetensors'))

    # Save config
    hf_config = lmt_config_to_hf(config, model_type)
    config_path = out_path / 'config.json'
    config_path.write_text(json.dumps(hf_config, indent=2) + '\n')
