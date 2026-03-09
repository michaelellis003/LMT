"""Model factory for creating architectures by name.

Provides a clean interface for creating any LMT model architecture
using a string name. Useful for automated experiments, sweeps, and
recipes that need to be architecture-agnostic.

Usage::

    from lmt.models.factory import create_model, list_architectures
    from lmt.models.config import ModelConfig

    config = ModelConfig(embed_dim=256, ...)
    model = create_model('qwen3', config)

    # See all available architectures
    print(list_architectures())  # ['deepseek', 'gemma', 'gpt', ...]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from lmt.models.config import ModelConfig

# Lazy registry: maps lowercase name to (module_path, class_name)
# Avoids importing all models at module load time.
_REGISTRY: dict[str, tuple[str, str]] = {
    'gpt': ('lmt.models.gpt.gpt', 'GPT'),
    'llama': ('lmt.models.llama.llama', 'LLaMA'),
    'qwen3': ('lmt.models.qwen3.qwen3', 'Qwen3'),
    'gemma': ('lmt.models.gemma.gemma', 'Gemma'),
    'mixtral': ('lmt.models.mixtral.mixtral', 'Mixtral'),
    'deepseek': ('lmt.models.deepseek.deepseek', 'DeepSeekV2'),
    'mamba': ('lmt.models.mamba.mamba', 'Mamba'),
    'kimi': ('lmt.models.kimi.kimi', 'Kimi'),
}


def create_model(arch: str, config: ModelConfig) -> nn.Module:
    """Create a model by architecture name.

    Args:
        arch: Architecture name (case-insensitive). One of:
            ``'gpt'``, ``'llama'``, ``'qwen3'``, ``'gemma'``,
            ``'mixtral'``, ``'deepseek'``, ``'mamba'``, ``'kimi'``.
        config: Model configuration.

    Returns:
        Instantiated model (``nn.Module``).

    Raises:
        ValueError: If architecture name is not recognized.
    """
    key = arch.lower()
    if key not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f'Unknown architecture: {arch!r}. Available: {available}'
        )

    module_path, class_name = _REGISTRY[key]

    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)


def list_architectures() -> list[str]:
    """List all available model architecture names.

    Returns:
        Sorted list of architecture name strings.
    """
    return sorted(_REGISTRY.keys())
