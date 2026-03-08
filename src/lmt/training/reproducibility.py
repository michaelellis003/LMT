"""Reproducibility utilities for experiments.

Provides deterministic seeding and experiment configuration logging.
Every experiment should be fully specified by a seed + config so that
results can be independently reproduced.

Usage::

    from lmt.training.reproducibility import set_seed, ExperimentConfig

    set_seed(42)
    config = ExperimentConfig(seed=42, model_type='qwen3', ...)
    config.save('experiment_config.json')
"""

import json
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for: Python random, NumPy (if available), PyTorch
    CPU, and PyTorch CUDA (all devices).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
        return 'unknown'
    except Exception:
        return 'unknown'


@dataclass
class ExperimentConfig:
    """Configuration that fully specifies a reproducible experiment.

    Attributes:
        seed: Random seed.
        model_type: Model architecture name.
        description: Human-readable description.
    """

    seed: int = 42
    model_type: str = ''
    description: str = ''

    def to_dict(self) -> dict:
        """Serialize to dict with environment info for reproducibility."""
        d = asdict(self)
        d['git_hash'] = _get_git_hash()
        d['python_version'] = platform.python_version()
        d['torch_version'] = torch.__version__
        d['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            d['cuda_device'] = torch.cuda.get_device_name(0)
        return d

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
