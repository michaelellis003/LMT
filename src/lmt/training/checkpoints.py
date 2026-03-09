"""Checkpoint manager for training and experiment tracking.

Saves and loads model checkpoints with metadata (step, metrics,
config). Supports best-model tracking for autoresearch experiments.

Usage::

    from lmt.training.checkpoints import CheckpointManager

    manager = CheckpointManager('runs/my_experiment/checkpoints')
    manager.save(model, step=100, metrics={'loss': 0.5, 'bpb': 1.2})

    # Later: load best or latest
    meta = manager.load_latest(model)
    # Or save only if improved:
    manager.save_if_best(model, step=200, metric_value=1.1, metric_name='bpb')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages model checkpoints with metadata.

    Saves checkpoints in a structured directory layout::

        base_dir/
            step_10/
                model.pt
                metadata.json
                optimizer.pt  (optional)
            step_20/
                model.pt
                metadata.json
            best/
                model.pt
                metadata.json

    Args:
        base_dir: Root directory for checkpoints.
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Initialize checkpoint manager.

        Args:
            base_dir: Root directory for checkpoints.
        """
        self.base_dir = Path(base_dir)
        self._best_values: dict[str, float] = {}

    def save(
        self,
        model: nn.Module,
        step: int,
        metrics: dict[str, float] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a checkpoint with metadata.

        Args:
            model: Model to save.
            step: Training step number.
            metrics: Optional metrics dict (loss, bpb, etc.).
            optimizer: Optional optimizer state to save.
            extra: Optional extra metadata.

        Returns:
            Path to the checkpoint directory.
        """
        ckpt_dir = self.base_dir / f'step_{step}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(model.state_dict(), ckpt_dir / 'model.pt')

        # Save optimizer if provided
        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / 'optimizer.pt')

        # Save metadata
        metadata: dict[str, Any] = {
            'step': step,
            'metrics': metrics or {},
        }
        if extra:
            metadata['extra'] = extra

        meta_path = ckpt_dir / 'metadata.json'
        meta_path.write_text(json.dumps(metadata, indent=2) + '\n')

        return ckpt_dir

    def load(
        self,
        model: nn.Module,
        step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint by step number.

        Args:
            model: Model to load weights into.
            step: Step number of checkpoint to load.
            optimizer: Optional optimizer to load state into.

        Returns:
            Metadata dict with step, metrics, etc.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        ckpt_dir = self.base_dir / f'step_{step}'
        return self._load_from_dir(ckpt_dir, model, optimizer)

    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any] | None:
        """Load the most recent checkpoint.

        Args:
            model: Model to load weights into.
            optimizer: Optional optimizer to load state into.

        Returns:
            Metadata dict, or None if no checkpoints exist.
        """
        ckpts = self.list_checkpoints()
        if not ckpts:
            return None

        latest = ckpts[-1]
        return self.load(model, latest['step'], optimizer)

    def save_if_best(
        self,
        model: nn.Module,
        step: int,
        metric_value: float,
        metric_name: str = 'loss',
        higher_is_better: bool = False,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> bool:
        """Save checkpoint only if metric improved.

        Args:
            model: Model to save.
            step: Training step number.
            metric_value: Current metric value.
            metric_name: Name of the metric to track.
            higher_is_better: If True, higher values are better.
            optimizer: Optional optimizer state.

        Returns:
            True if checkpoint was saved (metric improved).
        """
        prev_best = self._best_values.get(metric_name)

        is_better = prev_best is None or (
            (metric_value > prev_best)
            if higher_is_better
            else (metric_value < prev_best)
        )

        if is_better:
            self._best_values[metric_name] = metric_value
            best_dir = self.base_dir / 'best'
            best_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), best_dir / 'model.pt')

            if optimizer is not None:
                torch.save(
                    optimizer.state_dict(),
                    best_dir / 'optimizer.pt',
                )

            metadata = {
                'step': step,
                'metrics': {metric_name: metric_value},
                'is_best': True,
            }
            meta_path = best_dir / 'metadata.json'
            meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
            return True

        return False

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all saved checkpoints sorted by step.

        Returns:
            List of metadata dicts, sorted by step number.
        """
        if not self.base_dir.exists():
            return []

        checkpoints = []
        for ckpt_dir in sorted(self.base_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            if ckpt_dir.name == 'best':
                continue

            meta_path = ckpt_dir / 'metadata.json'
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                checkpoints.append(meta)

        return sorted(checkpoints, key=lambda m: m['step'])

    @staticmethod
    def _load_from_dir(
        ckpt_dir: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Load model and metadata from a checkpoint directory.

        Args:
            ckpt_dir: Path to checkpoint directory.
            model: Model to load weights into.
            optimizer: Optional optimizer to load state into.

        Returns:
            Metadata dict.
        """
        model_path = ckpt_dir / 'model.pt'
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)

        if optimizer is not None:
            opt_path = ckpt_dir / 'optimizer.pt'
            if opt_path.exists():
                opt_state = torch.load(opt_path, weights_only=True)
                optimizer.load_state_dict(opt_state)

        meta_path = ckpt_dir / 'metadata.json'
        with open(meta_path) as f:
            return json.load(f)
