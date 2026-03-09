"""Experiment runner for automated research iteration.

Core loop inspired by Karpathy's autoresearch:

1. Configure experiment (model, data, hyperparameters)
2. Train for N steps
3. Evaluate metrics
4. Log results, save checkpoint if improved
5. Repeat with modifications

The runner is model-agnostic -- it takes any ``nn.Module`` that
accepts token IDs and returns logits. This lets us iterate on
architecture, hyperparameters, and training recipes without
changing the runner itself.

Usage::

    from lmt.research.experiment import ExperimentRunConfig, ExperimentRunner

    config = ExperimentRunConfig(name='baseline', train_steps=100)
    runner = ExperimentRunner(output_dir='runs/')
    result = runner.run(model, train_data, config)
    print(f'Final loss: {result.final_loss:.4f}')
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as f


@dataclass
class ExperimentRunConfig:
    """Configuration for a single experiment run.

    Attributes:
        name: Unique experiment name (used for output directory).
        train_steps: Number of training steps.
        eval_interval: Evaluate on full training data every N steps.
            If larger than ``train_steps``, no evaluation occurs.
        lr: Learning rate.
        batch_size: Training batch size.
        save_checkpoint: Whether to save model checkpoint.
    """

    name: str = 'experiment'
    train_steps: int = 100
    eval_interval: int = 10
    lr: float = 1e-4
    batch_size: int = 4
    save_checkpoint: bool = False

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentRunConfig':
        """Deserialize from dict, ignoring unknown keys."""
        valid_keys = {field_name for field_name in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


class MetricTracker:
    """Tracks metrics over training with step numbers.

    Stores ``(step, value)`` pairs for each metric name.
    Supports querying best values and detecting improvement.
    """

    def __init__(self) -> None:
        """Initialize empty tracker."""
        self._history: dict[str, list[tuple[int, float]]] = {}

    def log(self, name: str, value: float, step: int) -> None:
        """Record a metric value at a given step.

        Args:
            name: Metric name (e.g., ``'loss'``, ``'bpb'``).
            value: Metric value.
            step: Training step number.
        """
        if name not in self._history:
            self._history[name] = []
        self._history[name].append((step, value))

    def get_history(self, name: str) -> list[tuple[int, float]]:
        """Get full history for a metric.

        Args:
            name: Metric name.

        Returns:
            List of ``(step, value)`` tuples, or empty list.
        """
        return self._history.get(name, [])

    def get_best(
        self,
        name: str,
        higher_is_better: bool = False,
    ) -> float | None:
        """Get the best value recorded for a metric.

        Args:
            name: Metric name.
            higher_is_better: If True, return max instead of min.

        Returns:
            Best value, or None if no history exists.
        """
        history = self.get_history(name)
        if not history:
            return None

        values = [v for _, v in history]
        return max(values) if higher_is_better else min(values)

    def is_improved(
        self,
        name: str,
        higher_is_better: bool = False,
    ) -> bool:
        """Check if the latest value is the best so far.

        Args:
            name: Metric name.
            higher_is_better: If True, check if latest is highest.

        Returns:
            True if the latest value is the best recorded.
        """
        history = self.get_history(name)
        if len(history) < 2:
            return True

        latest = history[-1][1]
        previous_values = [v for _, v in history[:-1]]

        if higher_is_better:
            return latest > max(previous_values)
        return latest < min(previous_values)

    def to_dict(self) -> dict[str, list[tuple[int, float]]]:
        """Serialize all metric histories."""
        return dict(self._history)


@dataclass
class ExperimentResult:
    """Result of a completed experiment run.

    Attributes:
        config: The experiment configuration.
        metrics: Full metric history.
        final_loss: Loss at the last training step.
    """

    config: ExperimentRunConfig
    metrics: MetricTracker
    final_loss: float | None = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'final_loss': self.final_loss,
        }


class ExperimentRunner:
    """Runs training experiments and tracks results.

    Args:
        output_dir: Base directory for experiment outputs.
    """

    def __init__(self, output_dir: str = 'runs') -> None:
        """Initialize runner.

        Args:
            output_dir: Base directory for saving results.
        """
        self.output_dir = Path(output_dir)
        self.results: list[ExperimentResult] = []

    def run(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        config: ExperimentRunConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> ExperimentResult:
        """Run a single training experiment.

        Trains the model on ``train_data`` for ``config.train_steps``
        steps, logging loss at each step and saving results.

        Args:
            model: Model to train (must accept token IDs, return logits).
            train_data: Token IDs ``[num_samples, seq_len]``.
            config: Experiment configuration.
            optimizer: Optional optimizer. Defaults to AdamW if not
                provided.

        Returns:
            ExperimentResult with metrics and final loss.
        """
        tracker = MetricTracker()
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        # Infer device from model parameters
        model_device = next(model.parameters()).device

        model.train()
        num_samples = train_data.shape[0]

        for step in range(config.train_steps):
            # Sample a batch and move to model device
            indices = torch.randint(0, num_samples, (config.batch_size,))
            batch = train_data[indices].to(model_device)

            # Forward pass (next-token prediction)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            loss = f.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tracker.log('train_loss', loss.item(), step=step)

            # Periodic evaluation (batched to avoid OOM)
            if (step + 1) % config.eval_interval == 0:
                model.eval()
                eval_total = 0.0
                eval_count = 0
                with torch.no_grad():
                    for i in range(0, num_samples, config.batch_size):
                        eb = train_data[i : i + config.batch_size].to(
                            model_device
                        )
                        el = model(eb[:, :-1])
                        eloss = f.cross_entropy(
                            el.reshape(-1, el.size(-1)),
                            eb[:, 1:].reshape(-1),
                            reduction='sum',
                        )
                        eval_total += eloss.item()
                        eval_count += eb[:, 1:].numel()
                avg_eval = eval_total / max(eval_count, 1)
                tracker.log('eval_loss', avg_eval, step=step)
                model.train()

        # Final loss
        final_loss = tracker.get_history('train_loss')[-1][1]

        result = ExperimentResult(
            config=config,
            metrics=tracker,
            final_loss=final_loss,
        )
        self.results.append(result)

        # Save results to disk
        exp_dir = self.output_dir / config.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        result_path = exp_dir / 'result.json'
        result_path.write_text(json.dumps(result.to_dict(), indent=2) + '\n')

        if config.save_checkpoint:
            ckpt_path = exp_dir / 'checkpoint.pt'
            torch.save(model.state_dict(), ckpt_path)

        return result
