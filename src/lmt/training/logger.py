"""Training metrics logger.

Provides a clean interface for logging and displaying training
progress. Tracks step-based metrics with formatting helpers.

Usage::

    from lmt.training.logger import TrainingLogger

    logger = TrainingLogger()
    for step in range(100):
        loss = train_step(...)
        logger.log('loss', loss, step=step)
        if step % 10 == 0:
            print(logger.step_summary(step))
    print(logger.summary())
"""

from __future__ import annotations


class TrainingLogger:
    """Logs and formats training metrics.

    Tracks ``(step, value)`` pairs for named metrics. Provides
    helpers for displaying progress during training.
    """

    def __init__(self) -> None:
        """Initialize empty logger."""
        self._history: dict[str, list[tuple[int, float]]] = {}

    def log(self, name: str, value: float, step: int) -> None:
        """Log a single metric value.

        Args:
            name: Metric name (e.g. ``'loss'``, ``'bpb'``).
            value: Metric value.
            step: Training step number.
        """
        if name not in self._history:
            self._history[name] = []
        self._history[name].append((step, value))

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dict mapping metric names to values.
            step: Training step number.
        """
        for name, value in metrics.items():
            self.log(name, value, step)

    def get_history(self, name: str) -> list[tuple[int, float]]:
        """Get full history for a metric.

        Args:
            name: Metric name.

        Returns:
            List of ``(step, value)`` tuples.
        """
        return self._history.get(name, [])

    def latest(self, name: str) -> float | None:
        """Get the most recent value for a metric.

        Args:
            name: Metric name.

        Returns:
            Latest value, or None if no history.
        """
        history = self.get_history(name)
        if not history:
            return None
        return history[-1][1]

    def step_summary(self, step: int) -> str:
        """Format a summary line for a specific step.

        Shows all metrics logged at the given step.

        Args:
            step: Training step number.

        Returns:
            Formatted string like ``Step 10 | loss: 2.50 | lr: 1.0e-04``.
        """
        parts = [f'Step {step}']
        for name in sorted(self._history.keys()):
            # Find the value at this step
            for s, v in reversed(self._history[name]):
                if s == step:
                    parts.append(_format_metric(name, v))
                    break
        return ' | '.join(parts)

    def summary(self) -> str:
        """Produce a summary of latest values for all metrics.

        Returns:
            Multi-line string with latest value per metric.
        """
        lines = ['Training Summary:', '-' * 40]
        for name in sorted(self._history.keys()):
            history = self._history[name]
            if history:
                latest = history[-1][1]
                lines.append(f'  {name}: {_format_metric(name, latest)}')
        return '\n'.join(lines)

    def reset(self) -> None:
        """Clear all logged metrics."""
        self._history.clear()

    def metric_names(self) -> list[str]:
        """List all logged metric names.

        Returns:
            Sorted list of metric names.
        """
        return sorted(self._history.keys())

    def to_dict(self) -> dict[str, list[tuple[int, float]]]:
        """Serialize all history.

        Returns:
            Dict mapping metric names to ``(step, value)`` lists.
        """
        return dict(self._history)


def _format_metric(name: str, value: float) -> str:
    """Format a metric value for display.

    Uses scientific notation for very small values (like learning
    rates) and fixed-point for normal values.

    Args:
        name: Metric name (used for format hints).
        value: Metric value.

    Returns:
        Formatted string.
    """
    if abs(value) < 1e-3 and value != 0:
        return f'{name}: {value:.2e}'
    return f'{name}: {value:.4f}'
