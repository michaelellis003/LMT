"""Early stopping monitor for training loops.

Tracks a metric over training steps and signals when to stop
if no improvement is seen within a patience window. Supports
both lower-is-better (loss, BPB) and higher-is-better (accuracy)
metrics.

Usage::

    from lmt.training.early_stopping import EarlyStopping

    es = EarlyStopping(patience=5, min_delta=0.01)
    for step in range(max_steps):
        loss = train_step()
        if es.step(loss):
            print(f'Early stopping at step {step}')
            break
"""

from __future__ import annotations


class EarlyStopping:
    """Monitor a metric and signal when training should stop.

    Args:
        patience: Number of steps without improvement before stopping.
        min_delta: Minimum change to count as improvement.
        higher_is_better: If True, higher metric values are better.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        higher_is_better: bool = False,
    ) -> None:
        """Initialize early stopping monitor.

        Args:
            patience: Number of steps without improvement before
                stopping. Must be positive.
            min_delta: Minimum change to count as improvement.
                Must be non-negative.
            higher_is_better: If True, higher metric values are
                better (e.g. accuracy). Default False (loss, BPB).
        """
        if patience < 1:
            raise ValueError(f'patience must be >= 1, got {patience}')
        if min_delta < 0:
            raise ValueError(f'min_delta must be >= 0, got {min_delta}')

        self.patience = patience
        self.min_delta = min_delta
        self.higher_is_better = higher_is_better
        self._best: float | None = None
        self._counter = 0

    @property
    def best_value(self) -> float | None:
        """Best metric value seen so far."""
        return self._best

    @property
    def counter(self) -> int:
        """Current number of steps without improvement."""
        return self._counter

    def step(self, metric_value: float) -> bool:
        """Record a metric value and check if training should stop.

        Args:
            metric_value: Current metric value.

        Returns:
            True if training should stop (patience exhausted).
        """
        if self._best is None:
            self._best = metric_value
            return False

        if self._is_improvement(metric_value):
            self._best = metric_value
            self._counter = 0
            return False

        self._counter += 1
        return self._counter >= self.patience

    def _is_improvement(self, value: float) -> bool:
        """Check if value is an improvement over best.

        Args:
            value: New metric value to compare.

        Returns:
            True if value improves on best by at least min_delta.
        """
        assert self._best is not None
        if self.higher_is_better:
            return value > self._best + self.min_delta
        return value < self._best - self.min_delta
