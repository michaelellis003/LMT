"""Persistent experiment registry.

Stores experiment results as JSON lines (one per line) for
tracking and comparing experiments across sessions. Each entry
is a full serialized ExperimentResult.

Usage::

    from lmt.research.registry import ExperimentRegistry

    reg = ExperimentRegistry('runs/registry.jsonl')
    reg.add(result)  # after each experiment
    reg.list()  # see all experiments
    reg.best(metric='val_bpb')  # find best
    reg.summary()  # formatted comparison
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lmt.research.experiment import ExperimentResult


class ExperimentRegistry:
    """Persistent registry of experiment results.

    Appends results to a JSONL file (one JSON object per line).
    Each read re-parses the file, so multiple processes can
    safely append without coordination.

    Args:
        path: Path to the JSONL registry file.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize registry.

        Args:
            path: Path to the JSONL file. Created on first write.
        """
        self.path = Path(path)

    def add(self, result: ExperimentResult) -> None:
        """Append an experiment result to the registry.

        Args:
            result: Completed experiment result to store.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')

    def list(self) -> list[dict[str, Any]]:
        """Load all experiment results from disk.

        Returns:
            List of serialized result dicts, in insertion order.
        """
        if not self.path.exists():
            return []

        results: list[dict[str, Any]] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    def count(self) -> int:
        """Return the number of registered experiments."""
        return len(self.list())

    def best(
        self,
        metric: str = 'val_bpb',
        higher_is_better: bool = False,
    ) -> dict[str, Any] | None:
        """Find the experiment with the best metric value.

        Args:
            metric: Metric name to compare.
            higher_is_better: If True, pick highest value.

        Returns:
            Best result dict, or None if registry is empty.
        """
        results = self.list()
        if not results:
            return None

        best_entry: dict[str, Any] | None = None
        best_val: float | None = None

        for entry in results:
            metrics = entry.get('metrics', {})
            history = metrics.get(metric, [])
            if not history:
                continue

            # Get best value from history (list of [step, value])
            values = [v for _, v in history]
            val = max(values) if higher_is_better else min(values)

            if best_val is None or (
                (val > best_val) if higher_is_better else (val < best_val)
            ):
                best_val = val
                best_entry = entry

        return best_entry

    def filter(
        self,
        name_contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter registered experiments.

        Args:
            name_contains: Only include experiments whose name
                contains this substring.

        Returns:
            Filtered list of result dicts.
        """
        results = self.list()

        if name_contains is not None:
            results = [
                r
                for r in results
                if name_contains in r.get('config', {}).get('name', '')
            ]

        return results

    def summary(
        self,
        metric: str = 'val_bpb',
        higher_is_better: bool = False,
    ) -> str:
        """Format a summary of all registered experiments.

        Args:
            metric: Primary metric to display and rank by.
            higher_is_better: If True, higher is better.

        Returns:
            Formatted multi-line string.
        """
        results = self.list()
        if not results:
            return 'No experiments registered.'

        lines = [f'Experiment Registry ({len(results)} experiments)']
        lines.append('=' * 60)

        header = f'{"Name":<25} {"LR":>10} {"Loss":>10} {metric:>10}'
        lines.append(header)
        lines.append('-' * 60)

        best_entry = self.best(metric, higher_is_better)
        best_name = (
            best_entry.get('config', {}).get('name') if best_entry else None
        )

        for entry in results:
            config = entry.get('config', {})
            name = config.get('name', '?')
            lr = config.get('lr', 0)
            loss = entry.get('final_loss')

            # Get best metric value
            history = entry.get('metrics', {}).get(metric, [])
            if history:
                values = [v for _, v in history]
                val = max(values) if higher_is_better else min(values)
                metric_str = f'{val:.4f}'
            else:
                metric_str = '-'

            loss_str = f'{loss:.4f}' if loss is not None else '-'
            marker = ' <-- best' if name == best_name else ''

            lines.append(
                f'{name:<25} {lr:>10.2e} {loss_str:>10} '
                f'{metric_str:>10}{marker}'
            )

        return '\n'.join(lines)
