"""Experiment results comparison and formatting.

Utilities for ranking and displaying autoresearch experiment results
in a readable table format.

Usage::

    from lmt.research.results_table import format_results_table, rank_results

    ranked = rank_results(results, metric='val_bpb')
    print(format_results_table(results, metric='val_bpb'))
"""

from __future__ import annotations

from lmt.research.experiment import ExperimentResult


def rank_results(
    results: list[ExperimentResult],
    metric: str = 'val_bpb',
    higher_is_better: bool = False,
) -> list[ExperimentResult]:
    """Rank experiment results by a metric.

    Results without the metric are placed last.

    Args:
        results: List of experiment results to rank.
        metric: Metric name to rank by.
        higher_is_better: If True, rank highest first.

    Returns:
        Sorted list of results (best first).
    """
    if not results:
        return []

    def sort_key(r: ExperimentResult) -> tuple[int, float]:
        val = r.metrics.get_best(metric, higher_is_better)
        if val is None:
            return (1, 0.0)
        return (0, -val if higher_is_better else val)

    return sorted(results, key=sort_key)


def format_results_table(
    results: list[ExperimentResult],
    metric: str = 'val_bpb',
    higher_is_better: bool = False,
    config_fields: list[str] | None = None,
) -> str:
    """Format experiment results as a comparison table.

    Args:
        results: Experiment results to display.
        metric: Primary metric to highlight and rank by.
        higher_is_better: If True, higher metric is better.
        config_fields: Config fields to show. Defaults to
            ``['lr', 'batch_size']``.

    Returns:
        Formatted table string.
    """
    if not results:
        return 'No results to display.'

    if config_fields is None:
        config_fields = ['lr', 'batch_size']

    ranked = rank_results(results, metric, higher_is_better)
    best_name = ranked[0].config.name if ranked else None

    # Build header
    columns = ['name'] + config_fields + ['final_loss', metric, '']
    rows: list[list[str]] = []

    for result in ranked:
        row = [result.config.name]

        # Config fields
        config_dict = result.config.to_dict()
        for field in config_fields:
            val = config_dict.get(field)
            if isinstance(val, float):
                row.append(f'{val:.2e}')
            else:
                row.append(str(val) if val is not None else '-')

        # Final loss
        if result.final_loss is not None:
            row.append(f'{result.final_loss:.4f}')
        else:
            row.append('-')

        # Primary metric
        metric_val = result.metrics.get_best(metric, higher_is_better)
        if metric_val is not None:
            row.append(f'{metric_val:.4f}')
        else:
            row.append('-')

        # Best marker
        if result.config.name == best_name:
            row.append('<-- best')
        else:
            row.append('')

        rows.append(row)

    # Calculate column widths
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Format
    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.ljust(widths[i]))
        return '  '.join(parts).rstrip()

    lines = [fmt_row(columns)]
    lines.append('  '.join('-' * w for w in widths))
    for row in rows:
        lines.append(fmt_row(row))

    return '\n'.join(lines)
