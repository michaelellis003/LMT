"""Tests for experiment results comparison table."""

from lmt.research.experiment import (
    ExperimentResult,
    ExperimentRunConfig,
    MetricTracker,
)
from lmt.research.results_table import format_results_table, rank_results


def _make_result(
    name: str,
    lr: float,
    final_loss: float,
    val_bpb: float | None = None,
) -> ExperimentResult:
    """Create a test result with metrics."""
    config = ExperimentRunConfig(name=name, lr=lr)
    tracker = MetricTracker()
    tracker.log('train_loss', final_loss, step=99)
    if val_bpb is not None:
        tracker.log('val_bpb', val_bpb, step=99)
    return ExperimentResult(
        config=config, metrics=tracker, final_loss=final_loss
    )


class TestRankResults:
    """Test rank_results function."""

    def test_ranks_by_metric(self):
        """Should rank results by metric (lower is better)."""
        results = [
            _make_result('a', 1e-3, 2.0, val_bpb=3.5),
            _make_result('b', 1e-4, 1.5, val_bpb=2.8),
            _make_result('c', 1e-2, 3.0, val_bpb=4.0),
        ]
        ranked = rank_results(results, metric='val_bpb')
        assert ranked[0].config.name == 'b'
        assert ranked[1].config.name == 'a'
        assert ranked[2].config.name == 'c'

    def test_ranks_higher_is_better(self):
        """Should rank results by metric (higher is better)."""
        results = [
            _make_result('a', 1e-3, 2.0, val_bpb=3.5),
            _make_result('b', 1e-4, 1.5, val_bpb=2.8),
        ]
        ranked = rank_results(results, metric='val_bpb', higher_is_better=True)
        assert ranked[0].config.name == 'a'

    def test_skips_missing_metrics(self):
        """Should put results with missing metrics last."""
        results = [
            _make_result('a', 1e-3, 2.0, val_bpb=3.5),
            _make_result('b', 1e-4, 1.5, val_bpb=None),
            _make_result('c', 1e-2, 3.0, val_bpb=2.0),
        ]
        ranked = rank_results(results, metric='val_bpb')
        assert ranked[0].config.name == 'c'
        assert ranked[1].config.name == 'a'
        assert ranked[2].config.name == 'b'

    def test_empty_results(self):
        """Should return empty for empty input."""
        assert rank_results([], metric='val_bpb') == []


class TestFormatResultsTable:
    """Test format_results_table function."""

    def test_basic_table(self):
        """Should format a readable table."""
        results = [
            _make_result('baseline', 1e-3, 2.5, val_bpb=3.2),
            _make_result('high_lr', 1e-2, 1.8, val_bpb=2.9),
        ]
        table = format_results_table(results)
        assert 'baseline' in table
        assert 'high_lr' in table
        assert 'val_bpb' in table

    def test_highlights_best(self):
        """Should mark the best result."""
        results = [
            _make_result('a', 1e-3, 2.0, val_bpb=3.5),
            _make_result('b', 1e-4, 1.5, val_bpb=2.8),
        ]
        table = format_results_table(results, metric='val_bpb')
        # Best result (b) should be marked
        assert '*' in table or '←' in table or 'best' in table.lower()

    def test_shows_config_fields(self):
        """Should show relevant config fields."""
        results = [_make_result('test', 1e-3, 2.0, val_bpb=3.0)]
        table = format_results_table(results, config_fields=['lr'])
        assert '1.00e-03' in table or '0.001' in table

    def test_empty_results(self):
        """Should handle empty results."""
        table = format_results_table([])
        assert 'no results' in table.lower() or table.strip() == ''

    def test_returns_string(self):
        """Should return a string."""
        results = [_make_result('test', 1e-3, 2.0)]
        assert isinstance(format_results_table(results), str)
