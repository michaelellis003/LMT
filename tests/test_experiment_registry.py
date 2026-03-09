"""Tests for persistent experiment registry."""

from pathlib import Path

from lmt.research.experiment import (
    ExperimentResult,
    ExperimentRunConfig,
    MetricTracker,
)
from lmt.research.registry import ExperimentRegistry


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


class TestExperimentRegistry:
    """Test ExperimentRegistry persistence."""

    def test_add_and_list(self, tmp_path: Path):
        """Should add results and list them."""
        reg = ExperimentRegistry(tmp_path / 'registry.jsonl')
        reg.add(_make_result('a', 1e-3, 2.0, val_bpb=3.5))
        reg.add(_make_result('b', 1e-4, 1.5, val_bpb=2.8))

        results = reg.list()
        assert len(results) == 2
        assert results[0]['config']['name'] == 'a'
        assert results[1]['config']['name'] == 'b'

    def test_persists_across_instances(self, tmp_path: Path):
        """Should persist data to disk between instances."""
        path = tmp_path / 'registry.jsonl'

        reg1 = ExperimentRegistry(path)
        reg1.add(_make_result('first', 1e-3, 2.0))

        reg2 = ExperimentRegistry(path)
        reg2.add(_make_result('second', 1e-4, 1.5))

        reg3 = ExperimentRegistry(path)
        results = reg3.list()
        assert len(results) == 2

    def test_best(self, tmp_path: Path):
        """Should find the best result by metric."""
        reg = ExperimentRegistry(tmp_path / 'registry.jsonl')
        reg.add(_make_result('a', 1e-3, 2.0, val_bpb=3.5))
        reg.add(_make_result('b', 1e-4, 1.5, val_bpb=2.8))
        reg.add(_make_result('c', 1e-2, 3.0, val_bpb=4.0))

        best = reg.best(metric='val_bpb')
        assert best is not None
        assert best['config']['name'] == 'b'

    def test_best_higher_is_better(self, tmp_path: Path):
        """Should support higher-is-better metrics."""
        reg = ExperimentRegistry(tmp_path / 'registry.jsonl')
        reg.add(_make_result('a', 1e-3, 2.0, val_bpb=3.5))
        reg.add(_make_result('b', 1e-4, 1.5, val_bpb=2.8))

        best = reg.best(metric='val_bpb', higher_is_better=True)
        assert best is not None
        assert best['config']['name'] == 'a'

    def test_best_empty(self, tmp_path: Path):
        """Should return None for empty registry."""
        reg = ExperimentRegistry(tmp_path / 'registry.jsonl')
        assert reg.best() is None

    def test_best_missing_metric(self, tmp_path: Path):
        """Should skip entries without the requested metric."""
        reg = ExperimentRegistry(tmp_path / 'registry.jsonl')
        reg.add(_make_result('no_bpb', 1e-3, 2.0, val_bpb=None))
        reg.add(_make_result('has_bpb', 1e-4, 1.5, val_bpb=3.0))

        best = reg.best(metric='val_bpb')
        assert best is not None
        assert best['config']['name'] == 'has_bpb'

    def test_summary(self, tmp_path: Path):
        """Should produce a readable summary string."""
        reg = ExperimentRegistry(tmp_path / 'registry.jsonl')
        reg.add(_make_result('a', 1e-3, 2.0, val_bpb=3.5))
        reg.add(_make_result('b', 1e-4, 1.5, val_bpb=2.8))

        summary = reg.summary(metric='val_bpb')
        assert isinstance(summary, str)
        assert 'a' in summary
        assert 'b' in summary

    def test_summary_empty(self, tmp_path: Path):
        """Should handle empty registry summary."""
        reg = ExperimentRegistry(tmp_path / 'registry.jsonl')
        summary = reg.summary()
        assert isinstance(summary, str)

    def test_count(self, tmp_path: Path):
        """Should report the number of experiments."""
        reg = ExperimentRegistry(tmp_path / 'registry.jsonl')
        assert reg.count() == 0
        reg.add(_make_result('a', 1e-3, 2.0))
        assert reg.count() == 1
        reg.add(_make_result('b', 1e-4, 1.5))
        assert reg.count() == 2

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Should create parent directories if needed."""
        path = tmp_path / 'nested' / 'dir' / 'registry.jsonl'
        reg = ExperimentRegistry(path)
        reg.add(_make_result('test', 1e-3, 2.0))
        assert path.exists()

    def test_filter_by_name(self, tmp_path: Path):
        """Should filter results by experiment name pattern."""
        reg = ExperimentRegistry(tmp_path / 'registry.jsonl')
        reg.add(_make_result('baseline_v1', 1e-3, 2.0))
        reg.add(_make_result('baseline_v2', 1e-4, 1.5))
        reg.add(_make_result('ablation_no_warmup', 1e-3, 2.5))

        baselines = reg.filter(name_contains='baseline')
        assert len(baselines) == 2

        ablations = reg.filter(name_contains='ablation')
        assert len(ablations) == 1
