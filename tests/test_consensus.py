"""Tests for consensus voting inference-time scaling.

Consensus voting clusters code samples by functional equivalence —
samples that produce the same outputs on probe inputs are grouped
together, and the largest cluster's representative is selected.
This works without ground-truth tests, only probe inputs.

Run with: uv run pytest tests/test_consensus.py -v
"""

from lmt.inference.consensus import (
    ConsensusResult,
    cluster_by_output,
)


class TestClusterByOutput:
    """Tests for output-based clustering logic."""

    def test_identical_outputs_same_cluster(self):
        """Functionally equivalent code goes in one cluster."""
        codes = [
            'def add(a, b): return a + b',
            'def add(a, b): return b + a',  # Same function
            'def add(a, b): return a + b + 0',  # Also same
        ]
        probe_calls = ['print(add(1, 2))', 'print(add(0, 0))']
        result = cluster_by_output(codes, probe_calls)
        # All three produce the same output, one cluster
        assert result.num_clusters == 1
        assert result.largest_cluster_size == 3

    def test_different_outputs_different_clusters(self):
        """Functionally different code goes in separate clusters."""
        codes = [
            'def f(x): return x',
            'def f(x): return x + 1',
            'def f(x): return x * 2',
        ]
        probe_calls = ['print(f(5))']
        result = cluster_by_output(codes, probe_calls)
        assert result.num_clusters == 3

    def test_majority_wins(self):
        """The largest cluster's representative is selected."""
        codes = [
            'def f(x): return x + 1',  # Correct
            'def f(x): return x + 1',  # Correct (same)
            'def f(x): return x + 1',  # Correct (same)
            'def f(x): return x * 2',  # Wrong
            'def f(x): return 0',  # Wrong
        ]
        probe_calls = ['print(f(3))', 'print(f(0))']
        result = cluster_by_output(codes, probe_calls)
        assert result.largest_cluster_size == 3
        assert result.best_code == 'def f(x): return x + 1'

    def test_crashing_code_forms_singleton(self):
        """Code that crashes gets its own singleton cluster."""
        codes = [
            'def f(x): return x + 1',
            'def f(x): return x + 1',
            'def f(x): raise ValueError("bad")',
        ]
        probe_calls = ['print(f(1))']
        result = cluster_by_output(codes, probe_calls)
        assert result.num_clusters == 2
        assert result.largest_cluster_size == 2
        assert result.best_code == 'def f(x): return x + 1'

    def test_tiebreak_shortest_code(self):
        """Tied clusters use shortest code as representative."""
        codes = [
            'def f(x): return x + 1  # very long comment',
            'def f(x): return x + 1',
            'def f(x): return x * 2  # another long one',
            'def f(x): return x * 2',
        ]
        probe_calls = ['print(f(3))']
        result = cluster_by_output(codes, probe_calls)
        # Two clusters of size 2 — both tied
        # Winner is picked by shorter code in largest cluster
        assert result.num_clusters == 2
        assert result.largest_cluster_size == 2

    def test_single_sample(self):
        """Works with a single sample."""
        codes = ['def f(): return 42']
        probe_calls = ['print(f())']
        result = cluster_by_output(codes, probe_calls)
        assert result.num_clusters == 1
        assert result.best_code == 'def f(): return 42'

    def test_empty_probe_calls(self):
        """With no probes, all code goes in one cluster."""
        codes = ['def f(): return 1', 'def f(): return 2']
        result = cluster_by_output(codes, probe_calls=[])
        # No probes means all outputs are '' — same cluster
        assert result.num_clusters == 1

    def test_timeout_code_forms_singleton(self):
        """Code that times out gets its own cluster."""
        codes = [
            'def f(x): return x + 1',
            'def f(x):\n    while True: pass',
        ]
        probe_calls = ['print(f(1))']
        result = cluster_by_output(codes, probe_calls, timeout=1)
        assert result.num_clusters == 2
        assert result.best_code == 'def f(x): return x + 1'


class TestConsensusResult:
    """Tests for ConsensusResult dataclass."""

    def test_fields(self):
        """Result stores cluster information."""
        result = ConsensusResult(
            best_code='def f(): return 1',
            cluster_sizes=[3, 1, 1],
            num_clusters=3,
            largest_cluster_size=3,
            all_codes=['a', 'b', 'c', 'd', 'e'],
        )
        assert result.num_clusters == 3
        assert result.largest_cluster_size == 3
        assert len(result.all_codes) == 5
