"""Tests for best-of-N inference-time scaling.

Best-of-N generates N code samples, executes each against tests,
and selects the sample with the highest reward (most tests passed).

Run with: uv run pytest tests/test_best_of_n.py -v
"""

import pytest

from lmt.eval.code_execution import ExecutionResult, SingleTestResult
from lmt.inference.best_of_n import (
    BestOfNConfig,
    BestOfNResult,
    best_of_n_select,
)


class TestBestOfNConfig:
    """Tests for BestOfNConfig defaults and validation."""

    def test_defaults(self):
        """Default config has sensible values."""
        config = BestOfNConfig()
        assert config.n_samples > 1
        assert config.temperature > 0.0
        assert config.timeout > 0

    def test_custom_config(self):
        """Custom values are stored correctly."""
        config = BestOfNConfig(
            n_samples=50,
            temperature=0.8,
            timeout=5,
        )
        assert config.n_samples == 50
        assert config.temperature == 0.8
        assert config.timeout == 5


class TestBestOfNResult:
    """Tests for BestOfNResult dataclass."""

    def test_basic_result(self):
        """Result stores selected code and all candidates."""
        result = BestOfNResult(
            best_code='def add(a, b): return a + b',
            best_reward=1.0,
            all_codes=[
                'def add(a, b): return 0',
                'def add(a, b): return a + b',
            ],
            all_rewards=[0.0, 1.0],
            n_samples=2,
            n_correct=1,
        )
        assert result.best_reward == 1.0
        assert result.n_correct == 1
        assert len(result.all_codes) == 2


class TestBestOfNSelect:
    """Tests for the selection logic (no model needed)."""

    def test_selects_highest_reward(self):
        """Picks the sample with the highest reward score."""
        codes = [
            'def f(): return 0',
            'def f(): return 1',
            'def f(): return 2',
        ]
        results = [
            _make_exec_result(passed=0, failed=2, total=2),
            _make_exec_result(passed=2, failed=0, total=2),
            _make_exec_result(passed=1, failed=1, total=2),
        ]
        selection = best_of_n_select(codes, results)
        assert selection.best_code == 'def f(): return 1'
        assert selection.best_reward == 1.0
        assert selection.n_correct == 1

    def test_tiebreak_shortest_code(self):
        """When rewards are equal, picks the shortest code."""
        codes = [
            'def f(): return a + b  # long comment here',
            'def f(): return a + b',
        ]
        results = [
            _make_exec_result(passed=2, failed=0, total=2),
            _make_exec_result(passed=2, failed=0, total=2),
        ]
        selection = best_of_n_select(codes, results)
        assert selection.best_code == 'def f(): return a + b'

    def test_all_failures(self):
        """When all samples fail, picks the best partial score."""
        codes = [
            'def f(): return 0',
            'def f(): return 1',
        ]
        results = [
            _make_exec_result(passed=0, failed=3, total=3),
            _make_exec_result(passed=1, failed=2, total=3),
        ]
        selection = best_of_n_select(codes, results)
        assert selection.best_code == 'def f(): return 1'
        assert selection.best_reward == pytest.approx(1 / 3)
        assert selection.n_correct == 0

    def test_single_sample(self):
        """Works with just one sample."""
        codes = ['def f(): return 42']
        results = [_make_exec_result(passed=1, failed=0, total=1)]
        selection = best_of_n_select(codes, results)
        assert selection.best_code == 'def f(): return 42'
        assert selection.n_samples == 1

    def test_empty_raises(self):
        """Empty inputs should raise."""
        with pytest.raises(ValueError, match='at least one'):
            best_of_n_select([], [])

    def test_mismatched_lengths_raises(self):
        """Codes and results must have same length."""
        with pytest.raises(ValueError, match='same length'):
            best_of_n_select(
                ['code1', 'code2'],
                [_make_exec_result(passed=1, failed=0, total=1)],
            )

    def test_counts_correct(self):
        """n_correct counts samples with reward == 1.0."""
        codes = ['a', 'b', 'c', 'd']
        results = [
            _make_exec_result(passed=2, failed=0, total=2),
            _make_exec_result(passed=0, failed=2, total=2),
            _make_exec_result(passed=2, failed=0, total=2),
            _make_exec_result(passed=1, failed=1, total=2),
        ]
        selection = best_of_n_select(codes, results)
        assert selection.n_correct == 2
        assert selection.n_samples == 4

    def test_preserves_all_rewards(self):
        """All rewards are stored for analysis."""
        codes = ['a', 'b', 'c']
        results = [
            _make_exec_result(passed=0, failed=1, total=1),
            _make_exec_result(passed=1, failed=0, total=1),
            _make_exec_result(passed=0, failed=1, total=1),
        ]
        selection = best_of_n_select(codes, results)
        assert selection.all_rewards == [0.0, 1.0, 0.0]


def _make_exec_result(passed: int, failed: int, total: int) -> ExecutionResult:
    """Helper to create ExecutionResult for testing."""
    return ExecutionResult(
        stdout='',
        stderr='',
        passed=passed,
        failed=failed,
        total=total,
        timed_out=False,
        test_results=[
            SingleTestResult(name=f'test_{i}', passed=i < passed)
            for i in range(total)
        ],
    )
