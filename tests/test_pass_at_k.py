"""Tests for the unbiased pass@k estimator.

The pass@k metric from Chen et al. (2021) "Evaluating Large Language Models
Trained on Code" computes the probability that at least one of k samples
is correct, given n total samples with c correct:

    pass@k = 1 - C(n-c, k) / C(n, k)

Run with: uv run pytest tests/test_pass_at_k.py -v
"""

import math

import pytest

from lmt.eval.pass_at_k import pass_at_k, pass_at_k_problems


class TestPassAtK:
    """Tests for single-problem pass@k."""

    def test_all_correct(self):
        """If all samples are correct, pass@k = 1.0 for any k."""
        assert pass_at_k(n=10, c=10, k=1) == 1.0
        assert pass_at_k(n=10, c=10, k=5) == 1.0
        assert pass_at_k(n=10, c=10, k=10) == 1.0

    def test_none_correct(self):
        """If no samples are correct, pass@k = 0.0 for any k."""
        assert pass_at_k(n=10, c=0, k=1) == 0.0
        assert pass_at_k(n=10, c=0, k=5) == 0.0
        assert pass_at_k(n=10, c=0, k=10) == 0.0

    def test_pass_at_1(self):
        """pass@1 = c/n (fraction of correct samples)."""
        assert pass_at_k(n=10, c=5, k=1) == pytest.approx(0.5)
        assert pass_at_k(n=10, c=1, k=1) == pytest.approx(0.1)
        assert pass_at_k(n=100, c=50, k=1) == pytest.approx(0.5)

    def test_pass_at_k_increases_with_k(self):
        """More samples means higher chance of success."""
        p1 = pass_at_k(n=10, c=3, k=1)
        p5 = pass_at_k(n=10, c=3, k=5)
        p10 = pass_at_k(n=10, c=3, k=10)
        assert p1 < p5 < p10

    def test_pass_at_k_known_value(self):
        """Verify against hand-calculated value.

        n=5, c=2, k=3:
        pass@3 = 1 - C(3,3)/C(5,3) = 1 - 1/10 = 0.9
        """
        assert pass_at_k(n=5, c=2, k=3) == pytest.approx(0.9)

    def test_pass_at_k_another_known_value(self):
        """n=10, c=3, k=2: 1 - C(7,2)/C(10,2) = 1 - 21/45 = 24/45."""
        # 1 - C(7,2)/C(10,2) = 1 - 21/45 = 24/45
        assert pass_at_k(n=10, c=3, k=2) == pytest.approx(24 / 45)

    def test_k_equals_n(self):
        """When k=n, pass@k = 1.0 if c >= 1, else 0.0."""
        assert pass_at_k(n=5, c=1, k=5) == 1.0
        assert pass_at_k(n=5, c=0, k=5) == 0.0

    def test_c_greater_than_k(self):
        """When c >= k, pass@k should be high but not necessarily 1.0.

        pass@k = 1.0 only when n-c < k (impossible to draw k without
        a correct one). With n=10, c=5, k=5: C(5,5)/C(10,5) = 1/252,
        so pass@5 = 251/252 ≈ 0.996.
        """
        assert pass_at_k(n=10, c=5, k=5) == pytest.approx(251 / 252)
        # But when n-c < k, it IS 1.0
        assert pass_at_k(n=10, c=8, k=5) == 1.0

    def test_k_greater_than_n_raises(self):
        """K > n is invalid and should raise."""
        with pytest.raises(ValueError, match='k.*cannot exceed.*n'):
            pass_at_k(n=5, c=3, k=10)

    def test_c_greater_than_n_raises(self):
        """C > n is invalid and should raise."""
        with pytest.raises(ValueError, match='c.*cannot exceed.*n'):
            pass_at_k(n=5, c=10, k=1)

    def test_negative_values_raise(self):
        """Negative inputs are invalid."""
        with pytest.raises(ValueError):
            pass_at_k(n=-1, c=0, k=1)
        with pytest.raises(ValueError):
            pass_at_k(n=5, c=-1, k=1)
        with pytest.raises(ValueError):
            pass_at_k(n=5, c=0, k=0)

    def test_large_n_numerical_stability(self):
        """Large n should not overflow or produce NaN.

        n=1000, c=100, k=10 — uses log-space computation internally.
        """
        result = pass_at_k(n=1000, c=100, k=10)
        assert 0.0 <= result <= 1.0
        assert not math.isnan(result)
        # With 10% success rate and 10 draws, should be very likely
        assert result > 0.5


class TestPassAtKProblems:
    """Tests for multi-problem pass@k averaging."""

    def test_single_problem(self):
        """Average of one problem = that problem's pass@k."""
        results = [(10, 5)]  # n=10, c=5
        assert pass_at_k_problems(results, k=1) == pytest.approx(0.5)

    def test_average_of_two(self):
        """Average across two problems."""
        results = [(10, 10), (10, 0)]  # One perfect, one zero
        assert pass_at_k_problems(results, k=1) == pytest.approx(0.5)

    def test_empty_list(self):
        """Empty problem list returns 0.0."""
        assert pass_at_k_problems([], k=1) == 0.0

    def test_all_perfect(self):
        """All-correct problems average to 1.0."""
        results = [(5, 5), (10, 10), (3, 3)]
        assert pass_at_k_problems(results, k=1) == 1.0

    def test_all_zero(self):
        """All-wrong problems average to 0.0."""
        results = [(5, 0), (10, 0), (3, 0)]
        assert pass_at_k_problems(results, k=1) == 0.0
