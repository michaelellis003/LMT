"""Tests for CodeT dual execution scoring.

CodeT scores code candidates by agreement with independently generated
tests. For each (code, test) pair, execution produces a pass/fail.
A code's score = fraction of generated tests it agrees with the
majority on. This is powerful because:
- No ground-truth tests needed at scoring time
- Agreement between independent code and tests is a strong signal
- Works as a reranking strategy on top of any generation method

Run with: uv run pytest tests/test_codet.py -v
"""

from lmt.inference.codet import (
    CodeTResult,
    codet_score,
)


class TestCodeTScore:
    """Tests for dual execution scoring logic."""

    def test_correct_majority_scores_highest(self):
        """Code matching the majority on all tests scores 1.0."""
        codes = [
            'def f(x): return x + 1',  # Correct
            'def f(x): return x + 1',  # Correct (majority)
            'def f(x): return x + 1',  # Correct (majority)
            'def f(x): return x * 2',  # Wrong
        ]
        gen_tests = [
            'assert f(0) == 1',
            'assert f(5) == 6',
        ]
        result = codet_score(codes, gen_tests)
        # 3/4 pass each test → majority=True
        # Correct codes agree on all → score=1.0
        assert result.best_code == 'def f(x): return x + 1'
        assert result.best_score == 1.0

    def test_majority_agreement(self):
        """Code agreeing with majority of tests wins."""
        codes = [
            'def f(x): return x + 1',  # Correct
            'def f(x): return x + 1',  # Correct (duplicate)
            'def f(x): return x * 2',  # Wrong
        ]
        gen_tests = [
            'assert f(1) == 2',  # Both correct and wrong pass this!
            'assert f(0) == 1',  # Only correct passes
            'assert f(3) == 4',  # Only correct passes
        ]
        result = codet_score(codes, gen_tests)
        assert result.best_code == 'def f(x): return x + 1'
        assert result.n_candidates == 3
        assert result.n_tests == 3

    def test_wrong_test_filtered_by_consensus(self):
        """A bad generated test doesn't hurt if majority disagree."""
        codes = [
            'def f(x): return x + 1',
            'def f(x): return x + 1',
            'def f(x): return x + 1',
        ]
        gen_tests = [
            'assert f(1) == 2',  # All pass — good test
            'assert f(0) == 99',  # All fail — bad test
        ]
        result = codet_score(codes, gen_tests)
        # All codes agree on both tests (pass first, fail second)
        # Agreement matrix: all identical → all score = 1.0
        assert result.best_score == 1.0

    def test_empty_tests_all_agree(self):
        """With no tests, all codes trivially agree."""
        codes = ['def f(): return 1', 'def f(): return 2']
        result = codet_score(codes, gen_tests=[])
        assert result.n_tests == 0
        # With no tests, fall back to first code
        assert result.best_code == codes[0]

    def test_single_code(self):
        """Single code candidate always wins."""
        codes = ['def f(x): return x + 1']
        gen_tests = ['assert f(1) == 2']
        result = codet_score(codes, gen_tests)
        assert result.best_code == codes[0]
        assert result.n_candidates == 1

    def test_tiebreak_shortest_code(self):
        """Tied scores broken by shortest code."""
        codes = [
            'def f(x): return x + 1  # long comment here',
            'def f(x): return x + 1',
        ]
        gen_tests = ['assert f(1) == 2']
        result = codet_score(codes, gen_tests)
        assert result.best_code == 'def f(x): return x + 1'

    def test_crashing_code_penalized(self):
        """Code that crashes gets low agreement score."""
        codes = [
            'def f(x): return x + 1',
            'def f(x): return x + 1',
            'def f(x): raise ValueError("bad")',
        ]
        gen_tests = [
            'assert f(1) == 2',
            'assert f(0) == 1',
        ]
        result = codet_score(codes, gen_tests, timeout=2)
        # Majority passes both tests → crasher disagrees
        assert result.best_code == 'def f(x): return x + 1'

    def test_result_has_all_scores(self):
        """Result contains per-candidate scores."""
        codes = [
            'def f(x): return x + 1',
            'def f(x): return x * 2',
        ]
        gen_tests = ['assert f(1) == 2', 'assert f(0) == 1']
        result = codet_score(codes, gen_tests)
        assert len(result.scores) == 2
        assert all(0.0 <= s <= 1.0 for s in result.scores)


class TestCodeTResult:
    """Tests for CodeTResult dataclass."""

    def test_fields(self):
        """Result stores all expected fields."""
        result = CodeTResult(
            best_code='def f(): return 1',
            best_score=0.8,
            scores=[0.8, 0.4],
            agreement_matrix=[[True, True], [True, False]],
            n_candidates=2,
            n_tests=2,
            all_codes=['a', 'b'],
        )
        assert result.n_candidates == 2
        assert result.n_tests == 2
        assert len(result.agreement_matrix) == 2
