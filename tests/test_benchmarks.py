"""Tests for the evaluation benchmarks wrapper."""

import pytest

from lmt.models.config import ModelConfigPresets
from lmt.models.gpt import GPT
from lmt.tokenizer import BPETokenizer

lm_eval = pytest.importorskip('lm_eval', reason='lm-eval not installed')


class TestTaskPresets:
    """Test that task preset names resolve correctly."""

    def test_leaderboard_preset_exists(self):
        """The leaderboard preset should map to known task names."""
        from lmt.eval.benchmarks import TASK_PRESETS

        assert 'leaderboard' in TASK_PRESETS
        tasks = TASK_PRESETS['leaderboard']
        assert isinstance(tasks, list)
        assert len(tasks) > 0

    def test_math_preset_exists(self):
        """The math preset should include math benchmarks."""
        from lmt.eval.benchmarks import TASK_PRESETS

        assert 'math' in TASK_PRESETS

    def test_code_preset_exists(self):
        """The code preset should include code benchmarks."""
        from lmt.eval.benchmarks import TASK_PRESETS

        assert 'code' in TASK_PRESETS

    def test_resolve_preset_expands(self):
        """resolve_tasks should expand a preset name into task list."""
        from lmt.eval.benchmarks import resolve_tasks

        tasks = resolve_tasks(['leaderboard'])
        assert isinstance(tasks, list)
        assert len(tasks) > 1  # preset expands to multiple tasks

    def test_resolve_plain_tasks_pass_through(self):
        """Plain task names should pass through unchanged."""
        from lmt.eval.benchmarks import resolve_tasks

        tasks = resolve_tasks(['hellaswag', 'piqa'])
        assert tasks == ['hellaswag', 'piqa']

    def test_resolve_mixed_presets_and_tasks(self):
        """Should handle a mix of presets and plain task names."""
        from lmt.eval.benchmarks import resolve_tasks

        tasks = resolve_tasks(['leaderboard', 'lambada_openai'])
        assert 'lambada_openai' in tasks
        assert len(tasks) > 2  # leaderboard expanded + lambada


class TestEvaluateModel:
    """Test the evaluate_model high-level API."""

    @pytest.fixture
    def small_model(self):
        """Create a small GPT model for testing."""
        config = ModelConfigPresets.small_gpt(context_length=64)
        return GPT(config)

    @pytest.fixture
    def tokenizer(self):
        """Create a BPE tokenizer."""
        return BPETokenizer()

    def test_evaluate_returns_dict(self, small_model, tokenizer):
        """evaluate_model should return a dict of task -> score."""
        from lmt.eval.benchmarks import evaluate_model

        results = evaluate_model(
            model=small_model,
            tokenizer=tokenizer,
            tasks=['hellaswag'],
            device='cpu',
            num_fewshot=0,
            limit=5,
        )
        assert isinstance(results, dict)
        assert 'hellaswag' in results

    def test_evaluate_scores_are_floats(self, small_model, tokenizer):
        """All scores should be floats between 0 and 1."""
        from lmt.eval.benchmarks import evaluate_model

        results = evaluate_model(
            model=small_model,
            tokenizer=tokenizer,
            tasks=['hellaswag'],
            device='cpu',
            num_fewshot=0,
            limit=5,
        )
        for task, score in results.items():
            if task != 'aggregate':
                assert isinstance(score, float), f'{task} score not float'
                assert 0.0 <= score <= 1.0, f'{task} score out of range'

    def test_evaluate_multiple_tasks(self, small_model, tokenizer):
        """Should handle multiple tasks at once."""
        from lmt.eval.benchmarks import evaluate_model

        results = evaluate_model(
            model=small_model,
            tokenizer=tokenizer,
            tasks=['hellaswag', 'piqa'],
            device='cpu',
            num_fewshot=0,
            limit=5,
        )
        assert 'hellaswag' in results
        assert 'piqa' in results

    def test_evaluate_with_bpb(self, small_model, tokenizer):
        """Should include BPB when include_bpb=True."""
        from lmt.eval.benchmarks import evaluate_model

        results = evaluate_model(
            model=small_model,
            tokenizer=tokenizer,
            tasks=['hellaswag'],
            device='cpu',
            num_fewshot=0,
            limit=5,
            include_bpb=True,
            bpb_texts=['Hello world. This is a test sentence.'],
        )
        assert 'bpb' in results
        assert isinstance(results['bpb'], float)
        assert results['bpb'] > 0.0

    def test_limit_parameter(self, small_model, tokenizer):
        """Limit should restrict the number of eval examples."""
        from lmt.eval.benchmarks import evaluate_model

        # Should complete quickly with limit=2
        results = evaluate_model(
            model=small_model,
            tokenizer=tokenizer,
            tasks=['hellaswag'],
            device='cpu',
            num_fewshot=0,
            limit=2,
        )
        assert 'hellaswag' in results
