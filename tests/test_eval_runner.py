"""Tests for unified evaluation runner."""

import torch

from lmt.data.code_data import CodeDataItem
from lmt.data.math_data import MathDataItem
from lmt.eval.runner import EvalConfig, EvalResults, EvalRunner
from lmt.models.config import ModelConfig
from lmt.training.reproducibility import set_seed


def _tiny_config() -> ModelConfig:
    """Create a tiny model config for testing."""
    return ModelConfig(
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        ffn_hidden_dim=64,
        vocab_size=64,
        context_length=32,
        tie_weights=True,
        dropout=0.0,
    )


class _SimpleTokenizer:
    """Minimal tokenizer mock for testing."""

    def encode(self, text: str) -> list[int]:
        return [hash(c) % 64 for c in text[:16]]

    def decode(self, ids: list[int]) -> str:
        return ''.join(chr(48 + (i % 26)) for i in ids)


class TestEvalConfig:
    """Test evaluation config dataclass."""

    def test_defaults(self):
        """Should have sensible defaults."""
        config = EvalConfig()
        assert config.max_new_tokens == 256
        assert config.temperature == 0.0
        assert config.device == 'cpu'

    def test_custom_values(self):
        """Should accept custom values."""
        config = EvalConfig(
            max_new_tokens=128,
            temperature=0.5,
            device='cuda',
        )
        assert config.max_new_tokens == 128
        assert config.temperature == 0.5


class TestEvalResults:
    """Test evaluation results dataclass."""

    def test_stores_metrics(self):
        """Should store metric dict."""
        results = EvalResults(
            metrics={'math_accuracy': 0.5, 'bpb': 1.2},
            details={'math': [{'correct': True}]},
        )
        assert results.metrics['math_accuracy'] == 0.5
        assert results.metrics['bpb'] == 1.2

    def test_summary_string(self):
        """Should produce a readable summary."""
        results = EvalResults(
            metrics={'math_accuracy': 0.75},
            details={},
        )
        summary = results.summary()
        assert 'math_accuracy' in summary
        assert '0.75' in summary


class TestEvalRunner:
    """Test the unified evaluation runner."""

    def _make_runner(self):
        """Create a runner with a tiny model."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)
        model = Qwen3(_tiny_config())
        tokenizer = _SimpleTokenizer()
        config = EvalConfig(max_new_tokens=4, device='cpu')
        return EvalRunner(model, tokenizer, config)

    def test_run_math_eval(self):
        """Should evaluate on math items."""
        runner = self._make_runner()
        items = [
            MathDataItem(prompt='What is 2+2?', ground_truth='4'),
            MathDataItem(prompt='What is 3+3?', ground_truth='6'),
        ]
        results = runner.run_math(items)
        assert 'math_accuracy' in results.metrics
        assert 'math_num_correct' in results.metrics
        assert 'math_num_total' in results.metrics
        assert results.metrics['math_num_total'] == 2

    def test_run_code_eval(self):
        """Should evaluate on code problems."""
        runner = self._make_runner()
        problems = [
            CodeDataItem(
                prompt='Write add(a, b).',
                tests='assert add(1, 2) == 3',
            ).to_code_problem(),
        ]
        results = runner.run_code(problems)
        assert 'code_accuracy' in results.metrics
        assert 'code_avg_reward' in results.metrics

    def test_run_bpb(self):
        """Should compute BPB on token sequences."""
        runner = self._make_runner()
        # Create dummy token sequences
        tokens = torch.randint(0, 64, (2, 16))
        results = runner.run_bpb(tokens, bytes_per_token=3.5)
        assert 'bpb' in results.metrics
        assert isinstance(results.metrics['bpb'], float)
        assert results.metrics['bpb'] > 0

    def test_run_all(self):
        """Should run all evaluations and merge results."""
        runner = self._make_runner()
        math_items = [
            MathDataItem(prompt='What is 1+1?', ground_truth='2'),
        ]
        code_problems = [
            CodeDataItem(
                prompt='Write noop().',
                tests='assert True',
            ).to_code_problem(),
        ]
        tokens = torch.randint(0, 64, (1, 16))

        results = runner.run_all(
            math_items=math_items,
            code_problems=code_problems,
            bpb_tokens=tokens,
            bpb_bytes_per_token=3.5,
        )
        assert 'math_accuracy' in results.metrics
        assert 'code_accuracy' in results.metrics
        assert 'bpb' in results.metrics

    def test_run_all_partial(self):
        """Should handle running only some evaluations."""
        runner = self._make_runner()
        results = runner.run_all(
            math_items=[
                MathDataItem(prompt='1+1?', ground_truth='2'),
            ],
        )
        assert 'math_accuracy' in results.metrics
        assert 'code_accuracy' not in results.metrics
        assert 'bpb' not in results.metrics
