"""Tests for math reasoning evaluation."""

from lmt.data.math_data import MathDataItem
from lmt.eval.math_eval import evaluate_math_accuracy
from lmt.models.config import ModelConfig


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


class TestMathEvaluation:
    """Test math evaluation pipeline."""

    def test_evaluate_returns_accuracy(self):
        """evaluate_math_accuracy should return accuracy and details."""
        from lmt.models.qwen3.qwen3 import Qwen3

        model = Qwen3(_tiny_config())

        items = [
            MathDataItem(prompt='What is 2+2?', ground_truth='4'),
            MathDataItem(prompt='What is 3+3?', ground_truth='6'),
        ]

        # Mock tokenizer
        class MockTokenizer:
            """Mock tokenizer for testing."""

            def encode(self, text: str) -> list[int]:
                """Simple encode."""
                return [ord(c) % 64 for c in text[:16]]

            def decode(self, ids: list[int]) -> str:
                """Return fixed response (won't match ground truth)."""
                return 'some random text'

        result = evaluate_math_accuracy(
            model=model,
            items=items,
            tokenizer=MockTokenizer(),
            max_new_tokens=4,
        )

        assert 'accuracy' in result
        assert 'num_correct' in result
        assert 'num_total' in result
        assert result['num_total'] == 2
        assert 0.0 <= result['accuracy'] <= 1.0

    def test_correct_answers_give_high_accuracy(self):
        """When model generates correct boxed answers, accuracy should be 1."""
        from lmt.models.qwen3.qwen3 import Qwen3

        model = Qwen3(_tiny_config())

        items = [
            MathDataItem(prompt='What is 2+2?', ground_truth='4'),
        ]

        # Mock tokenizer that always returns correct answer
        class CorrectTokenizer:
            """Tokenizer that decodes to correct answer."""

            def encode(self, text: str) -> list[int]:
                """Simple encode."""
                return [1, 2, 3]

            def decode(self, ids: list[int]) -> str:
                """Return correct boxed answer."""
                return r'The answer is \boxed{4}'

        result = evaluate_math_accuracy(
            model=model,
            items=items,
            tokenizer=CorrectTokenizer(),
            max_new_tokens=4,
        )

        assert result['accuracy'] == 1.0
        assert result['num_correct'] == 1

    def test_empty_items_returns_zero(self):
        """Empty items list should return 0 accuracy."""
        from lmt.models.qwen3.qwen3 import Qwen3

        model = Qwen3(_tiny_config())

        class MockTokenizer:
            """Mock tokenizer."""

            def encode(self, text: str) -> list[int]:
                """Simple encode."""
                return [1]

            def decode(self, ids: list[int]) -> str:
                """Simple decode."""
                return ''

        result = evaluate_math_accuracy(
            model=model,
            items=[],
            tokenizer=MockTokenizer(),
            max_new_tokens=4,
        )

        assert result['accuracy'] == 0.0
        assert result['num_total'] == 0
