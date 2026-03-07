"""Tests for lm-eval-harness wrapper."""

import pytest

from lmt.models.config import ModelConfigPresets
from lmt.models.gpt import GPT
from lmt.tokenizer import BPETokenizer

lm_eval = pytest.importorskip('lm_eval', reason='lm-eval not installed')
from lmt.training.lm_eval_adapter import LMTEvalModel  # noqa: E402


class TestLMTEvalModel:
    """Test the lm-eval-harness adapter for LMT models."""

    @pytest.fixture
    def small_model(self):
        """Create a small GPT model for testing."""
        config = ModelConfigPresets.small_gpt(context_length=64)
        model = GPT(config)
        return model

    @pytest.fixture
    def tokenizer(self):
        """Create a BPE tokenizer."""
        return BPETokenizer()

    @pytest.fixture
    def adapter(self, small_model, tokenizer):
        """Create an LMTEvalModel adapter."""
        return LMTEvalModel(small_model, tokenizer, device='cpu')

    def test_creates_adapter(self, adapter):
        """Should create an adapter instance."""
        assert adapter is not None

    def test_loglikelihood_basic(self, adapter):
        """Loglikelihood should return (float, bool) tuples."""
        from lm_eval.api.instance import Instance

        requests = [
            Instance(
                request_type='loglikelihood',
                doc={},
                arguments=('The cat sat on', ' the mat'),
                idx=0,
            ),
        ]
        results = adapter.loglikelihood(requests)
        assert len(results) == 1
        ll, is_greedy = results[0]
        assert isinstance(ll, float)
        assert isinstance(is_greedy, bool)
        assert ll <= 0.0  # log-likelihood is non-positive

    def test_loglikelihood_empty_context(self, adapter):
        """Should handle empty context string."""
        from lm_eval.api.instance import Instance

        requests = [
            Instance(
                request_type='loglikelihood',
                doc={},
                arguments=('', 'Hello'),
                idx=0,
            ),
        ]
        results = adapter.loglikelihood(requests)
        assert len(results) == 1
        ll, is_greedy = results[0]
        assert isinstance(ll, float)

    def test_loglikelihood_rolling(self, adapter):
        """loglikelihood_rolling should return floats."""
        from lm_eval.api.instance import Instance

        requests = [
            Instance(
                request_type='loglikelihood_rolling',
                doc={},
                arguments=('The cat sat on the mat.',),
                idx=0,
            ),
        ]
        results = adapter.loglikelihood_rolling(requests)
        assert len(results) == 1
        assert isinstance(results[0], float)
        assert results[0] <= 0.0  # negative log-likelihood

    def test_generate_until(self, adapter):
        """generate_until should return strings."""
        from lm_eval.api.instance import Instance

        requests = [
            Instance(
                request_type='generate_until',
                doc={},
                arguments=(
                    'Once upon a time',
                    {'until': ['\n'], 'do_sample': False, 'max_gen_toks': 10},
                ),
                idx=0,
            ),
        ]
        results = adapter.generate_until(requests)
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_multiple_requests(self, adapter):
        """Should handle multiple requests in batch."""
        from lm_eval.api.instance import Instance

        requests = [
            Instance(
                request_type='loglikelihood',
                doc={},
                arguments=('Hello', ' world'),
                idx=0,
            ),
            Instance(
                request_type='loglikelihood',
                doc={},
                arguments=('Good', ' morning'),
                idx=1,
            ),
        ]
        results = adapter.loglikelihood(requests)
        assert len(results) == 2
        for ll, is_greedy in results:
            assert isinstance(ll, float)
            assert isinstance(is_greedy, bool)
