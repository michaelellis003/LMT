"""Tests for HuggingFace tokenizer bridge.

Validates that HF tokenizers can be used with LMT's reward pipeline
and GRPO training loop.
"""

import pytest

try:
    import transformers as _transformers  # noqa: F401

    _has_transformers = True
except ImportError:
    _has_transformers = False


@pytest.mark.skipif(not _has_transformers, reason='transformers not installed')
class TestHFTokenizerBridge:
    """Test HF tokenizer integration with LMT reward pipeline."""

    def test_wrap_hf_tokenizer(self):
        """Wrapped HF tokenizer should have encode/decode methods."""
        from lmt.tokenizer.hf_tokenizer import HFTokenizerWrapper

        wrapper = HFTokenizerWrapper.from_pretrained('Qwen/Qwen3-0.6B')
        assert hasattr(wrapper, 'encode')
        assert hasattr(wrapper, 'decode')
        assert wrapper.vocab_size > 0

    def test_encode_decode_roundtrip(self):
        """Encode then decode should preserve text content."""
        from lmt.tokenizer.hf_tokenizer import HFTokenizerWrapper

        wrapper = HFTokenizerWrapper.from_pretrained('Qwen/Qwen3-0.6B')
        text = r'The answer is \boxed{42}'
        ids = wrapper.encode(text)
        decoded = wrapper.decode(ids)

        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert r'\boxed{42}' in decoded

    def test_reward_bridge_with_hf_tokenizer(self):
        """Real HF tokenizer should work with create_math_reward_fn."""
        import torch

        from lmt.data.math_data import MathDataItem
        from lmt.recipes.reasoning import create_math_reward_fn
        from lmt.tokenizer.hf_tokenizer import HFTokenizerWrapper

        wrapper = HFTokenizerWrapper.from_pretrained('Qwen/Qwen3-0.6B')

        item = MathDataItem(prompt='What is 2+2?', ground_truth='4')
        reward_fn = create_math_reward_fn(item, wrapper)

        # Encode a correct response then pass as tensor
        correct_text = r'Let me think... The answer is \boxed{4}'
        response_ids = torch.tensor(wrapper.encode(correct_text))
        prompt_ids = torch.tensor(wrapper.encode(item.prompt))

        reward = reward_fn(prompt_ids, response_ids)
        assert reward == 1.0

    def test_wrong_answer_with_hf_tokenizer(self):
        """Wrong answer should get 0.0 reward with real tokenizer."""
        import torch

        from lmt.data.math_data import MathDataItem
        from lmt.recipes.reasoning import create_math_reward_fn
        from lmt.tokenizer.hf_tokenizer import HFTokenizerWrapper

        wrapper = HFTokenizerWrapper.from_pretrained('Qwen/Qwen3-0.6B')

        item = MathDataItem(prompt='What is 2+2?', ground_truth='4')
        reward_fn = create_math_reward_fn(item, wrapper)

        wrong_text = r'I think the answer is \boxed{5}'
        response_ids = torch.tensor(wrapper.encode(wrong_text))
        prompt_ids = torch.tensor(wrapper.encode(item.prompt))

        reward = reward_fn(prompt_ids, response_ids)
        assert reward == 0.0

    def test_vocab_size_matches_qwen3(self):
        """Qwen3 tokenizer should report correct vocab size."""
        from lmt.tokenizer.hf_tokenizer import HFTokenizerWrapper

        wrapper = HFTokenizerWrapper.from_pretrained('Qwen/Qwen3-0.6B')
        assert wrapper.vocab_size == 151643

    def test_eos_token(self):
        """Wrapper should expose EOS token ID."""
        from lmt.tokenizer.hf_tokenizer import HFTokenizerWrapper

        wrapper = HFTokenizerWrapper.from_pretrained('Qwen/Qwen3-0.6B')
        assert wrapper.eos_token_id is not None
        assert isinstance(wrapper.eos_token_id, int)
