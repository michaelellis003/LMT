"""Tests for BabyLM pretraining recipe components."""

import pytest

tokenizers_mod = pytest.importorskip('tokenizers')
transformers_mod = pytest.importorskip('transformers')

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn import functional as f  # noqa: E402

from lmt.data.text_dataset import TextDataset  # noqa: E402
from lmt.models.config import ModelConfig  # noqa: E402
from lmt.models.factory import create_model  # noqa: E402
from lmt.models.sizing import suggest_config  # noqa: E402
from lmt.tokenizer.trainer import train_bpe_tokenizer  # noqa: E402
from lmt.training.train_config import TrainConfig  # noqa: E402
from lmt.training.train_loop import train_loop  # noqa: E402


def _mock_babylm_texts() -> list[str]:
    """Small corpus mimicking BabyLM data."""
    return [
        'just like your book at home .',
        'do you want to look at that ?',
        'see the dog running in the park .',
        'mommy loves you very much .',
        'where is the ball ?',
        'the cat sat on the mat .',
        'one two three four five .',
        'can you say hello ?',
        'look at the big red truck .',
        'daddy is coming home soon .',
    ] * 50  # 500 lines


class TestBabyLMPipeline:
    """Test the BabyLM end-to-end pipeline components."""

    def test_tokenizer_training_on_babylm(self):
        """Should train a BPE tokenizer on BabyLM-like text."""
        texts = _mock_babylm_texts()
        tok = train_bpe_tokenizer(texts, vocab_size=256)
        assert tok.vocab_size <= 256

        # Should encode and decode correctly
        ids = tok.encode('the cat sat on the mat .')
        decoded = tok.decode(ids)
        assert 'cat' in decoded
        assert 'mat' in decoded

    def test_text_dataset_from_tokenized_babylm(self):
        """Should create a TextDataset from tokenized BabyLM data."""
        texts = _mock_babylm_texts()
        tok = train_bpe_tokenizer(texts, vocab_size=256)

        dataset = TextDataset(texts, tok, context_length=32)
        assert len(dataset) > 0

        sample = dataset[0]
        assert sample.shape == (33,)  # context_length + 1

    def test_model_sizing_for_babylm(self):
        """Should suggest a model config within BabyLM budget."""
        # BabyLM strict-small: 10M words ~ roughly 5-10M params model
        config = suggest_config(target_params=1_000_000, vocab_size=256)
        assert config['embed_dim'] > 0
        assert config['num_layers'] > 0

    def test_end_to_end_train(self):
        """Should train a small model on mock BabyLM data."""
        texts = _mock_babylm_texts()
        tok = train_bpe_tokenizer(texts, vocab_size=256)

        # Create dataset
        dataset = TextDataset(texts[:100], tok, context_length=32)
        train_data = torch.stack(
            [dataset[i] for i in range(min(64, len(dataset)))]
        )

        # Create small model
        config = ModelConfig(
            embed_dim=32,
            num_heads=2,
            num_kv_heads=2,
            num_layers=1,
            ffn_hidden_dim=64,
            vocab_size=tok.vocab_size,
            context_length=32,
            tie_weights=True,
            dropout=0.0,
        )
        model = create_model('gpt', config)

        # Train
        def loss_fn(m: nn.Module, batch: torch.Tensor) -> torch.Tensor:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = m(inputs)
            return f.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        train_config = TrainConfig(total_steps=5, lr=1e-3)
        state = train_loop(
            model=model,
            train_data=train_data,
            loss_fn=loss_fn,
            config=train_config,
            batch_size=8,
        )
        assert state.step == 5
        assert len(state.losses) == 5

    def test_word_count_budget(self):
        """Should count words to verify budget compliance."""
        texts = _mock_babylm_texts()
        word_count = sum(len(t.split()) for t in texts)
        # 500 lines * ~8 words = ~4000 words (well under 10M)
        assert word_count < 10_000_000
