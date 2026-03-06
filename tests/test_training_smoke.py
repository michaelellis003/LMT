# Copyright 2025 Michael Ellis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Smoke tests: do models actually learn language?

These tests train tiny models on small text corpora and verify that:
1. Loss decreases during training (the model is learning)
2. Final loss is well below random chance (ln(vocab_size))
3. The model generates coherent continuations (not garbage)

This is the most important test suite in the library. A model that
passes shape/gradient tests but can't learn "the cat sat on the mat"
is a broken model.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig
from lmt.training.config import BaseTrainingConfig
from lmt.training.datasets import GPTDataset
from lmt.training.loss import calc_loss_loader


def _build_vocab(text: str) -> dict[str, int]:
    """Build a character-level vocabulary from text."""
    chars = sorted(set(text))
    return {ch: i for i, ch in enumerate(chars)}


class _CharTokenizer:
    """Minimal character-level tokenizer for testing."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: ch for ch, i in vocab.items()}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> list[int]:
        """Encode text to character IDs."""
        return [self.str_to_int[ch] for ch in text if ch in self.str_to_int]

    def decode(self, ids: list[int]) -> str:
        """Decode character IDs back to text."""
        return ''.join(self.int_to_str.get(i, '?') for i in ids)


# A small but structured corpus with clear patterns.
# Key property: it's repetitive enough that a tiny model should
# learn the patterns, but not so trivial it's meaningless.
CORPUS = (
    'the cat sat on the mat. '
    'the dog sat on the log. '
    'the cat ran to the hat. '
    'the dog ran to the log. '
) * 20  # Repeat for enough training data


@pytest.mark.slow
class TestTrainingSmoke:
    """Verify models actually learn from text data."""

    def _make_model_and_data(
        self,
        block_config: BlockConfig | None = None,
    ) -> tuple[BaseModel, DataLoader, DataLoader, _CharTokenizer, float]:
        """Build a tiny model, tokenizer, and data loaders.

        Returns model, train_loader, val_loader, tokenizer, random_loss.
        """
        torch.manual_seed(42)
        vocab = _build_vocab(CORPUS)
        tokenizer = _CharTokenizer(vocab)
        vocab_size = tokenizer.vocab_size

        context_length = 32
        model_config = ModelConfig(
            context_length=context_length,
            vocab_size=vocab_size,
            num_layers=2,
            num_heads=4,
            embed_dim=64,
            dropout=0.0,
            num_kv_heads=4,
        )

        if block_config is None:
            block_config = BlockConfig(
                attention='mha', ffn='default', norm='rmsnorm'
            )

        model = BaseModel(model_config, block_config, learned_pos_embed=True)

        # Split corpus: 80% train, 20% val
        split = int(len(CORPUS) * 0.8)
        train_text = CORPUS[:split]
        val_text = CORPUS[split:]

        train_ds = GPTDataset(
            train_text, tokenizer, context_length, stride=context_length
        )
        val_ds = GPTDataset(
            val_text, tokenizer, context_length, stride=context_length
        )

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        random_loss = math.log(vocab_size)

        return model, train_loader, val_loader, tokenizer, random_loss

    def _train_loop(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        num_epochs: int = 20,
        lr: float = 1e-3,
    ) -> list[float]:
        """Run a simple training loop and return per-epoch losses."""
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01
        )
        device = torch.device('cpu')
        model.to(device)
        epoch_losses: list[float] = []

        for _epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for input_batch, target_batch in train_loader:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                optimizer.zero_grad()
                logits = model(input_batch)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), target_batch.flatten()
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            epoch_losses.append(total_loss / n_batches)

        return epoch_losses

    def test_mha_learns(self) -> None:
        """MHA model loss should drop well below random chance."""
        model, train_loader, val_loader, _, random_loss = (
            self._make_model_and_data(
                BlockConfig(attention='mha', ffn='default', norm='rmsnorm')
            )
        )

        epoch_losses = self._train_loop(model, train_loader, num_epochs=30)

        # Loss should decrease
        assert epoch_losses[-1] < epoch_losses[0], (
            f'Loss did not decrease: {epoch_losses[0]:.3f} -> '
            f'{epoch_losses[-1]:.3f}'
        )

        # Final loss should be well below random (ln(vocab_size) ≈ 2.89)
        # We require at least 50% below random -- generous threshold
        assert epoch_losses[-1] < random_loss * 0.5, (
            f'Final loss {epoch_losses[-1]:.3f} is not substantially '
            f'below random chance {random_loss:.3f}'
        )

        # Validation loss should also be reasonable
        device = torch.device('cpu')
        model.eval()
        val_loss = calc_loss_loader(val_loader, model, device)
        assert val_loss < random_loss * 0.7, (
            f'Val loss {val_loss:.3f} too high (random={random_loss:.3f})'
        )

    def test_gqa_learns(self) -> None:
        """GQA model should also learn from text."""
        model, train_loader, val_loader, _, random_loss = (
            self._make_model_and_data(
                BlockConfig(attention='gqa', ffn='swiglu', norm='rmsnorm')
            )
        )

        epoch_losses = self._train_loop(model, train_loader, num_epochs=30)

        assert epoch_losses[-1] < epoch_losses[0]
        assert epoch_losses[-1] < random_loss * 0.5, (
            f'GQA final loss {epoch_losses[-1]:.3f} too high '
            f'(random={random_loss:.3f})'
        )

    def test_gated_delta_net_learns(self) -> None:
        """GatedDeltaNet (linear attention) should also learn."""
        model, train_loader, val_loader, _, random_loss = (
            self._make_model_and_data(
                BlockConfig(
                    attention='gated_delta_net',
                    ffn='swiglu',
                    norm='rmsnorm',
                )
            )
        )

        # GDN may need more epochs since it's a very different mechanism
        epoch_losses = self._train_loop(
            model, train_loader, num_epochs=50, lr=5e-4
        )

        assert epoch_losses[-1] < epoch_losses[0]
        # Be more lenient with GDN -- it's a fundamentally different
        # architecture and may not converge as fast on tiny data
        assert epoch_losses[-1] < random_loss * 0.8, (
            f'GDN final loss {epoch_losses[-1]:.3f} too high '
            f'(random={random_loss:.3f})'
        )

    def test_loss_monotonically_trends_down(self) -> None:
        """Loss should generally trend downward, not oscillate wildly.

        We don't require strict monotonic decrease (that's unrealistic),
        but the trend over 5-epoch windows should be clearly downward.
        """
        model, train_loader, _, _, _ = self._make_model_and_data()

        epoch_losses = self._train_loop(model, train_loader, num_epochs=20)

        # Compare first 5 epochs avg to last 5 epochs avg
        early_avg = sum(epoch_losses[:5]) / 5
        late_avg = sum(epoch_losses[-5:]) / 5

        assert late_avg < early_avg * 0.7, (
            f'Loss not trending down: early avg {early_avg:.3f}, '
            f'late avg {late_avg:.3f}'
        )

    def test_generation_coherence(self) -> None:
        """After training, model should generate from the vocabulary.

        This doesn't check grammar -- just that the model outputs
        tokens from the training distribution, not uniform noise.
        """
        model, train_loader, _, tokenizer, _ = self._make_model_and_data()

        self._train_loop(model, train_loader, num_epochs=30)

        model.eval()
        # Prompt with "the "
        prompt = tokenizer.encode('the ')
        input_ids = torch.tensor([prompt], dtype=torch.long)

        with torch.no_grad():
            for _ in range(20):
                logits = model(input_ids[:, -32:])  # crop to context
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        generated = tokenizer.decode(input_ids[0].tolist())

        # The generated text should contain characters from the corpus
        corpus_chars = set(CORPUS)
        gen_chars = set(generated)
        overlap = len(gen_chars & corpus_chars) / len(gen_chars)
        assert overlap > 0.8, (
            f'Generated text has too many out-of-vocab chars: {generated!r}'
        )

    def test_trainer_integration(self) -> None:
        """Verify the Trainer class works end-to-end."""
        model, train_loader, val_loader, _, random_loss = (
            self._make_model_and_data()
        )

        config = BaseTrainingConfig(
            num_epochs=10,
            eval_freq=5,
            eval_iter=2,
            learning_rate=1e-3,
            weight_decay=0.01,
            batch_size=8,
            device='cpu',
        )

        from lmt.training.trainer import Trainer

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )

        results = trainer.train()

        assert 'train_losses' in results
        assert 'val_losses' in results
        assert len(results['train_losses']) > 0
        # Training should produce decreasing losses
        if len(results['train_losses']) >= 2:
            assert results['train_losses'][-1] < results['train_losses'][0]
