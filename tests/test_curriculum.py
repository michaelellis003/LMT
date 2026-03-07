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
"""Tests for curriculum learning schedules."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lmt.models.config import ModelConfig
from lmt.models.gpt import GPT
from lmt.training.config import BaseTrainingConfig
from lmt.training.curriculum import (
    CurriculumSchedule,
    SequenceLengthCurriculum,
)
from lmt.training.trainer import Trainer


class TestCurriculumSchedule:
    """Tests for the CurriculumSchedule base class."""

    def test_is_abstract(self):
        """CurriculumSchedule cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CurriculumSchedule()  # type: ignore


class TestSequenceLengthCurriculum:
    """Tests for the SequenceLengthCurriculum."""

    def test_init_basic(self):
        """Can initialize with min/max length and total steps."""
        c = SequenceLengthCurriculum(
            min_length=16, max_length=128, warmup_steps=100
        )
        assert c.min_length == 16
        assert c.max_length == 128
        assert c.warmup_steps == 100

    def test_length_at_step_zero(self):
        """At step 0, length should be min_length."""
        c = SequenceLengthCurriculum(
            min_length=16, max_length=128, warmup_steps=100
        )
        assert c.get_length(step=0) == 16

    def test_length_at_final_step(self):
        """At warmup_steps, length should be max_length."""
        c = SequenceLengthCurriculum(
            min_length=16, max_length=128, warmup_steps=100
        )
        assert c.get_length(step=100) == 128

    def test_length_beyond_warmup(self):
        """After warmup, length should stay at max_length."""
        c = SequenceLengthCurriculum(
            min_length=16, max_length=128, warmup_steps=100
        )
        assert c.get_length(step=200) == 128

    def test_length_at_midpoint(self):
        """At halfway through warmup, length is between min and max."""
        c = SequenceLengthCurriculum(
            min_length=16, max_length=128, warmup_steps=100
        )
        mid_length = c.get_length(step=50)
        assert 16 < mid_length < 128

    def test_length_is_monotonically_increasing(self):
        """Length should never decrease as step increases."""
        c = SequenceLengthCurriculum(
            min_length=8, max_length=64, warmup_steps=50
        )
        prev = 0
        for step in range(60):
            length = c.get_length(step)
            assert length >= prev, f'Decreased at step {step}'
            prev = length

    def test_length_is_always_divisible_by_alignment(self):
        """Length should always be a multiple of alignment."""
        c = SequenceLengthCurriculum(
            min_length=8, max_length=64, warmup_steps=50, alignment=8
        )
        for step in range(60):
            length = c.get_length(step)
            assert length % 8 == 0, f'step {step}: {length} not aligned'

    def test_default_alignment_is_1(self):
        """Default alignment of 1 means any integer length is valid."""
        c = SequenceLengthCurriculum(
            min_length=10, max_length=100, warmup_steps=50
        )
        # Just verify it doesn't crash and returns integers
        for step in range(55):
            length = c.get_length(step)
            assert isinstance(length, int)

    def test_min_equals_max(self):
        """When min_length == max_length, always returns that length."""
        c = SequenceLengthCurriculum(
            min_length=64, max_length=64, warmup_steps=100
        )
        for step in range(110):
            assert c.get_length(step) == 64

    def test_invalid_min_greater_than_max(self):
        """Raises ValueError if min_length > max_length."""
        with pytest.raises(ValueError, match='min_length'):
            SequenceLengthCurriculum(
                min_length=128, max_length=64, warmup_steps=100
            )

    def test_invalid_warmup_steps_zero(self):
        """Raises ValueError if warmup_steps <= 0."""
        with pytest.raises(ValueError, match='warmup_steps'):
            SequenceLengthCurriculum(
                min_length=16, max_length=128, warmup_steps=0
            )

    def test_truncate_batch(self):
        """truncate_batch crops input and target to the curriculum length."""
        c = SequenceLengthCurriculum(
            min_length=8, max_length=32, warmup_steps=100
        )
        inp = torch.randint(0, 100, (4, 32))  # batch=4, seq=32
        tgt = torch.randint(0, 100, (4, 32))

        # At step 0, should truncate to min_length=8
        inp_t, tgt_t = c.truncate_batch(inp, tgt, step=0)
        assert inp_t.shape == (4, 8)
        assert tgt_t.shape == (4, 8)

        # At step 100, should keep full length
        inp_t, tgt_t = c.truncate_batch(inp, tgt, step=100)
        assert inp_t.shape == (4, 32)
        assert tgt_t.shape == (4, 32)

    def test_truncate_preserves_data(self):
        """Truncated batch should be a prefix of the original."""
        c = SequenceLengthCurriculum(
            min_length=4, max_length=16, warmup_steps=100
        )
        inp = torch.arange(16).unsqueeze(0)  # [1, 16]
        tgt = torch.arange(1, 17).unsqueeze(0)

        inp_t, tgt_t = c.truncate_batch(inp, tgt, step=0)
        assert torch.equal(inp_t, inp[:, :4])
        assert torch.equal(tgt_t, tgt[:, :4])


class TestCurriculumTrainerIntegration:
    """Tests that curriculum integrates correctly with the Trainer."""

    @pytest.fixture()
    def setup(self):
        """Create a minimal model and data for integration tests."""
        torch.manual_seed(42)
        config = ModelConfig(
            vocab_size=50, embed_dim=32, num_heads=2,
            num_layers=1, context_length=16, dropout=0.0,
        )
        model = GPT(config)

        # Fake data: random token sequences
        n_samples = 20
        inp = torch.randint(0, 50, (n_samples, 16))
        tgt = torch.randint(0, 50, (n_samples, 16))
        ds = TensorDataset(inp, tgt)
        loader = DataLoader(ds, batch_size=4)

        train_config = BaseTrainingConfig(
            num_epochs=1, eval_freq=5, eval_iter=2,
            learning_rate=1e-3, weight_decay=0.0,
            device='cpu',
        )
        return model, loader, train_config

    def test_trainer_accepts_curriculum(self, setup):
        """Trainer can be constructed with a curriculum parameter."""
        model, loader, train_config = setup
        curriculum = SequenceLengthCurriculum(
            min_length=4, max_length=16, warmup_steps=10
        )
        trainer = Trainer(
            model, loader, loader, train_config,
            curriculum=curriculum,
        )
        assert trainer.curriculum is curriculum

    def test_trainer_without_curriculum(self, setup):
        """Trainer works normally without curriculum."""
        model, loader, train_config = setup
        trainer = Trainer(model, loader, loader, train_config)
        assert trainer.curriculum is None
        result = trainer.train()
        assert len(result['train_losses']) > 0

    def test_trainer_with_curriculum_trains(self, setup):
        """Trainer with curriculum completes training without error."""
        model, loader, train_config = setup
        curriculum = SequenceLengthCurriculum(
            min_length=4, max_length=16, warmup_steps=10
        )
        trainer = Trainer(
            model, loader, loader, train_config,
            curriculum=curriculum,
        )
        result = trainer.train()
        assert len(result['train_losses']) > 0
        assert result['train_losses'][-1] < 10.0  # not diverged
