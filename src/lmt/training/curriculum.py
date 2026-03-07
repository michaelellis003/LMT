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
r"""Curriculum learning schedules for training.

Curriculum learning trains on easier examples first, then gradually
increases difficulty. The most common strategy for language models is
**sequence length warmup**: start with short sequences and linearly
increase to the full context length.

Reference:
    Bengio et al., 2009. "Curriculum Learning."
    Li et al., 2022. "Stability and Scalability of Sequence Length Warmup."
"""

from abc import ABC, abstractmethod

import torch


class CurriculumSchedule(ABC):
    """Abstract base class for curriculum learning schedules.

    A curriculum schedule controls some aspect of training difficulty
    (e.g., sequence length, data complexity) as a function of the
    current training step.
    """

    @abstractmethod
    def get_length(self, step: int) -> int:
        """Return the sequence length for the given training step.

        Args:
            step: The current global training step.

        Returns:
            The target sequence length for this step.
        """


class SequenceLengthCurriculum(CurriculumSchedule):
    """Linearly increases sequence length from min to max over warmup steps.

    This implements the simplest and most widely-used curriculum strategy:
    start training with short sequences and linearly ramp up to the full
    context length. Short sequences are "easier" because the model needs
    to track fewer dependencies.

    The intuition: early in training, the model learns local patterns
    (character n-grams, common words). Long-range dependencies only become
    learnable once local patterns are solid. Starting with short sequences
    focuses early gradient signal on local patterns.

    Args:
        min_length: Starting sequence length.
        max_length: Final sequence length (full context).
        warmup_steps: Number of steps to linearly increase from min to max.
        alignment: Round lengths to multiples of this value (useful for
            hardware efficiency). Defaults to 1.

    Example:
        >>> curriculum = SequenceLengthCurriculum(
        ...     min_length=16, max_length=128, warmup_steps=1000
        ... )
        >>> curriculum.get_length(0)
        16
        >>> curriculum.get_length(500)
        72
        >>> curriculum.get_length(1000)
        128
    """

    def __init__(
        self,
        min_length: int,
        max_length: int,
        warmup_steps: int,
        alignment: int = 1,
    ) -> None:
        """Initializes the SequenceLengthCurriculum.

        Args:
            min_length: Starting sequence length.
            max_length: Final sequence length (full context).
            warmup_steps: Steps to linearly increase from min to max.
            alignment: Round lengths to multiples of this value.
        """
        if min_length > max_length:
            raise ValueError(
                f'min_length ({min_length}) must be <= max_length '
                f'({max_length})'
            )
        if warmup_steps <= 0:
            raise ValueError(f'warmup_steps must be > 0, got {warmup_steps}')

        self.min_length = min_length
        self.max_length = max_length
        self.warmup_steps = warmup_steps
        self.alignment = alignment

    def get_length(self, step: int) -> int:
        """Return the sequence length for the given training step.

        Linearly interpolates between min_length and max_length over
        warmup_steps, then clamps to max_length.

        Args:
            step: The current global training step.

        Returns:
            The target sequence length, rounded down to nearest alignment.
        """
        progress = min(step / self.warmup_steps, 1.0)
        raw = self.min_length + progress * (self.max_length - self.min_length)

        # Round down to alignment, but never below min_length
        aligned = int(raw) // self.alignment * self.alignment
        return max(aligned, self.min_length)

    def truncate_batch(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate a batch to the curriculum length for this step.

        Takes the first `get_length(step)` tokens from each sequence.

        Args:
            input_batch: Input tensor of shape [batch, seq_len].
            target_batch: Target tensor of shape [batch, seq_len].
            step: Current global training step.

        Returns:
            Tuple of truncated (input, target) tensors.
        """
        length = self.get_length(step)
        return input_batch[:, :length], target_batch[:, :length]
