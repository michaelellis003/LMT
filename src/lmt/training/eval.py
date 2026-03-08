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
"""Evaluation utilities for language models.

This module provides functions for computing standard evaluation metrics
like perplexity on token sequences.
"""

import torch
import torch.nn as nn
from torch.nn import functional as f


def compute_perplexity(
    model: nn.Module,
    token_sequences: torch.Tensor,
    device: str = 'cpu',
    stride: int | None = None,
) -> float:
    r"""Compute perplexity of a language model on token sequences.

    Perplexity is defined as :math:`\exp(\text{avg cross-entropy loss})`
    where the average is taken over all predicted tokens. Lower perplexity
    indicates a better model.

    For sequences longer than the model's context length, a sliding window
    with the given ``stride`` is used. Tokens within the overlap region
    use the full available context but only the non-overlapping tokens
    contribute to the loss, following the HuggingFace convention.

    Args:
        model: A language model that returns logits of shape
            ``[batch, seq_len, vocab_size]``.
        token_sequences: Token IDs of shape ``[num_sequences, seq_len]``.
            Each row is an independent sequence.
        device: Device to run evaluation on.
        stride: Sliding window stride for sequences longer than the
            model's context length. Defaults to the context length
            (no overlap).

    Returns:
        The perplexity as a float.
    """
    was_training = model.training
    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)

    # Get context length from model config
    config = getattr(model, 'config', None)
    if config is not None and hasattr(config, 'context_length'):
        ctx_len: int = config.context_length
    else:
        ctx_len = token_sequences.shape[1] - 1

    step = stride if stride is not None else ctx_len

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for seq in token_sequences:
            seq_len = seq.shape[0]

            # Sliding window over the sequence
            for begin in range(0, seq_len - 1, step):
                end = min(begin + ctx_len + 1, seq_len)
                inp = seq[begin : end - 1].unsqueeze(0).to(device_obj)
                tgt = seq[begin + 1 : end].to(device_obj)

                logits = model(inp).squeeze(0)

                # Only score tokens that are new in this window
                if begin > 0:
                    num_to_score = min(step, end - 1 - begin)
                    score_start = (end - 1 - begin) - num_to_score
                else:
                    score_start = 0

                loss = f.cross_entropy(
                    logits[score_start:],
                    tgt[score_start:],
                    reduction='sum',
                )
                total_nll += loss.item()
                total_tokens += tgt[score_start:].numel()

    ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()

    # Restore original training mode
    if was_training:
        model.train()

    return ppl
