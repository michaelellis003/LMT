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
r"""Adapter for EleutherAI's lm-evaluation-harness.

This module provides an adapter class that wraps any LMT model so it
can be evaluated using the ``lm-eval`` framework, giving access to
200+ benchmarks (HellaSwag, PIQA, ARC, LAMBADA, etc.).

Usage::

    from lmt.models.gpt import GPT
    from lmt.models.config import ModelConfigPresets
    from lmt.tokenizer import BPETokenizer
    from lmt.training.lm_eval_adapter import LMTEvalModel
    from lm_eval import simple_evaluate

    model = GPT(ModelConfigPresets.small_gpt())
    tokenizer = BPETokenizer()
    adapter = LMTEvalModel(model, tokenizer, device='cpu')

    results = simple_evaluate(
        model=adapter,
        tasks=['hellaswag', 'piqa'],
        num_fewshot=0,
    )

Requires the ``lm-eval`` package: ``uv sync --group eval``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.tokenizer.base import BaseTokenizer

try:
    from lm_eval.api.model import LM as _LM
    from lm_eval.api.registry import register_model as _register_model

    _LM_EVAL_AVAILABLE = True
except ImportError:
    _LM_EVAL_AVAILABLE = False
    _LM = object  # type: ignore[assignment,misc]

    def _register_model(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when lm-eval is not installed."""
        return lambda cls: cls


@_register_model('lmt')
class LMTEvalModel(_LM):  # type: ignore[misc]
    """Adapter bridging LMT models to the lm-eval-harness interface.

    Wraps any LMT model (GPT, LLaMA, Mamba, etc.) that returns logits
    of shape ``[batch, seq_len, vocab_size]`` and exposes the three
    methods required by ``lm_eval``: ``loglikelihood``,
    ``loglikelihood_rolling``, and ``generate_until``.

    Args:
        model: An LMT model (``nn.Module``).
        tokenizer: An LMT tokenizer with ``encode``/``decode`` methods.
        device: Device string (``'cpu'``, ``'cuda'``, ``'mps'``).
        batch_size: Batch size for evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        device: str = 'cpu',
        batch_size: int = 1,
    ):
        """Initialize the adapter.

        Args:
            model: An LMT model (``nn.Module``).
            tokenizer: An LMT tokenizer with ``encode``/``decode``.
            device: Device string.
            batch_size: Batch size for evaluation.
        """
        if not _LM_EVAL_AVAILABLE:
            raise ImportError(
                'lm-eval is required for LMTEvalModel. '
                'Install with: uv sync --group eval'
            )
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()
        self._batch_size = batch_size

        # Get context length from model config
        config = getattr(model, 'config', None)
        if config is not None and hasattr(config, 'context_length'):
            self._max_length = config.context_length
        else:
            self._max_length = 1024

    @property
    def eot_token_id(self) -> int:
        """End-of-text token ID."""
        # GPT-2 BPE uses 50256 as EOT
        tok = getattr(self._tokenizer, 'tokenizer', None)
        if tok is not None and hasattr(tok, 'eot_token'):
            return tok.eot_token
        return 50256

    @property
    def max_length(self) -> int:
        """Maximum context length of the model."""
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        """Maximum number of tokens to generate."""
        return 256

    @property
    def batch_size(self) -> int:
        """Batch size for evaluation."""
        return self._batch_size

    @property
    def device(self) -> torch.device:
        """Device the model is on."""
        return self._device

    def tok_encode(self, string: str) -> list[int]:
        """Tokenize a string."""
        return self._tokenizer.encode(string)

    def tok_decode(self, tokens: list[int]) -> str:
        """Decode token IDs to string."""
        return self._tokenizer.decode(tokens)

    def _model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits from the model.

        Args:
            input_ids: Shape ``[batch, seq_len]``.

        Returns:
            Logits of shape ``[batch, seq_len, vocab_size]``.
        """
        with torch.no_grad():
            return self._model(input_ids.to(self._device))

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Compute log-likelihood of continuations given contexts.

        For each request ``(context, continuation)``, tokenize both,
        concatenate, run through the model, and compute the log-prob
        of the continuation tokens.

        Returns:
            List of ``(log_likelihood, is_greedy)`` tuples.
        """
        results = []
        for req in requests:
            context, continuation = req.args

            ctx_ids = self.tok_encode(context) if context else []
            cont_ids = self.tok_encode(continuation)

            # Concatenate context + continuation
            all_ids = ctx_ids + cont_ids

            # Truncate from the left if too long
            if len(all_ids) > self._max_length:
                start = max(0, len(all_ids) - self._max_length)
                all_ids = all_ids[start:]
                cont_len = min(len(cont_ids), len(all_ids) - 1)
            else:
                cont_len = len(cont_ids)

            # Empty context: we can only score cont tokens
            # starting from the second one (need at least 1
            # token as input to predict the next)
            if len(ctx_ids) == 0 and cont_len == len(all_ids):
                cont_len = len(all_ids) - 1

            input_tensor = torch.tensor([all_ids], dtype=torch.long)
            logits = self._model_logits(input_tensor)

            # continuation starts at position (len(all_ids) - cont_len)
            # logits at position i predict token at position i+1
            start_pos = len(all_ids) - cont_len
            cont_logits = logits[0, start_pos - 1 : -1]  # [cont_len, V]
            cont_targets = torch.tensor(
                all_ids[start_pos:], dtype=torch.long
            ).to(self._device)

            log_probs = f.log_softmax(cont_logits.float(), dim=-1)
            token_log_probs = log_probs[torch.arange(cont_len), cont_targets]

            total_ll = token_log_probs.sum().item()

            # Check if greedy decoding would produce the continuation
            greedy_tokens = cont_logits.argmax(dim=-1)
            is_greedy = bool((greedy_tokens == cont_targets).all().item())

            results.append((total_ll, is_greedy))

        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        """Compute rolling log-likelihood for perplexity.

        For each request containing a full text string, compute the
        total log-likelihood using a sliding window over the model's
        context length.

        Returns:
            List of total log-likelihoods (negative values).
        """
        results = []
        for req in requests:
            (text,) = req.args
            token_ids = self.tok_encode(text)

            total_ll = 0.0

            # Sliding window with stride = max_length
            for i in range(0, len(token_ids) - 1, self._max_length):
                end = min(i + self._max_length, len(token_ids))
                chunk = token_ids[max(0, end - self._max_length) : end]

                input_tensor = torch.tensor([chunk], dtype=torch.long)
                logits = self._model_logits(input_tensor)

                # Score tokens: for the first chunk, start from
                # position 0; for subsequent chunks, only score
                # new tokens (non-overlapping part)
                if i == 0:
                    score_start = 0
                else:
                    overlap = len(chunk) - (end - i)
                    score_start = overlap

                targets = torch.tensor(
                    chunk[score_start + 1 :], dtype=torch.long
                ).to(self._device)
                pred_logits = logits[0, score_start:-1]  # [num_scored, V]

                if targets.numel() > 0:
                    log_probs = f.log_softmax(pred_logits.float(), dim=-1)
                    token_lls = log_probs[
                        torch.arange(targets.numel()), targets
                    ]
                    total_ll += token_lls.sum().item()

            results.append(total_ll)

        return results

    def generate_until(self, requests) -> list[str]:
        """Generate text until a stop sequence.

        For each request ``(context, gen_kwargs)``, generate tokens
        autoregressively until a stop string is produced or the
        maximum number of tokens is reached.

        Returns:
            List of generated continuation strings.
        """
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get('until', [])
            if isinstance(until, str):
                until = [until]
            max_gen = gen_kwargs.get('max_gen_toks', self.max_gen_toks)
            do_sample = gen_kwargs.get('do_sample', False)
            temperature = gen_kwargs.get('temperature', 1.0)

            ctx_ids = self.tok_encode(context) if context else []

            # Truncate context from the left
            if len(ctx_ids) > self._max_length - 1:
                ctx_ids = ctx_ids[-(self._max_length - 1) :]

            generated_ids = list(ctx_ids)

            for _ in range(max_gen):
                # Use last max_length tokens as input
                input_ids = generated_ids[-self._max_length :]
                input_tensor = torch.tensor([input_ids], dtype=torch.long)
                logits = self._model_logits(input_tensor)
                next_logits = logits[0, -1, :]

                if do_sample and temperature > 0:
                    probs = torch.softmax(
                        next_logits.float() / temperature, dim=-1
                    )
                    next_id = int(
                        torch.multinomial(probs, num_samples=1).item()
                    )
                else:
                    next_id = int(next_logits.argmax().item())

                generated_ids.append(next_id)

                # Check for EOT
                if next_id == self.eot_token_id:
                    break

                # Check for stop sequences
                decoded_so_far = self.tok_decode(generated_ids[len(ctx_ids) :])
                if any(s in decoded_so_far for s in until):
                    break

            # Return only the generated continuation
            continuation = self.tok_decode(generated_ids[len(ctx_ids) :])

            # Trim at the first stop sequence
            for stop in until:
                if stop in continuation:
                    continuation = continuation[: continuation.index(stop)]

            results.append(continuation)

        return results
